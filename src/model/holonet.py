from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from doe_eval import pearson_torch, ssim_global_torch
from doe_eval.config import DISTANCES, PIXEL, WAVELENGTH, N
from doe_eval.metrics import (
    amplitude_shape_error_torch,
    plane_crosstalk_error_torch,
    ssim_local_7x7_torch,
)
from doe_eval.physics import (
    fresnel_inverse_complex,
    fresnel_propagate_complex,
    fresnel_reconstruct,
    normalize_reconstructions,
    ste_quantize,
)

# Вспомогательные веса; главный член loss — ошибка формы амплитуды.
METRIC_WEIGHTS = (0.20, 0.10, 0.30)  # Pearson, global SSIM, local SSIM
LOSS_VERSION = "visual-fit-v2"


def fresnel_backward_to_slm(
    target_field: torch.Tensor,
    plane_weights: torch.Tensor | None = None,
    distances: Sequence[float] = DISTANCES,
    wavelength: float = WAVELENGTH,
    pixel: float = PIXEL,
    equalize_power: bool = False,
) -> torch.Tensor:
    """Обратное распространение каждой плоскости и взвешенная сумма в ДОЭ.

    Полный inverse включает 1/prefactor, поэтому масштабирование плоскостей
    согласовано с классическим GS и основным оператором Френеля.
    """
    if equalize_power:
        # При единичной амплитуде ДОЭ RMS выходного поля равен N/(lambda*z).
        # Без этого маленький объект получает меньшую энергию, чем большая фигура.
        rms = (
            target_field.abs()
            .square()
            .mean((-2, -1), keepdim=True)
            .clamp_min(1e-16)
            .sqrt()
        )
        z = torch.as_tensor(
            distances, device=target_field.device, dtype=torch.float32
        ).view(-1, 1, 1)
        target_field = (target_field / rms) * (N / (wavelength * z))
    per_plane_slm = fresnel_inverse_complex(target_field, distances, wavelength, pixel)

    if plane_weights is not None:
        w = F.softmax(plane_weights, dim=0).view(1, -1, 1, 1)
        return (per_plane_slm * w).sum(dim=1)
    return per_plane_slm.mean(dim=1)


class SpectralConv2d(nn.Module):
    """Глобальный спектральный слой Fourier Neural Operator (рецептивное поле 128x128)."""

    def __init__(self, channels: int, modes: int = 24) -> None:
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            x_f32 = x.float()
            _, _, h, w = x_f32.shape
            x_ft = torch.fft.rfft2(x_f32, norm="ortho")
            out_ft = torch.zeros_like(x_ft)
            m = min(self.modes, h, x_ft.shape[-1])
            w_c = torch.complex(
                self.weights_real[:, :, :m, :m], self.weights_imag[:, :, :m, :m]
            )
            out_ft[:, :, :m, :m] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :m, :m], w_c
            )
            out = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return out.to(orig_dtype)


class ConvNeXtSEBlock(nn.Module):
    """Блок ConvNeXt (7x7 depthwise) + Squeeze-and-Excitation внимание между плоскостями."""

    def __init__(self, dim: int, expansion: int = 3) -> None:
        super().__init__()
        hidden = dim * expansion
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        self.pwconv1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden, dim, kernel_size=1)

        se_hidden = max(8, dim // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, se_hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(1e-2 * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * self.se(x)
        return residual + self.gamma * x


class TargetPlaneCrosstalkGate(nn.Module):
    """
    Модуль подавления осевого кросс-тока в пространстве 4 целевых плоскостей (Object Domain).
    Принимает [targets(4), recon(4), cos_theta(4), sin_theta(4)] = 16 каналов,
    предсказывает скорректированную целевую амплитуду (с подавлением фона) и поправку фазы плоскости.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            ConvNeXtSEBlock(hidden, expansion=2),
            nn.Conv2d(hidden, 8, kernel_size=1),  # 4 amp_gain + 4 phase_delta
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        targets: torch.Tensor,
        recon_norm: torch.Tensor,
        plane_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat(
            [targets, recon_norm, torch.cos(plane_phase), torch.sin(plane_phase)],
            dim=1,
        )
        out = self.net(inp)
        amp_delta, phase_delta = out[:, :4].float(), out[:, 4:].float()

        # В светлых зонах усиливаем контраст, в тёмных зонах (targets == 0) жёстко гасим паразитное эхо
        fg_mask = (targets > 0.03).float()
        boosted_amp = (
            targets * (1.0 + 0.4 * torch.tanh(amp_delta))
            + 0.25 * (targets - recon_norm) * fg_mask
        )
        eff_amp = torch.clamp(boosted_amp, min=0.0)
        eff_phase = plane_phase + 0.35 * torch.tanh(phase_delta)
        return eff_amp, eff_phase


class FourierUNetRefiner(nn.Module):
    """Мультимасштабный Fourier-ConvNeXt U-Net рефайнер в плоскости ДОЭ."""

    def __init__(self, in_channels: int, base_dim: int = 56) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
            ConvNeXtSEBlock(base_dim),
            ConvNeXtSEBlock(base_dim),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=2, stride=2),
            nn.GroupNorm(8, base_dim * 2),
            nn.GELU(),
            ConvNeXtSEBlock(base_dim * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=2, stride=2),
            nn.GroupNorm(8, base_dim * 4),
            nn.GELU(),
            ConvNeXtSEBlock(base_dim * 4),
        )

        self.spectral = SpectralConv2d(base_dim * 4, modes=16)
        self.bottleneck = nn.Sequential(
            ConvNeXtSEBlock(base_dim * 4),
            ConvNeXtSEBlock(base_dim * 4),
        )

        self.up2 = nn.ConvTranspose2d(
            base_dim * 4, base_dim * 2, kernel_size=2, stride=2
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dim * 2),
            nn.GELU(),
            ConvNeXtSEBlock(base_dim * 2),
        )
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
            ConvNeXtSEBlock(base_dim),
            ConvNeXtSEBlock(base_dim),
        )
        self.head = nn.Conv2d(base_dim, 2, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        b = self.bottleneck(e3 + self.spectral(e3))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class HoloUnrollNet(nn.Module):
    """
    Экспериментальная архитектура: Dual-Domain Deep Unrolled Fourier-ConvNeXt Network.
    Работает одновременно в пространстве 4 плоскостей объектов (гашение кросс-тока и спекла)
    и в частотной плоскости ДОЭ (глобальный FNO + фазорная коррекция на единичной окружности).
    """

    def __init__(
        self,
        unroll_steps: int = 8,
        base_dim: int = 56,
        equalize_plane_power: bool = True,
        phase_carrier_seed: int | None = None,
    ) -> None:
        super().__init__()
        if unroll_steps < 1 or base_dim < 8 or base_dim % 8:
            raise ValueError(
                "unroll_steps >= 1; base_dim должен быть кратен 8 и не меньше 8"
            )
        self.unroll_steps = unroll_steps
        self.base_dim = base_dim
        self.equalize_plane_power = equalize_plane_power
        self.phase_carrier_seed = phase_carrier_seed
        if phase_carrier_seed is not None:
            generator = torch.Generator().manual_seed(phase_carrier_seed)
            carrier = (2 * torch.rand(4, N, N, generator=generator) - 1) * torch.pi
            self.register_buffer("phase_carrier", carrier, persistent=False)

        # 1. Инициализация гладкой фазы на 4 целевых плоскостях
        self.target_phase_init = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ConvNeXtSEBlock(32),
            ConvNeXtSEBlock(32),
            nn.Conv2d(32, 8, kernel_size=1),
        )

        self.plane_logits = nn.Parameter(torch.zeros(unroll_steps + 1, 4))
        self.step_alphas = nn.Parameter(torch.full((unroll_steps,), 0.70))

        # 2. Двухдоменные блоки внутри каждой развёрнутой итерации:
        #    - obj_gates: фильтруют кросс-ток на 4 плоскостях объектов
        #    - slm_blocks: фильтруют фазовый фронт непосредственно на самом ДОЭ
        self.obj_gates = nn.ModuleList(
            [TargetPlaneCrosstalkGate(hidden=32) for _ in range(unroll_steps)]
        )
        self.slm_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(6, 24, kernel_size=3, padding=1),
                    nn.GroupNorm(4, 24),
                    nn.GELU(),
                    ConvNeXtSEBlock(24, expansion=2),
                    nn.Conv2d(24, 2, kernel_size=1),
                )
                for _ in range(unroll_steps)
            ]
        )
        for block in self.slm_blocks:
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

        # 3. Финальный глобальный Fourier-ConvNeXt U-Net рефайнер
        self.refiner = FourierUNetRefiner(in_channels=12, base_dim=base_dim)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        bsz = targets.shape[0]

        # Шаг 1: Предсказание начальной фазы в плоскостях z1..z4 и перенос в z=0
        init_raw = self.target_phase_init(targets).view(bsz, 4, 2, N, N)
        if self.phase_carrier_seed is None:
            init_angle = torch.atan2(
                init_raw[:, :, 1].float(), init_raw[:, :, 0].float()
            )
        else:
            # atan2 двух близких к нулю выходов CNN даёт неограниченный градиент.
            # Фиксированная фаза распределяет поле по апертуре, CNN учит поправку.
            correction = (init_raw[:, :, 1].float() - init_raw[:, :, 0].float()) / (
                2.0**0.5
            )
            init_angle = self.phase_carrier + 0.25 * torch.tanh(correction)

        with torch.amp.autocast("cuda", enabled=False):
            targets_f32 = targets.float()
            target_field = torch.polar(targets_f32, init_angle.float())
            slm_field = fresnel_backward_to_slm(
                target_field,
                self.plane_logits[0],
                equalize_power=self.equalize_plane_power,
            )
            phase = slm_field.angle()

        # Шаг 2: K развёрнутых двухдоменных итераций (Object Domain + SLM Domain)
        for k in range(self.unroll_steps):
            with torch.amp.autocast("cuda", enabled=False):
                prop_field = fresnel_propagate_complex(phase)
                prop_norm = normalize_reconstructions(
                    prop_field.abs(), norm="max", compare="amplitude"
                )
                plane_phase = prop_field.angle()

            # 2a. Подавление осевого кросс-тока в пространстве 4 целевых плоскостей
            eff_amp, eff_phase = self.obj_gates[k](
                targets,
                prop_norm.to(targets.dtype),
                plane_phase.to(targets.dtype),
            )

            with torch.amp.autocast("cuda", enabled=False):
                updated_target_field = torch.polar(eff_amp.float(), eff_phase.float())
                back_field = fresnel_backward_to_slm(
                    updated_target_field,
                    self.plane_logits[k + 1],
                    equalize_power=self.equalize_plane_power,
                )
                back_angle = back_field.angle()

                alpha = torch.sigmoid(self.step_alphas[k])
                mixed_re = (1.0 - alpha) * torch.cos(phase) + alpha * torch.cos(
                    back_angle
                )
                mixed_im = (1.0 - alpha) * torch.sin(phase) + alpha * torch.sin(
                    back_angle
                )

            # 2b. Нейросетевая коррекция фазора в плоскости ДОЭ
            step_in = torch.cat(
                [
                    mixed_re.unsqueeze(1),
                    mixed_im.unsqueeze(1),
                    torch.cos(back_angle).unsqueeze(1),
                    torch.sin(back_angle).unsqueeze(1),
                    targets.mean(dim=1, keepdim=True),
                    (targets - prop_norm.to(targets.dtype))
                    .abs()
                    .mean(dim=1, keepdim=True),
                ],
                dim=1,
            ).to(targets.dtype)

            delta = self.slm_blocks[k](step_in).float()
            u_re = mixed_re + 0.25 * torch.tanh(delta[:, 0])
            u_im = mixed_im + 0.25 * torch.tanh(delta[:, 1])
            phase = torch.atan2(u_im, u_re)

        # Шаг 3: Финальное уточнение через Fourier-ConvNeXt U-Net
        with torch.amp.autocast("cuda", enabled=False):
            final_prop = fresnel_propagate_complex(phase)
            final_recon = normalize_reconstructions(
                final_prop.abs(), norm="max", compare="amplitude"
            )
            recon_err = targets_f32 - final_recon
            cos_p, sin_p = torch.cos(phase), torch.sin(phase)
            back_re, back_im = torch.cos(back_angle), torch.sin(back_angle)

        refiner_in = torch.cat(
            [
                cos_p.unsqueeze(1),
                sin_p.unsqueeze(1),
                back_re.unsqueeze(1),
                back_im.unsqueeze(1),
                targets_f32,
                recon_err,
            ],
            dim=1,
        ).to(targets.dtype)

        ref_delta = self.refiner(refiner_in).float()
        if self.phase_carrier_seed is not None:
            # Радиус поправки <= sqrt(2)/2 < 1: итоговый фазор не обнуляется.
            ref_delta = torch.tanh(ref_delta)
        final_re = cos_p + 0.5 * ref_delta[:, 0]
        final_im = sin_p + 0.5 * ref_delta[:, 1]
        return torch.atan2(final_im, final_re)


def _image_objective(
    recon_amp: torch.Tensor,
    targets: torch.Tensor,
    w_pearson: float = METRIC_WEIGHTS[0],
    w_ssim_global: float = METRIC_WEIGHTS[1],
    w_ssim_local: float = METRIC_WEIGHTS[2],
    w_bg_suppress: float = 0.0,
    w_worst_plane: float = 0.15,
):
    """Общий критерий обучения и уточнения; форма важнее совпадения чёрного фона."""
    recon = normalize_reconstructions(recon_amp)
    shape = amplitude_shape_error_torch(recon_amp, targets)
    p = pearson_torch(recon, targets)
    sg = ssim_global_torch(recon, targets)
    sl = ssim_local_7x7_torch(recon, targets)
    foreground = targets.square()
    foreground_error = ((recon - targets).square() * foreground).sum(
        (-2, -1)
    ) / foreground.sum((-2, -1)).clamp_min(1e-8)
    crosstalk = plane_crosstalk_error_torch(recon, targets)
    visual = shape + 1.5 * foreground_error + 10.0 * crosstalk
    loss = (
        visual
        + w_pearson * (1 - p)
        + w_ssim_global * (1 - sg)
        + w_ssim_local * (1 - sl)
    ).mean()
    loss = loss + w_worst_plane * shape.amax(dim=-1).mean()
    if w_bg_suppress:
        bg = (targets < 0.02).float()
        loss = (
            loss
            + w_bg_suppress
            * (
                (recon.square() * bg).sum((-2, -1)) / bg.sum((-2, -1)).clamp_min(1)
            ).mean()
        )
    planes = {
        "pearson": p,
        "ssim_global": sg,
        "ssim_local": sl,
        "shape_error": shape,
        "foreground_error": foreground_error,
        "crosstalk": crosstalk,
        "visual_error": visual,
    }
    return loss, recon, planes


def compute_sota_physics_loss(
    pred_phase: torch.Tensor,
    targets: torch.Tensor,
    w_pearson: float = METRIC_WEIGHTS[0],
    w_ssim_global: float = METRIC_WEIGHTS[1],
    w_ssim_local: float = METRIC_WEIGHTS[2],
    w_bg_suppress: float = 0.0,
    w_worst_plane: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Квантованная физика: RMS-ошибка формы и вспомогательные метрики.

    RMS-нормировка применяется только в обучающем критерии. Выходные изображения
    и официальные метрики по-прежнему используют amplitude/max из ТЗ проекта.
    """
    with torch.amp.autocast("cuda", enabled=False):
        amplitude = fresnel_reconstruct(ste_quantize(pred_phase.float()))
        loss, recon, planes = _image_objective(
            amplitude,
            targets.float(),
            w_pearson,
            w_ssim_global,
            w_ssim_local,
            w_bg_suppress,
            w_worst_plane,
        )
    metrics = {name: value.mean().detach() for name, value in planes.items()}
    metrics["loss"] = loss.detach()
    return loss, recon.detach(), metrics


def hybrid_infer_phase(
    model: nn.Module,
    targets: torch.Tensor,
    polish_steps: int = 20,
    polish_lr: float = 0.06,
) -> torch.Tensor:
    """Прямой проход сети и необязательное уточнение квантованной фазы.

    Выбирает лучший кандидат отдельно для каждого примера, включая исходный
    прогноз и результат последнего шага. Скорость измеряется на целевом GPU.
    """
    if polish_steps < 0 or polish_lr <= 0:
        raise ValueError("polish_steps >= 0 и polish_lr > 0")
    model.eval()
    with torch.no_grad():
        init_phase = model(targets)
        if init_phase.ndim == 4 and init_phase.shape[1] == 1:
            init_phase = init_phase.squeeze(1)

    if polish_steps <= 0:
        return init_phase

    with torch.enable_grad():
        targets_f32 = targets.float()

        phase = init_phase.detach().clone().float().requires_grad_(True)
        optimizer = torch.optim.Adam([phase], lr=polish_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=polish_steps, eta_min=polish_lr * 0.15
        )

        best_phase = phase.detach().clone()
        best_score = torch.full((targets.shape[0],), -1e9, device=targets.device)
        best_ssim = torch.full_like(best_score, -1e9)

        for step in range(polish_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            wrapped = torch.atan2(torch.sin(phase), torch.cos(phase))
            q_phase = ste_quantize(wrapped)
            recon_amp = fresnel_reconstruct(q_phase)
            loss, _, planes = _image_objective(recon_amp, targets_f32)
            quality_scores = -planes["visual_error"].mean(dim=1)
            sl_val = planes["ssim_local"].mean(dim=1)

            with torch.no_grad():
                improved = (quality_scores > best_score) | (
                    (quality_scores == best_score) & (sl_val > best_ssim)
                )
                best_score = torch.where(improved, quality_scores, best_score)
                best_ssim = torch.where(improved, sl_val, best_ssim)
                best_phase[improved] = wrapped.detach()[improved]

            if step == polish_steps:
                break

            loss.backward()
            optimizer.step()
            scheduler.step()

    return best_phase
