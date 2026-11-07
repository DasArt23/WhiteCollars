from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from .config import (
    DEFAULT_COMPARE,
    DEFAULT_NORM,
    DISTANCES,
    EPS,
    PIXEL,
    WAVELENGTH,
    N,
)

_FRESNEL_CACHE: dict[tuple[torch.device, tuple[float, ...], float, float], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_fresnel_kernels(
    device: torch.device,
    distances: Sequence[float] = DISTANCES,
    wavelength: float = WAVELENGTH,
    pixel: float = PIXEL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Кеширует неизменяемые комплексные ядра Френеля прямо в памяти GPU."""
    if len(distances) == 0 or any(not math.isfinite(float(z)) or z <= 0 for z in distances):
        raise ValueError("Расстояния должны быть конечными положительными числами")
    if not math.isfinite(wavelength) or not math.isfinite(pixel) or wavelength <= 0 or pixel <= 0:
        raise ValueError("Длина волны и шаг пикселя должны быть конечными и положительными")
    key = (device, tuple(float(d) for d in distances), float(wavelength), float(pixel))
    if key not in _FRESNEL_CACHE:
        # Угол kz достигает ~3e5 рад: float32 теряет точность комплексного поля.
        # Считаем константы один раз в float64, сам оператор работает в complex64.
        idx = torch.arange(-N // 2, N // 2, device=device, dtype=torch.float64)
        u = idx * pixel
        uu, vv = torch.meshgrid(u, u, indexing="ij")
        u2_v2 = (uu.square() + vv.square()).unsqueeze(0)  # [1, N, N]

        z = torch.tensor(distances, device=device, dtype=torch.float64).view(-1, 1, 1)
        k = 2.0 * math.pi / wavelength

        quad_in_angle = (math.pi / (wavelength * z)) * u2_v2
        quad_in = torch.polar(torch.ones_like(quad_in_angle), quad_in_angle).unsqueeze(0)  # [1, D, N, N]

        dx = wavelength * z / (N * pixel)
        idx2 = (idx.square().unsqueeze(1) + idx.square().unsqueeze(0)).unsqueeze(0)
        x2_y2 = idx2 * dx.square()

        quad_out_angle = k * z + (math.pi / (wavelength * z)) * x2_y2
        quad_out = torch.polar(torch.ones_like(quad_out_angle), quad_out_angle)
        prefactor = (quad_out / (1j * wavelength * z)).unsqueeze(0)  # [1, D, N, N]

        _FRESNEL_CACHE[key] = (
            quad_in.to(torch.complex64).contiguous(),
            prefactor.to(torch.complex64).contiguous(),
        )
    return _FRESNEL_CACHE[key]


def fresnel_propagate_complex(
    phase: torch.Tensor,
    distances: Sequence[float] = DISTANCES,
    wavelength: float = WAVELENGTH,
    pixel: float = PIXEL,
) -> torch.Tensor:
    unbatched = phase.ndim == 2
    phase_b = phase.unsqueeze(0) if unbatched else phase
    if phase_b.ndim != 3 or phase_b.shape[-2:] != (N, N):
        raise ValueError(f"Ожидался тензор [N,N] или [B,N,N] ({N}x{N}), получено {tuple(phase.shape)}")

    phase_f32 = phase_b.to(dtype=torch.float32)
    quad_in, prefactor = _get_fresnel_kernels(phase_f32.device, distances, wavelength, pixel)

    # H(u,v) = exp(j * phi)
    h_field = torch.polar(torch.ones_like(phase_f32), phase_f32).unsqueeze(1)  # [B, 1, N, N]
    inner = h_field * quad_in  # [B, 4, N, N]

    fft_term = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(inner, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    field = prefactor * fft_term
    return field.squeeze(0) if unbatched else field


def fresnel_inverse_complex(
    field: torch.Tensor,
    distances: Sequence[float] = DISTANCES,
    wavelength: float = WAVELENGTH,
    pixel: float = PIXEL,
) -> torch.Tensor:
    """Обратный оператор для каждой плоскости; сохраняет ось расстояний.

    Включает деление на полный префактор, а не только компенсацию его фазы.
    Это inverse, а не эрмитово-сопряжённый (adjoint) оператор.
    """
    if field.ndim not in (3, 4) or field.shape[-3:] != (len(distances), N, N):
        raise ValueError("Ожидалось комплексное поле [D,N,N] или [B,D,N,N]")
    unbatched = field.ndim == 3
    field_b = field.unsqueeze(0) if unbatched else field
    quad_in, prefactor = _get_fresnel_kernels(field.device, distances, wavelength, pixel)
    inner = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(field_b / prefactor, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    result = inner * quad_in.conj()
    return result.squeeze(0) if unbatched else result

def fresnel_reconstruct(
    phase: torch.Tensor,
    distances: Sequence[float] = DISTANCES,
    wavelength: float = WAVELENGTH,
    pixel: float = PIXEL,
) -> torch.Tensor:
    """
    Возвращает модуль |K(x,y)| на заданных расстояниях: [4, N, N] или [B, 4, N, N].
    Полностью дифференцируема по фазе.
    """
    field = fresnel_propagate_complex(phase, distances=distances, wavelength=wavelength, pixel=pixel)
    return torch.abs(field)


def normalize_reconstructions(
    recon: torch.Tensor,
    norm: str = DEFAULT_NORM,
    compare: str = DEFAULT_COMPARE,
) -> torch.Tensor:
    """
    Приводит восстановленные амплитуды к [0, 1] с учётом флагов --compare и --norm.
    """
    if compare == "intensity":
        val = recon.square()
    elif compare == "amplitude":
        val = recon
    else:
        raise ValueError(f"Неизвестный режим compare='{compare}'. Допустимо: amplitude | intensity")

    if norm == "max":
        denom = val.amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
        return val / denom
    if norm == "minmax":
        vmin = val.amin(dim=(-2, -1), keepdim=True)
        vmax = val.amax(dim=(-2, -1), keepdim=True)
        return (val - vmin) / (vmax - vmin).clamp_min(EPS)
    raise ValueError(f"Неизвестный режим norm='{norm}'. Допустимо: max | minmax")


def quantize_phase_u8(phase: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Отображение [-pi, pi] -> uint8 [0, 255]: u8 = round((phi + pi) / (2*pi) * 255)."""
    if isinstance(phase, np.ndarray):
        scaled = np.rint((phase + math.pi) / (2.0 * math.pi) * 255.0)
        return np.clip(scaled, 0, 255).astype(np.uint8)
    scaled = torch.round((phase + math.pi) / (2.0 * math.pi) * 255.0)
    return torch.clamp(scaled, 0.0, 255.0).to(torch.uint8)


def dequantize_phase_u8(u8: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Обратное отображение uint8 [0, 255] -> [-pi, pi]: phi = u8 / 255 * 2*pi - pi."""
    if isinstance(u8, np.ndarray):
        return (u8.astype(np.float32) / 255.0) * (2.0 * math.pi) - math.pi
    return (u8.to(torch.float32) / 255.0) * (2.0 * math.pi) - math.pi


def ste_quantize(phase: torch.Tensor) -> torch.Tensor:
    """Квантование фазы с прямым пропуском градиента (Straight-Through Estimator)."""
    wrapped = torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
    # Фаза сети уже в диапазоне. Повторное remainder сдвигает float32 на несколько
    # ulp и может поменять округление в PNG у границы шага квантования.
    phase_wrapped = torch.where((phase >= -math.pi) & (phase <= math.pi), phase, wrapped)
    q_phase = dequantize_phase_u8(quantize_phase_u8(phase_wrapped))
    return phase_wrapped + (q_phase - phase_wrapped).detach()
