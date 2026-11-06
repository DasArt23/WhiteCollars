from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .config import (
    DEFAULT_COMPARE,
    DEFAULT_NORM,
    DISTANCES,
    EPS,
    N,
    PIXEL,
    WAVELENGTH,
)


def fresnel_propagate_complex(
    phase: torch.Tensor,
    distances: Sequence[float] = DISTANCES,
    wavelength: float = WAVELENGTH,
    pixel: float = PIXEL,
) -> torch.Tensor:
    """
    Прямой расчёт дифракции Френеля с одним БПФ (возвращает комплексное поле K(x,y)).
    Вход: phase [N, N] или [B, N, N] (радианы).
    Выход: complex64 [4, N, N] или [B, 4, N, N].
    """
    unbatched = phase.ndim == 2
    if unbatched:
        phase_b = phase.unsqueeze(0)
    elif phase.ndim == 3:
        phase_b = phase
    else:
        raise ValueError(f"Ожидался тензор фазы [N,N] или [B,N,N], получено {tuple(phase.shape)}")

    _, height, width = phase_b.shape
    if height != N or width != N:
        raise ValueError(f"Размер фазы должен быть {N}x{N}, получено {height}x{width}")

    device = phase_b.device
    phase_f32 = phase_b.to(dtype=torch.float32)

    idx = torch.arange(-N // 2, N // 2, device=device, dtype=torch.float32)
    u = idx * pixel
    uu, vv = torch.meshgrid(u, u, indexing="ij")
    u2_v2 = (uu.square() + vv.square()).unsqueeze(0)  # [1, N, N]

    z = torch.tensor(distances, device=device, dtype=torch.float32).view(-1, 1, 1)  # [D, 1, 1]
    k = 2.0 * math.pi / wavelength

    # H(u,v) = exp(j * phi)
    h_field = torch.polar(torch.ones_like(phase_f32), phase_f32).unsqueeze(1)  # [B, 1, N, N]

    quad_in_angle = (math.pi / (wavelength * z)) * u2_v2  # [D, N, N]
    quad_in = torch.polar(torch.ones_like(quad_in_angle), quad_in_angle).unsqueeze(0)  # [1, D, N, N]

    inner = h_field * quad_in  # [B, D, N, N]
    fft_term = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(inner, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )

    # Шаг сетки в плоскости наблюдения: dx = lambda * z / (N * PIXEL)
    dx = wavelength * z / (N * pixel)  # [D, 1, 1]
    idx2 = (idx.square().unsqueeze(1) + idx.square().unsqueeze(0)).unsqueeze(0)  # [1, N, N]
    x2_y2 = idx2 * dx.square()  # [D, N, N]

    quad_out_angle = k * z + (math.pi / (wavelength * z)) * x2_y2
    quad_out = torch.polar(torch.ones_like(quad_out_angle), quad_out_angle)
    prefactor = quad_out / (1j * wavelength * z)  # [D, N, N]

    field = prefactor.unsqueeze(0) * fft_term.to(torch.complex64)
    return field.squeeze(0) if unbatched else field


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
    phase_wrapped = torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
    q_phase = dequantize_phase_u8(quantize_phase_u8(phase_wrapped))
    return phase_wrapped + (q_phase - phase_wrapped).detach()