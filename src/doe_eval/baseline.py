from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .config import DISTANCES, EPS, PIXEL, WAVELENGTH, N
from .physics import fresnel_inverse_complex, fresnel_propagate_complex


def gerchberg_saxton_multiplane(
    targets: torch.Tensor,
    iters: int = 200,
    distances: Sequence[float] = DISTANCES,
    wavelength: float = WAVELENGTH,
    pixel: float = PIXEL,
    seed: int = 42,
) -> torch.Tensor:
    """
    Многоплоскостной алгоритм Герчберга–Сэкстона (GS) для 4 плоскостей.
    targets: [4, N, N] float32 в [0, 1].
    Возвращает фазу ДОЭ [N, N] в [-pi, pi].
    """
    device = targets.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if targets.shape != (len(distances), N, N):
        raise ValueError(f"Ожидались цели [{len(distances)}, {N}, {N}]")
    if iters < 1:
        raise ValueError("Число итераций GS должно быть положительным")

    phase = (torch.rand((N, N), generator=gen, device=device, dtype=torch.float32) * 2.0 - 1.0) * math.pi
    target_amp = targets.to(torch.float32).clamp_min(0.0)

    for _ in range(iters):
        field_z = fresnel_propagate_complex(phase, distances, wavelength, pixel)

        # Замена амплитуды на целевую с сохранением фазы в каждой плоскости наблюдения
        unit_field_z = field_z / torch.abs(field_z).clamp_min(EPS)
        updated_z = target_amp * unit_field_z

        # Обратный проход в плоскость ДОЭ
        back_h = fresnel_inverse_complex(updated_z, distances, wavelength, pixel)
        mean_field = back_h.mean(dim=0)
        phase = torch.angle(mean_field)

    return phase