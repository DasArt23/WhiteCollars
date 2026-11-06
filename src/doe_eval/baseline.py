from __future__ import annotations

import math
from typing import Sequence

import torch

from .config import DISTANCES, EPS, N, PIXEL, WAVELENGTH


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

    idx = torch.arange(-N // 2, N // 2, device=device, dtype=torch.float32)
    u = idx * pixel
    uu, vv = torch.meshgrid(u, u, indexing="ij")
    u2_v2 = (uu.square() + vv.square()).unsqueeze(0)  # [1, N, N]

    z = torch.tensor(distances, device=device, dtype=torch.float32).view(-1, 1, 1)
    k = 2.0 * math.pi / wavelength
    dx = wavelength * z / (N * pixel)
    idx2 = (idx.square().unsqueeze(1) + idx.square().unsqueeze(0)).unsqueeze(0)
    x2_y2 = idx2 * dx.square()

    quad_in_angle = (math.pi / (wavelength * z)) * u2_v2
    quad_in = torch.polar(torch.ones_like(quad_in_angle), quad_in_angle)

    quad_out_angle = k * z + (math.pi / (wavelength * z)) * x2_y2
    quad_out = torch.polar(torch.ones_like(quad_out_angle), quad_out_angle)
    prefactor = quad_out / (1j * wavelength * z)

    phase = (torch.rand((N, N), generator=gen, device=device, dtype=torch.float32) * 2.0 - 1.0) * math.pi
    target_amp = targets.to(torch.float32).clamp_min(0.0)

    for _ in range(iters):
        h_doe = torch.polar(torch.ones_like(phase), phase).unsqueeze(0)  # [1, N, N]
        inner = h_doe * quad_in
        fft_term = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(inner, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        field_z = prefactor * fft_term

        # Замена амплитуды на целевую с сохранением фазы в каждой плоскости наблюдения
        unit_field_z = field_z / torch.abs(field_z).clamp_min(EPS)
        updated_z = target_amp * unit_field_z

        # Обратный проход в плоскость ДОЭ
        inv_fft_input = updated_z / prefactor
        back_inner = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(inv_fft_input, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        back_h = back_inner * torch.conj(quad_in)
        mean_field = back_h.mean(dim=0)
        phase = torch.angle(mean_field)

    return phase