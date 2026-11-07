from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import DISTANCES, PIXEL, WAVELENGTH, N


def fresnel_reconstruct_numpy(
    phase: np.ndarray,
    distances: Sequence[float] = DISTANCES,
    wavelength: float = WAVELENGTH,
    pixel: float = PIXEL,
) -> np.ndarray:
    """
    Независимая эталонная реализация дифракции Френеля на numpy (float64/complex128)
    через явные матрицы ДПФ (без использования np.fft и без импорта physics.py).
    """
    phase64 = np.asarray(phase, dtype=np.float64)
    if phase64.shape != (N, N):
        raise ValueError(f"Ожидалась фаза ({N}, {N}), получено {phase64.shape}")

    idx = np.arange(-N // 2, N // 2, dtype=np.float64)
    u = idx * pixel
    uu, vv = np.meshgrid(u, u, indexing="ij")
    u2_v2 = uu**2 + vv**2

    # Явная матрица центрированного ДПФ: W[p, m] = exp(-j * 2*pi * idx[p] * idx[m] / N)
    w_dft = np.exp(-1j * (2.0 * np.pi / N) * np.outer(idx, idx))

    h_doe = np.exp(1j * phase64)
    k = 2.0 * np.pi / wavelength

    outputs = []
    for z in distances:
        dx = wavelength * z / (N * pixel)
        x = idx * dx
        xx, yy = np.meshgrid(x, x, indexing="ij")
        x2_y2 = xx**2 + yy**2

        inner = h_doe * np.exp(1j * np.pi * u2_v2 / (wavelength * z))
        # Двумерное ДПФ через умножение матриц: W @ inner @ W^T
        dft_2d = w_dft @ inner @ w_dft.T

        prefactor = (np.exp(1j * k * z) / (1j * wavelength * z)) * np.exp(
            1j * np.pi * x2_y2 / (wavelength * z)
        )
        field = prefactor * dft_2d
        outputs.append(np.abs(field))

    return np.stack(outputs, axis=0)