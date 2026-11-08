from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import IMAGE_FILENAMES, N
from .io import save_recon_png


def generate_synthetic_targets() -> np.ndarray:
    """Генерирует 4 синтетические картинки [4, N, N] в [0, 1]: круг, квадрат, полосы, градиент."""
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, N), np.linspace(-1.0, 1.0, N), indexing="ij")

    # 1. Круг
    img1 = ((xx**2 + yy**2) <= 0.45**2).astype(np.float32)

    # 2. Квадрат (рамка)
    outer = (np.abs(xx) <= 0.55) & (np.abs(yy) <= 0.55)
    inner = (np.abs(xx) <= 0.25) & (np.abs(yy) <= 0.25)
    img2 = (outer & ~inner).astype(np.float32)

    # 3. Полосы
    img3 = (0.5 * (1.0 + np.cos(4.0 * np.pi * (xx + 0.5 * yy)))).astype(np.float32)

    # 4. Плавный градиент с гауссовой огибающей
    img4 = (((xx + 1.0) * 0.5) * np.exp(-1.5 * (xx**2 + yy**2))).astype(np.float32)
    img4 /= max(float(img4.max()), 1e-8)

    return np.stack([img1, img2, img3, img4], axis=0)


def save_demo_inputs(out_dir: str | Path) -> list[Path]:
    """Сохраняет 4 демо-картинки в указанную директорию и возвращает список путей."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    targets = generate_synthetic_targets()
    paths: list[Path] = []
    for img, fname in zip(targets, IMAGE_FILENAMES):
        p = base / fname
        save_recon_png(img, p)
        paths.append(p)
    return paths