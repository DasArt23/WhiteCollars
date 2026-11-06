from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

from .config import EPS, SSIM_K1, SSIM_K2


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Коэффициент корреляции Пирсона между двумя картинками в [0, 1]."""
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    a_diff = a64 - a64.mean()
    b_diff = b64 - b64.mean()
    denom = np.sqrt(np.sum(a_diff**2) * np.sum(b_diff**2))
    if denom < 1e-15:
        return 1.0 if np.allclose(a64, b64) else 0.0
    return float(np.sum(a_diff * b_diff) / denom)


def ssim_global(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Глобальный SSIM по формуле из ТЗ (без скользящего окна)."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    mu_a = float(np.mean(a64))
    mu_b = float(np.mean(b64))
    sigma_a2 = float(np.mean((a64 - mu_a) ** 2))
    sigma_b2 = float(np.mean((b64 - mu_b) ** 2))
    sigma_ab = float(np.mean((a64 - mu_a) * (b64 - mu_b)))

    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2

    num = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return float(num / den)


def ssim_skimage(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Стандартный оконный SSIM из scikit-image."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    return float(structural_similarity(a64, b64, data_range=data_range))


def pearson_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Дифференцируемый пакетный Пирсон по последним двум осям (..., H, W)."""
    a_flat = a.flatten(start_dim=-2)
    b_flat = b.flatten(start_dim=-2)
    a_centered = a_flat - a_flat.mean(dim=-1, keepdim=True)
    b_centered = b_flat - b_flat.mean(dim=-1, keepdim=True)
    num = (a_centered * b_centered).sum(dim=-1)
    den = torch.sqrt((a_centered.square().sum(dim=-1) * b_centered.square().sum(dim=-1)).clamp_min(EPS))
    return num / den


def ssim_global_torch(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Дифференцируемый пакетный глобальный SSIM по последним двум осям (..., H, W)."""
    a_flat = a.flatten(start_dim=-2)
    b_flat = b.flatten(start_dim=-2)
    mu_a = a_flat.mean(dim=-1)
    mu_b = b_flat.mean(dim=-1)
    da = a_flat - mu_a.unsqueeze(-1)
    db = b_flat - mu_b.unsqueeze(-1)
    sigma_a2 = da.square().mean(dim=-1)
    sigma_b2 = db.square().mean(dim=-1)
    sigma_ab = (da * db).mean(dim=-1)

    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2

    num = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    den = (mu_a.square() + mu_b.square() + c1) * (sigma_a2 + sigma_b2 + c2)
    return num / den