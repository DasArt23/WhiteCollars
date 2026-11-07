from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

from .config import SSIM_K1, SSIM_K2


def amplitude_shape_error_torch(
    amplitude: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Ошибка формы после независимой RMS-нормировки, по каждой плоскости.

    Для ненулевых положительных изображений равна 2 - 2*cosine_similarity.
    Один яркий пик не может улучшить ошибку простым затемнением всего кадра,
    как при max-нормировке. Официальные метрики при этом не изменяются.
    """
    if amplitude.shape != target.shape:
        raise ValueError("Формы амплитуды и цели должны совпадать")
    tiny = torch.finfo(amplitude.dtype).tiny
    rms_a = amplitude.square().mean(dim=(-2, -1), keepdim=True).clamp_min(tiny).sqrt()
    rms_t = target.square().mean(dim=(-2, -1), keepdim=True).clamp_min(tiny).sqrt()
    return ((amplitude / rms_a) - (target / rms_t)).square().mean(dim=(-2, -1))


def plane_crosstalk_error_torch(
    recon: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Штраф сходства с чужой целью сильнее, чем со своей; возвращает [B,D]."""
    if recon.shape != target.shape or recon.ndim != 4:
        raise ValueError("Ожидались одинаковые тензоры [B,D,H,W]")
    tiny = torch.finfo(recon.dtype).tiny
    r, t = recon.flatten(-2), target.flatten(-2)
    r = r - r.mean(-1, keepdim=True)
    t = t - t.mean(-1, keepdim=True)
    r = r / r.square().sum(-1, keepdim=True).clamp_min(tiny).sqrt()
    t = t / t.square().sum(-1, keepdim=True).clamp_min(tiny).sqrt()
    correlation = r @ t.transpose(-1, -2)
    similarity = t @ t.transpose(-1, -2)
    # Одинаковые цели не требуется искусственно разделять между плоскостями.
    weights = (1 - similarity).clamp(0, 1)
    eye = torch.eye(target.shape[1], device=target.device, dtype=target.dtype)
    weights = weights * (1 - eye)
    correct = correlation.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    confusion = (correlation - correct).clamp_min(0).square()
    return (weights * confusion).sum(-1) / weights.sum(-1).clamp_min(1e-8)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Коэффициент корреляции Пирсона между двумя картинками в [0, 1]."""
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    if np.shape(a) != np.shape(b):
        raise ValueError("Формы изображений для Пирсона должны совпадать")
    a_diff = a64 - a64.mean()
    b_diff = b64 - b64.mean()
    denom = np.sqrt(np.sum(a_diff**2) * np.sum(b_diff**2))
    if denom == 0:
        return 1.0 if np.array_equal(a64, b64) else 0.0
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
    if a.shape != b.shape:
        raise ValueError("Формы изображений для Пирсона должны совпадать")
    a_flat = a.flatten(start_dim=-2)
    b_flat = b.flatten(start_dim=-2)
    a_centered = a_flat - a_flat.mean(dim=-1, keepdim=True)
    b_centered = b_flat - b_flat.mean(dim=-1, keepdim=True)
    num = (a_centered * b_centered).sum(dim=-1)
    energy = a_centered.square().sum(dim=-1) * b_centered.square().sum(dim=-1)
    # Единая с numpy договорённость для констант; clamp до sqrt исключает NaN в backward.
    den = torch.sqrt(energy.clamp_min(torch.finfo(a.dtype).tiny))
    identical = (a_flat == b_flat).all(dim=-1).to(a.dtype)
    return torch.where(energy > 0, num / den, identical)


def ssim_global_torch(
    a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0
) -> torch.Tensor:
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


def ssim_local_7x7_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 7,
) -> torch.Tensor:
    """Пакетный аналог skimage SSIM: равномерное окно, выборочная ковариация.

    Граничные окна исключаются, как в structural_similarity; результат [B,C].
    Константы C1/C2 уже стабилизируют знаменатель, дополнительный epsilon
    искажает SSIM на тёмном фоне (для двух нулевых окон должно получаться 1).
    """
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError("SSIM ожидает два одинаковых тензора [B,C,H,W]")
    if win_size < 3 or win_size % 2 != 1 or min(x.shape[-2:]) < win_size:
        raise ValueError(
            "Размер окна SSIM должен быть нечётным и помещаться в изображении"
        )
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    mu_x = F.avg_pool2d(x, win_size, stride=1)
    mu_y = F.avg_pool2d(y, win_size, stride=1)
    cov_norm = win_size**2 / (win_size**2 - 1)
    var_x = cov_norm * (F.avg_pool2d(x.square(), win_size, stride=1) - mu_x.square())
    var_y = cov_norm * (F.avg_pool2d(y.square(), win_size, stride=1) - mu_y.square())
    cov_xy = cov_norm * (F.avg_pool2d(x * y, win_size, stride=1) - mu_x * mu_y)
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return score.mean(dim=(-2, -1))
