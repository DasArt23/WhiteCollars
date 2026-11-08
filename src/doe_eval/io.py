from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import NUM_PLANES, N
from .physics import dequantize_phase_u8, quantize_phase_u8


def load_gray_u8_png(path: str | Path) -> np.ndarray:
    """Строгая загрузка 8-битного одноканального PNG (режим 'L') размера 128x128."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Файл не найден: {p}")

    with Image.open(p) as img:
        if img.format != "PNG":
            raise ValueError(f"{p}: ожидался формат PNG, получен {img.format}")
        if img.mode != "L":
            raise ValueError(
                f"{p}: ожидался одноканальный 8-битный режим 'L', получен режим '{img.mode}'"
            )
        if img.size != (N, N):
            raise ValueError(f"{p}: ожидался размер ({N}, {N}), получен {img.size}")
        arr = np.array(img, dtype=np.uint8)
    return arr


def load_target_png(path: str | Path) -> np.ndarray:
    """Загружает целевую картинку как float32 в [0, 1]."""
    return load_gray_u8_png(path).astype(np.float32) / 255.0


def load_four_targets(paths: Sequence[str | Path]) -> np.ndarray:
    """Загружает 4 целевые картинки в массив [4, N, N] float32 в [0, 1]."""
    if len(paths) != NUM_PLANES:
        raise ValueError(f"Ожидалось {NUM_PLANES} файлов, получено {len(paths)}")
    return np.stack([load_target_png(p) for p in paths], axis=0)


def save_u8_png(arr_u8: np.ndarray, path: str | Path) -> None:
    """Сохраняет uint8 массив [N, N] как 8-битный PNG режима 'L'."""
    if arr_u8.shape != (N, N):
        raise ValueError(f"Ожидался массив ({N}, {N}), получено {arr_u8.shape}")
    if arr_u8.dtype != np.uint8:
        raise ValueError("save_u8_png ожидает массив uint8")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr_u8)
    img.save(p, format="PNG")


def save_doe_png(phase: np.ndarray, path: str | Path) -> None:
    """Квантует фазу [-pi, pi] в uint8 [0, 255] и сохраняет в PNG."""
    if not np.isfinite(phase).all():
        raise ValueError("Фаза содержит NaN или Inf")
    u8 = quantize_phase_u8(np.asarray(phase, dtype=np.float32))
    save_u8_png(u8, path)


def load_doe_png(path: str | Path) -> np.ndarray:
    """Загружает ДОЭ из 8-битного PNG и переводит в фазу [-pi, pi] float32."""
    u8 = load_gray_u8_png(path)
    return dequantize_phase_u8(u8)


def save_recon_png(recon_01: np.ndarray, path: str | Path) -> None:
    """Сохраняет нормированную картинку [0, 1] в 8-битный PNG."""
    if not np.isfinite(recon_01).all():
        raise ValueError("Восстановление содержит NaN или Inf")
    clipped = np.clip(np.asarray(recon_01, dtype=np.float32), 0.0, 1.0)
    u8 = np.rint(clipped * 255.0).astype(np.uint8)
    save_u8_png(u8, path)


def save_comparison_png(targets: np.ndarray, recons: np.ndarray, path: str | Path) -> None:
    """Панель для визуальной проверки: цели сверху, восстановления снизу."""
    if targets.shape != (NUM_PLANES, N, N) or recons.shape != targets.shape:
        raise ValueError("Ожидались две четвёрки [4,N,N]")
    panel = Image.new("L", (NUM_PLANES * N, 2 * N + 50), 255)
    draw = ImageDraw.Draw(panel)
    for i, (target, recon) in enumerate(zip(targets, recons)):
        for image, offset, label in ((target, 20, "Target"), (recon, N + 50, "Reconstruction")):
            pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
            panel.paste(Image.fromarray(pixels), (i * N, offset))
            draw.text((i * N + 4, offset - 16), f"{label} {i + 1}", fill=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
