from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from doe_eval.config import DISTANCES, PIXEL, WAVELENGTH, N
from doe_eval.demo import save_demo_inputs
from doe_eval.io import load_gray_u8_png, save_doe_png, save_recon_png
from doe_eval.metrics import (
    pearson,
    pearson_torch,
    ssim_global,
    ssim_global_torch,
    ssim_skimage,
)
from doe_eval.physics import (
    dequantize_phase_u8,
    fresnel_reconstruct,
    normalize_reconstructions,
    quantize_phase_u8,
)
from doe_eval.reference_numpy import fresnel_reconstruct_numpy


# 1. Сверка основной реализации (physics.py) и независимой (reference_numpy.py)
def test_physics_matches_reference_numpy() -> None:
    rng = np.random.default_rng(123)
    phase_np = rng.uniform(-math.pi, math.pi, size=(N, N)).astype(np.float32)

    ref_out = fresnel_reconstruct_numpy(phase_np)
    torch_out = fresnel_reconstruct(torch.from_numpy(phase_np)).numpy()

    rel_diff = np.linalg.norm(torch_out - ref_out) / np.linalg.norm(ref_out)
    assert rel_diff < 1e-4, f"Относительная разница {rel_diff:.2e} превышает 1e-4"


# 2. Физическая проверка собирающей линзой: phi = -pi*(u^2+v^2)/(lambda*f)
def test_fresnel_lens_focusing() -> None:
    focus_idx = 0
    f = DISTANCES[focus_idx]

    idx = np.arange(-N // 2, N // 2, dtype=np.float32)
    u = idx * PIXEL
    uu, vv = np.meshgrid(u, u, indexing="ij")
    raw_phase = -math.pi * (uu**2 + vv**2) / (WAVELENGTH * f)
    # Оборачивание фазы в [-pi, pi]
    wrapped_phase = ((raw_phase + math.pi) % (2.0 * math.pi)) - math.pi

    recon = fresnel_reconstruct(torch.from_numpy(wrapped_phase)).numpy()
    c = N // 2

    # Обоснование порогов:
    # 1) В плоскости фокуса (z0 = f = 24.1 мм) квадратичная фаза полностью компенсируется,
    #    давая дифракционный пик |K(c,c)| = N^2 / (lambda * f) ≈ 1.278e12.
    #    Отношение пика к среднему по кадру (peak-to-mean) близко к N^2 = 16384 (порог > 5000).
    # 2) В ближайшей соседней плоскости (z1 = 25.3 мм, dz = 1.2 мм) при апертуре L = 1.28 мм
    #    набег фазы на краю составляет ~1.52*pi. Интеграл Френеля даёт падение амплитуды |K|
    #    в 3.997 раза (интенсивности |K|^2 — в 16 раз), а отношение peak-to-mean падает
    #    более чем в 15 раз из-за размытия энергии в пятно. Поэтому порог по амплитуде >= 3.5,
    #    а по остроте фокуса (peak-to-mean) >= 10x относительно всех остальных плоскостей.
    focus_plane = recon[focus_idx]
    peak_focus = float(focus_plane[c, c])
    ratio_focus = peak_focus / float(focus_plane.mean())

    assert focus_plane.argmax() == c * N + c, "Максимум не в центре сетки (N//2, N//2)"
    assert ratio_focus > 5000.0, f"Недостаточно острый фокус: peak/mean = {ratio_focus:.1f}"

    for other_idx in range(1, len(DISTANCES)):
        other_plane = recon[other_idx]
        peak_other = float(other_plane.max())
        ratio_other = peak_other / float(other_plane.mean())

        assert peak_focus > 3.5 * peak_other, (
            f"Пик амплитуды в фокусе ({peak_focus:.2e}) должен быть минимум в 3.5 раза выше "
            f"расфокусированной плоскости {other_idx} ({peak_other:.2e})"
        )
        assert ratio_focus > 10.0 * ratio_other, (
            f"Острота пика (peak/mean) в фокусе ({ratio_focus:.1f}) должна быть минимум на порядок выше "
            f"расфокусированной плоскости {other_idx} ({ratio_other:.1f})"
        )

# 3. Проверка метрик: совпадение с numpy.corrcoef и skimage, единица на одинаковых картинках
def test_metrics_correctness() -> None:
    rng = np.random.default_rng(7)
    a = rng.uniform(0.0, 1.0, size=(N, N)).astype(np.float32)
    b = rng.uniform(0.0, 1.0, size=(N, N)).astype(np.float32)

    np_corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert math.isclose(pearson(a, b), np_corr, rel_tol=1e-6, abs_tol=1e-6)

    sk_val = float(structural_similarity(a.astype(np.float64), b.astype(np.float64), data_range=1.0))
    assert math.isclose(ssim_skimage(a, b), sk_val, rel_tol=1e-6, abs_tol=1e-6)

    assert math.isclose(pearson(a, a), 1.0, abs_tol=1e-6)
    assert math.isclose(ssim_global(a, a), 1.0, abs_tol=1e-6)
    assert math.isclose(ssim_skimage(a, a), 1.0, abs_tol=1e-6)


# 4. Отображение фаза <-> uint8: граничные точки и шаг квантования
def test_phase_quantization_roundtrip() -> None:
    endpoints = np.array([-math.pi, math.pi], dtype=np.float32)
    u8 = quantize_phase_u8(endpoints)
    assert int(u8[0]) == 0
    assert int(u8[1]) == 255

    deq = dequantize_phase_u8(u8)
    assert math.isclose(float(deq[0]), -math.pi, abs_tol=1e-6)
    assert math.isclose(float(deq[1]), math.pi, abs_tol=1e-6)

    rng = np.random.default_rng(99)
    phases = rng.uniform(-math.pi, math.pi, size=(N, N)).astype(np.float32)
    restored = dequantize_phase_u8(quantize_phase_u8(phases))
    half_step = 0.5 * (2.0 * math.pi / 255.0)
    max_err = float(np.max(np.abs(restored - phases)))
    assert max_err <= half_step + 1e-5, f"Ошибка квантования {max_err} > {half_step}"


# 5. Формат PNG: 128x128, режим L, 8 бит; понятные ошибки при неверном входе
def test_png_io_validation(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.png"
    save_recon_png(np.zeros((N, N), dtype=np.float32), valid_path)

    with Image.open(valid_path) as img:
        assert img.format == "PNG"
        assert img.mode == "L"
        assert img.size == (N, N)

    # Неверный режим (RGB вместо L)
    rgb_path = tmp_path / "bad_rgb.png"
    Image.new("RGB", (N, N)).save(rgb_path, format="PNG")
    with pytest.raises(ValueError, match="режим 'L'"):
        load_gray_u8_png(rgb_path)

    # Неверный размер (64x64 вместо 128x128)
    size_path = tmp_path / "bad_size.png"
    Image.new("L", (64, 64)).save(size_path, format="PNG")
    with pytest.raises(ValueError, match=f"({N}, {N})"):
        load_gray_u8_png(size_path)


# 6. Дифференцируемость: градиент по фазе через fresnel_reconstruct и торчевые метрики
def test_differentiability_and_gradients() -> None:
    torch.manual_seed(42)
    phase = torch.empty((2, N, N), dtype=torch.float32).uniform_(-math.pi, math.pi).requires_grad_(True)
    target = torch.rand((2, len(DISTANCES), N, N), dtype=torch.float32)

    recon = fresnel_reconstruct(phase)
    recon_norm = normalize_reconstructions(recon, norm="max", compare="amplitude")

    loss_p = 1.0 - pearson_torch(recon_norm, target).mean()
    loss_s = 1.0 - ssim_global_torch(recon_norm, target).mean()
    loss = loss_p + loss_s
    loss.backward()

    assert phase.grad is not None
    assert not torch.isnan(phase.grad).any(), "В градиенте обнаружены NaN"
    assert not torch.isinf(phase.grad).any(), "В градиенте обнаружены Inf"
    assert float(phase.grad.abs().sum()) > 0.0, "Градиент равен нулю"


# 7. Сквозной тест: baseline на синтетических картинках -> reconstruct с файлами лучше случайной фазы
def test_end_to_end_baseline_and_reconstruct(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    input_paths = save_demo_inputs(inputs_dir)

    # Случайный ДОЭ
    rng = np.random.default_rng(1234)
    rand_doe_path = tmp_path / "rand_doe.png"
    save_doe_png(rng.uniform(-math.pi, math.pi, size=(N, N)).astype(np.float32), rand_doe_path)

    rand_out = tmp_path / "rand_out"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "doe_eval.cli",
            "reconstruct",
            "--inputs",
            *[str(p) for p in input_paths],
            "--doe",
            str(rand_doe_path),
            "--out",
            str(rand_out),
            "--device",
            "cpu",
        ],
        check=True,
    )
    rand_metrics = json.loads((rand_out / "metrics.json").read_text(encoding="utf-8"))

    # Baseline GS (30 итераций для быстрого теста)
    base_out = tmp_path / "base_out"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "doe_eval.cli",
            "baseline",
            "--inputs",
            *[str(p) for p in input_paths],
            "--out",
            str(base_out),
            "--iters",
            "30",
            "--device",
            "cpu",
        ],
        check=True,
    )

    # Повторный вызов reconstruct по сохранённому baseline doe.png
    rec_out = tmp_path / "rec_out"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "doe_eval.cli",
            "reconstruct",
            "--inputs",
            *[str(p) for p in input_paths],
            "--doe",
            str(base_out / "doe.png"),
            "--out",
            str(rec_out),
            "--device",
            "cpu",
        ],
        check=True,
    )
    rec_metrics = json.loads((rec_out / "metrics.json").read_text(encoding="utf-8"))

    for i in range(1, 5):
        assert (rec_out / f"recon_{i}.png").is_file()

    assert rec_metrics["mean"]["pearson"] > rand_metrics["mean"]["pearson"] + 0.3
    assert rec_metrics["mean"]["ssim_global"] > rand_metrics["mean"]["ssim_global"] + 0.2