from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from doe_eval.config import DEFAULT_MANIFEST
from model import (
    ManifestQuartetDataset,
    build_dataloaders,
    export_split_predictions,
    forward_physics_loss,
    load_manifest,
)


def _make_valid_target_png(path: Path, seed: int) -> None:
    """Создаёт валидный 8-битный PNG 128x128 режима 'L' с чёрным фоном и светлой фигурой."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("L", (128, 128), 0)
    draw = ImageDraw.Draw(im)
    offset = (seed * 7) % 40
    draw.ellipse(
        (24 + offset, 24 + offset, 70 + offset, 70 + offset), fill=200 + (seed % 55)
    )
    im.save(path, format="PNG")


@pytest.fixture()
def mock_anya_dataset(tmp_path: Path) -> Path:
    """Собирает мини-датасет по контракту doe-manifest/1 (1 diag, 10 train, 2 val, 2 test)."""
    data_dir = tmp_path / "data"
    distances = [24.1, 25.3, 26.5, 27.7]
    samples = []
    seed_counter = 1

    # 1. Diagnostic
    diag_imgs = []
    for c in range(1, 5):
        rel = f"diagnostic/0000/{c:02d}.png"
        _make_valid_target_png(data_dir / rel, seed_counter)
        seed_counter += 1
        diag_imgs.append(rel)
    samples.append(
        {
            "sample_id": "diagnostic_0000",
            "split": "diagnostic",
            "images": diag_imgs,
            "distances_mm": distances,
            "seed": 0,
            "generator_version": "0.2.0",
            "params": {"purpose": "fixed_A1_debug_only"},
        }
    )

    # 2. Train (10), Validation (2), Test (2)
    for split, count in [("train", 10), ("validation", 2), ("test", 2)]:
        for idx in range(1, count + 1):
            imgs = []
            for c in range(1, 5):
                rel = f"{split}/{idx:06d}/{c:02d}.png"
                _make_valid_target_png(data_dir / rel, seed_counter)
                seed_counter += 1
                imgs.append(rel)
            samples.append(
                {
                    "sample_id": f"{split}_{idx:06d}",
                    "split": split,
                    "images": imgs,
                    "distances_mm": distances,
                    "seed": idx,
                    "generator_version": "0.2.0",
                    "params": {"targets": []},
                }
            )

    manifest = {
        "format": "doe-manifest/1",
        "dataset_name": "procedural_mixed_targets",
        "dataset_version": "v2-seed3701-train10-val2-test2",
        "generator_version": "0.2.0",
        "distances_mm": distances,
        "master_seed": 3701,
        "target_interpretation": "normalized amplitude = png/255",
        "samples": samples,
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class DummyDOENet(torch.nn.Module):
    """Минимальная тестовая нейросеть: [B, 4, 128, 128] -> [B, 128, 128] в [-pi, pi]."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.pi * torch.tanh(self.conv(x).squeeze(1))


# 1. Проверка контракта load_manifest (совместимость с однострочником Ани/Максима)
def test_load_manifest_contract(mock_anya_dataset: Path) -> None:
    d = load_manifest(mock_anya_dataset)
    assert d.report.counts == {"diagnostic": 1, "train": 10, "validation": 2, "test": 2}
    assert d.report.total_images == 15 * 4
    assert d.report.warnings == []
    assert len(d.by_split("train")) == 10


# 2. Проверка отлова ошибок при нарушении контракта манифеста
def test_load_manifest_rejects_invalid_data(mock_anya_dataset: Path) -> None:
    raw = json.loads(mock_anya_dataset.read_text(encoding="utf-8"))

    # Попытка подсунуть картинку из previews/
    bad_preview = dict(raw)
    bad_sample = dict(raw["samples"][0])
    bad_sample["images"] = [
        "previews/contact_sheet_01.png",
        *raw["samples"][0]["images"][1:],
    ]
    bad_preview["samples"] = [bad_sample]
    bad_path = mock_anya_dataset.parent / "bad_manifest.json"
    bad_path.write_text(json.dumps(bad_preview), encoding="utf-8")

    with pytest.raises(ValueError, match="previews/"):
        load_manifest(bad_path)


# 3. Проверка режимов train_smoke (8 четвёрок) и v1 (полный сплит)
def test_dataloaders_smoke_and_v1(mock_anya_dataset: Path) -> None:
    smoke_loaders = build_dataloaders(
        mock_anya_dataset, batch_size=4, num_workers=0, mode="train_smoke"
    )
    assert len(smoke_loaders["train"].dataset) == 8

    v1_loaders = build_dataloaders(
        mock_anya_dataset, batch_size=4, num_workers=0, mode="v1"
    )
    assert len(v1_loaders["train"].dataset) == 10
    assert len(v1_loaders["validation"].dataset) == 2
    assert len(v1_loaders["test"].dataset) == 2
    assert len(v1_loaders["diagnostic"].dataset) == 1

    batch_targets, batch_ids = next(iter(smoke_loaders["train"]))
    assert batch_targets.shape == (4, 4, 128, 128)
    assert batch_targets.dtype == torch.float32
    assert 0.0 <= float(batch_targets.min()) and float(batch_targets.max()) <= 1.0
    assert len(batch_ids) == 4


# 4. Проверка правил аугментации: запрещены на val/test, не меняют файлы на диске
def test_augmentations_safety(mock_anya_dataset: Path) -> None:
    with pytest.raises(ValueError, match="ТОЛЬКО для split='train'"):
        ManifestQuartetDataset(
            mock_anya_dataset, split="validation", augment_channel_shuffle=True
        )

    manifest_hash_before = hashlib.sha256(mock_anya_dataset.read_bytes()).hexdigest()
    ds = ManifestQuartetDataset(
        mock_anya_dataset,
        split="train",
        augment_channel_shuffle=True,
        augment_dihedral=True,
    )
    img_path = mock_anya_dataset.parent / ds.samples[0].images[0]
    img_hash_before = hashlib.sha256(img_path.read_bytes()).hexdigest()

    tensor, sid = ds[0]
    assert tensor.shape == (4, 128, 128)
    assert sid.startswith("train_")

    # Файлы на диске и манифест должны остаться нетронутыми
    assert (
        hashlib.sha256(mock_anya_dataset.read_bytes()).hexdigest()
        == manifest_hash_before
    )
    assert hashlib.sha256(img_path.read_bytes()).hexdigest() == img_hash_before


# 5. Проверка связки Модель -> forward_physics_loss -> backward() (обучение)
def test_model_training_step_with_connector(mock_anya_dataset: Path) -> None:
    loaders = build_dataloaders(
        mock_anya_dataset, batch_size=2, num_workers=0, mode="train_smoke"
    )
    targets, _ = next(iter(loaders["train"]))

    net = DummyDOENet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    pred_phase = net(targets)
    loss, recon_01, metrics = forward_physics_loss(pred_phase, targets)

    optimizer.zero_grad()
    loss.backward()

    assert recon_01.shape == (2, 4, 128, 128)
    assert "loss" in metrics and "pearson" in metrics and "ssim_global" in metrics
    assert net.conv.weight.grad is not None
    assert not torch.isnan(net.conv.weight.grad).any()
    assert float(net.conv.weight.grad.abs().sum()) > 0.0


# 6. Проверка экспорта предсказаний и совместимости с CLI `doe-eval evaluate`
def test_export_predictions_and_cli_eval(
    mock_anya_dataset: Path, tmp_path: Path
) -> None:
    loaders = build_dataloaders(
        mock_anya_dataset, batch_size=2, num_workers=0, mode="v1"
    )
    net = DummyDOENet()
    preds_dir = tmp_path / "preds_val"

    saved = export_split_predictions(
        net, loaders["validation"], preds_dir, torch.device("cpu")
    )
    assert len(saved) == 2
    assert (preds_dir / "validation_000001.png").is_file()
    assert (preds_dir / "validation_000002.png").is_file()

    out_csv = tmp_path / "val_results.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "doe_eval.cli",
            "evaluate",
            "--data",
            str(mock_anya_dataset.parent / "validation"),
            "--doe-dir",
            str(preds_dir),
            "--out",
            str(out_csv),
            "--device",
            "cpu",
        ],
        check=True,
    )
    assert out_csv.is_file()
    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # заголовок + 2 примера из validation


# 7. Автоматическая проверка реального датасета Ани (если папка data/ уже лежит в проекте)
@pytest.mark.skipif(
    not DEFAULT_MANIFEST.is_file(), reason="Реальный датасет не установлен"
)
def test_real_anya_dataset_if_present() -> None:
    d = load_manifest(DEFAULT_MANIFEST, verify_images=True)
    assert d.report.counts["diagnostic"] == 1
    assert d.report.counts["train"] >= 200
    assert d.report.counts["validation"] == 100
    assert d.report.counts["test"] == 100
    assert d.report.warnings == []


@pytest.mark.parametrize("kind", ["exchange_A", "physical_forward"])
def test_streamed_physical_quartets_keep_plane_order(
    mock_anya_dataset: Path, kind
) -> None:
    raw = json.loads(mock_anya_dataset.read_text())
    for record in raw["samples"]:
        if record["split"] == "train":
            record["params"] = (
                {"exchange": {"kind": "A"}}
                if kind == "exchange_A"
                else {"kind": "physical_forward"}
            )
    mock_anya_dataset.write_text(json.dumps(raw))
    plain = ManifestQuartetDataset(mock_anya_dataset, split="train")
    streamed = ManifestQuartetDataset(mock_anya_dataset, split="train", preload=False)
    for i in range(len(plain)):
        torch.testing.assert_close(streamed[i][0], plain[i][0])
