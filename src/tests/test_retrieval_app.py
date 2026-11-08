import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from app.cli import main
from doe_eval.dataset import sample_image_paths
from doe_eval.io import load_four_targets
from model.retrieval import plane_metrics, reconstruct


def test_packaged_model_predicts_a_quantized_doe(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    out = tmp_path / "prediction"
    assert main(["predict", "--device", "cpu", "--out", str(out)]) == out

    with Image.open(out / "doe.png") as image:
        assert (image.format, image.mode, image.size) == ("PNG", "L", (128, 128))
        mask = torch.tensor(
            list(image.get_flattened_data()), dtype=torch.uint8
        ).reshape(128, 128)
    for i in range(1, 5):
        with Image.open(out / f"restored_{i}.png") as image:
            assert (image.format, image.mode, image.size) == ("PNG", "L", (128, 128))

    report = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert report["sample_id"] == "test_000001"
    assert [plane["distance_mm"] for plane in report["planes"]] == [
        24.1,
        25.3,
        26.5,
        27.7,
    ]
    paths = sample_image_paths(root / "src/dataset/example/test/000001")
    targets = torch.from_numpy(load_four_targets(paths))
    expected = plane_metrics(reconstruct(mask), targets)
    for plane, (ssim, pearson) in zip(report["planes"], expected.tolist(), strict=True):
        assert plane["ssim"] == pytest.approx(ssim, abs=1e-6)
        assert plane["pearson"] == pytest.approx(pearson, abs=1e-6)
