"""Evaluate saved model weights on a held-out quartet."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from model.data_connector import ManifestQuartetDataset, export_split_predictions

from model.prob import main as predict

MODEL = Path(__file__).resolve().parent / "model"


def save_comparison(output):
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    sheet = Image.new("RGB", (742, 314), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 6), f"{report['sample_id']} / top: targets / bottom: model", fill="black")
    for i, (path, plane) in enumerate(zip(report["inputs"], report["planes"], strict=True)):
        x = 12 + 146 * i
        draw.text((x, 27), f"{plane['distance_mm']:.1f} mm", fill="black")
        for source, y in ((path, 44), (output / f"restored_{i + 1}.png", 176)):
            with Image.open(source) as image:
                sheet.paste(image, (x, y))
    draw.text((596, 27), "Model phase DOE", fill="black")
    with Image.open(output / "doe.png") as mask:
        sheet.paste(mask, (596, 44))
    comparison = output / "comparison.png"
    sheet.save(comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Use a specific saved .pt file")
    parser.add_argument("--sample", default="test_000002", help="Test sample ID")
    args = parser.parse_args()
    candidates = [
        *(MODEL / "weights").glob("best-*.pt"),
        *(MODEL / "results").glob("training-*/best.pt"),
    ]
    weights = args.weights or max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)
    if weights is None or not weights.is_file():
        parser.error("No saved weights found. Run src/model/train.py or pass --weights.")
    output, model = predict(["--weights", str(weights), "--split", "test"], sample_id=args.sample)
    validation = ManifestQuartetDataset(MODEL.parent / "dataset/data/manifest.json", split="validation")
    loader = DataLoader(validation, batch_size=8, shuffle=False)
    exported = export_split_predictions(model, loader, "preds/val", torch.device("cpu"))
    print(f"Exported {len(exported)} validation DOEs to preds/val", flush=True)
    comparison = save_comparison(output)
    print(f"Comparison: {comparison}", flush=True)
    explorer, wslpath = shutil.which("explorer.exe"), shutil.which("wslpath")
    if explorer and wslpath:
        path = subprocess.check_output([wslpath, "-w", str(comparison)], text=True).strip()
        subprocess.Popen([explorer, path], stdin=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
