"""Synthesize a DOE and reconstruct all four observation planes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(SRC))

from doe_eval import DISTANCES, quantize_phase_u8, dequantize_phase_u8
from model.data_connector import ManifestData, ManifestQuartetDataset, ManifestReport, ManifestSample
from model.importance import ARCHITECTURE, LEGACY_ARCHITECTURE, ImportanceDOE
from model.train_importance import reconstruct, score


def main(argv=None, *, sample_id=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="Four PNGs in increasing distance order")
    parser.add_argument("--weights", type=Path, help="ImportanceDOE checkpoint")
    parser.add_argument("--manifest", type=Path, default=SRC / "dataset/data/manifest.json")
    parser.add_argument("--split", choices=("diagnostic", "test"), default="diagnostic")
    args = parser.parse_args(argv)
    distances = tuple(round(z * 1000, 1) for z in DISTANCES)
    if args.images:
        if len(args.images) != 4:
            parser.error("Supply four images ordered by 24.1, 25.3, 26.5 and 27.7 mm")
        sample = ManifestSample("custom", "diagnostic", tuple(str(p.resolve()) for p in args.images),
                                distances, 0, "", {})
        manifest = ManifestData(Path("custom"), Path.cwd(), "custom", "", 0, [sample],
                                ManifestReport({"diagnostic": 1}, 4))
        split = "diagnostic"
    else:
        manifest, split = args.manifest, args.split

    dataset = ManifestQuartetDataset(manifest, split=split, preload=False)
    index = next((i for i, sample in enumerate(dataset.samples) if sample.sample_id == sample_id), None)
    if sample_id is not None and index is None:
        parser.error(f"Sample {sample_id} is absent from split {split}")
    index = 0 if index is None else index
    targets, sample_id = dataset[index]

    results = ROOT / "results"
    candidates = results.glob("training-importance-*/best.pt")
    weights = args.weights or max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)
    if weights is None or not weights.is_file():
        parser.error("Run train_importance.py or supply --weights")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in (ARCHITECTURE, LEGACY_ARCHITECTURE):
        parser.error("The checkpoint is not from the pixel importance model")
    torch.set_num_threads(4)
    model = ImportanceDOE(checkpoint["projections"], learn_phase=architecture == ARCHITECTURE)
    model.load_state_dict(checkpoint["state"])
    model.eval()

    output = results / f"network-importance-{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True)
    with torch.no_grad():
        mask = quantize_phase_u8(model(targets.unsqueeze(0))[0])
        Image.fromarray(mask.numpy()).save(output / "doe.png")
        planes = reconstruct(dequantize_phase_u8(mask))
        values = score(planes, targets).tolist()
        for i, image in enumerate(planes, 1):
            Image.fromarray((image * 255).round().to(torch.uint8).numpy()).save(output / f"restored_{i}.png")
    report = {
        "sample_id": sample_id, "split": split, "weights": str(weights.resolve()),
        "epoch": checkpoint["epoch"],
        "inputs": [str((dataset.root_dir / path).resolve()) for path in dataset.samples[index].images],
        "planes": [{"distance_mm": distance, "ssim": ssim, "pearson": pearson}
                   for distance, (ssim, pearson) in zip(distances, values, strict=True)],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Sample {sample_id} | output {output}", flush=True)
    return output, model


if __name__ == "__main__":
    main()
