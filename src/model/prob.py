"""Predict one DOE from four images and display the saved results."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
from PIL import Image

SRC = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(SRC))

from model.data_connector import (
    ManifestData,
    ManifestQuartetDataset,
    ManifestReport,
    ManifestSample,
    build_dataloaders,
)
from model.fresnel import DISTANCES, dequantize_phase_u8, quantize_phase_u8, reconstruct
from model.metrics import compare_images
from model.network import DOENet, PhaseNet
from model.train import fit, save_checkpoint


def main(argv=None, *, sample_id=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="Four PNGs ordered by distance")
    parser.add_argument("--weights", type=Path, help="Use a specific trained .pt file")
    parser.add_argument("--manifest", type=Path, default=SRC / "dataset/data/manifest.json")
    parser.add_argument("--split", choices=("diagnostic", "test"), default="diagnostic")
    args = parser.parse_args(argv)
    if args.images and len(args.images) != 4:
        parser.error("Supply exactly four PNGs for 24.1, 25.3, 26.5 and 27.7 mm")
    if args.images:
        sample = ManifestSample(
            sample_id="custom",
            split="diagnostic",
            images=tuple(str(path.resolve()) for path in args.images),
            distances_mm=tuple(round(z * 1000, 1) for z in DISTANCES),
            seed=0,
            generator_version="",
            params={},
        )
        manifest = ManifestData(
            manifest_path=Path("custom"),
            root_dir=Path.cwd(),
            dataset_name="custom",
            dataset_version="",
            master_seed=0,
            samples=[sample],
            report=ManifestReport({"diagnostic": 1}, 4),
        )
    else:
        manifest = args.manifest
    split = "diagnostic" if args.images else args.split
    dataset = ManifestQuartetDataset(manifest, split=split)
    index = next((i for i, sample in enumerate(dataset.samples) if sample.sample_id == sample_id), None)
    if sample_id is not None and index is None:
        parser.error(f"Sample {sample_id} is absent from split {split}")
    index = 0 if index is None else index
    targets, sample_id = dataset[index]
    input_paths = [str((dataset.root_dir / image).resolve()) for image in dataset.samples[index].images]
    print(f"Input sample: {sample_id} ({split})", flush=True)

    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device("cpu")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = SRC / "model" / "results" / f"network-{stamp}"
    weights_dir = SRC / "model" / "weights"
    weights = args.weights
    checkpoint = None
    if weights is not None:
        checkpoint = torch.load(weights, map_location=device, weights_only=True)
    else:
        candidates = [*weights_dir.glob("best-*.pt"), *(output.parent.glob("training-*/best.pt"))]
        for candidate in sorted(candidates, key=lambda path: path.stat().st_mtime_ns, reverse=True):
            state = torch.load(candidate, map_location=device, weights_only=True)
            if state.get("architecture") in (2, 3, 4):
                weights, checkpoint = candidate, state
                break
    if checkpoint is None:
        model = PhaseNet(correction_limit=1.0)
        loaders = build_dataloaders(args.manifest, batch_size=2, num_workers=0, pin_memory=False)
        print(f"Training neural network on {len(loaders['train'].dataset)} quartets...", flush=True)
        output.mkdir(parents=True)
        best_epoch = fit(model, loaders, device, epochs=10, report_dir=output)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights = weights_dir / f"best-{stamp}.pt"
        save_checkpoint(weights, model, best_epoch)
        print(f"Saved weights: {weights}", flush=True)
    else:
        architecture = checkpoint.get("architecture")
        if architecture in (3, 4):
            model = PhaseNet(
                projection_iterations=checkpoint.get("projection_iterations", 12),
                aligned=architecture == 4,
                correction_limit=checkpoint.get("correction_limit"),
            )
        elif architecture == 2:
            model = DOENet(projection_iterations=checkpoint.get("projection_iterations", 6))
        else:
            parser.error(f"Incompatible weights: {weights}")
        model.load_state_dict(checkpoint["state"])
        best_epoch = checkpoint["epoch"]
        print(f"Loaded weights: {weights}", flush=True)
    model.eval()

    with torch.inference_mode():
        pixels = quantize_phase_u8(model(targets.unsqueeze(0))[0])

        output.mkdir(parents=True, exist_ok=True)
        mask_path = output / "doe.png"
        Image.fromarray(pixels.numpy()).save(mask_path)
        with Image.open(mask_path) as mask:
            saved = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8).reshape(pixels.shape)
        phase = dequantize_phase_u8(saved)
        planes = reconstruct(phase)
        scores = compare_images(planes, targets).tolist()

        report = {
            "trained": best_epoch > 0, "epoch": best_epoch, "sample_id": sample_id,
            "split": split, "inputs": input_paths, "weights": str(weights.resolve()),
            "planes": [],
        }
        for index, (plane, z, (ssim, pearson)) in enumerate(
            zip(planes, DISTANCES, scores, strict=True), 1
        ):
            restored = (plane * 255).round().to(torch.uint8)
            Image.fromarray(restored.numpy()).save(output / f"restored_{index}.png")
            report["planes"].append({
                "distance_mm": round(z * 1000, 1), "pearson": pearson, "ssim": ssim,
            })

    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Best validation epoch: {best_epoch}. Output: {output}", flush=True)
    return output, model


if __name__ == "__main__":
    main()
