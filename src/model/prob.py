"""Generate a phase DOE and its four Fresnel reconstructions."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from model.data_connector import ManifestQuartetDataset
from model.inference import (
    DEFAULT_MANIFEST,
    DEFAULT_WEIGHTS,
    PredictionService,
    load_predictor,
)
from model.retrieval import save_example
from model.train_retrieval import is_zakhar_a, split_a


def main(argv=None, *, sample_id=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--images", nargs=4, type=Path)
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "diagnostic"), default="test"
    )
    parser.add_argument("--sample", help="Sample ID from the selected split")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--optical-steps", type=int)
    args = parser.parse_args(argv)
    try:
        model, device = load_predictor(args.weights, args.device, args.optical_steps)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if args.images:
        inputs = args.images
        planes = []
        for path in inputs:
            with Image.open(path) as image:
                if (image.format, image.mode, image.size) != ("PNG", "L", (128, 128)):
                    parser.error(f"Expected a 128x128 grayscale PNG: {path}")
                planes.append(torch.from_numpy(np.array(image, dtype=np.float32)) / 255)
        targets = torch.stack(planes)
        sid = "custom"
    else:
        dataset = ManifestQuartetDataset(args.manifest, split=args.split, preload=False)
        requested = sample_id or args.sample
        if requested:
            index = next(
                (
                    i
                    for i, row in enumerate(dataset.samples)
                    if row.sample_id == requested
                ),
                None,
            )
            if index is None:
                parser.error(f"Sample not in {args.split}: {requested}")
        else:
            index = 0
            if args.split == "train" and any(
                is_zakhar_a(row) for row in dataset.samples
            ):
                index = split_a(dataset)[1][0]
        targets, sid = dataset[index]
        inputs = [dataset.root_dir / path for path in dataset.samples[index].images]
    result = PredictionService(model, device).predict(targets)
    out = args.out or Path("outputs") / f"prediction-{datetime.now():%Y%m%d-%H%M%S-%f}"
    out.mkdir(parents=True, exist_ok=False)
    save_example(
        out,
        result.mask,
        result.reconstructions,
        result.scores,
        sid,
        [path.resolve() for path in inputs],
    )
    np.save(out / "weights.npy", result.importance_maps.cpu().numpy())
    scores = result.scores
    print(
        f"{sid} | SSIM min {scores[:, 0].min().item():.4f} | "
        f"Pearson min {scores[:, 1].min().item():.4f} | {result.synthesis_ms:.1f} ms | {out.resolve()}",
        flush=True,
    )
    return out, model


if __name__ == "__main__":
    main()
