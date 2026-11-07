"""Check phase refinement on fixed validation examples; no network training.

--predictions points to a directory with validation_<id>/doe.png.
"""

import argparse
import json
from pathlib import Path

import torch

from doe_eval.cli import evaluate_pair_arrays
from doe_eval.config import IMAGE_FILENAMES
from doe_eval.io import (
    load_doe_png,
    load_four_targets,
    save_comparison_png,
    save_doe_png,
)
from model.holonet import compute_sota_physics_loss, hybrid_infer_phase

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--predictions", type=Path, default=Path("previews/previews"))
parser.add_argument("--out", type=Path, default=Path("build/current_polish"))
parser.add_argument("--steps", type=int, default=200)
args = parser.parse_args()
torch.set_num_threads(2)
ids = [1, 34, 67, 100]
targets = torch.stack(
    [
        torch.from_numpy(
            load_four_targets(
                [f"src/dataset/data/validation/{i:06d}/{n}" for n in IMAGE_FILENAMES]
            )
        )
        for i in ids
    ]
)
initial = torch.stack(
    [
        torch.from_numpy(load_doe_png(args.predictions / f"validation_{i:06d}/doe.png"))
        for i in ids
    ]
)


class Fixed(torch.nn.Module):
    def forward(self, t):
        return initial.clone()


result = hybrid_infer_phase(Fixed(), targets, polish_steps=args.steps, polish_lr=0.035)
root = args.out
root.mkdir(parents=True, exist_ok=True)
report = {
    "scope": "direct phase refinement of four uploaded DOE PNGs, not network training",
    "steps": args.steps,
    "samples": [],
}
for i, sid in enumerate(ids):
    row = {"id": f"validation_{sid:06d}"}
    for name, phase in [("before", initial[i]), ("after", result[i])]:
        out = root / f"{sid:06d}" / name
        out.mkdir(parents=True, exist_ok=True)
        save_doe_png(phase.numpy(), out / "doe.png")
        loaded = torch.from_numpy(load_doe_png(out / "doe.png"))[None]
        _, recon, metrics = compute_sota_physics_loss(loaded, targets[i : i + 1])
        save_comparison_png(
            targets[i].numpy(), recon[0].numpy(), out / "comparison.png"
        )
        row[name] = {
            **evaluate_pair_arrays(targets[i].numpy(), recon[0].numpy())["mean"],
            **{k: float(v) for k, v in metrics.items()},
        }
        row[name]["object_brightness"] = (
            (
                (recon * targets[i : i + 1]).sum((-2, -1))
                / targets[i : i + 1].sum((-2, -1)).clamp_min(1e-8)
            )
            .mean()
            .item()
        )
    report["samples"].append(row)
    print(
        sid,
        {
            k: {
                x: round(v[x], 4)
                for x in [
                    "pearson",
                    "ssim_local",
                    "foreground_error",
                    "background_error",
                    "edge_error",
                    "object_brightness",
                ]
            }
            for k, v in row.items()
            if isinstance(v, dict)
        },
        flush=True,
    )
(root / "report.json").write_text(json.dumps(report, indent=2))
