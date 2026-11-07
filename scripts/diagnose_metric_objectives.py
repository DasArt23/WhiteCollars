"""Direct phase baselines for the fundamental audit; no network training.

Example: python scripts/diagnose_metric_objectives.py --objective global_joint
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from doe_eval.cli import evaluate_pair_arrays
from doe_eval.config import IMAGE_FILENAMES
from doe_eval.io import (
    load_doe_png,
    load_four_targets,
    save_comparison_png,
    save_doe_png,
)
from doe_eval.metrics import pearson_torch, ssim_global_torch, ssim_local_7x7_torch
from doe_eval.physics import (
    fresnel_reconstruct,
    normalize_reconstructions,
    ste_quantize,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--objective",
    choices=["pearson_only", "joint", "single_plane", "global_joint", "global_only"],
    default="global_joint",
)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--out", type=Path, default=Path("build/metric_objective_audit"))
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")
torch.set_num_threads(2)
ids = [1, 34, 67, 100]
targets = torch.stack(
    [
        torch.from_numpy(
            load_four_targets(
                [
                    f"src/dataset/data/validation/{sid:06d}/{name}"
                    for name in IMAGE_FILENAMES
                ]
            )
        )
        for sid in ids
    ]
)
initial = torch.stack(
    [
        torch.from_numpy(
            load_doe_png(f"reports/suppression_inputs/validation_{sid:06d}/doe.png")
        )
        for sid in ids
    ]
)
root = args.out
root.mkdir(parents=True, exist_ok=True)
report = {
    "steps": args.steps,
    "lr": 0.04,
    "scope": "direct optimization of phase, not neural-network training; not a metric upper bound",
    "methods": {},
}
for method in [args.objective]:
    p = initial.clone().requires_grad_()
    opt = torch.optim.Adam([p], lr=0.04)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=0.002)
    best = torch.full((len(ids),), -float("inf"))
    bestp = initial.clone()
    start = time.perf_counter()
    for step in range(args.steps + 1):
        opt.zero_grad()
        r = normalize_reconstructions(fresnel_reconstruct(ste_quantize(p)))
        pval = pearson_torch(r, targets)
        if method == "single_plane":
            score = pval[:, 0]
        elif method == "pearson_only":
            score = pval.mean(1)
        elif method == "global_joint":
            score = pval.mean(1) + ssim_global_torch(r, targets).mean(1)
        elif method == "global_only":
            score = ssim_global_torch(r, targets).mean(1)
        else:
            score = pval.mean(1) + ssim_local_7x7_torch(r, targets).mean(1)
        with torch.no_grad():
            improved = score > best
            best = torch.maximum(best, score)
            bestp[improved] = p.detach()[improved]
        if step % 200 == 0:
            print(method, step, pval.detach().mean(1).tolist(), flush=True)
        if step == args.steps:
            break
        (-score.mean()).backward()
        opt.step()
        sched.step()
    r = normalize_reconstructions(fresnel_reconstruct(ste_quantize(bestp)))
    rows = []
    for i, sid in enumerate(ids):
        out = root / method / f"{sid:06d}"
        out.mkdir(parents=True, exist_ok=True)
        wrapped = torch.atan2(bestp[i].sin(), bestp[i].cos())
        save_doe_png(wrapped.detach().numpy(), out / "doe.png")
        saved = torch.from_numpy(load_doe_png(out / "doe.png"))
        rec = normalize_reconstructions(fresnel_reconstruct(saved)).detach().numpy()
        save_comparison_png(targets[i].numpy(), rec, out / "comparison.png")
        rows.append({"id": sid, **evaluate_pair_arrays(targets[i].numpy(), rec)})
    report["methods"][method] = {
        "seconds": time.perf_counter() - start,
        "rows": rows,
        "mean": {
            key: float(np.mean([row["mean"][key] for row in rows]))
            for key in rows[0]["mean"]
        },
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print("FINAL", method, report["methods"][method]["mean"], flush=True)
