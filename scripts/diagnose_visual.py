"""Reproduce the five-quartet visual diagnostic: optimize phase, not network weights.

Run from repository root: uv run --frozen python scripts/diagnose_visual.py
Outputs: build/diagnosis/multi_visual/. Uses CPU and fixed seed 42.
"""

import json
from pathlib import Path

import torch

from doe_eval.cli import evaluate_pair_arrays, reconstruct_from_doe_file
from doe_eval.config import IMAGE_FILENAMES, N
from doe_eval.io import load_four_targets, save_comparison_png, save_doe_png
from doe_eval.physics import (
    fresnel_inverse_complex,
    fresnel_propagate_complex,
    fresnel_reconstruct,
    ste_quantize,
)
from model.holonet import _image_objective

torch.set_num_threads(2)
ids = [1, 2, 34, 67, 100]
targets = torch.stack(
    [
        torch.from_numpy(
            load_four_targets(
                [
                    f"src/dataset/data/validation/{sid:06}/{name}"
                    for name in IMAGE_FILENAMES
                ]
            )
        )
        for sid in ids
    ]
)
gen = torch.Generator().manual_seed(42)
phase = (torch.rand(len(ids), N, N, generator=gen) * 2 - 1) * torch.pi
for _ in range(100):
    field = fresnel_propagate_complex(phase)
    rms = field.abs().square().mean((-2, -1), keepdim=True).sqrt()
    trms = targets.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    updated = targets / trms * rms * torch.exp(1j * field.angle())
    phase = fresnel_inverse_complex(updated).mean(1).angle()
initial = phase.clone()
phase.requires_grad_()
opt = torch.optim.Adam([phase], lr=0.03)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400, eta_min=0.003)
best = torch.full((len(ids),), float("inf"))
bestphase = phase.detach().clone()
for step in range(401):
    opt.zero_grad()
    amp = fresnel_reconstruct(ste_quantize(phase))
    loss, _, planes = _image_objective(amp, targets)
    score = planes["visual_error"].mean(1)
    with torch.no_grad():
        mask = score < best
        best = torch.minimum(best, score)
        bestphase[mask] = phase.detach()[mask]
    if step % 100 == 0:
        print("step", step, "visual", score.detach().tolist(), flush=True)
    if step == 400:
        break
    loss.backward()
    opt.step()
    sched.step()
root = Path("build/diagnosis/multi_visual")
root.mkdir(parents=True, exist_ok=True)
report = {
    "scope": "five validation quartets; direct phase optimization, not neural-network training",
    "steps": 400,
    "samples": [],
}
for i, sid in enumerate(ids):
    row = {"id": f"validation_{sid:06}"}
    for name, p in [("initial", initial[i]), ("visual", bestphase[i])]:
        d = root / f"{sid:06}" / name
        d.mkdir(parents=True, exist_ok=True)
        save_doe_png(p.numpy(), d / "doe.png")
        r = reconstruct_from_doe_file(
            d / "doe.png", torch.device("cpu"), "max", "amplitude"
        )
        save_comparison_png(targets[i].numpy(), r, d / "comparison.png")
        m = evaluate_pair_arrays(targets[i].numpy(), r)
        row[name] = m
    report["samples"].append(row)
    print(sid, {n: row[n]["mean"] for n in ["initial", "visual"]}, flush=True)
(root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
