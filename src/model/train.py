"""Train the phase network on independently optimized phase targets."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from model.data_connector import build_dataloaders
from model.fresnel import DISTANCES, dequantize_phase_u8, quantize_phase_u8, reconstruct
from model.metrics import compare_images
from model.network import PhaseNet
from model.progress import save_progress
from model.teacher import make_supervised_loader


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    scores = []
    for targets, _ in loader:
        targets = targets.to(device)
        phase = dequantize_phase_u8(quantize_phase_u8(model(targets)))
        recon = reconstruct(phase)
        scores.append(compare_images(recon, targets).cpu())
    per_plane = torch.cat(scores).mean(dim=0)
    ssim, pearson = per_plane.mean(dim=0).tolist()
    return {
        "ssim": ssim,
        "pearson": pearson,
        "planes": [
            {"distance_mm": round(z * 1000, 1), "ssim": s, "pearson": p}
            for z, (s, p) in zip(DISTANCES, per_plane.tolist(), strict=True)
        ],
    }


def fit(model, loaders, device, *, epochs, lr=0.001, teacher_steps=24, out=None, report_dir=None):
    """Fit neural weights and restore the best validation epoch in memory."""
    report_dir = out if report_dir is None else report_dir
    train_loader = make_supervised_loader(model, loaders["train"], device, steps=teacher_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    baseline = validate(model, loaders["validation"], device)
    history = []
    best = None
    best_state = None
    best_epoch = 0
    print(f"Initial validation SSIM {baseline['ssim']:.4f} | Pearson {baseline['pearson']:.4f}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for step, (features, initial, desired) in enumerate(train_loader, 1):
            features, initial, desired = (item.to(device) for item in (features, initial, desired))
            phase = model.predict_phase(features, initial)
            loss = (1 - torch.cos(phase - desired)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(desired)
            seen += len(desired)
            if step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{epochs} | batch {step}/{len(train_loader)} | "
                      f"loss {loss_sum / seen:.4f}", flush=True)

        score = validate(model, loaders["validation"], device)
        loss_mean = loss_sum / seen
        history.append({"epoch": epoch, "phase_loss": loss_mean, "validation": score})
        rank = (score["ssim"], score["pearson"])
        if best is None or rank > best:
            best = rank
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            if out is not None:
                checkpoint = out / "best.pt"
                temp = checkpoint.with_suffix(".tmp")
                torch.save({
                    "architecture": 4, "state": best_state, "epoch": epoch, "validation": score,
                    "projection_iterations": model.projection_iterations,
                }, temp)
                temp.replace(checkpoint)
        print(f"Epoch {epoch:02d} | phase loss {loss_mean:.4f} | "
              f"val SSIM {score['ssim']:.4f} | Pearson {score['pearson']:.4f}", flush=True)
        if report_dir is not None:
            (report_dir / "history.json").write_text(
                json.dumps({"baseline": baseline, "epochs": history}, indent=2) + "\n",
                encoding="utf-8",
            )

    model.load_state_dict(best_state)
    if report_dir is not None:
        progress = report_dir / "progress.png"
        save_progress(progress, baseline, history)
        print(f"Saved training chart: {progress}", flush=True)
    return best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "dataset/data/manifest.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--projection-iterations", type=int, default=12)
    parser.add_argument("--teacher-steps", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.teacher_steps, args.threads) < 1 or not 0 < args.lr < 1:
        parser.error("Counts must be positive and learning rate must be between zero and one")
    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        args.out = ROOT / "results" / f"training-{stamp}"
    if args.out.exists():
        parser.error(f"Output directory already exists: {args.out}")

    if args.projection_iterations < 2:
        parser.error("Use at least two model projections")

    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    loaders = build_dataloaders(
        args.manifest, batch_size=args.batch_size, num_workers=0,
        mode="train_smoke" if args.smoke else "v1", pin_memory=device.type == "cuda",
    )
    model = PhaseNet(projection_iterations=args.projection_iterations).to(device)
    args.out.mkdir(parents=True)
    print(f"Training output: {args.out}", flush=True)
    fit(model, loaders, device, epochs=args.epochs, lr=args.lr,
        teacher_steps=args.teacher_steps, out=args.out)
    print(f"Saved weights: {args.out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
