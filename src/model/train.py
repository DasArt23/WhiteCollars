"""Train the phase network on independently optimized phase targets."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from model.data_connector import build_dataloaders, export_split_predictions
from model.fresnel import (
    DISTANCES, dequantize_phase_u8, quantize_phase_ste, quantize_phase_u8, reconstruct,
)
from model.metrics import balanced_optical_loss, compare_images
from model.network import PhaseNet
from model.progress import save_progress
from model.teacher import make_supervised_loader


@torch.inference_mode()
def precompute_val_cache(model, loader, device):
    """Один раз считает тяжелые optical_features для валидации и держит их на GPU."""
    cached = []
    for targets, _ in loader:
        targets = targets.to(device, non_blocking=True)
        features, initial, _ = model.optical_features(targets)
        cached.append((features, initial, targets))
    return cached


@torch.inference_mode()
def validate_cached(model, val_cache):
    model.eval()
    scores = []
    for features, initial, targets in val_cache:
        raw_phase = model.predict_phase(features, initial)
        phase = dequantize_phase_u8(quantize_phase_u8(raw_phase))
        recon = reconstruct(phase)
        scores.append(compare_images(recon, targets))
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


def aligned_phase_loss(phase, desired):
    delta = (phase - desired).flatten(start_dim=1)
    mean_cos = delta.cos().mean(dim=1)
    mean_sin = delta.sin().mean(dim=1)
    agreement = torch.stack((mean_cos, mean_sin), dim=1).norm(dim=1)
    return (1.0 - agreement.clamp_max(1.0)).mean()


def save_checkpoint(path, model, epoch, validation=None):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "architecture": 4, "state": model.state_dict(), "epoch": epoch,
        "validation": validation, "projection_iterations": model.projection_iterations,
        "correction_limit": model.correction_limit,
    }, temporary)
    temporary.replace(path)


def fit(model, loaders, device, *, epochs, lr=0.003, teacher_steps=24, train_batch_size=16, out=None, report_dir=None):
    """Fit neural weights and restore the best validation epoch in memory."""
    report_dir = out if report_dir is None else report_dir
    train_loader = make_supervised_loader(
        model, loaders["train"], device, steps=teacher_steps, train_batch_size=train_batch_size
    )
    val_cache = precompute_val_cache(model, loaders["validation"], device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    baseline = validate_cached(model, val_cache)
    history = []
    best = baseline
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    if out is not None:
        save_checkpoint(out / "best.pt", model, best_epoch, best)
    print(f"Initial validation SSIM {baseline['ssim']:.4f} | Pearson {baseline['pearson']:.4f}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        phase_weight = 0.5 - 0.45 * (epoch - 1) / (epochs - 1) if epochs > 1 else 0.05
        loss_sums = torch.zeros(3, device=device)
        seen = 0
        for features, initial, desired, targets, initial_scores in train_loader:
            phase = model.predict_phase(features, initial)
            phase_loss = aligned_phase_loss(phase, desired)
            recon = reconstruct(quantize_phase_ste(phase))
            scores = compare_images(recon, targets).mean(dim=1)
            optical_loss = balanced_optical_loss(scores, initial_scores)
            loss = phase_weight * phase_loss + (1.0 - phase_weight) * optical_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sums += torch.stack((loss, phase_loss, optical_loss)).detach() * len(desired)
            seen += len(desired)

        scheduler.step()
        score = validate_cached(model, val_cache)
        loss_mean, phase_mean, optical_mean = (loss_sums / seen).tolist()
        history.append({
            "epoch": epoch, "loss": loss_mean, "phase_loss": phase_mean,
            "optical_loss": optical_mean, "phase_weight": phase_weight, "validation": score,
        })
        if score["ssim"] > best["ssim"] and score["pearson"] > best["pearson"]:
            best = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            if out is not None:
                save_checkpoint(out / "best.pt", model, best_epoch, best)
        print(f"Epoch {epoch:02d}/{epochs} | loss {loss_mean:.4f} | "
              f"val SSIM {score['ssim']:.4f} | Pearson {score['pearson']:.4f}", flush=True)
        if report_dir is not None:
            (report_dir / "history.json").write_text(
                json.dumps({"baseline": baseline, "best_epoch": best_epoch, "epochs": history}, indent=2) + "\n",
                encoding="utf-8",
            )

    model.load_state_dict(best_state)
    if report_dir is not None:
        progress = report_dir / "progress.png"
        save_progress(progress, baseline, history)
        print(f"Saved training chart: {progress}", flush=True)
    return best_epoch


def main():
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "dataset/data/manifest.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for GPU teacher & validation")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Mini-batch size for neural updates")
    parser.add_argument("--projection-iterations", type=int, default=12)
    parser.add_argument("--teacher-steps", type=int, default=24)
    parser.add_argument("--phase-range", type=float, default=1.0, help="Maximum phase correction in radians")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_dev)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.train_batch_size, args.teacher_steps, args.threads) < 1 or not 0 < args.lr < 1:
        parser.error("Counts must be positive and learning rate must be between zero and one")
    if not 0 < args.phase_range <= math.pi:
        parser.error("Phase range must be greater than zero and no more than pi")
    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        args.out = ROOT / "results" / f"training-{stamp}"
    if args.out.exists():
        parser.error(f"Output directory already exists: {args.out}")
    if args.projection_iterations < 2:
        parser.error("Use at least two model projections")

    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    loaders = build_dataloaders(
        args.manifest, batch_size=args.batch_size, num_workers=0,
        mode="train_smoke" if args.smoke else "v1", pin_memory=(device.type == "cuda"),
    )
    model = PhaseNet(
        projection_iterations=args.projection_iterations, correction_limit=args.phase_range
    ).to(device)
    args.out.mkdir(parents=True)
    print(f"Training output: {args.out}", flush=True)

    best_epoch = fit(
        model, loaders, device,
        epochs=args.epochs, lr=args.lr,
        teacher_steps=args.teacher_steps,
        train_batch_size=args.train_batch_size,
        out=args.out,
    )
    print(f"Best validation epoch: {best_epoch}. Saved weights: {args.out / 'best.pt'}", flush=True)

    # Экспорт предсказаний в preds/val и preds/diag для проверки через doe-eval
    export_split_predictions(model, loaders["validation"], "preds/val", device)
    export_split_predictions(model, loaders["diagnostic"], "preds/diag", device)
    print("Exported predictions to preds/val and preds/diag", flush=True)


if __name__ == "__main__":
    main()
