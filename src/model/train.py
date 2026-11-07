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

from model.data_connector import build_dataloaders, export_split_predictions
from model.fresnel import DISTANCES, dequantize_phase_u8, quantize_phase_u8, reconstruct
from model.metrics import compare_images
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
    best = None
    best_state = None
    best_epoch = 0
    print(f"Initial validation SSIM {baseline['ssim']:.4f} | Pearson {baseline['pearson']:.4f}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for step, (features, initial, desired, targets) in enumerate(train_loader, 1):
            phase = model.predict_phase(features, initial)
            phase_loss = (1.0 - torch.cos(phase - desired)).mean()

            # Прямой оптический лосс через дифракцию Френеля для роста SSIM и Pearson
            recon = reconstruct(phase)
            scores = compare_images(recon, targets).mean(dim=1)
            optical_loss = 1.0 - (0.8 * scores[:, 0] + 0.2 * scores[:, 1]).mean()

            loss = 0.5 * phase_loss + 0.5 * optical_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.detach().item() * len(desired)
            seen += len(desired)

        scheduler.step()
        score = validate_cached(model, val_cache)
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
        print(f"Epoch {epoch:02d}/{epochs} | loss {loss_mean:.4f} | "
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
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "dataset/data/manifest.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for GPU teacher & validation")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Mini-batch size for neural updates")
    parser.add_argument("--projection-iterations", type=int, default=12)
    parser.add_argument("--teacher-steps", type=int, default=24)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_dev)
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
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    loaders = build_dataloaders(
        args.manifest, batch_size=args.batch_size, num_workers=0,
        mode="train_smoke" if args.smoke else "v1", pin_memory=(device.type == "cuda"),
    )
    model = PhaseNet(projection_iterations=args.projection_iterations).to(device)
    args.out.mkdir(parents=True)
    print(f"Training output: {args.out}", flush=True)

    fit(
        model, loaders, device,
        epochs=args.epochs, lr=args.lr,
        teacher_steps=args.teacher_steps,
        train_batch_size=args.train_batch_size,
        out=args.out,
    )
    print(f"Saved weights: {args.out / 'best.pt'}", flush=True)

    # Экспорт предсказаний в preds/val и preds/diag для проверки через doe-eval
    export_split_predictions(model, loaders["validation"], "preds/val", device)
    export_split_predictions(model, loaders["diagnostic"], "preds/diag", device)
    print("Exported predictions to preds/val and preds/diag", flush=True)


if __name__ == "__main__":
    main()
