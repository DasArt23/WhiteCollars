from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from model.data_connector import (
    build_dataloaders,
    export_split_predictions,
    forward_physics_loss,
)
from model.holonet import HoloUnrollNet, compute_sota_physics_loss, hybrid_infer_phase
from doe_eval.io import save_doe_png


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_pearson = 0.0
    total_ssim = 0.0
    total_samples = 0

    for targets, _ in loader:
        bsz = targets.shape[0]
        targets = targets.to(device, non_blocking=True)
        pred_phase = model(targets)
        _, _, metrics = forward_physics_loss(pred_phase, targets, use_ste_quantize=True)

        total_loss += float(metrics["loss"].item()) * bsz
        total_pearson += float(metrics["pearson"].item()) * bsz
        total_ssim += float(metrics["ssim_global"].item()) * bsz
        total_samples += bsz

    denom = max(1, total_samples)
    return {
        "loss": total_loss / denom,
        "pearson": total_pearson / denom,
        "ssim_global": total_ssim / denom,
        "score": 0.5 * (total_pearson / denom) + 0.5 * (total_ssim / denom),
    }


def export_split_with_optional_polish(
    model: torch.nn.Module,
    loader: DataLoader,
    out_doe_dir: str | Path,
    device: torch.device,
    polish_steps: int = 0,
) -> list[Path]:
    """Экспортирует предсказания через export_split_predictions (или с гибридным polish)."""
    if polish_steps <= 0:
        return export_split_predictions(model, loader, out_doe_dir, device)

    out_dir = Path(out_doe_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for targets, sample_ids in loader:
        targets = targets.to(device, non_blocking=True)
        pred_phase = hybrid_infer_phase(model, targets, polish_steps=polish_steps)
        pred_np = pred_phase.detach().cpu().numpy()
        for phase_2d, sid in zip(pred_np, sample_ids):
            p = out_dir / f"{sid}.png"
            save_doe_png(phase_2d, p)
            saved.append(p)
    return saved


def train_model(
    manifest_path: str | Path = "src/dataset/data/manifest.json",
    checkpoint_out: str | Path = "checkpoints/best_model.pt",
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 2e-3,
    unroll_steps: int = 6,
    base_dim: int = 48,
    mode: str = "v1",
    device_str: str = "auto",
    polish_export_steps: int = 25,
) -> Path:
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) else (
            "cpu" if device_str == "auto" else device_str
        )
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    manifest_p = Path(manifest_path)
    if not manifest_p.is_file() and Path("data/manifest.json").is_file():
        manifest_p = Path("data/manifest.json")

    nw = 4 if device.type == "cuda" else 0
    loaders = build_dataloaders(
        manifest_path=manifest_p,
        batch_size=batch_size,
        num_workers=nw,
        mode=mode,
        augment_train=True,
        pin_memory=(device.type == "cuda"),
    )

    model = HoloUnrollNet(unroll_steps=unroll_steps, base_dim=base_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(loaders["train"]),
        pct_start=0.15,
        anneal_strategy="cos",
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = Path(checkpoint_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[HoloUnrollNet] Device: {device} "
        f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}) | "
        f"Mode: {mode} | Unroll: {unroll_steps} | BaseDim: {base_dim}",
        flush=True,
    )

    best_score = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_p_sum = 0.0
        train_s_sum = 0.0
        seen = 0

        for targets, _ in loaders["train"]:
            bsz = targets.shape[0]
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred_phase = model(targets)
                loss, _, metrics = compute_sota_physics_loss(pred_phase, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += float(metrics["loss"].item()) * bsz
            train_p_sum += float(metrics["pearson"].item()) * bsz
            train_s_sum += float(metrics["ssim_global"].item()) * bsz
            seen += bsz

        val_metrics = evaluate_loader(model, loaders["validation"], device)
        dt = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / seen,
            "train_pearson": train_p_sum / seen,
            "train_ssim": train_s_sum / seen,
            "val_loss": val_metrics["loss"],
            "val_pearson": val_metrics["pearson"],
            "val_ssim_global": val_metrics["ssim_global"],
            "val_score": val_metrics["score"],
            "time_sec": round(dt, 2),
        }
        history.append(row)

        is_best = val_metrics["score"] > best_score
        if is_best:
            best_score = val_metrics["score"]
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "model_type": "HoloUnrollNet",
                    "unroll_steps": unroll_steps,
                    "base_dim": base_dim,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "state_dict": best_state,
                },
                ckpt_path,
            )

        mark = "*" if is_best else " "
        print(
            f"Epoch {epoch:03d}/{epochs:03d} [{dt:5.2f}s]{mark} | "
            f"Train P: {row['train_pearson']:.4f} SSIM: {row['train_ssim']:.4f} | "
            f"Val P: {val_metrics['pearson']:.4f} SSIM: {val_metrics['ssim_global']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    (ckpt_path.parent / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nСохранены лучшие веса модели: {ckpt_path} (Best Val Score: {best_score:.4f})")
    print("Экспорт предсказаний в preds/val и preds/diag...")
    export_split_with_optional_polish(
        model, loaders["validation"], "preds/val", device, polish_steps=polish_export_steps
    )
    export_split_with_optional_polish(
        model, loaders["diagnostic"], "preds/diag", device, polish_steps=polish_export_steps
    )
    print("Готово! Папки preds/val и preds/diag сформированы.")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение SOTA модели HoloUnrollNet для синтеза ДОЭ")
    parser.add_argument("--manifest", default="src/dataset/data/manifest.json", help="Путь к manifest.json")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Куда сохранить веса")
    parser.add_argument("--epochs", type=int, default=60, help="Число эпох обучения")
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча")
    parser.add_argument("--lr", type=float, default=2e-3, help="Максимальный learning rate")
    parser.add_argument("--unroll-steps", type=int, default=6, help="Число развёрнутых физических итераций")
    parser.add_argument("--base-dim", type=int, default=48, help="Базовое число каналов Fourier U-Net")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="Быстрый тест на 8 примерах (train_smoke)")
    parser.add_argument("--polish-steps", type=int, default=25, help="Шаги финального дожима при экспорте PNG")
    args = parser.parse_args()

    train_model(
        manifest_path=args.manifest,
        checkpoint_out=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unroll_steps=args.unroll_steps,
        base_dim=args.base_dim,
        mode="train_smoke" if args.smoke else "v1",
        device_str=args.device,
        polish_export_steps=args.polish_steps,
    )


if __name__ == "__main__":
    main()
