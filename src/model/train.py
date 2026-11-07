from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from doe_eval.cli import evaluate_pair_arrays
from doe_eval.config import DEFAULT_MANIFEST
from doe_eval.io import load_doe_png, save_comparison_png, save_doe_png
from doe_eval.physics import fresnel_reconstruct, normalize_reconstructions
from model.data_connector import (
    build_dataloaders,
    export_split_predictions,
)
from model.holonet import (
    LOSS_VERSION,
    METRIC_WEIGHTS,
    PRIMARY_SSIM,
    WORST_PLANE_WEIGHT,
    HoloUnrollNet,
    compute_sota_physics_loss,
    hybrid_infer_phase,
)


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
    total_local = 0.0
    total_shape = 0.0
    total_visual = 0.0
    total_selection = 0.0
    total_worst_pearson = 0.0
    total_worst_ssim = 0.0
    fidelity_totals = {
        key: 0.0
        for key in (
            "foreground_error",
            "background_error",
            "background_energy",
            "foreign_energy",
            "power_error",
            "edge_error",
            "multiscale_error",
        )
    }
    total_samples = 0

    for targets, _ in loader:
        bsz = targets.shape[0]
        targets = targets.to(device, non_blocking=True)
        pred_phase = model(targets)
        _, _, metrics = compute_sota_physics_loss(pred_phase, targets)
        if any(not torch.isfinite(value).all() for value in metrics.values()):
            raise FloatingPointError(
                "Неконечная метрика на validation; checkpoint не сохраняется"
            )

        total_loss += float(metrics["loss"].item()) * bsz
        total_pearson += float(metrics["pearson"].item()) * bsz
        total_ssim += float(metrics["ssim_global"].item()) * bsz
        total_local += float(metrics["ssim_local"].item()) * bsz
        total_shape += float(metrics["shape_error"].item()) * bsz
        total_visual += float(metrics["visual_error"].item()) * bsz
        total_selection += float(metrics["selection_error"].item()) * bsz
        total_worst_pearson += float(metrics["worst_pearson"].item()) * bsz
        total_worst_ssim += float(metrics["worst_ssim_local"].item()) * bsz
        for key in fidelity_totals:
            fidelity_totals[key] += float(metrics[key]) * bsz
        total_samples += bsz

    if total_samples == 0:
        raise ValueError("Нельзя оценить пустую выборку")
    denom = total_samples
    return {
        "loss": total_loss / denom,
        "pearson": total_pearson / denom,
        "ssim_global": total_ssim / denom,
        "ssim_skimage": total_local / denom,
        "shape_error": total_shape / denom,
        "visual_error": total_visual / denom,
        "selection_error": total_selection / denom,
        "worst_pearson": total_worst_pearson / denom,
        "worst_ssim_skimage": total_worst_ssim / denom,
        "score": -total_selection / denom,
        **{key: value / denom for key, value in fidelity_totals.items()},
    }


@torch.no_grad()
def save_validation_previews(model, loader, device, directory: Path) -> None:
    """Фиксированные примеры показывают потерю объектов, скрытую средним SSIM."""
    indices = np.linspace(
        0, len(loader.dataset) - 1, min(4, len(loader.dataset)), dtype=int
    )
    examples = [loader.dataset[int(i)] for i in indices]
    targets = torch.stack([target for target, _ in examples]).to(device)
    phases = model(targets).cpu().numpy()
    for phase, (target, sid) in zip(phases, examples):
        destination = directory / str(sid)
        doe = destination / "doe.png"
        save_doe_png(phase, doe)
        restored = fresnel_reconstruct(torch.from_numpy(load_doe_png(doe)).to(device))
        restored = normalize_reconstructions(restored).cpu().numpy()
        save_comparison_png(target.numpy(), restored, destination / "comparison.png")
        report = evaluate_pair_arrays(target.numpy(), restored)
        (destination / "metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


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


def initialize_training_model(
    unroll_steps: int,
    base_dim: int,
    seed: int,
    init_checkpoint: str | Path | None = None,
) -> HoloUnrollNet:
    """Загружает веса для дообучения; состояния Adam и scheduler создаются заново."""
    payload = None
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(
                "Для дообучения нужен checkpoint с метаданными и state_dict"
            )
        if (
            payload.get("unroll_steps") != unroll_steps
            or payload.get("base_dim") != base_dim
        ):
            raise ValueError(
                "unroll_steps/base_dim должны совпадать с исходным checkpoint"
            )
    model = HoloUnrollNet(
        unroll_steps=unroll_steps,
        base_dim=base_dim,
        equalize_plane_power=payload.get("equalize_plane_power", False)
        if payload
        else True,
        phase_carrier_seed=payload.get("phase_carrier_seed") if payload else seed,
    )
    if payload:
        model.load_state_dict(payload["state_dict"], strict=True)
    return model


def train_model(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    checkpoint_out: str | Path = "checkpoints/best_model.pt",
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 2e-3,
    unroll_steps: int = 6,
    base_dim: int = 48,
    mode: str = "v1",
    device_str: str = "auto",
    polish_export_steps: int = 25,
    seed: int = 3701,
    precision: str = "fp32",
    init_checkpoint: str | Path | None = None,
    lr_schedule: str = "onecycle",
) -> Path:
    if epochs < 1 or batch_size < 1 or lr <= 0 or polish_export_steps < 0:
        raise ValueError("epochs и batch_size >= 1, lr > 0, polish_export_steps >= 0")
    if precision not in ("fp32", "amp"):
        raise ValueError("precision: fp32 | amp")
    if lr_schedule not in ("onecycle", "plateau"):
        raise ValueError("lr_schedule: onecycle | plateau")
    if (
        init_checkpoint is not None
        and Path(init_checkpoint).resolve() == Path(checkpoint_out).resolve()
    ):
        raise ValueError(
            "Для дообучения задайте новый выходной checkpoint, сохранив исходный"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        "cuda"
        if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "auto" else device_str)
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    manifest_p = Path(manifest_path)

    nw = 4 if device.type == "cuda" else 0
    loaders = build_dataloaders(
        manifest_path=manifest_p,
        batch_size=batch_size,
        num_workers=nw,
        mode=mode,
        augment_train=True,
        pin_memory=(device.type == "cuda"),
        seed=seed,
    )

    model = initialize_training_model(unroll_steps, base_dim, seed, init_checkpoint).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = (
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(loaders["train"]),
            pct_start=0.15,
            anneal_strategy="cos",
        )
        if lr_schedule == "onecycle"
        else torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=min(lr, 1e-6)
        )
    )

    use_amp = device.type == "cuda" and precision == "amp"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = Path(checkpoint_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[HoloUnrollNet] Device: {device} "
        f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}) | "
        f"Mode: {mode} | Unroll: {unroll_steps} | BaseDim: {base_dim} | Precision: {precision}",
        flush=True,
    )

    initial_metrics = evaluate_loader(model, loaders["validation"], device)
    if lr_schedule == "plateau":
        scheduler.step(initial_metrics["selection_error"])
    (ckpt_path.parent / "initial_validation.json").write_text(
        json.dumps(initial_metrics, indent=2), encoding="utf-8"
    )
    save_validation_previews(
        model, loaders["validation"], device, ckpt_path.parent / "initial_previews"
    )
    print(
        f"Initial | P: {initial_metrics['pearson']:.4f} "
        f"SSIM global: {initial_metrics['ssim_global']:.4f} "
        f"SSIM skimage: {initial_metrics['ssim_skimage']:.4f} Visual: {initial_metrics['visual_error']:.4f}",
        flush=True,
    )

    def save_checkpoint(epoch: int, val_metrics: dict[str, float]):
        state = deepcopy(model.state_dict())
        torch.save(
            {
                "model_type": "HoloUnrollNet",
                "unroll_steps": unroll_steps,
                "base_dim": base_dim,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "seed": seed,
                "manifest_sha256": hashlib.sha256(manifest_p.read_bytes()).hexdigest(),
                "metric_weights": METRIC_WEIGHTS,
                "primary_ssim": PRIMARY_SSIM,
                "loss_version": LOSS_VERSION,
                "equalize_plane_power": model.equalize_plane_power,
                "phase_carrier_seed": model.phase_carrier_seed,
                "training_revision": "stable-carrier-v3",
                "selection_revision": "global-metrics-spatial-worst-plane-v5",
                "worst_plane_weight": WORST_PLANE_WEIGHT,
                "precision": precision,
                "lr_schedule": lr_schedule,
                "init_checkpoint": str(init_checkpoint)
                if init_checkpoint is not None
                else None,
                "initial_validation": initial_metrics,
                "checkpoint_criterion": "global-fidelity-v5 loss; global SSIM breaks ties",
                "quantization": "uint8-ste",
                "state_dict": state,
            },
            ckpt_path,
        )
        return state

    # Не заменяем хорошую инициализацию ухудшившейся первой эпохой.
    best_score = initial_metrics["score"]
    best_ssim = initial_metrics[PRIMARY_SSIM]
    best_epoch = 0
    best_state = save_checkpoint(0, initial_metrics)
    save_validation_previews(
        model, loaders["validation"], device, ckpt_path.parent / "previews"
    )
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_p_sum = 0.0
        train_s_sum = 0.0
        train_local_sum = 0.0
        train_shape_sum = 0.0
        seen = 0
        updates = 0
        skipped = 0
        grad_norm_sum = 0.0
        nonfinite_grad_batches = 0

        for targets, _ in loaders["train"]:
            bsz = targets.shape[0]
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred_phase = model(targets)
                loss, _, metrics = compute_sota_physics_loss(pred_phase, targets)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Неконечный loss на эпохе {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Инициализатор не должен подавлять градиенты всех последующих блоков.
            initial_norm = torch.nn.utils.clip_grad_norm_(
                model.target_phase_init.parameters(),
                max_norm=1.0,
                error_if_nonfinite=not use_amp,
            )
            remaining = [
                p
                for n, p in model.named_parameters()
                if not n.startswith("target_phase_init.")
            ]
            rest_norm = torch.nn.utils.clip_grad_norm_(
                remaining, max_norm=1.0, error_if_nonfinite=not use_amp
            )
            grad_norm = torch.sqrt(initial_norm.square() + rest_norm.square())
            if torch.isfinite(grad_norm):
                grad_norm_sum += float(grad_norm.item())
            else:
                nonfinite_grad_batches += 1
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                if lr_schedule == "onecycle":
                    scheduler.step()
                updates += 1
            else:
                skipped += 1

            train_loss_sum += float(metrics["loss"].item()) * bsz
            train_p_sum += float(metrics["pearson"].item()) * bsz
            train_s_sum += float(metrics["ssim_global"].item()) * bsz
            train_local_sum += float(metrics["ssim_local"].item()) * bsz
            train_shape_sum += float(metrics["shape_error"].item()) * bsz
            seen += bsz

        if updates == 0:
            raise FloatingPointError(
                "Оптимизатор не выполнил ни одного обновления за эпоху. Используйте --precision fp32."
            )
        val_metrics = evaluate_loader(model, loaders["validation"], device)
        metric_guard = (
            val_metrics["pearson"] >= initial_metrics["pearson"]
            and val_metrics[PRIMARY_SSIM] >= initial_metrics[PRIMARY_SSIM]
        )
        if lr_schedule == "plateau":
            monitor = val_metrics["selection_error"]
            if not metric_guard:
                monitor = max(initial_metrics["selection_error"], monitor)
            scheduler.step(monitor)
        dt = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "metric_guard_passed": metric_guard,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss_sum / seen,
            "train_pearson": train_p_sum / seen,
            "train_ssim": train_s_sum / seen,
            "train_ssim_skimage": train_local_sum / seen,
            "train_shape_error": train_shape_sum / seen,
            "val_loss": val_metrics["loss"],
            "val_pearson": val_metrics["pearson"],
            "val_ssim_global": val_metrics["ssim_global"],
            "val_ssim_skimage": val_metrics["ssim_skimage"],
            "val_score": val_metrics["score"],
            "val_shape_error": val_metrics["shape_error"],
            "val_visual_error": val_metrics["visual_error"],
            "val_selection_error": val_metrics["selection_error"],
            "val_worst_pearson": val_metrics["worst_pearson"],
            "val_worst_ssim_skimage": val_metrics["worst_ssim_skimage"],
            **{
                f"val_{key}": val_metrics[key]
                for key in (
                    "foreground_error",
                    "background_error",
                    "background_energy",
                    "foreign_energy",
                    "power_error",
                    "edge_error",
                    "multiscale_error",
                )
            },
            "optimizer_updates": updates,
            "amp_skipped": skipped,
            "amp_scale": scaler.get_scale(),
            "grad_norm_mean": grad_norm_sum
            / max(1, len(loaders["train"]) - nonfinite_grad_batches),
            "nonfinite_grad_batches": nonfinite_grad_batches,
            "visual_gain_from_initial": initial_metrics["visual_error"]
            - val_metrics["visual_error"],
            "time_sec": round(dt, 2),
        }
        history.append(row)
        # Прерывание больше не теряет журнал уже законченных эпох.
        (ckpt_path.parent / "training_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        is_best = metric_guard and (
            val_metrics["score"],
            val_metrics[PRIMARY_SSIM],
        ) > (
            best_score,
            best_ssim,
        )
        if is_best:
            best_score = val_metrics["score"]
            best_ssim = val_metrics[PRIMARY_SSIM]
            best_epoch = epoch
            best_state = save_checkpoint(epoch, val_metrics)
            save_validation_previews(
                model, loaders["validation"], device, ckpt_path.parent / "previews"
            )

        mark = "*" if is_best else " "
        print(
            f"Epoch {epoch:03d}/{epochs:03d} [{dt:5.2f}s]{mark} | "
            f"Train P: {row['train_pearson']:.4f} SSIM global: {row['train_ssim']:.4f} "
            f"SSIM skimage: {row['train_ssim_skimage']:.4f} | "
            f"Val P: {val_metrics['pearson']:.4f} SSIM global: {val_metrics['ssim_global']:.4f} "
            f"SSIM skimage: {val_metrics['ssim_skimage']:.4f} "
            f"Shape: {val_metrics['shape_error']:.4f} Visual: {val_metrics['visual_error']:.4f} | "
            f"Select: {val_metrics['selection_error']:.4f} "
            f"BG: {val_metrics['background_error']:.4f} Ghost: {val_metrics['foreign_energy']:.4f} "
            f"Worst P: {val_metrics['worst_pearson']:.4f} "
            f"LR: {row['lr']:.2g} Guard: {'OK' if metric_guard else 'FAIL'} "
            f"Updates: {updates}/{len(loaders['train'])} AMP skipped: {skipped} "
            f"Gain: {row['visual_gain_from_initial']:+.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    (ckpt_path.parent / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\nСохранены лучшие веса модели: {ckpt_path} "
        f"(эпоха {best_epoch}; Selection Error: {-best_score:.4f}; меньше — лучше)"
    )
    if best_epoch == 0:
        print(
            "Нет допустимого улучшения начальной модели по метрикам и критерию выбора; "
            "сохранён epoch=0 текущего запуска."
        )
    print("Экспорт предсказаний в preds/val и preds/diag...")
    export_split_with_optional_polish(
        model,
        loaders["validation"],
        "preds/val",
        device,
        polish_steps=polish_export_steps,
    )
    export_split_with_optional_polish(
        model,
        loaders["diagnostic"],
        "preds/diag",
        device,
        polish_steps=polish_export_steps,
    )
    print("Готово! Папки preds/val и preds/diag сформированы.")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение HoloUnrollNet для синтеза ДОЭ"
    )
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST), help="Путь к manifest.json"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pt", help="Куда сохранить веса"
    )
    parser.add_argument("--epochs", type=int, default=60, help="Число эпох обучения")
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча")
    parser.add_argument(
        "--lr", type=float, default=2e-3, help="Максимальный learning rate"
    )
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=6,
        help="Число развёрнутых физических итераций",
    )
    parser.add_argument(
        "--base-dim", type=int, default=48, help="Базовое число каналов Fourier U-Net"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--smoke", action="store_true", help="Быстрый тест на 8 примерах (train_smoke)"
    )
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    parser.add_argument(
        "--lr-schedule", choices=["onecycle", "plateau"], default="onecycle"
    )
    parser.add_argument(
        "--init-checkpoint", help="Начальные веса для дообучения с новым оптимизатором"
    )
    parser.add_argument(
        "--polish-steps",
        type=int,
        default=25,
        help="Шаги финального дожима при экспорте PNG",
    )
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
        seed=args.seed,
        precision=args.precision,
        init_checkpoint=args.init_checkpoint,
        lr_schedule=args.lr_schedule,
    )


if __name__ == "__main__":
    main()
