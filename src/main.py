from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from doe_eval.cli import evaluate_pair_arrays
from doe_eval.config import (
    DEFAULT_COMPARE,
    DEFAULT_MANIFEST,
    DEFAULT_NORM,
    DISTANCES,
)
from doe_eval.dataset import sample_image_paths
from doe_eval.io import (
    load_doe_png,
    load_four_targets,
    save_comparison_png,
    save_doe_png,
    save_recon_png,
)
from doe_eval.physics import (
    fresnel_reconstruct,
    normalize_reconstructions,
)
from model.data_connector import build_dataloaders
from model.holonet import HoloUnrollNet, hybrid_infer_phase
from model.train import export_split_with_optional_polish, train_model


def load_trained_holonet(
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    device_str: str = "auto",
) -> tuple[HoloUnrollNet, torch.device, dict[str, Any]]:
    device = torch.device(
        "cuda"
        if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "auto" else device_str)
    )
    ckpt_p = Path(checkpoint_path)
    meta: dict[str, Any] = {"checkpoint": str(ckpt_p), "loaded": False}

    unroll_steps = 6
    base_dim = 48
    equalize_plane_power = False  # Старые checkpoint сохраняют прежний forward.
    phase_carrier_seed = None
    state_dict = None

    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"Чекпоинт не найден: {ckpt_p}. Сначала выполните project train."
        )
    if ckpt_p.is_file():
        payload = torch.load(ckpt_p, map_location=device, weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            unroll_steps = int(payload.get("unroll_steps", 6))
            base_dim = int(payload.get("base_dim", 48))
            equalize_plane_power = bool(payload.get("equalize_plane_power", False))
            phase_carrier_seed = payload.get("phase_carrier_seed")
            state_dict = payload["state_dict"]
            meta["loaded"] = True
            meta["epoch"] = payload.get("epoch")
            meta["val_metrics"] = payload.get("val_metrics")
        elif isinstance(payload, dict):
            state_dict = payload
            meta["loaded"] = True

    model = HoloUnrollNet(
        unroll_steps=unroll_steps,
        base_dim=base_dim,
        equalize_plane_power=equalize_plane_power,
        phase_carrier_seed=phase_carrier_seed,
    ).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"{ckpt_p}: отсутствует state_dict модели")
    model.eval()
    return model, device, meta


def synthesize_doe_and_reconstruct(
    model: HoloUnrollNet,
    device: torch.device,
    targets_01: np.ndarray,
    polish_steps: int = 25,
) -> dict[str, Any]:
    """
    Принимает 4 целевых изображения [4, 128, 128] в [0, 1],
    рассчитывает 8-битный ДОЭ и 4 восстановленных изображения по Френелю + метрики ТЗ.
    """
    targets_t = torch.from_numpy(targets_01).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred_phase_t = hybrid_infer_phase(model, targets_t, polish_steps=polish_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    phase_np = pred_phase_t.squeeze(0).detach().cpu().numpy()
    # Проверяем ровно тот формат, который передаётся экспертам: PNG на диске.
    with tempfile.TemporaryDirectory(prefix="doe_roundtrip_") as temporary:
        doe_path = Path(temporary) / "doe.png"
        save_doe_png(phase_np, doe_path)
        phase_quant_np = load_doe_png(doe_path)
        with Image.open(doe_path) as image:
            doe_u8 = np.array(image)

    # Восстановление 4 изображений из 8-битного ДОЭ численным расчётом дифракции Френеля
    recon_amp = fresnel_reconstruct(torch.from_numpy(phase_quant_np).to(device))
    recons_01 = (
        normalize_reconstructions(recon_amp, norm=DEFAULT_NORM, compare=DEFAULT_COMPARE)
        .detach()
        .cpu()
        .numpy()
    )

    metrics_report = evaluate_pair_arrays(targets_01, recons_01)

    return {
        "phase_rad": phase_quant_np,
        "doe_u8": doe_u8,
        "recons_01": recons_01,
        "report": {
            "device": str(device),
            "polish_steps": polish_steps,
            "inference_ms": round(elapsed_ms, 2),
            "norm": DEFAULT_NORM,
            "compare": DEFAULT_COMPARE,
            **metrics_report,
        },
    }


def cmd_predict(args: argparse.Namespace) -> None:
    if args.sample_dir is not None:
        sdir = Path(args.sample_dir)
        input_paths = sample_image_paths(sdir)
    else:
        input_paths = [Path(p) for p in args.inputs]

    targets_01 = load_four_targets(input_paths)
    model, device, meta = load_trained_holonet(args.checkpoint, args.device)
    result = synthesize_doe_and_reconstruct(
        model, device, targets_01, polish_steps=args.polish_steps
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_doe_png(result["phase_rad"], out_dir / "doe.png")
    save_comparison_png(
        targets_01, result["recons_01"], out_dir / "preview" / "comparison.png"
    )
    for i in range(len(DISTANCES)):
        save_recon_png(result["recons_01"][i], out_dir / f"recon_{i + 1}.png")

    result["report"]["checkpoint"] = meta["checkpoint"]
    result["report"]["checkpoint_loaded"] = meta["loaded"]
    (out_dir / "metrics.json").write_text(
        json.dumps(result["report"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rep = result["report"]
    print("=" * 74)
    print(
        f" Чекпоинт: {meta['checkpoint']} ({'загружен' if meta['loaded'] else 'init'}) | "
        f"Устройство: {rep['device']} | Время: {rep['inference_ms']:.2f} мс"
    )
    print("-" * 74)
    print(
        f"{'Plane':<8} {'z (mm)':<10} {'Pearson':>12} {'SSIM (global)':>16} {'SSIM (skimage)':>16}"
    )
    print("-" * 74)
    for row in rep["planes"]:
        print(
            f"{row['plane']:<8} {row['distance_mm']:<10.1f} "
            f"{row['pearson']:>12.4f} {row['ssim_global']:>16.4f} {row['ssim_skimage']:>16.4f}"
        )
    print("-" * 74)
    m = rep["mean"]
    print(
        f"{'Mean':<19} {m['pearson']:>12.4f} {m['ssim_global']:>16.4f} {m['ssim_skimage']:>16.4f}"
    )
    print("=" * 74)
    print(
        f"Файлы сохранены в {out_dir.resolve()}: doe.png, recon_1..4.png, preview/comparison.png, metrics.json"
    )


def cmd_export(args: argparse.Namespace) -> None:
    model, device, _ = load_trained_holonet(args.checkpoint, args.device)
    loaders = build_dataloaders(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=0,
        mode="v1",
        pin_memory=(device.type == "cuda"),
    )
    saved = export_split_with_optional_polish(
        model, loaders[args.split], args.out, device, polish_steps=args.polish_steps
    )
    print(
        f"Экспортировано {len(saved)} ДОЭ-масок для сплита '{args.split}' в папку: {args.out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="project",
        description="Единый интерфейс обучения и использования модели синтеза ДОЭ (HoloUnrollNet)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Команда predict (основной интерфейс взаимодействия с готовой моделью на 4 картинках)
    p_pred = sub.add_parser(
        "predict", help="Синтезировать ДОЭ и восстановить 4 плоскости из 4 входных PNG"
    )
    grp = p_pred.add_mutually_exclusive_group(required=True)
    grp.add_argument("--inputs", nargs=4, help="Пути к 4 целевым PNG (128x128, 8 бит)")
    grp.add_argument(
        "--sample-dir",
        help="Папка с файлами 01.png..04.png (например, src/dataset/data/diagnostic/0000)",
    )
    p_pred.add_argument(
        "--out",
        default="outputs/predict",
        help="Папка для сохранения doe.png, recon_1..4.png и metrics.json",
    )
    p_pred.add_argument(
        "--checkpoint", default="checkpoints/best_model.pt", help="Путь к весам модели"
    )
    p_pred.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_pred.add_argument(
        "--polish-steps",
        type=int,
        default=25,
        help="Шаги физического дожима (0 = только нейросеть)",
    )
    p_pred.set_defaults(func=cmd_predict)

    # 2. Команда export (пакетная генерация ДОЭ для validation / test / diagnostic)
    p_exp = sub.add_parser(
        "export", help="Экспортировать предсказанные ДОЭ для всего сплита датасета"
    )
    p_exp.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_exp.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test", "diagnostic", "train"],
    )
    p_exp.add_argument("--out", default="preds/val")
    p_exp.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_exp.add_argument("--batch-size", type=int, default=16)
    p_exp.add_argument("--polish-steps", type=int, default=25)
    p_exp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_exp.set_defaults(func=cmd_export)

    # 3. Команда train (запуск обучения модели)
    p_tr = sub.add_parser("train", help="Обучить модель HoloUnrollNet")
    p_tr.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_tr.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_tr.add_argument("--epochs", type=int, default=60)
    p_tr.add_argument("--batch-size", type=int, default=16)
    p_tr.add_argument("--lr", type=float, default=2e-3)
    p_tr.add_argument("--unroll-steps", type=int, default=6)
    p_tr.add_argument("--base-dim", type=int, default=48)
    p_tr.add_argument("--polish-steps", type=int, default=25)
    p_tr.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_tr.add_argument("--smoke", action="store_true")
    p_tr.add_argument("--seed", type=int, default=3701)
    p_tr.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    p_tr.set_defaults(
        func=lambda a: train_model(
            manifest_path=a.manifest,
            checkpoint_out=a.checkpoint,
            epochs=a.epochs,
            batch_size=a.batch_size,
            lr=a.lr,
            unroll_steps=a.unroll_steps,
            base_dim=a.base_dim,
            mode="train_smoke" if a.smoke else "v1",
            device_str=a.device,
            polish_export_steps=a.polish_steps,
            seed=a.seed,
            precision=a.precision,
        )
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
