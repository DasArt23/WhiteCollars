from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baseline import gerchberg_saxton_multiplane
from .config import DEFAULT_COMPARE, DEFAULT_NORM, DISTANCES, N
from .dataset import FourImageDataset
from .io import (
    load_doe_png,
    load_four_targets,
    save_comparison_png,
    save_doe_png,
    save_recon_png,
)
from .metrics import pearson, ssim_global, ssim_skimage
from .physics import (
    dequantize_phase_u8,
    fresnel_reconstruct,
    normalize_reconstructions,
    quantize_phase_u8,
)
from .reference_numpy import fresnel_reconstruct_numpy


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def evaluate_pair_arrays(
    targets: np.ndarray,
    recons_01: np.ndarray,
) -> dict[str, Any]:
    per_plane = []
    for i, z in enumerate(DISTANCES):
        p_val = pearson(targets[i], recons_01[i])
        sg_val = ssim_global(targets[i], recons_01[i], data_range=1.0)
        ss_val = ssim_skimage(targets[i], recons_01[i], data_range=1.0)
        per_plane.append(
            {
                "plane": i + 1,
                "distance_mm": round(z * 1e3, 2),
                "pearson": p_val,
                "ssim_global": sg_val,
                "ssim_skimage": ss_val,
            }
        )

    mean_metrics = {
        "pearson": float(np.mean([m["pearson"] for m in per_plane])),
        "ssim_global": float(np.mean([m["ssim_global"] for m in per_plane])),
        "ssim_skimage": float(np.mean([m["ssim_skimage"] for m in per_plane])),
    }
    return {"planes": per_plane, "mean": mean_metrics}


def print_metrics_table(report: dict[str, Any]) -> None:
    header = f"{'Plane':<8} {'z (mm)':<10} {'Pearson':>10} {'SSIM (global)':>15} {'SSIM (skimage)':>15}"
    print(header)
    print("-" * len(header))
    for row in report["planes"]:
        print(
            f"{row['plane']:<8} {row['distance_mm']:<10.1f} "
            f"{row['pearson']:>10.4f} {row['ssim_global']:>15.4f} {row['ssim_skimage']:>15.4f}"
        )
    print("-" * len(header))
    m = report["mean"]
    print(f"{'Mean':<19} {m['pearson']:>10.4f} {m['ssim_global']:>15.4f} {m['ssim_skimage']:>15.4f}")


def reconstruct_from_doe_file(
    doe_path: Path,
    device: torch.device,
    norm: str,
    compare: str,
) -> np.ndarray:
    phase_np = load_doe_png(doe_path)
    phase_t = torch.from_numpy(phase_np).to(device)
    recon_amp = fresnel_reconstruct(phase_t)
    recon_01 = normalize_reconstructions(recon_amp, norm=norm, compare=compare)
    return recon_01.detach().cpu().numpy()


def cmd_reconstruct(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_four_targets(args.inputs)
    recons_01 = reconstruct_from_doe_file(Path(args.doe), device, args.norm, args.compare)

    for i in range(len(DISTANCES)):
        save_recon_png(recons_01[i], out_dir / f"recon_{i + 1}.png")

    report = evaluate_pair_arrays(targets, recons_01)
    save_comparison_png(targets, recons_01, out_dir / "preview" / "comparison.png")
    report["norm"] = args.norm
    report["compare"] = args.compare

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_metrics_table(report)


def cmd_evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    dataset = FourImageDataset(args.data, split=args.split)
    doe_dir = Path(args.doe_dir)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for targets_t, sample_id in dataset:
        doe_path = doe_dir / f"{sample_id}.png"
        # Совместимость с экспортом по имени папки и существующими manifest ID.
        if not doe_path.is_file():
            paths, _ = dataset.items[len(rows)]
            alternatives = [doe_dir / f"{paths[0].parent.name}.png",
                            doe_dir / f"{paths[0].parent.parent.name}_{sample_id}.png"]
            doe_path = next((p for p in alternatives if p.is_file()), doe_path)
        recons_01 = reconstruct_from_doe_file(doe_path, device, args.norm, args.compare)
        report = evaluate_pair_arrays(targets_t.numpy(), recons_01)
        row = {
            "id": sample_id,
            "norm": args.norm,
            "compare": args.compare,
            "pearson_mean": report["mean"]["pearson"],
            "ssim_global_mean": report["mean"]["ssim_global"],
            "ssim_skimage_mean": report["mean"]["ssim_skimage"],
        }
        for p in report["planes"]:
            idx = p["plane"]
            row[f"pearson_{idx}"] = p["pearson"]
            row[f"ssim_global_{idx}"] = p["ssim_global"]
            row[f"ssim_skimage_{idx}"] = p["ssim_skimage"]
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    p_mean = float(np.mean([r["pearson_mean"] for r in rows]))
    sg_mean = float(np.mean([r["ssim_global_mean"] for r in rows]))
    ss_mean = float(np.mean([r["ssim_skimage_mean"] for r in rows]))
    print(f"Evaluated {len(rows)} samples -> saved to {out_csv}")
    print(f"Dataset Mean | Pearson: {p_mean:.4f} | SSIM (global): {sg_mean:.4f} | SSIM (skimage): {ss_mean:.4f}")


def cmd_baseline(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_four_targets(args.inputs)
    targets_t = torch.from_numpy(targets).to(device)

    phase_t = gerchberg_saxton_multiplane(targets_t, iters=args.iters)
    doe_path = out_dir / "doe.png"
    save_doe_png(phase_t.detach().cpu().numpy(), doe_path)

    # Оценка строго из сохранённого 8-битного PNG (после квантования)
    recons_01 = reconstruct_from_doe_file(doe_path, device, args.norm, args.compare)
    for i in range(len(DISTANCES)):
        save_recon_png(recons_01[i], out_dir / f"recon_{i + 1}.png")

    report = evaluate_pair_arrays(targets, recons_01)
    save_comparison_png(targets, recons_01, out_dir / "preview" / "comparison.png")
    report["iters"] = args.iters
    report["norm"] = args.norm
    report["compare"] = args.compare

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_metrics_table(report)


def cmd_selfcheck(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    rng = np.random.default_rng(0)
    phase_np = rng.uniform(-math.pi, math.pi, size=(N, N)).astype(np.float32)

    ref = fresnel_reconstruct_numpy(phase_np)
    tor = fresnel_reconstruct(torch.from_numpy(phase_np).to(device)).cpu().numpy()
    rel_err = float(np.linalg.norm(tor - ref) / np.linalg.norm(ref))
    assert rel_err < 1e-4, f"Ошибка сверки с эталоном: {rel_err}"

    u8 = quantize_phase_u8(np.array([-math.pi, math.pi], dtype=np.float32))
    assert int(u8[0]) == 0 and int(u8[1]) == 255
    back = dequantize_phase_u8(quantize_phase_u8(phase_np))
    step = 2.0 * math.pi / 255.0
    assert float(np.max(np.abs(back - phase_np))) <= 0.5 * step + 1e-5

    print(f"[OK] selfcheck passed on device={device} (rel_err={rel_err:.2e})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doe-eval", description="Оценка и синтез ДОЭ")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--norm", choices=["max", "minmax"], default=DEFAULT_NORM)
        p.add_argument("--compare", choices=["amplitude", "intensity"], default=DEFAULT_COMPARE)
        p.add_argument("--device", default="auto", help="auto | cpu | cuda")

    p_rec = sub.add_parser("reconstruct", help="Восстановление 4 картинок из ДОЭ и расчёт метрик")
    p_rec.add_argument("--inputs", nargs=4, required=True, help="Пути к 4 исходным PNG")
    p_rec.add_argument("--doe", required=True, help="Путь к PNG с фазой ДОЭ")
    p_rec.add_argument("--out", required=True, help="Директория для сохранения результатов")
    add_common_flags(p_rec)
    p_rec.set_defaults(func=cmd_reconstruct)

    p_eval = sub.add_parser("evaluate", help="Пакетная проверка предсказанных ДОЭ на датасете")
    p_eval.add_argument("--data", required=True, help="Путь к папке датасета (data/test)")
    p_eval.add_argument("--doe-dir", required=True, help="Папка с предсказанными <id>.png")
    p_eval.add_argument("--split", help="Фильтр split при чтении manifest.json")
    p_eval.add_argument("--out", default="results.csv", help="Путь к итоговому CSV")
    add_common_flags(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_base = sub.add_parser("baseline", help="Классический алгоритм Герчберга–Сэкстона")
    p_base.add_argument("--inputs", nargs=4, required=True, help="Пути к 4 исходным PNG")
    p_base.add_argument("--out", required=True, help="Директория для сохранения ДОЭ и реконструкций")
    p_base.add_argument("--iters", type=int, default=200, help="Число итераций GS")
    add_common_flags(p_base)
    p_base.set_defaults(func=cmd_baseline)

    p_self = sub.add_parser("selfcheck", help="Быстрая самопроверка физики и квантования")
    p_self.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p_self.set_defaults(func=cmd_selfcheck)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
