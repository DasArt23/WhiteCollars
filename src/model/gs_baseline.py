from __future__ import annotations

"""
src/model/gs_baseline.py — классический эталон Герчберга–Сэкстона (GS) для 4 плоскостей.

Ничего не обучает: для каждой четвёрки мишеней отдельно считает ДОЭ итерациями
  1) поле на ДОЭ = exp(i*phase)  ->  Френель на 4 плоскости;
  2) на каждой плоскости заменяем амплитуду на целевую, фазу оставляем;
  3) обратный Френель на ДОЭ, усреднение 4 плоскостей, берём фазу;
и повторяет --iters раз. Метрики считаются так же, как в train.py: по 8-битному ДОЭ
(Pearson, SSIM global, SSIM local 7x7).

Запуск:
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python src/model/gs_baseline.py \
      --manifest <путь>/manifest.json --split validation --iters 200 --device cuda
"""

import argparse
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from doe_eval import (  # noqa: E402
    fresnel_reconstruct,
    normalize_reconstructions,
    pearson_torch,
    ssim_global_torch,
    ste_quantize,
)
from doe_eval.physics import fresnel_propagate_complex  # noqa: E402
from model.data_connector import build_dataloaders  # noqa: E402
from model.holonet import fresnel_backward_to_slm, ssim_local_7x7_torch  # noqa: E402


@torch.no_grad()
def gs_phase(targets: torch.Tensor, iters: int = 200, seed: int = 0) -> torch.Tensor:
    """targets [B, 4, N, N] в [0, 1] -> фаза ДОЭ [B, N, N] в [-pi, pi]."""
    g = torch.Generator(device=targets.device).manual_seed(seed)
    b, _, n, _ = targets.shape
    phase = (torch.rand(b, n, n, device=targets.device, generator=g) * 2 - 1) * torch.pi
    for _ in range(iters):
        field = fresnel_propagate_complex(phase)  # [B, 4, N, N]
        new_field = torch.polar(targets, field.angle())  # амплитуда мишени, фаза текущая
        phase = fresnel_backward_to_slm(new_field, None).angle()
    return phase


@torch.no_grad()
def metrics_for(phase: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    recon = normalize_reconstructions(fresnel_reconstruct(ste_quantize(phase)), norm="max", compare="amplitude")
    return {
        "pearson": pearson_torch(recon, targets).mean(dim=1),
        "ssim_global": ssim_global_torch(recon, targets).mean(dim=1),
        "ssim_local": ssim_local_7x7_torch(recon, targets).mean(dim=1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="GS-эталон на сплите датасета")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test", "diagnostic"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="Только первые N четвёрок (0 = все)")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    loaders = build_dataloaders(args.manifest, batch_size=args.batch_size, num_workers=0, pin_memory=False)
    loader = loaders[args.split]

    sums = {"pearson": 0.0, "ssim_global": 0.0, "ssim_local": 0.0}
    n = 0
    t_total = 0.0
    for targets, _ in loader:
        if args.limit and n >= args.limit:
            break
        targets = targets.to(device).float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        phase = gs_phase(targets, iters=args.iters)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        m = metrics_for(phase, targets)
        for k in sums:
            sums[k] += float(m[k].sum().item())
        n += targets.shape[0]

    print(f"GS-{args.iters} на {args.split}: {n} четвёрок")
    print(
        f"  Pearson {sums['pearson'] / n:.4f} | SSIM global {sums['ssim_global'] / n:.4f} | "
        f"SSIM local {sums['ssim_local'] / n:.4f}"
    )
    print(f"  время: {t_total / n * 1000:.1f} мс на четвёрку (батчем по {args.batch_size})")


if __name__ == "__main__":
    main()
