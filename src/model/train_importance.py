"""Train importance maps with one, two, then four active Fresnel planes."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from doe_eval import (dequantize_phase_u8, fresnel_reconstruct, normalize_reconstructions,
                      pearson_torch, quantize_phase_u8, ssim_global_torch)
from model.data_connector import build_dataloaders, export_split_predictions
from model.importance import ImportanceDOE


def reconstruct(phase):
    return normalize_reconstructions(fresnel_reconstruct(phase), norm="max", compare="amplitude")


def score(recon, targets):
    return torch.stack((ssim_global_torch(recon, targets), pearson_torch(recon, targets)), dim=-1)


def quantized(phase):
    return dequantize_phase_u8(quantize_phase_u8(phase))


@torch.no_grad()
def prepare_batches(model, loader, device, planes=4):
    batches = []
    rng = torch.Generator().manual_seed(42 + planes)
    for targets, _ in loader:
        targets = targets.to(device, non_blocking=True)
        active = (torch.rand(len(targets), 4, generator=rng).argsort(dim=1) < planes).to(device)
        features, fields = model.prepare(targets, active if planes < 4 else None)
        baseline = score(reconstruct(quantized(model.phase_from_fields(fields))), targets)
        batches.append((features, fields, targets, active, baseline))
    return batches


@torch.no_grad()
def validation(model, batches, equal_weights=False):
    model.eval()
    scores = []
    for features, fields, targets, _, baseline in batches:
        if equal_weights:
            scores.append(baseline)
        else:
            phase = model.predict_prepared(features, fields)
            scores.append(score(reconstruct(quantized(phase)), targets))
    per_plane = torch.cat(scores).mean(dim=0)
    return {"ssim": float(per_plane[:, 0].mean()), "pearson": float(per_plane[:, 1].mean()),
            "planes": per_plane.tolist()}


def optical_objective(scores, baseline, active):
    gains = (scores - baseline) * active[..., None]
    mean_gains = gains.sum(dim=1) / active.sum(dim=1, keepdim=True)
    joint = 0.02 * torch.logsumexp(-mean_gains / 0.02, dim=1)
    pearson_drop = ((-gains[..., 1]).relu().sum(dim=1) / active.sum(dim=1))
    return (joint + 0.25 * pearson_drop).mean(), mean_gains


def corrected_scores(model, fields, corrections, targets):
    phase = model.phase_from_corrections(fields, corrections)
    effective_phase = phase + (quantized(phase) - phase).detach()
    return score(reconstruct(effective_phase), targets)


def make_teacher(model, fields, targets, active, baseline, steps):
    # Optimize the same bounded corrections that the network can predict.
    logits = targets.new_zeros((len(targets), 8, *targets.shape[-2:]), requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.12)
    best = torch.zeros_like(logits, dtype=torch.float16)
    best_gain = targets.new_zeros(len(targets))
    for step in range(steps + 1):
        corrections = logits.tanh()
        scores = corrected_scores(model, fields, corrections, targets)
        loss, gains = optical_objective(scores, baseline, active)
        with torch.no_grad():
            gain = gains.clamp_min(0).prod(dim=1)
            improved = gain > best_gain
            best_gain = torch.maximum(best_gain, gain)
            best = torch.where(improved[:, None, None, None], corrections.half(), best)
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            (loss + 0.005 * corrections.square().mean()).backward()
            optimizer.step()
    return best, best_gain > 0


def train_epoch(model, loader, optimizer, teacher_weight):
    model.train()
    loss_sum = torch.zeros((), device=next(model.parameters()).device)
    seen = 0
    for features, fields, targets, active, baseline, *teacher in loader:
        corrections = model.corrections(features)
        scores = corrected_scores(model, fields, corrections, targets)
        loss, _ = optical_objective(scores, baseline, active)
        if teacher_weight and teacher:
            maps, accepted = teacher
            support = targets * active[..., None, None] * accepted[:, None, None, None]
            weight_delta, phase_delta = (corrections - maps).chunk(2, dim=1)
            # A common phase offset leaves the reconstructed amplitudes unchanged.
            dims = (1, 2, 3)
            phase_offset = (phase_delta * support).sum(dims, keepdim=True) / support.sum(dims, keepdim=True).clamp_min(1)
            map_error = weight_delta.square() + (phase_delta - phase_offset).square()
            fit_maps = (map_error * support).sum() / (2 * support.sum()).clamp_min(1)
            loss = loss + teacher_weight * fit_maps
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.detach() * len(targets)
        seen += len(targets)
    return (loss_sum / seen).item()


def save_best(path, model, epoch, metrics):
    temporary = path.with_suffix(".tmp")
    torch.save({"architecture": model.architecture, "state": model.state_dict(),
                "projections": model.projections, "epoch": epoch, "validation": metrics}, temporary)
    temporary.replace(path)


def fit(model, loaders, device, stages, batch_size, lr, out, teacher_steps=12):
    val_cache = prepare_batches(model, loaders["validation"], device)
    equal = validation(model, val_cache, equal_weights=True)
    best = equal
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    save_best(out / "best.pt", model, 0, equal)
    print(f"Equal weights | SSIM {equal['ssim']:.5f} | Pearson {equal['pearson']:.5f}", flush=True)
    history = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    stages = tuple(stages)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sum(count for _, count in stages), eta_min=lr * 0.1,
    )
    for planes, count in stages:
        if count == 0:
            continue
        batches = prepare_batches(model, loaders["train"], device, planes)
        if teacher_steps:
            batches = [
                (*batch, *make_teacher(model, *batch[1:], steps=teacher_steps))
                for batch in batches
            ]
            accepted = sum(batch[-1].sum() for batch in batches).item()
            print(f"Teacher accepted {accepted}/{sum(len(batch[2]) for batch in batches)} "
                  f"quartets on {planes} active planes", flush=True)
        dataset = TensorDataset(*(torch.cat(parts) for parts in zip(*batches, strict=True)))
        del batches
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for stage_epoch in range(count):
            epoch = len(history) + 1
            # Finish on the task metrics alone once all four planes are active.
            teacher_weight = 0.2
            if planes == 4:
                teacher_weight *= max(0.0, 1.0 - 2.0 * (stage_epoch + 1) / count)
            loss = train_epoch(model, loader, optimizer, teacher_weight)
            scheduler.step()
            metrics = validation(model, val_cache)
            history.append({"epoch": epoch, "active_planes": planes, "loss": loss, "teacher_weight": teacher_weight if teacher_steps else 0.0,
                            "validation": metrics})
            if planes == 4 and metrics["ssim"] > best["ssim"] and metrics["pearson"] > best["pearson"]:
                best, best_epoch = metrics, epoch
                best_state = deepcopy(model.state_dict())
                save_best(out / "best.pt", model, epoch, best)
            print(f"Epoch {epoch:02d} | train planes {planes} | "
                  f"val SSIM {metrics['ssim']:.5f} | Pearson {metrics['pearson']:.5f}", flush=True)
        del loader, dataset
    model.load_state_dict(best_state)
    (out / "history.json").write_text(
        json.dumps({"baseline": equal, "best_epoch": best_epoch, "epochs": history}, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "dataset/data/manifest.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", nargs=3, type=int, default=(2, 2, 16), metavar=("ONE", "TWO", "FOUR"))
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for preparing optical fields")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--projections", type=int, default=12)
    parser.add_argument("--teacher-steps", type=int, default=12, help="Offline optimization steps per quartet; 0 disables")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.teacher_steps < 0:
        parser.error("--teacher-steps must be nonnegative")
    if min(args.epochs) < 0 or args.epochs[-1] < 1:
        parser.error("Use nonnegative epoch counts and at least one four-plane epoch")
    if min(args.batch_size, args.train_batch_size, args.projections, args.threads) < 1 or not 0 < args.lr < 1:
        parser.error("Batch sizes, projections and threads must be positive, and LR must be in (0, 1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("PyTorch does not see a CUDA device")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out = args.out or ROOT / "results" / f"training-importance-{datetime.now():%Y%m%d-%H%M%S-%f}"
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})",
          flush=True)
    loaders = build_dataloaders(args.manifest, batch_size=args.batch_size, num_workers=0,
                               pin_memory=device.type == "cuda")
    out.mkdir(parents=True)
    model = ImportanceDOE(args.projections).to(device)
    fit(model, loaders, device, zip((1, 2, 4), args.epochs, strict=True),
        args.train_batch_size, args.lr, out, args.teacher_steps)
    export_split_predictions(model, loaders["validation"], "preds/val", device)
    print(f"Weights: {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
