"""Train the pixel-weight retrieval model using manifest splits or the legacy ZAKHAR-A holdout."""

import argparse
import os
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from .data_connector import ManifestQuartetDataset
from .retrieval import (
    RetrievalDOE,
    plane_metrics,
    reconstruct,
    save_example,
    write_json,
)

ROOT = Path(__file__).resolve().parent


def is_zakhar_a(sample):
    return (
        sample.params.get("source") == "ZAKHAR_FULL_TRAIN_train512"
        and sample.params.get("kind") == "A"
    )


def split_a(dataset, reserve=False):
    records = [
        (i, sample.params["origin_id"])
        for i, sample in enumerate(dataset.samples)
        if is_zakhar_a(sample)
    ]
    if len({origin for _, origin in records}) != len(records) or len(records) <= 44:
        raise ValueError("Expected independent ZAKHAR-A origins for a held-out split")
    # Exclude the twelve examples previously inspected to choose an optical teacher.
    pilot = {index for index, _ in records[:6] + records[-6:]}
    ordered = sorted(records, key=lambda item: sha256(item[1].encode()).digest())
    holdout = [index for index, _ in ordered if index not in pilot][:32]
    training = [index for index, _ in ordered if index not in holdout]
    # Select only from the tail of the old training order, excluding inspected pilot cases.
    reserve_indices = [index for index in training[64:] if index not in pilot][-32:]
    return [
        index for index in training if index not in reserve_indices
    ], reserve_indices if reserve else holdout


def available_cpu_threads():
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def image_loader(dataset, batch_size, workers, device):
    options = {}
    if workers:
        # Workers decode PNGs on CPU even when the main process uses CUDA.
        options["multiprocessing_context"] = "spawn"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        **options,
    )


def plane_mse(model, pupil, images):
    return (reconstruct(model.encode(pupil)) - images).square().mean(dim=(-2, -1))


def teacher_selection(error, initial_error, max_mse):
    worst = error.amax(dim=1)
    finite = torch.isfinite(worst)
    strict = finite & (worst < max_mse)
    if initial_error is None:
        return strict, strict
    # A lower mean alone can hide a deteriorating worst plane.
    improves = (error.mean(dim=1) < 0.95 * initial_error.mean(dim=1)) & (
        worst <= initial_error.amax(dim=1)
    )
    relative = finite & torch.isfinite(initial_error).all(dim=1) & improves
    return strict | relative, strict


def json_mse(values):
    return [
        [value if np.isfinite(value) else None for value in row]
        for row in values.tolist()
    ]


@torch.no_grad()
def probe_teacher(model, loader, device, steps, max_mse, out, *, allow_relative):
    model.eval()
    sample_ids = []
    initial_batches = []
    errors = {count: [] for count in steps}
    seconds = dict.fromkeys(steps, 0.0)
    for batch, (images, ids) in enumerate(loader, 1):
        images = images.to(device, non_blocking=loader.pin_memory)
        _, amplitude, basis = model.prepare(images)
        initial_batches.append(
            plane_mse(model, model._back_project(basis), images).cpu()
        )
        sample_ids.extend(ids)
        for count in steps:
            start = perf_counter()
            pupil = model.optical_solve(amplitude, count)
            errors[count].append(plane_mse(model, pupil, images).cpu())
            seconds[count] += perf_counter() - start
        print(
            f"Teacher probe | {batch}/{len(loader)} | {len(sample_ids)} train quartets",
            flush=True,
        )

    initial = torch.cat(initial_batches)
    result = {
        "sample_ids": sample_ids,
        "threshold": max_mse,
        "min_relative_mean_gain": 0.05 if allow_relative else None,
        "initial_per_plane_mse": json_mse(initial),
        "teacher": {},
    }
    for count in steps:
        error = torch.cat(errors[count])
        useful, strict = teacher_selection(
            error, initial if allow_relative else None, max_mse
        )
        finite = torch.isfinite(error).all(dim=1)
        valid = error[finite]
        mean_mse = valid.mean().item() if len(valid) else None
        worst = valid.amax(dim=1)
        summary = {
            "accepted": useful.sum().item(),
            "strict": strict.sum().item(),
            "nonfinite": (~finite).sum().item(),
            "mean_mse": mean_mse,
            "median_worst_mse": worst.quantile(0.5).item() if len(worst) else None,
            "best_worst_mse": worst.min().item() if len(worst) else None,
            "seconds": seconds[count],
            "accepted_ids": [
                sid
                for sid, keep in zip(sample_ids, useful.tolist(), strict=True)
                if keep
            ],
            "per_plane_mse": json_mse(error),
        }
        result["teacher"][str(count)] = summary
        median = summary["median_worst_mse"]
        median_text = f"{median:.5f}" if median is not None else "n/a"
        print(
            f"{count} steps | accepted {summary['accepted']}/{len(sample_ids)} "
            f"({summary['strict']} below {max_mse:g}) | median worst-plane MSE {median_text} "
            f"| {seconds[count]:.1f}s",
            flush=True,
        )
    write_json(out, result)
    print(f"Report: {out.resolve()}", flush=True)


@torch.no_grad()
def prepare_labels(model, loader, device, *, allow_relative, teacher_steps, max_mse):
    examples, bases, pupils = [], [], []
    retained = strict_count = seen = 0
    best_worst = float("inf")
    for batch, (images, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=loader.pin_memory)
        targets, amplitude, basis = model.prepare(images)
        teacher = model.optical_solve(amplitude, teacher_steps)
        error = plane_mse(model, teacher, images)
        initial_error = (
            plane_mse(model, model._back_project(basis), images)
            if allow_relative
            else None
        )
        useful, strict = teacher_selection(error, initial_error, max_mse)
        worst = error.amax(dim=1)
        finite = torch.isfinite(worst)
        indices = useful.nonzero(as_tuple=True)[0]
        if indices.numel():
            examples.append(targets.index_select(0, indices).cpu())
            bases.append(basis.index_select(0, indices).cpu())
            pupils.append(teacher.index_select(0, indices).cpu())
        seen += len(images)
        retained += indices.numel()
        strict_count += strict.sum().item()
        best_worst = min(
            best_worst, worst.masked_fill(~finite, float("inf")).min().item()
        )
        if batch == 1 or batch % 10 == 0 or batch == len(loader):
            print(
                f"Optical teacher | {batch}/{len(loader)} | accepted {retained}/{seen} "
                f"({strict_count} below {max_mse:g}) | best worst-plane MSE {best_worst:.5g}",
                flush=True,
            )
    if not retained:
        raise RuntimeError(
            f"Teacher accepted 0/{seen} masks. Best worst-plane MSE: {best_worst:.5g}; "
            f"threshold: {max_mse:g}. Check the targets or adjust --teacher-max-mse explicitly."
        )
    targets = torch.cat(examples)
    examples.clear()
    basis = torch.cat(bases)
    bases.clear()
    return TensorDataset(targets, basis, torch.cat(pupils))


@torch.no_grad()
def evaluate(model, loader, device, out, *, split=None):
    model.eval()
    scores, sample_ids = {}, []
    for batch, (images, ids) in enumerate(loader, 1):
        images = images.to(device, non_blocking=loader.pin_memory)
        if split:
            mask, weights = model.predict_weighted(images)
            masks = {"weighted_refined": mask}
        else:
            masks, weights = model.candidates(images)
        sample_ids.extend(ids)
        for name, mask in masks.items():
            recon = reconstruct(mask)
            values = plane_metrics(recon, images)
            scores.setdefault(name, []).append(values.cpu())
            if batch == 1 and name == "weighted_refined":
                save_example(out / "example", mask[0], recon[0], values[0], ids[0])
                np.save(out / "example/weights.npy", weights[0].cpu().numpy())
        print(f"{split or 'Held-out A'} | {batch}/{len(loader)}", flush=True)
    result = {}
    for name, batches in scores.items():
        values = torch.cat(batches)
        floor = values.amin(dim=(1, 2))
        row = {
            "ssim": values[..., 0].mean().item(),
            "pearson": values[..., 1].mean().item(),
            "pass_0.9": (floor > 0.9).sum().item(),
            "pass_0.95": (floor > 0.95).sum().item(),
            "per_quartet": {
                sid: metrics.tolist()
                for sid, metrics in zip(sample_ids, values, strict=True)
            },
        }
        result[name] = row
        print(
            f"{name} | SSIM {row['ssim']:.4f} | Pearson {row['pearson']:.4f} | "
            f">0.90 {row['pass_0.9']}/{len(values)} | >0.95 {row['pass_0.95']}/{len(values)}",
            flush=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT.parent / "dataset/data/manifest.json"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=available_cpu_threads(),
        help="Total CPU thread budget (default: all visible logical CPUs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="PNG loader processes (default: automatic; 0 disables multiprocessing)",
    )
    parser.add_argument(
        "--optical-steps",
        type=int,
        help="Refinement steps (default: 300 or checkpoint setting)",
    )
    parser.add_argument(
        "--teacher-steps",
        type=int,
        help="Teacher iterations (default: same as --optical-steps)",
    )
    parser.add_argument(
        "--teacher-probe",
        type=int,
        nargs="+",
        metavar="STEPS",
        help="Compare teacher iteration counts on the same TRAIN quartets without training",
    )
    parser.add_argument(
        "--teacher-max-mse",
        type=float,
        default=0.005,
        help="Absolute worst-plane MSE threshold for teacher labels (default: 0.005)",
    )
    parser.add_argument(
        "--amplitude-steps",
        type=int,
        default=0,
        help="Experimental adjoint refinement when evaluating an original split",
    )
    parser.add_argument(
        "--retry-steps", type=int, help="Set the optical retry budget; 0 disables it"
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        help="Evaluate the entire original split",
    )
    parser.add_argument(
        "--reserve", action="store_true", help="Evaluate on the independent A reserve"
    )
    parser.add_argument("--images", type=Path, nargs=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or (
        args.train_limit is not None and args.train_limit < 1
    ):
        parser.error("Invalid training settings")
    if args.optical_steps is not None and args.optical_steps < 10:
        parser.error("--optical-steps must be at least 10")
    if args.teacher_steps is not None and args.teacher_steps < 10:
        parser.error("--teacher-steps must be at least 10")
    if args.teacher_probe and (
        min(args.teacher_probe) < 10
        or len(set(args.teacher_probe)) != len(args.teacher_probe)
    ):
        parser.error("--teacher-probe requires distinct counts of at least 10")
    if args.teacher_probe and (
        args.weights or args.split or args.reserve or args.images or args.teacher_steps
    ):
        parser.error(
            "--teacher-probe uses TRAIN and cannot be combined with weights, split, reserve, images or teacher-steps"
        )
    if not 0 < args.teacher_max_mse <= 1:
        parser.error("--teacher-max-mse must be in (0, 1]")
    if args.amplitude_steps < 0 or (
        args.retry_steps is not None and args.retry_steps < 0
    ):
        parser.error("Refinement counts must be nonnegative")
    if args.amplitude_steps and not args.split:
        parser.error("--amplitude-steps requires an original evaluation split")
    if (args.reserve or args.split) and not args.weights:
        parser.error("--reserve and --split require --weights")
    if args.split and (args.reserve or args.images):
        parser.error("--split is incompatible with --reserve and --images")
    if args.images and not args.weights:
        parser.error("--images requires --weights")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    workers = args.workers
    if workers is None:
        if args.device == "cuda":
            workers = min(8, args.cpu_threads - 1)
        else:
            workers = min(4, (args.cpu_threads - 1) // 4)
    if not 0 <= workers < args.cpu_threads:
        parser.error("--workers must be nonnegative and smaller than --cpu-threads")
    decode_threads = args.cpu_threads - workers
    torch.manual_seed(42)
    torch.set_num_threads(decode_threads)
    device = torch.device(args.device)
    out = args.out or ROOT / "results" / f"retrieval-{datetime.now():%Y%m%d-%H%M%S-%f}"
    checkpoint = (
        torch.load(args.weights, map_location="cpu", weights_only=True)
        if args.weights
        else None
    )
    if (
        checkpoint is not None
        and checkpoint.get("architecture") != "pixel_retrieval_v1"
    ):
        parser.error("Incompatible weights")
    config = dict(checkpoint["config"]) if checkpoint else {"optical_steps": 300}
    if args.optical_steps is not None:
        config["optical_steps"] = args.optical_steps
    if args.retry_steps is not None:
        config["retry_steps"] = args.retry_steps
    if args.amplitude_steps:
        config["amplitude_steps"] = args.amplitude_steps
    if args.images:
        from .prob import main as predict

        argv = [
            "--weights",
            str(args.weights),
            "--device",
            args.device,
            "--out",
            str(out),
        ]
        if args.optical_steps is not None:
            argv.extend(("--optical-steps", str(args.optical_steps)))
        predict([*argv, "--images", *(str(path) for path in args.images)])
        return
    model = RetrievalDOE(**config).to(device)
    config["retry_steps"] = model.retry_steps
    if args.teacher_probe:
        dataset = ManifestQuartetDataset(args.manifest, split="train", preload=False)
        zakhar = any(is_zakhar_a(sample) for sample in dataset.samples)
        indices = split_a(dataset)[0] if zakhar else range(len(dataset))
        loader = image_loader(
            Subset(dataset, indices[: args.train_limit or 64]),
            args.batch_size,
            workers,
            device,
        )
        out.mkdir(parents=True, exist_ok=False)
        probe_teacher(
            model,
            loader,
            device,
            args.teacher_probe,
            args.teacher_max_mse,
            out / "teacher_probe.json",
            allow_relative=not zakhar,
        )
        return
    out.mkdir(parents=True, exist_ok=False)
    if checkpoint:
        model.load_state_dict(checkpoint["state"])
    dataset = ManifestQuartetDataset(
        args.manifest, split=args.split or "train", preload=False
    )
    evaluation_dataset = dataset
    evaluation_split = args.split
    if args.split:
        holdout = list(range(len(dataset)))
    elif args.reserve or any(is_zakhar_a(sample) for sample in dataset.samples):
        training, holdout = split_a(dataset, reserve=args.reserve)
    else:
        training = list(range(len(dataset)))
        evaluation_split = "validation"
        evaluation_dataset = ManifestQuartetDataset(
            args.manifest, split=evaluation_split, preload=False
        )
        holdout = list(range(len(evaluation_dataset)))
    history = {
        "held_out": [evaluation_dataset.samples[i].sample_id for i in holdout],
        "epochs": [],
        "config": config,
        "split": evaluation_split or ("A-reserve" if args.reserve else "A-holdout"),
    }
    if not checkpoint:
        if args.train_limit is not None:
            training = training[: args.train_limit]
        history["training_candidates"] = [
            dataset.samples[i].sample_id for i in training
        ]
        print(
            f"Optical teacher | {decode_threads} PyTorch threads | {workers} PNG workers",
            flush=True,
        )
        loader = image_loader(
            Subset(dataset, training), args.batch_size, workers, device
        )
        teacher_steps = args.teacher_steps or model.optical_steps
        labels = prepare_labels(
            model,
            loader,
            device,
            allow_relative=evaluation_dataset is not dataset,
            teacher_steps=teacher_steps,
            max_mse=args.teacher_max_mse,
        )
        history["teacher_masks"] = len(labels)
        history["teacher_steps"] = teacher_steps
        history["teacher_max_mse"] = args.teacher_max_mse
        # The teacher's accepted examples are already in RAM.
        torch.set_num_threads(args.cpu_threads)
        loader = DataLoader(
            labels,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
        )
        print(
            f"Training | {args.cpu_threads} PyTorch threads | {len(labels)} quartets in RAM",
            flush=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for targets, basis, reference in loader:
                data = (
                    targets.to(device, non_blocking=loader.pin_memory),
                    None,
                    basis.to(device, non_blocking=loader.pin_memory),
                )
                loss = model.training_loss(
                    data, reference.to(device, non_blocking=loader.pin_memory)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                total += loss.item()
            average = total / len(loader)
            history["epochs"].append(average)
            print(
                f"Epoch {epoch}/{args.epochs} | weight Smooth L1 {average:.6f}",
                flush=True,
            )
            torch.save(
                {
                    "architecture": "pixel_retrieval_v1",
                    "config": config,
                    "state": model.state_dict(),
                    "epoch": epoch,
                },
                out / "weights.pt",
            )
            write_json(out / "history.json", history)
    torch.set_num_threads(decode_threads)
    loader = image_loader(
        Subset(evaluation_dataset, holdout), args.batch_size, workers, device
    )
    history["evaluation"] = evaluate(model, loader, device, out, split=evaluation_split)
    write_json(out / "history.json", history)
    print(f"Results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
