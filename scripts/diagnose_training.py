"""Bounded TRAIN-only overfit diagnostic. Never uses reference phases as predictions."""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

import model.holonet as holo
from doe_eval.config import N
from doe_eval.io import load_doe_png, save_comparison_png, save_doe_png
from doe_eval.physics import fresnel_reconstruct, normalize_reconstructions
from model.data_connector import ManifestQuartetDataset


def stats(value):
    if isinstance(value, (tuple, list)):
        return [stats(v) for v in value]
    if not isinstance(value, torch.Tensor):
        return str(value)
    observed = value.detach().abs() if value.is_complex() else value.detach()
    observed = observed.float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(observed.min()),
        "max": float(observed.max()),
        "mean": float(observed.mean()),
        "std": float(observed.std()),
        "finite": bool(torch.isfinite(observed).all()),
        "requires_grad": value.requires_grad,
        "grad_fn": type(value.grad_fn).__name__ if value.grad_fn else None,
    }


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(3701)
    device = torch.device(args.device)
    dataset = ManifestQuartetDataset(
        args.manifest,
        split="train",
        preload=False,
        augment_channel_shuffle=False,
        augment_dihedral=False,
    )
    indices = [
        i for i, r in enumerate(dataset.samples) if r.params.get("kind") == args.group
    ][: args.count]
    if len(indices) != args.count:
        raise ValueError("Not enough TRAIN examples in requested group")
    examples = [dataset[i] for i in indices]
    targets = [t[None].to(device) for t, _ in examples]
    if args.checkpoint:
        from main import load_trained_holonet

        model, _, _ = load_trained_holonet(args.checkpoint, args.device)
    else:
        model = holo.HoloUnrollNet(
            unroll_steps=args.unroll_steps,
            base_dim=args.base_dim,
            phase_carrier_seed=3701,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert expected == actual
    report = {
        "scope": "bounded TRAIN-only diagnostic; no reference-phase supervision",
        "group": args.group,
        "ids": [sid for _, sid in examples],
        "manifest": str(args.manifest.resolve()),
        "device": str(device),
        "torch": str(torch.__version__),
        "precision": "fp32/complex64",
        "unroll_steps": model.unroll_steps,
        "base_dim": model.base_dim,
        "initial_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "loss_version": holo.LOSS_VERSION,
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_parameters_match_model": True,
        "scheduler": "none; constant LR diagnostic",
        "lr": args.lr,
        "max_steps": args.steps,
        "max_seconds": args.seconds,
        "tensor_stats": {},
        "history": [],
    }
    destination = args.out
    destination.mkdir(parents=True, exist_ok=True)

    def write():
        (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    @torch.no_grad()
    def evaluate():
        model.eval()
        results = []
        for target in targets:
            phase = model(target)
            _, _, metrics = holo.compute_sota_physics_loss(phase, target)
            metrics = {k: float(v) for k, v in metrics.items()}
            metrics["phase_circular_variance"] = float(
                1 - torch.polar(torch.ones_like(phase), phase).mean().abs()
            )
            results.append(metrics)
        model.train()
        return {
            key: sum(row[key] for row in results) / len(results) for key in results[0]
        }

    report["initial"] = evaluate()
    print("INITIAL", args.group, args.count, json.dumps(report["initial"]), flush=True)
    write()
    probes = {
        "initializer_head": model.target_phase_init[-1].weight,
        "refiner_head": model.refiner.head.weight,
        "plane_logits": model.plane_logits,
    }
    initial_probes = {key: p.detach().clone() for key, p in probes.items()}
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        if step > 1 and time.perf_counter() - start > args.seconds:
            break
        optimizer.zero_grad(set_to_none=True)
        target = targets[(step - 1) % len(targets)]
        tape = {}

        def capture_function(name, function, tensor_tape=tape):
            def wrapped(*a, **kw):
                result = function(*a, **kw)
                report["tensor_stats"].setdefault(name, []).append(
                    {"input": stats(a), "output": stats(result)}
                )
                if name == "ste_quantize":
                    result.retain_grad()
                    tensor_tape["quantized"] = result
                return result

            return wrapped

        def capture_module(name):
            def hook(module, inputs, output):
                report["tensor_stats"][name] = {
                    "input": stats(inputs),
                    "output": stats(output),
                }
                if name == "target_phase_init":
                    raw = output.float().reshape(-1, 4, 2, N, N)
                    correction = (raw[:, :, 1] - raw[:, :, 0]) / 2**0.5
                    report["initializer_tanh_saturation_first_step"] = float(
                        (correction.abs() > 3).float().mean()
                    )
                if name == "refiner":
                    report["refiner_tanh_saturation_first_step"] = float(
                        (output.abs() > 3).float().mean()
                    )

            return hook

        with ExitStack() as stack:
            if step == 1:
                report["tensor_stats"]["targets"] = stats(target)
                for name in (
                    "fresnel_propagate_complex",
                    "fresnel_backward_to_slm",
                    "normalize_reconstructions",
                    "ste_quantize",
                ):
                    function = getattr(holo, name)
                    stack.enter_context(
                        patch.object(holo, name, capture_function(name, function))
                    )
                for name, module in model.named_modules():
                    if (
                        name in ("target_phase_init", "refiner")
                        or name.startswith(("obj_gates.", "slm_blocks."))
                        and name.count(".") == 1
                    ):
                        handle = module.register_forward_hook(capture_module(name))
                        stack.callback(handle.remove)
            phase = model(target)
            phase.retain_grad()
            loss, _, metrics = holo.compute_sota_physics_loss(phase, target)
            assert loss.requires_grad and torch.isfinite(loss)
            loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        grad_norm = sum(float(g.square().sum()) for g in grads) ** 0.5
        assert grad_norm > 0
        if step == 1:
            report["phase_gradient"] = stats(phase.grad)
            report["ste_output_gradient"] = stats(tape["quantized"].grad)
            report["first_step_zero_gradient_parameters"] = [
                name
                for name, p in model.named_parameters()
                if p.grad is not None and not p.grad.any()
            ]
        initial_norm = torch.nn.utils.clip_grad_norm_(
            model.target_phase_init.parameters(), 1.0, error_if_nonfinite=True
        )
        remaining = [
            p
            for name, p in model.named_parameters()
            if not name.startswith("target_phase_init.")
        ]
        rest_norm = torch.nn.utils.clip_grad_norm_(
            remaining, 1.0, error_if_nonfinite=True
        )
        before = {key: p.detach().clone() for key, p in probes.items()}
        optimizer.step()
        update = {
            key: float((p.detach() - before[key]).norm()) for key, p in probes.items()
        }
        row = {
            "step": step,
            "sample_id": examples[(step - 1) % len(examples)][1],
            "lr": optimizer.param_groups[0]["lr"],
            **{k: float(v) for k, v in metrics.items()},
            "gradient_norm_before_clip": grad_norm,
            "initializer_gradient_norm": float(initial_norm),
            "remaining_gradient_norm": float(rest_norm),
            "phase_gradient_norm": float(phase.grad.norm()),
            "parameter_update_l2": update,
        }
        report["history"].append(row)
        if step == 1 or step % 10 == 0:
            print(
                "STEP",
                args.group,
                args.count,
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "step",
                            "loss",
                            "pearson",
                            "ssim_global",
                            "ssim_local",
                            "lr",
                            "gradient_norm_before_clip",
                            "parameter_update_l2",
                        )
                    }
                ),
                flush=True,
            )
            write()
        del phase, loss, metrics, grads, tape
    report["elapsed_update_seconds"] = time.perf_counter() - start
    report["completed_steps"] = len(report["history"])
    report["parameter_change_from_start_l2"] = {
        key: float((p.detach() - initial_probes[key]).norm())
        for key, p in probes.items()
    }
    report["final"] = evaluate()
    model.eval()
    for index, (target, (_, sid)) in enumerate(zip(targets, examples)):
        with torch.no_grad():
            phase = model(target)[0].cpu().numpy()
        doe = destination / sid / "network_doe.png"
        save_doe_png(phase, doe)
        restored = (
            normalize_reconstructions(
                fresnel_reconstruct(torch.from_numpy(load_doe_png(doe)).to(device))
            )
            .cpu()
            .numpy()
        )
        save_comparison_png(
            target[0].cpu().numpy(), restored, doe.parent / "comparison.png"
        )
    write()
    print("FINAL", args.group, args.count, json.dumps(report["final"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("src/dataset/data_soft600/manifest.json")
    )
    parser.add_argument(
        "--group", choices=["physical_forward", "soft_independent"], required=True
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unroll-steps", type=int, default=6)
    parser.add_argument("--base-dim", type=int, default=48)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Actual checkpoint to diagnose; otherwise random initialization",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.seconds <= 0 or args.count < 1:
        parser.error("positive steps/seconds/count required")
    run(args)
