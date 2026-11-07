"""Necessary compatibility test for a uniformly illuminated phase-only DOE.

This is not a metric ceiling or a sufficient feasibility certificate. It checks
whether low-frequency intensity moments can change as much as the targets demand.
The bound is independent of the unknown DOE phase, using the autocorrelation
identity for Fourier intensity and the triangle inequality. See FUNDAMENTALS.md.
"""

import argparse
import json
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DEFAULT_MANIFEST, DISTANCES, PIXEL, WAVELENGTH, N

LAGS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2), (0, 3), (3, 0))


@lru_cache(maxsize=1)
def intensity_moment_bounds():
    coordinate = (np.arange(N) - N // 2) * PIXEL
    radius2 = coordinate[:, None] ** 2 + coordinate[None, :] ** 2
    chirp = np.exp(
        1j * np.pi * radius2[None] / (WAVELENGTH * np.array(DISTANCES)[:, None, None])
    )
    pairs = tuple(combinations(range(len(DISTANCES)), 2))
    bounds = np.array(
        [
            [
                np.abs(
                    chirp[i] * np.roll(chirp[i].conj(), lag, axis=(0, 1))
                    - chirp[j] * np.roll(chirp[j].conj(), lag, axis=(0, 1))
                ).mean()
                for lag in LAGS
            ]
            for i, j in pairs
        ]
    )
    return pairs, bounds


def phase_only_compatibility(targets: np.ndarray, tolerance: float = 0.01) -> dict:
    """Check amplitude targets; ignores per-plane positive scale factors.

    A positive violation beyond tolerance rules out exact matching under our
    discrete propagation model. Passing does NOT prove a DOE exists. Tolerance
    allows small changes caused by storing simulated target amplitudes in 8 bits.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if (
        targets.shape != (len(DISTANCES), N, N)
        or not np.isfinite(targets).all()
        or np.any(targets < 0)
    ):
        raise ValueError(
            f"Expected finite nonnegative amplitude [{len(DISTANCES)},{N},{N}]"
        )
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    power = np.square(targets)
    total = power.sum((-2, -1), keepdims=True)
    if np.any(total == 0):
        raise ValueError(
            "An entirely black target plane cannot carry the nonzero illumination power"
        )
    coefficients = np.fft.fft2(power / total)
    pairs, bounds = intensity_moment_bounds()
    differences = np.array(
        [
            [abs(coefficients[i, dy, dx] - coefficients[j, dy, dx]) for dy, dx in LAGS]
            for i, j in pairs
        ]
    )
    violations = differences - bounds
    pair_index, lag_index = np.unravel_index(np.argmax(violations), violations.shape)
    violation = float(violations[pair_index, lag_index])
    return {
        "exact_targets_ruled_out": violation > tolerance,
        "passing_is_not_a_feasibility_certificate": True,
        "tolerance": tolerance,
        "max_moment_violation": violation,
        "planes": [index + 1 for index in pairs[pair_index]],
        "lag": list(LAGS[lag_index]),
        "target_difference": float(differences[pair_index, lag_index]),
        "allowed_difference": float(bounds[pair_index, lag_index]),
        "pair_intensity_l1_error_sum_lower_bound": max(0.0, violation),
    }


def audit_manifest(manifest_path: Path, split: str = "validation") -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = [round(z * 1000, 1) for z in DISTANCES]
    if (
        manifest.get("format") != "doe-manifest/1"
        or manifest.get("distances_mm") != expected
    ):
        raise ValueError("Invalid manifest format or distances")
    rows = []
    for record in manifest["samples"]:
        if record["split"] != split:
            continue
        if record.get("distances_mm") != expected or len(record["images"]) != len(
            DISTANCES
        ):
            raise ValueError("Invalid plane count or ordering")
        images = []
        for relative in record["images"]:
            path = (manifest_path.parent / relative).resolve()
            if not path.is_relative_to(manifest_path.parent.resolve()):
                raise ValueError("Image path outside dataset")
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "L" or image.size != (N, N):
                    raise ValueError(f"Invalid target PNG: {path}")
                images.append(np.asarray(image))
        kind = record.get("params", {}).get("exchange", {}).get("kind")
        group = (
            f"exchange_{kind}" if kind else record.get("generator_version", "unknown")
        )
        rows.append(
            {
                "id": record["sample_id"],
                "group": group,
                **phase_only_compatibility(np.stack(images)),
            }
        )
    if not rows:
        raise ValueError("Empty split")
    totals = Counter(row["group"] for row in rows)
    ruled_out = Counter(row["group"] for row in rows if row["exact_targets_ruled_out"])
    return {
        "split": split,
        "samples": len(rows),
        "exact_targets_ruled_out": sum(ruled_out.values()),
        "groups": {
            key: {"total": count, "ruled_out": ruled_out[key]}
            for key, count in totals.items()
        },
        "note": "Necessary condition only. Does not give a Pearson/SSIM ceiling or optimizer convergence guarantee.",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "diagnostic"],
        default="validation",
    )
    parser.add_argument("--out", type=Path, default=Path("build/feasibility.json"))
    args = parser.parse_args()
    report = audit_manifest(args.manifest, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
