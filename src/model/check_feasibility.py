"""Conservative SSIM feasibility certificates for the fixed four-plane Fresnel model.

For a unit-energy field in two planes, restrict the propagation operator to
the high-amplitude regions of two targets. Schur and Frobenius bounds give an
upper bound on the overlap of those subspaces. Global SSIM gives an upper
bound on each field's angle to the corresponding target-amplitude constraint.
When those angles cannot bridge the subspace gap, that SSIM is impossible
on both planes. The test relaxes phase-only and 8-bit restrictions.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from doe_eval import DISTANCES, N
from doe_eval.config import SSIM_K1, SSIM_K2
from doe_eval.physics import _get_fresnel_kernels

from .data_connector import ManifestQuartetDataset
from .retrieval import THRESHOLDS, reconstruct

MANIFEST = Path(__file__).resolve().parent.parent / "dataset/data/manifest.json"
CUTOFFS = (0.001, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90)
# Leave generous room for float32 differences in the evaluator's Fresnel kernels.
MIN_MARGIN = 0.01


class FeasibilityCheck:
    def __init__(self):
        chirp, _ = _get_fresnel_kernels(torch.device("cpu"))
        chirp = torch.fft.ifftshift(chirp[0], dim=(-2, -1)).to(torch.complex128)
        chirp = chirp / chirp.abs()
        self.pairs = []
        for i, j in combinations(range(len(DISTANCES)), 2):
            kernel = torch.fft.fft2(chirp[j] * chirp[i].conj()).abs() / (N * N)
            self.pairs.append(
                (i, j, torch.fft.fft2(kernel), torch.fft.fft2(kernel.square()))
            )
        self.cutoffs = torch.tensor(CUTOFFS, dtype=torch.float64)[:, None, None]

    def _pair_bounds(self, targets):
        targets = targets.double()
        power = targets.square().mean(dim=(-2, -1))
        mask = (targets[:, None] > self.cutoffs).double()
        captured = (mask * targets[:, None].square()).mean(dim=(-2, -1)) / power[
            :, None
        ]
        subspace_angle = captured.sqrt().clamp(0, 1).acos()
        spectra = torch.fft.fft2(mask)
        bounds = []
        for i, j, kernel, squared_kernel in self.pairs:
            # Absolute row/column sums and squared entry sums of P_j U P_i.
            columns = torch.fft.ifft2(kernel.conj() * spectra[j]).real.clamp_min(0)
            rows = torch.fft.ifft2(kernel * spectra[i]).real.clamp_min(0)
            max_col = (columns[None] * mask[i, :, None]).amax(dim=(-2, -1))
            max_row = (rows[:, None] * mask[j, None]).amax(dim=(-2, -1))
            energy = torch.fft.ifft2(squared_kernel * spectra[i]).real.clamp_min(0)
            frobenius_sq = (energy[:, None] * mask[j, None]).sum(dim=(-2, -1))
            overlap = (
                torch.minimum(max_col * max_row, frobenius_sq).sqrt() + 1e-5
            ).clamp(max=1)
            gap = overlap.acos() - subspace_angle[i, :, None] - subspace_angle[j, None]
            a, b = divmod(gap.argmax().item(), len(CUTOFFS))
            bounds.append((i, j, gap[a, b].item(), a, b, overlap[a, b].item()))
        return power, bounds

    def check(self, targets):
        power, bounds = self._pair_bounds(targets)
        witnesses = {}
        for i, j, gap, a, b, overlap in bounds:
            for threshold in THRESHOLDS:
                # If SSIM >= t, both its luminance and contrast/structure factors are >= t.
                # Thus MSE <= (1-t)*(rms_x^2 + rms_y^2 + C1 + C2).
                # Minimizing the resulting amplitude-cosine bound over rms_x gives this angle.
                cos_sq = threshold * (
                    threshold - (1 - threshold) * (SSIM_K1**2 + SSIM_K2**2) / power
                )
                angle = cos_sq.clamp(0, 1).sqrt().acos()
                margin = gap - angle[i].item() - angle[j].item()
                key = str(threshold)
                if key not in witnesses or margin > witnesses[key]["margin_rad"]:
                    witnesses[key] = {
                        "excluded": margin > MIN_MARGIN,
                        "margin_rad": margin,
                        "distances_mm": [round(DISTANCES[k] * 1000, 1) for k in (i, j)],
                        "cutoffs": [CUTOFFS[a], CUTOFFS[b]],
                        "overlap_upper": overlap,
                    }
        return witnesses


def mean_ssim_upper_bound(check, targets, grid):
    """Bound the mean over four planes while allowing any complex pupil."""
    power, pairs = check._pair_bounds(targets)
    gap = np.zeros((4, 4))
    # Rounding each actual angle down makes its SSIM bound larger. Relax the
    # pair inequalities by two grid intervals so that rounding cannot exclude it.
    slack = 2 * grid[1] + MIN_MARGIN
    for i, j, angle, *_ in pairs:
        gap[i, j] = max(0, angle - slack)
    a, b, c = np.ix_(grid, grid, grid)
    valid = (a + b >= gap[0, 1]) & (a + c >= gap[0, 2]) & (b + c >= gap[1, 2])
    d = np.maximum(
        np.maximum(gap[0, 3] - a, gap[1, 3] - b), np.maximum(gap[2, 3] - c, 0)
    )
    stabilizers = ((SSIM_K1**2 + SSIM_K2**2) / power).numpy()

    def ssim_bound(angle, q):
        # Positive root of (1+q)*SSIM**2 - q*SSIM <= cos(angle)**2.
        return (q + np.sqrt(q * q + 4 * (1 + q) * np.cos(angle) ** 2)) / (2 * (1 + q))

    scores = (
        sum(
            ssim_bound(angle, q)
            for angle, q in zip((a, b, c, d), stabilizers, strict=True)
        )
        / 4
    )
    return float(scores[valid].max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--mean-ssim",
        action="store_true",
        help="Also bound mean SSIM across the four planes",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    check = FeasibilityCheck()
    rng = torch.Generator().manual_seed(42)
    grid = np.linspace(0, np.pi / 2, 81)
    for _ in range(8):
        known_mask = torch.randint(256, (N, N), dtype=torch.uint8, generator=rng)
        targets = (reconstruct(known_mask) * 255).round() / 255
        assert not any(w["excluded"] for w in check.check(targets).values())
        if args.mean_ssim:
            assert mean_ssim_upper_bound(check, targets, grid) > 0.999
    dataset = ManifestQuartetDataset(args.manifest, split="validation")
    rows = [
        {"sample_id": sid, "witnesses": check.check(images)} for images, sid in dataset
    ]
    counts = {
        str(t): sum(row["witnesses"][str(t)]["excluded"] for row in rows)
        for t in THRESHOLDS
    }
    report = {
        "quartets": len(rows),
        "excluded": counts,
        "margin_required_rad": MIN_MARGIN,
        "control_masks": 8,
        "witnesses": rows,
    }
    if args.mean_ssim:
        mean_bounds = [
            mean_ssim_upper_bound(check, images, grid) for images, _ in dataset
        ]
        for row, bound in zip(rows, mean_bounds, strict=True):
            row["mean_ssim_upper_bound"] = bound
        report["mean_ssim_upper_bound"] = float(np.mean(mean_bounds))
        print(
            f"Mean SSIM upper bound: {report['mean_ssim_upper_bound']:.6f}", flush=True
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Excluded by SSIM alone: {counts} of {len(rows)} quartets; 8 realizable controls pass."
    )


if __name__ == "__main__":
    main()
