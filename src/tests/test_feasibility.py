"""Necessary-condition audit must accept actual propagated phase-only fields."""

import numpy as np
import pytest
import torch

from doe_eval.config import DISTANCES, PIXEL, WAVELENGTH, N
from doe_eval.feasibility import phase_only_compatibility
from doe_eval.physics import fresnel_reconstruct


@pytest.mark.parametrize("kind", ["random", "lens", "quantized_phase"])
def test_actual_fresnel_outputs_satisfy_moment_bound(kind):
    generator = torch.Generator().manual_seed(3701)
    phase = (torch.rand(N, N, generator=generator) * 2 - 1) * torch.pi
    if kind == "lens":
        coordinate = (torch.arange(N) - N // 2) * PIXEL
        r2 = coordinate[:, None].square() + coordinate[None, :].square()
        phase = -torch.pi * r2 / (WAVELENGTH * DISTANCES[1])
    elif kind == "quantized_phase":
        phase = (
            torch.round((phase + torch.pi) * 255 / (2 * torch.pi))
            * (2 * torch.pi / 255)
            - torch.pi
        )
    amplitude = fresnel_reconstruct(phase).numpy()
    result = phase_only_compatibility(amplitude, tolerance=1e-5)
    assert not result["exact_targets_ruled_out"]
    normalized = amplitude / amplitude.max((-2, -1), keepdims=True)
    saved = np.rint(normalized * 255).astype(np.uint8)
    assert not phase_only_compatibility(saved)["exact_targets_ruled_out"]


def test_independent_opposite_spots_violate_bound_and_positive_scale_does_not_change_it():
    target = np.zeros((4, N, N))
    for plane in range(4):
        target[plane, N // 2, N // 4 if plane % 2 else 3 * N // 4] = 1
    result = phase_only_compatibility(target)
    assert result["exact_targets_ruled_out"]
    assert result["pair_intensity_l1_error_sum_lower_bound"] > 1.8
    scaled = phase_only_compatibility(target * np.array([2, 3, 4, 5])[:, None, None])
    assert scaled["max_moment_violation"] == pytest.approx(
        result["max_moment_violation"]
    )


def test_invalid_targets_rejected():
    with pytest.raises(ValueError, match="black target"):
        phase_only_compatibility(np.zeros((4, N, N)))
    with pytest.raises(ValueError):
        phase_only_compatibility(np.full((4, N, N), np.nan))
