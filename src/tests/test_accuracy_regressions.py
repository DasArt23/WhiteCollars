"""Регрессии реальных ошибок метрик, распространения и обучения."""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from doe_eval.config import DISTANCES, PIXEL, WAVELENGTH, N
from doe_eval.dataset import FourImageDataset
from doe_eval.demo import generate_synthetic_targets
from doe_eval.io import load_doe_png, save_doe_png, save_recon_png
from doe_eval.metrics import (
    amplitude_shape_error_torch,
    pearson,
    pearson_torch,
    plane_crosstalk_error_torch,
    spatial_fidelity_errors_torch,
    ssim_local_7x7_torch,
)
from doe_eval.physics import (
    fresnel_inverse_complex,
    fresnel_propagate_complex,
    fresnel_reconstruct,
    normalize_reconstructions,
    quantize_phase_u8,
    ste_quantize,
)
from main import load_trained_holonet, synthesize_doe_and_reconstruct
from model.holonet import (
    HoloUnrollNet,
    compute_sota_physics_loss,
    hybrid_infer_phase,
)
from model.train import (
    evaluate_loader,
    initialize_training_model,
    save_validation_previews,
)


@pytest.mark.parametrize("kind", ["random", "sparse", "black", "white"])
def test_local_ssim_matches_skimage_including_dark_background(kind):
    rng = np.random.default_rng(47)
    a = rng.random((2, 4, N, N))
    b = rng.random(a.shape)
    if kind == "sparse":
        a[a < 0.95] = 0
        b[b < 0.95] = 0
    elif kind in ("black", "white"):
        a.fill(0 if kind == "black" else 1)
        b = a.copy()
    actual = ssim_local_7x7_torch(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    expected = np.array(
        [
            [structural_similarity(x, y, data_range=1.0) for x, y in zip(aa, bb)]
            for aa, bb in zip(a, b)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_torch_pearson_constants_match_numpy_with_finite_backward(value):
    a = torch.full((2, 4, N, N), value, requires_grad=True)
    b = torch.full_like(a, value)
    score = pearson_torch(a, b)
    assert torch.all(score == 1)
    assert pearson(a.detach().numpy()[0, 0], b.numpy()[0, 0]) == 1
    score.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.all(pearson_torch(a, 1 - b) == 0)


def test_pearson_low_contrast_not_suppressed_by_epsilon():
    a = torch.rand(2, 4, N, N) * 1e-6
    torch.testing.assert_close(pearson_torch(a, a), torch.ones(2, 4))


def test_complex_fresnel_matches_direct_dft_and_inverse():
    phase = (
        np.random.default_rng(9).uniform(-math.pi, math.pi, (N, N)).astype(np.float32)
    )
    index = np.arange(N) - N // 2
    r2 = (index[:, None] ** 2 + index[None, :] ** 2) * PIXEL**2
    dft = np.exp(-2j * np.pi * np.outer(index, index) / N)
    fields = []
    for z in DISTANCES:
        dx = WAVELENGTH * z / (N * PIXEL)
        out_r2 = (index[:, None] ** 2 + index[None, :] ** 2) * dx**2
        inner = np.exp(1j * (phase.astype(np.float64) + np.pi * r2 / (WAVELENGTH * z)))
        factor = np.exp(
            1j * (2 * np.pi * z / WAVELENGTH + np.pi * out_r2 / (WAVELENGTH * z))
        ) / (1j * WAVELENGTH * z)
        fields.append(factor * (dft @ inner @ dft.T))
    expected = np.stack(fields)
    actual = fresnel_propagate_complex(torch.from_numpy(phase))
    assert actual.dtype == torch.complex64
    assert np.linalg.norm(actual.numpy() - expected) / np.linalg.norm(expected) < 2e-6
    recovered = fresnel_inverse_complex(actual)
    h = torch.exp(1j * torch.from_numpy(phase)).expand_as(recovered)
    torch.testing.assert_close(recovered, h, atol=1e-6, rtol=1e-6)
    batched = fresnel_inverse_complex(actual.unsqueeze(0).expand(2, -1, -1, -1))
    torch.testing.assert_close(batched[1], h, atol=1e-6, rtol=1e-6)


def test_folder_dataset_ids_and_custom_names(tmp_path):
    directory = tmp_path / "test" / "sample42"
    directory.mkdir(parents=True)
    targets = generate_synthetic_targets()
    names = ("a.png", "b.png", "c.png", "d.png")
    for target, name in zip(targets, names):
        save_recon_png(target, directory / name)
    dataset = FourImageDataset(directory.parent, filenames=names)
    target, sample_id = dataset[0]
    assert sample_id == "sample42"
    assert target.shape == (4, N, N)


def test_neural_model_backward_and_single_physics_loss_pass():
    torch.manual_seed(3)
    model = HoloUnrollNet(unroll_steps=1, base_dim=8)
    targets = torch.from_numpy(generate_synthetic_targets()).unsqueeze(0)
    phase = model(targets)
    assert phase.shape == (1, N, N)
    assert phase.abs().max() <= math.pi
    with patch(
        "model.holonet.fresnel_reconstruct", wraps=fresnel_reconstruct
    ) as propagate:
        loss, _, metrics = compute_sota_physics_loss(phase, targets)
    assert propagate.call_count == 1  # прежний loss дважды запускал всю физику
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    assert all(torch.isfinite(v) for v in metrics.values())


def _scores(phase, targets):
    return torch.stack(
        [
            -compute_sota_physics_loss(p.unsqueeze(0), t.unsqueeze(0))[2][
                "selection_error"
            ]
            for p, t in zip(phase, targets)
        ]
    )


def test_polish_keeps_best_per_sample_and_evaluates_last_step():
    class FixedPhase(torch.nn.Module):
        def forward(self, targets):
            return initial.clone()

    torch.manual_seed(8)
    initial = torch.rand(2, N, N) * (2 * math.pi) - math.pi
    targets = torch.rand(2, 4, N, N)
    before = _scores(initial, targets)
    timings = {}
    with patch(
        "model.holonet.fresnel_reconstruct", wraps=fresnel_reconstruct
    ) as propagate:
        improved = hybrid_infer_phase(
            FixedPhase(), targets, polish_steps=1, timings=timings
        )
    assert timings["network_ms"] >= 0
    assert timings["refinement_ms"] > 0
    assert (
        propagate.call_count == 2
    )  # начальная фаза и результат единственного шага Adam
    after = _scores(improved, targets)
    assert torch.all(after >= before - 1e-6)
    for p0, p1, target in zip(initial, improved, targets):
        before_metrics = compute_sota_physics_loss(p0[None], target[None])[2]
        after_metrics = compute_sota_physics_loss(p1[None], target[None])[2]
        assert after_metrics["pearson"] >= before_metrics["pearson"] - 1e-6
        assert after_metrics["ssim_local"] >= before_metrics["ssim_local"] - 1e-6
        for key in (
            "foreground_error",
            "brightness_error",
            "background_error",
            "edge_error",
        ):
            assert after_metrics[key] <= before_metrics[key] + 1e-6


def test_validation_reports_same_local_ssim_as_final_evaluation():
    class FixedPhase(torch.nn.Module):
        def forward(self, targets):
            return torch.zeros(targets.shape[0], N, N)

    targets = torch.from_numpy(generate_synthetic_targets()).unsqueeze(0)
    metrics = evaluate_loader(FixedPhase(), [(targets, ["demo"])], torch.device("cpu"))
    recon = normalize_reconstructions(
        fresnel_reconstruct(ste_quantize(torch.zeros(1, N, N)))
    )
    expected = np.mean(
        [
            structural_similarity(t.numpy(), r.numpy(), data_range=1.0)
            for t, r in zip(targets[0], recon[0])
        ]
    )
    assert metrics["ssim_skimage"] == pytest.approx(expected, abs=2e-5)
    assert metrics["score"] == -metrics["selection_error"]
    assert metrics["selection_error"] > metrics["visual_error"]


def test_coherent_interference_global_phase_and_fresnel_power():
    generator = torch.Generator().manual_seed(3701)
    phase = torch.rand(N, N, generator=generator) * (2 * math.pi) - math.pi
    field = fresnel_propagate_complex(phase)
    opposite = fresnel_propagate_complex(phase + math.pi)
    scale = torch.tensor([N / (WAVELENGTH * z) for z in DISTANCES])[:, None, None]
    normalized = field / scale
    # Для FFT без norm: mean(|K|²) = N²/(lambda*z)² при |H|=1.
    # Формула из ТЗ не содержит множителя площади входного пикселя.
    torch.testing.assert_close(
        normalized.abs().square().mean((-2, -1)), torch.ones(4), rtol=5e-6, atol=5e-6
    )
    # Противофаза меняет знак поля, но не интенсивность отдельного пучка.
    assert torch.linalg.vector_norm(
        (opposite + field) / scale
    ) < 1e-6 * torch.linalg.vector_norm(normalized)
    torch.testing.assert_close(
        ((field + field) / scale).abs().square(),
        4 * normalized.abs().square(),
    )
    shifted = fresnel_propagate_complex(phase + 0.63) / scale
    assert torch.linalg.vector_norm(
        shifted.abs() - normalized.abs()
    ) < 1e-6 * torch.linalg.vector_norm(normalized.abs())


def test_selection_includes_worst_plane_and_matches_training_penalty():
    from model.holonet import METRIC_WEIGHTS, WORST_PLANE_WEIGHT, _image_objective

    generator = torch.Generator().manual_seed(15)
    target = torch.rand(2, 4, N, N, generator=generator)
    amplitude = torch.rand(2, 4, N, N, generator=generator)
    loss, _, planes = _image_objective(amplitude, target)
    expected = planes["visual_error"].mean(-1) + WORST_PLANE_WEIGHT * planes[
        "shape_error"
    ].amax(-1)
    auxiliary = sum(
        weight * (1 - planes[key]).mean()
        for weight, key in zip(METRIC_WEIGHTS, ["pearson", "ssim_global", "ssim_local"])
    )
    torch.testing.assert_close(
        planes["selection_error"].mean(), expected.mean() + auxiliary
    )
    torch.testing.assert_close(loss, planes["selection_error"].mean())
    torch.testing.assert_close(planes["worst_pearson"], planes["pearson"].amin(-1))


def test_shape_error_rejects_dim_objects_and_unrelated_bright_peak():
    target = torch.zeros(1, 4, N, N)
    target[:, :, 40:80, 40:80] = 1
    visible = (target + 0.08).clamp_max(1)
    degenerate = 0.01 * target
    degenerate[:, :, 4, 4] = 1
    # Проверяем известный контрпример: фон повышает оконный SSIM, объект теряется.
    assert (
        ssim_local_7x7_torch(degenerate, target).mean()
        > ssim_local_7x7_torch(visible, target).mean()
    )
    assert (
        amplitude_shape_error_torch(degenerate, target).mean()
        > amplitude_shape_error_torch(visible, target).mean()
    )
    torch.testing.assert_close(
        amplitude_shape_error_torch(5 * target, target), torch.zeros(1, 4)
    )
    zeros = torch.zeros_like(target, requires_grad=True)
    amplitude_shape_error_torch(zeros, target).mean().backward()
    assert torch.isfinite(zeros.grad).all()


def test_power_equalization_does_not_prioritize_large_targets():
    from model.holonet import fresnel_backward_to_slm

    fields = torch.zeros(1, 4, N, N, dtype=torch.complex64)
    fields[:, :, 30:70, 30:70] = 1
    logits = torch.full((4,), -100.0)
    logits[0] = 0
    base = fresnel_backward_to_slm(fields, logits, equalize_power=True)
    fields[:, 0] *= 0.03
    scaled = fresnel_backward_to_slm(fields, logits, equalize_power=True)
    torch.testing.assert_close(base, scaled, atol=1e-6, rtol=1e-5)
    assert float(base.abs().square().mean()) == pytest.approx(1.0, abs=1e-5)


def test_best_checkpoint_previews_are_saved_from_png(tmp_path):
    class FixedPhase(torch.nn.Module):
        def forward(self, targets):
            return torch.zeros(targets.shape[0], N, N)

    target = torch.from_numpy(generate_synthetic_targets())
    loader = torch.utils.data.DataLoader([(target, "case")], batch_size=1)
    save_validation_previews(FixedPhase(), loader, torch.device("cpu"), tmp_path)
    assert (tmp_path / "case/doe.png").is_file()
    assert (tmp_path / "case/comparison.png").is_file()
    assert (tmp_path / "case/metrics.json").is_file()


def test_checkpoint_is_required_and_state_must_match(tmp_path):
    with pytest.raises(FileNotFoundError, match="Чекпоинт"):
        load_trained_holonet(tmp_path / "missing.pt", "cpu")
    model = HoloUnrollNet(unroll_steps=1, base_dim=8)
    checkpoint = tmp_path / "model.pt"
    payload = {"unroll_steps": 1, "base_dim": 8, "state_dict": model.state_dict()}
    torch.save(payload, checkpoint)
    restored, device, meta = load_trained_holonet(checkpoint, "cpu")
    assert meta["loaded"] and device.type == "cpu"
    assert restored.equalize_plane_power is False
    payload["equalize_plane_power"] = True
    torch.save(payload, checkpoint)
    restored, _, _ = load_trained_holonet(checkpoint, "cpu")
    assert restored.equalize_plane_power is True
    payload["state_dict"].pop("step_alphas")
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_trained_holonet(checkpoint, "cpu")
    result = synthesize_doe_and_reconstruct(
        restored, device, generate_synthetic_targets(), polish_steps=0
    )
    png = tmp_path / "doe.png"
    save_doe_png(result["phase_rad"], png)
    with Image.open(png) as image:
        assert image.mode == "L" and image.size == (N, N)
    recon = normalize_reconstructions(
        fresnel_reconstruct(torch.from_numpy(load_doe_png(png)))
    )
    np.testing.assert_allclose(recon.numpy(), result["recons_01"], atol=1e-7)


def test_numpy_reference_does_not_import_torch_or_physics():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import doe_eval.reference_numpy; assert 'torch' not in sys.modules; assert 'doe_eval.physics' not in sys.modules",
        ],
        check=True,
    )


def test_ste_and_png_quantization_agree_near_rounding_boundaries():
    boundaries = ((torch.arange(255) + 0.5) / 255) * (2 * math.pi) - math.pi
    values = torch.cat(
        [
            boundaries - 1e-7,
            boundaries,
            boundaries + 1e-7,
            torch.tensor([-math.pi, math.pi]),
        ]
    )
    png_codes = quantize_phase_u8(values.numpy())
    torch.testing.assert_close(
        quantize_phase_u8(ste_quantize(values)), torch.from_numpy(png_codes)
    )


def test_depth_confusion_penalizes_swapped_images_and_allows_identical_targets():
    targets = torch.from_numpy(generate_synthetic_targets()).unsqueeze(0)
    swapped = targets.roll(1, dims=1).clone().requires_grad_()
    assert plane_crosstalk_error_torch(targets, targets).max() < 1e-8
    error = plane_crosstalk_error_torch(swapped, targets).mean()
    assert error > 0.1
    error.backward()
    assert torch.isfinite(swapped.grad).all() and swapped.grad.abs().sum() > 0
    same = targets[:, :1].expand(-1, 4, -1, -1)
    assert plane_crosstalk_error_torch(same, same).max() < 1e-8


def test_spatial_fidelity_detects_visible_ghost_even_when_own_correlation_wins():
    target = torch.zeros(1, 4, 32, 32)
    for plane in range(4):
        target[:, plane, 4:12, plane * 7 : plane * 7 + 5] = 1
    ghost = target + 0.5 * target.roll(1, dims=1)
    # Старый сравнительный штраф не видит этот паразитный объект.
    assert plane_crosstalk_error_torch(ghost, target).max() < 1e-8
    errors = spatial_fidelity_errors_torch(ghost.requires_grad_(), target)
    assert errors["foreign_energy"].min() > 0
    assert errors["background_energy"].min() > 0
    sum(x.mean() for x in errors.values()).backward()
    assert torch.isfinite(ghost.grad).all() and ghost.grad.abs().sum() > 0
    identical = spatial_fidelity_errors_torch(target, target)
    assert all(x.max() < 1e-8 for x in identical.values())
    same = target[:, :1].expand(-1, 4, -1, -1)
    assert spatial_fidelity_errors_torch(same, same)["foreign_energy"].max() == 0


def test_spatial_fidelity_preserves_gray_targets_and_penalizes_speckle():
    target = torch.full((1, 4, 32, 32), 0.5)
    target[..., 8:20, 8:20] = 1
    noise = torch.zeros_like(target)
    noise[..., ::2, ::2] = 0.15
    exact = spatial_fidelity_errors_torch(target, target)
    noisy = spatial_fidelity_errors_torch(target + noise, target)
    assert exact["background_energy"].max() == 0
    assert exact["foreign_energy"].max() == 0
    assert exact["edge_error"].max() == 0
    assert noisy["edge_error"].min() > 0
    assert noisy["power_error"].min() > 0
    zero = torch.zeros_like(target, requires_grad=True)
    sum(x.mean() for x in spatial_fidelity_errors_torch(zero, zero).values()).backward()
    assert torch.isfinite(zero.grad).all()


def test_finetune_preserves_carrier_forward_and_checks_architecture(tmp_path):
    model = HoloUnrollNet(unroll_steps=1, base_dim=8, phase_carrier_seed=57)
    path = tmp_path / "initial.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "unroll_steps": 1,
            "base_dim": 8,
            "phase_carrier_seed": 57,
            "equalize_plane_power": True,
        },
        path,
    )
    restored = initialize_training_model(1, 8, 3701, path)
    assert restored.phase_carrier_seed == 57
    target = torch.from_numpy(generate_synthetic_targets()).unsqueeze(0)
    with torch.no_grad():
        torch.testing.assert_close(model(target), restored(target))
    with pytest.raises(ValueError, match="base_dim"):
        initialize_training_model(1, 16, 3701, path)


def test_stable_phase_initialization_handles_zero_encoder_outputs():
    torch.manual_seed(3701)
    model = HoloUnrollNet(unroll_steps=1, base_dim=8, phase_carrier_seed=3701)
    with torch.no_grad():
        model.target_phase_init[-1].weight.zero_()
        model.target_phase_init[-1].bias.zero_()
    targets = torch.from_numpy(generate_synthetic_targets()).unsqueeze(0)
    phase = model(targets)
    loss, _, _ = compute_sota_physics_loss(phase, targets)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert torch.isfinite(phase).all()
    assert model.target_phase_init[-1].weight.grad.abs().sum() > 0
    assert "phase_carrier" not in model.state_dict()


def test_carrier_seed_roundtrips_without_changing_legacy_state_format(tmp_path):
    torch.manual_seed(77)
    model = HoloUnrollNet(unroll_steps=1, base_dim=8, phase_carrier_seed=123)
    path = tmp_path / "stable.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "unroll_steps": 1,
            "base_dim": 8,
            "equalize_plane_power": True,
            "phase_carrier_seed": 123,
        },
        path,
    )
    restored, _, _ = load_trained_holonet(path, "cpu")
    torch.testing.assert_close(restored.phase_carrier, model.phase_carrier)
    target = torch.from_numpy(generate_synthetic_targets()).unsqueeze(0)
    with torch.no_grad():
        torch.testing.assert_close(model(target), restored(target))
