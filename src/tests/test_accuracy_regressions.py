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
    pearson,
    pearson_torch,
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
    balanced_quality_score,
    compute_sota_physics_loss,
    hybrid_infer_phase,
)
from model.train import evaluate_loader


@pytest.mark.parametrize("kind", ["random", "sparse", "black", "white"])
def test_local_ssim_matches_skimage_including_dark_background(kind):
    rng = np.random.default_rng(47)
    a = rng.random((2, 4, N, N))
    b = rng.random(a.shape)
    if kind == "sparse":
        a[a < .95] = 0
        b[b < .95] = 0
    elif kind in ("black", "white"):
        a.fill(0 if kind == "black" else 1)
        b = a.copy()
    actual = ssim_local_7x7_torch(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    expected = np.array([[structural_similarity(x, y, data_range=1.0) for x, y in zip(aa, bb)] for aa, bb in zip(a, b)])
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("value", [0., 1.])
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
    phase = np.random.default_rng(9).uniform(-math.pi, math.pi, (N, N)).astype(np.float32)
    index = np.arange(N) - N // 2
    r2 = (index[:, None] ** 2 + index[None, :] ** 2) * PIXEL**2
    dft = np.exp(-2j * np.pi * np.outer(index, index) / N)
    fields = []
    for z in DISTANCES:
        dx = WAVELENGTH * z / (N * PIXEL)
        out_r2 = (index[:, None] ** 2 + index[None, :] ** 2) * dx**2
        inner = np.exp(1j * (phase.astype(np.float64) + np.pi * r2 / (WAVELENGTH * z)))
        factor = np.exp(1j * (2 * np.pi * z / WAVELENGTH + np.pi * out_r2 / (WAVELENGTH * z))) / (1j * WAVELENGTH * z)
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
    with patch("model.holonet.fresnel_reconstruct", wraps=fresnel_reconstruct) as propagate:
        loss, _, metrics = compute_sota_physics_loss(phase, targets)
    assert propagate.call_count == 1  # прежний loss дважды запускал всю физику
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    assert all(torch.isfinite(v) for v in metrics.values())


def _scores(phase, targets):
    recon = normalize_reconstructions(fresnel_reconstruct(ste_quantize(phase)))
    return balanced_quality_score(pearson_torch(recon, targets).mean(dim=1),
                                  ssim_local_7x7_torch(recon, targets).mean(dim=1))


def test_polish_keeps_best_per_sample_and_evaluates_last_step():
    class FixedPhase(torch.nn.Module):
        def forward(self, targets):
            return initial.clone()
    torch.manual_seed(8)
    initial = torch.rand(2, N, N) * (2 * math.pi) - math.pi
    targets = torch.rand(2, 4, N, N)
    before = _scores(initial, targets)
    with patch("model.holonet.fresnel_reconstruct", wraps=fresnel_reconstruct) as propagate:
        improved = hybrid_infer_phase(FixedPhase(), targets, polish_steps=1)
    assert propagate.call_count == 2  # начальная фаза и результат единственного шага Adam
    after = _scores(improved, targets)
    assert torch.all(after >= before - 1e-6)
    assert torch.any(after > before + 1e-6)


def test_validation_reports_same_local_ssim_as_final_evaluation():
    class FixedPhase(torch.nn.Module):
        def forward(self, targets):
            return torch.zeros(targets.shape[0], N, N)
    targets = torch.from_numpy(generate_synthetic_targets()).unsqueeze(0)
    metrics = evaluate_loader(FixedPhase(), [(targets, ["demo"])], torch.device("cpu"))
    recon = normalize_reconstructions(fresnel_reconstruct(ste_quantize(torch.zeros(1, N, N))))
    expected = np.mean([structural_similarity(t.numpy(), r.numpy(), data_range=1.) for t, r in zip(targets[0], recon[0])])
    assert metrics["ssim_skimage"] == pytest.approx(expected, abs=2e-5)
    assert metrics["score"] == balanced_quality_score(metrics["pearson"], metrics["ssim_skimage"])


def test_high_ssim_with_poor_pearson_is_not_preferred():
    assert balanced_quality_score(.1, .84) < balanced_quality_score(.5657, .1267)
    assert balanced_quality_score(-.1, .84) == 0


def test_checkpoint_is_required_and_state_must_match(tmp_path):
    with pytest.raises(FileNotFoundError, match="Чекпоинт"):
        load_trained_holonet(tmp_path / "missing.pt", "cpu")
    model = HoloUnrollNet(unroll_steps=1, base_dim=8)
    checkpoint = tmp_path / "model.pt"
    payload = {"unroll_steps": 1, "base_dim": 8, "state_dict": model.state_dict()}
    torch.save(payload, checkpoint)
    restored, device, meta = load_trained_holonet(checkpoint, "cpu")
    assert meta["loaded"] and device.type == "cpu"
    payload["state_dict"].pop("step_alphas")
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_trained_holonet(checkpoint, "cpu")
    result = synthesize_doe_and_reconstruct(restored, device, generate_synthetic_targets(), polish_steps=0)
    png = tmp_path / "doe.png"
    save_doe_png(result["phase_rad"], png)
    with Image.open(png) as image:
        assert image.mode == "L" and image.size == (N, N)
    recon = normalize_reconstructions(fresnel_reconstruct(torch.from_numpy(load_doe_png(png))))
    np.testing.assert_allclose(recon.numpy(), result["recons_01"], atol=1e-7)


def test_numpy_reference_does_not_import_torch_or_physics():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", "import sys; import doe_eval.reference_numpy; assert 'torch' not in sys.modules; assert 'doe_eval.physics' not in sys.modules"], check=True)


def test_ste_and_png_quantization_agree_near_rounding_boundaries():
    boundaries = ((torch.arange(255) + .5) / 255) * (2 * math.pi) - math.pi
    values = torch.cat([boundaries - 1e-7, boundaries, boundaries + 1e-7,
                        torch.tensor([-math.pi, math.pi])])
    png_codes = quantize_phase_u8(values.numpy())
    torch.testing.assert_close(quantize_phase_u8(ste_quantize(values)), torch.from_numpy(png_codes))
