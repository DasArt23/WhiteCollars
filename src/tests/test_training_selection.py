"""Check orchestration with a tiny fake model; no HoloUnrollNet training."""

from copy import deepcopy

import pytest
import torch

from model import train


@pytest.mark.parametrize("regression", ["pearson", "global_ssim"])
def test_metric_regression_preserves_initial_checkpoint_and_plateau_reduces_lr(
    tmp_path, monkeypatch, regression
):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.target_phase_init = torch.nn.Linear(1, 1)
            self.offset = torch.nn.Parameter(torch.zeros(()))
            self.equalize_plane_power = True
            self.phase_carrier_seed = 3701

        def forward(self, targets):
            value = self.target_phase_init.weight.sum() + self.offset
            return value.expand(targets.shape[0], 128, 128)

    model = Tiny()
    starting_state = deepcopy(model.state_dict())
    monkeypatch.setattr(train, "initialize_training_model", lambda *args: model)
    batch = [(torch.zeros(1, 4, 128, 128), ["sample"])]
    monkeypatch.setattr(
        train,
        "build_dataloaders",
        lambda **kwargs: {"train": batch, "validation": batch, "diagnostic": batch},
    )
    monkeypatch.setattr(train, "save_validation_previews", lambda *args: None)
    monkeypatch.setattr(
        train, "export_split_with_optional_polish", lambda *args, **kwargs: []
    )
    metric_names = ("loss", "pearson", "ssim_global", "ssim_local", "shape_error")

    def fake_loss(phase, target):
        loss = (phase - 1).square().mean()
        return loss, phase.detach(), {key: loss.detach() for key in metric_names}

    monkeypatch.setattr(train, "compute_sota_physics_loss", fake_loss)
    calls = 0

    def evaluate(*args):
        nonlocal calls
        calls += 1
        selection = 1.0 if calls == 1 else 0.9
        return {
            "loss": selection,
            "pearson": 0.6 if calls == 1 or regression == "global_ssim" else 0.5,
            "ssim_global": 0.2 if calls == 1 or regression == "pearson" else 0.1,
            "ssim_skimage": 0.2 if calls == 1 else 0.3,
            "shape_error": 1.0,
            "visual_error": 1.0,
            "selection_error": selection,
            "score": -selection,
            "worst_pearson": 0.3,
            "worst_ssim_skimage": 0.1,
            **dict.fromkeys(
                (
                    "foreground_error",
                    "background_error",
                    "background_energy",
                    "foreign_energy",
                    "power_error",
                    "edge_error",
                    "multiscale_error",
                ),
                0.1,
            ),
        }

    monkeypatch.setattr(train, "evaluate_loader", evaluate)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    destination = tmp_path / "run/best.pt"
    train.train_model(
        manifest,
        destination,
        epochs=4,
        batch_size=1,
        lr=1e-3,
        device_str="cpu",
        polish_export_steps=0,
        lr_schedule="plateau",
    )
    checkpoint = torch.load(destination, weights_only=True)
    assert checkpoint["epoch"] == 0  # Better scalar loss, worse Pearson: rejected.
    for name, value in starting_state.items():
        torch.testing.assert_close(checkpoint["state_dict"][name], value)
    import json

    history = json.loads((destination.parent / "training_history.json").read_text())
    assert all(not row["metric_guard_passed"] for row in history)
    assert history[-1]["lr"] < history[0]["lr"]
    assert all(row["optimizer_updates"] == 1 for row in history)
