"""Shared inference pipeline for the terminal and browser interfaces."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from .retrieval import RetrievalDOE, plane_metrics, reconstruct

DEFAULT_WEIGHTS = Path("weights/best-retrieval.pt")
DEFAULT_MANIFEST = Path("src/dataset/example/manifest.json")


def load_predictor(
    weights: Path, device_name: str, optical_steps: int | None = None
) -> tuple[RetrievalDOE, torch.device]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    device = torch.device(device_name)
    torch.set_num_threads(4)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "pixel_retrieval_v1":
        raise ValueError("Expected a pixel_retrieval_v1 checkpoint")
    config = dict(checkpoint["config"])
    if optical_steps is not None:
        if optical_steps < 10:
            raise ValueError("--optical-steps must be at least 10")
        config["optical_steps"] = optical_steps
    model = RetrievalDOE(**config).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, device


@dataclass(frozen=True)
class Prediction:
    mask: torch.Tensor
    reconstructions: torch.Tensor
    scores: torch.Tensor
    importance_maps: torch.Tensor
    synthesis_ms: float


class PredictionService:
    """Run one quartet through the existing model and optical evaluator."""

    def __init__(self, model: RetrievalDOE, device: torch.device) -> None:
        self.model = model
        self.device = device

    def predict(self, targets: torch.Tensor) -> Prediction:
        batch = targets[None].to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        mask, maps = self.model.predict_weighted(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        synthesis_ms = (perf_counter() - start) * 1000
        reconstructions = reconstruct(mask)
        scores = plane_metrics(reconstructions, batch)
        return Prediction(mask[0], reconstructions[0], scores[0], maps[0], synthesis_ms)
