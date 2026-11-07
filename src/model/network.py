"""Direct four-plane phase prediction."""
import torch
from torch import nn

from .fresnel import DIMS, N, input_chirp, project_planes, projection_amplitudes


class DOENet(nn.Module):
    """Map [B, 4, 128, 128] targets to [B, 128, 128] phase in radians."""

    def __init__(self, projection_iterations=6):
        super().__init__()
        self.projection_iterations = projection_iterations
        self.register_buffer("chirp", input_chirp(), persistent=False)
        rng = torch.Generator().manual_seed(42)
        seed = torch.rand((1, N, N), generator=rng) * (2 * torch.pi)
        self.register_buffer("seed_field", torch.exp(1j * seed), persistent=False)
        self.features = nn.Sequential(
            nn.Conv2d(12, 24, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=4, dilation=4),
            nn.SiLU(),
        )
        self.field = nn.Conv2d(24, 2, 1)
        nn.init.zeros_(self.field.weight)
        nn.init.zeros_(self.field.bias)

    @torch.no_grad()
    def optical_features(self, targets):
        amplitudes = projection_amplitudes(targets)
        field = self.seed_field
        for _ in range(self.projection_iterations - 1):
            projected = project_planes(amplitudes, field, self.chirp).mean(dim=1)
            field = torch.exp(1j * projected.angle())
        planes = project_planes(amplitudes, field, self.chirp) * N
        base = planes.mean(dim=1)
        phase = torch.fft.fftshift(base.angle(), dim=DIMS)
        direction = torch.fft.fftshift((base * field.conj()).angle(), dim=DIMS)
        planes = torch.fft.fftshift(planes, dim=DIMS)
        features = torch.cat((targets, planes.real, planes.imag), dim=1)
        return features, phase, direction

    def forward(self, targets):
        features, phase, direction = self.optical_features(targets)
        gain, residual = self.field(self.features(features)).unbind(dim=1)
        phase = phase + 3 * gain.tanh() * direction + 0.1 * residual.tanh()
        return torch.atan2(phase.sin(), phase.cos())


class PhaseNet(DOENet):
    """Predict the full phase correction from four targets and optical features."""

    def __init__(self, projection_iterations=12, *, aligned=True, correction_limit=None):
        super().__init__(projection_iterations)
        self.aligned = aligned
        self.correction_limit = (0.5 if aligned else torch.pi) if correction_limit is None else correction_limit
        self.features = nn.Sequential(
            nn.Conv2d(13 if aligned else 12, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        self.field = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.field.weight)
        nn.init.zeros_(self.field.bias)

    @torch.no_grad()
    def optical_features(self, targets):
        features, phase, direction = super().optical_features(targets)
        if self.aligned:
            amplitudes, real, imag = features.chunk(3, dim=1)
            cos, sin = phase.cos().unsqueeze(1), phase.sin().unsqueeze(1)
            features = torch.cat((
                amplitudes, real * cos + imag * sin, imag * cos - real * sin,
                direction.unsqueeze(1),
            ), dim=1)
        return features, phase, direction

    def predict_phase(self, features, initial_phase):
        correction = self.correction_limit * self.field(self.features(features)).squeeze(1).tanh()
        phase = initial_phase + correction
        return torch.atan2(phase.sin(), phase.cos())

    def forward(self, targets):
        features, phase, _ = self.optical_features(targets)
        return self.predict_phase(features, phase)
