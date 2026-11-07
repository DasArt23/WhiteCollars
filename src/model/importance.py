"""Predict pixel importance, then synthesize phase by summing optical fields."""
import math

import torch
from torch import nn

from doe_eval import DISTANCES, N, PIXEL, WAVELENGTH

DIMS = (-2, -1)
ARCHITECTURE = "importance_v2"
LEGACY_ARCHITECTURE = "importance_v1"


class ImportanceDOE(nn.Module):
    def __init__(self, projections=12, *, learn_phase=True):
        super().__init__()
        if projections < 1:
            raise ValueError("At least one optical projection is required")
        self.projections = projections
        self.learn_phase = learn_phase
        self.architecture = ARCHITECTURE if learn_phase else LEGACY_ARCHITECTURE
        axis = (torch.arange(N) - N // 2) * PIXEL
        radius_sq = axis[:, None].square() + axis.square()
        z = torch.tensor(DISTANCES)[:, None, None]
        chirp = torch.exp(1j * (math.pi / (WAVELENGTH * z)) * radius_sq)
        self.register_buffer("chirp", torch.fft.ifftshift(chirp, dim=DIMS), persistent=False)
        seed = torch.rand(N, N, generator=torch.Generator().manual_seed(42)) * (2 * math.pi)
        self.register_buffer("seed_field", torch.exp(1j * seed), persistent=False)
        self.head = nn.Sequential(
            nn.Conv2d(12, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 8 if learn_phase else 4, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if learn_phase:
            self.global_head = nn.Linear(24, 8, bias=False)
            nn.init.zeros_(self.global_head.weight)

    def back_project(self, plane_fields):
        fields = torch.fft.ifft2(plane_fields, dim=DIMS) * self.chirp.conj()
        return fields.sum(dim=1)

    @torch.no_grad()
    def prepare(self, targets, active=None):
        if active is not None:
            targets = targets * active[..., None, None]
        amplitude = torch.fft.ifftshift(targets, dim=DIMS)
        rms = amplitude.square().mean(dim=DIMS, keepdim=True).clamp_min(1e-16).sqrt()
        amplitude = amplitude / rms

        field = self.seed_field.expand(targets.shape[0], -1, -1)
        for step in range(self.projections):
            observed = torch.fft.fft2(field[:, None] * self.chirp, dim=DIMS)
            magnitude = observed.abs()
            # Ignore the unit-modulus output chirp here. It cancels in the adjoint.
            plane_fields = amplitude * (observed / magnitude.clamp_min(1e-8))
            if step + 1 < self.projections:
                field = torch.exp(1j * self.back_project(plane_fields).angle())

        preview = magnitude / magnitude.amax(dim=DIMS, keepdim=True).clamp_min(1e-8)
        preview = torch.fft.fftshift(preview, dim=DIMS)
        if active is not None:
            preview = preview * active[..., None, None]
        features = torch.cat((targets, preview, targets - preview), dim=1)
        return features, plane_fields

    def corrections(self, features):
        if not self.learn_phase:
            return self.head(features).float().tanh()
        hidden = features
        for layer in list(self.head)[:-1]:
            hidden = layer(hidden)
        local = self.head[-1](hidden)
        global_offsets = self.global_head(hidden.mean(dim=DIMS))[..., None, None]
        return (local + global_offsets).float().tanh()

    def importance(self, features):
        return torch.exp(math.log(2) * self.corrections(features)[:, :4])

    def phase_from_fields(self, fields):
        return torch.fft.fftshift(self.back_project(fields).angle(), dim=DIMS)

    def phase_from_corrections(self, fields, corrections):
        log_weight = math.log(2) * corrections[:, :4]
        amplitude = log_weight.exp()
        if self.learn_phase:
            correction_phase = (math.pi / 2) * corrections[:, 4:]
            amplitude = torch.polar(amplitude, correction_phase)
        return self.phase_from_fields(fields * torch.fft.ifftshift(amplitude, dim=DIMS))

    def predict_prepared(self, features, fields):
        return self.phase_from_corrections(fields, self.corrections(features))

    def forward(self, targets):
        return self.predict_prepared(*self.prepare(targets))
