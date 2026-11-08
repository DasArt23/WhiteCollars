"""Learn pixel weights for Fresnel phase retrieval from four amplitude images."""

import json
import math

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from doe_eval import (
    DISTANCES,
    dequantize_phase_u8,
    fresnel_reconstruct,
    normalize_reconstructions,
    pearson_torch,
    quantize_phase_u8,
    ssim_global_torch,
)
from doe_eval.physics import _get_fresnel_kernels

DIMS = (-2, -1)
EPS = 1e-8
THRESHOLDS = (0.90, 0.95)


def reconstruct(mask):
    return normalize_reconstructions(
        fresnel_reconstruct(dequantize_phase_u8(mask)), norm="max", compare="amplitude"
    )


def plane_metrics(recon, targets):
    return torch.stack(
        (ssim_global_torch(recon, targets), pearson_torch(recon, targets)), dim=-1
    )


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_example(out, mask, recon, values, sample_id, inputs=None):
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.cpu().numpy()).save(out / "doe.png")
    for i, plane in enumerate(recon, 1):
        pixels = (plane.clamp(0, 1) * 255).round().byte().cpu().numpy()
        Image.fromarray(pixels).save(out / f"restored_{i}.png")
    planes = [
        {"distance_mm": round(z * 1000, 1), "ssim": s, "pearson": p}
        for z, (s, p) in zip(DISTANCES, values.tolist(), strict=True)
    ]
    report = {"sample_id": sample_id, "planes": planes}
    if inputs is not None:
        report["inputs"] = [str(path) for path in inputs]
    write_json(out / "metrics.json", report)


def unit(field):
    return torch.exp(1j * field.angle())


class RetrievalDOE(nn.Module):
    """Trainable amplitude weighting followed by a phase-only optical projection."""

    def __init__(
        self, pilot_steps=24, optical_steps=300, retry_steps=1200, amplitude_steps=0
    ):
        super().__init__()
        if (
            pilot_steps < 1
            or optical_steps < 10
            or min(retry_steps, amplitude_steps) < 0
        ):
            raise ValueError("Invalid projection counts")
        self.pilot_steps = pilot_steps
        self.optical_steps = optical_steps
        self.retry_steps = retry_steps
        self.amplitude_steps = amplitude_steps
        chirp, _ = _get_fresnel_kernels(torch.device("cpu"))
        self.register_buffer(
            "chirp", torch.fft.ifftshift(chirp, dim=DIMS), persistent=False
        )
        # Preserve compatibility with the existing trained checkpoint.
        self.controller = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(16, 4, 1),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def _propagate(self, pupil):
        return torch.fft.fft2(pupil[:, None] * self.chirp, dim=DIMS, norm="ortho")

    def _back_project(self, fields):
        return (
            torch.fft.ifft2(fields, dim=DIMS, norm="ortho") * self.chirp.conj()
        ).sum(dim=1)

    @staticmethod
    def _update_weights(weights, delta):
        return (weights * delta.exp()).clamp(0.5, 2.0)

    @torch.no_grad()
    def optical_solve(self, amplitude, steps, pupil=None):
        if pupil is None:
            pupil = torch.ones_like(amplitude[:, 0], dtype=torch.complex64)
        field = self._propagate(pupil)
        reflect_steps = int(steps * 0.8)
        for step in range(steps):
            projected = amplitude * unit(field)
            if step < reflect_steps:
                reflected_pupil = unit(self._back_project(2 * projected - field))
                field = (
                    0.85 * (field + self._propagate(reflected_pupil)) - 0.70 * projected
                )
            else:
                field = self._propagate(unit(self._back_project(projected)))
        return unit(self._back_project(amplitude * unit(field)))

    @torch.no_grad()
    def prepare(self, targets):
        targets = torch.fft.ifftshift(targets, dim=DIMS)
        amplitude = (
            targets
            / targets.square().mean(dim=DIMS, keepdim=True).clamp_min(EPS).sqrt()
        )
        pilot = self.optical_solve(amplitude, self.pilot_steps)
        basis = amplitude * unit(self._propagate(pilot))
        return targets, amplitude, basis

    def _features(self, targets, pupil, weights):
        field = self._propagate(pupil)
        field = field / field.abs().amax(dim=DIMS, keepdim=True).clamp_min(EPS)
        amplitude = field.abs()
        features = torch.cat(
            (
                targets,
                weights.log(),
                amplitude,
                amplitude - targets,
                field.real,
                field.imag,
            ),
            dim=1,
        )
        return torch.fft.fftshift(features, dim=DIMS)

    def _delta(self, targets, pupil, weights):
        correction = self.controller(self._features(targets, pupil, weights))
        return 0.5 * torch.fft.ifftshift(correction, dim=DIMS).tanh() * (targets > 0)

    @torch.no_grad()
    def _teacher_weights(self, reference, pupil, basis, weights):
        summed = self._back_project(basis * weights)
        # Relative phase is invariant to an arbitrary piston of either mask.
        aligned = reference * unit(
            (reference.conj() * pupil).mean(dim=DIMS, keepdim=True)
        )
        phase_grad = (pupil * aligned.conj()).imag
        pupil_grad = 1j * pupil * phase_grad / summed.abs().clamp_min(EPS)
        gradient = (basis.conj() * self._propagate(pupil_grad)).real * weights
        rms = gradient.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(EPS)
        delta = (-0.12 * gradient / rms).clamp(-0.5, 0.5)
        best = weights
        best_fit = (pupil.conj() * reference).mean(dim=DIMS).abs()
        for scale in (0.25, 0.5, 1.0):
            proposal = self._update_weights(weights, scale * delta)
            proposed_pupil = unit(self._back_project(basis * proposal))
            fit = (proposed_pupil.conj() * reference).mean(dim=DIMS).abs()
            keep = fit > best_fit
            best_fit = torch.where(keep, fit, best_fit)
            best = torch.where(keep[:, None, None, None], proposal, best)
        return best

    def training_loss(self, data, reference):
        targets, _, basis = data
        weights = torch.ones_like(targets)
        support = targets > 0
        losses = []
        for _ in range(2):
            with torch.no_grad():
                pupil = unit(self._back_project(basis * weights))
                teacher = self._teacher_weights(reference, pupil, basis, weights)
            proposed = self._update_weights(
                weights, self._delta(targets, pupil, weights)
            )
            error = F.smooth_l1_loss(
                (proposed / weights).log(),
                (teacher / weights).log(),
                beta=0.05,
                reduction="none",
            )
            losses.append((error * support).sum() / support.sum().clamp_min(1))
            weights = proposed.detach()
        return torch.stack(losses).mean()

    @torch.no_grad()
    def weighted_pupil(self, data):
        targets, _, basis = data
        weights = torch.ones_like(targets)
        for _ in range(2):
            pupil = unit(self._back_project(basis * weights))
            weights = self._update_weights(
                weights, self._delta(targets, pupil, weights)
            )
        return unit(self._back_project(basis * weights)), weights

    @staticmethod
    def encode(pupil):
        phase = torch.fft.fftshift(pupil.angle(), dim=DIMS)
        return quantize_phase_u8((phase + math.pi).remainder(2 * math.pi) - math.pi)

    @staticmethod
    def _physical_error(mask, targets):
        mse = (reconstruct(mask) - targets).square().mean(dim=DIMS)
        variance = targets.var(dim=DIMS, correction=0).clamp_min(EPS)
        relative = mse / variance
        return relative.amax(dim=1)

    @torch.no_grad()
    def _refined_mask(self, targets, amplitude, pupil):
        mask = self.encode(self.optical_solve(amplitude, self.optical_steps, pupil))
        if self.retry_steps <= self.optical_steps:
            return mask
        error = self._physical_error(mask, targets)
        hard = error > 0.08
        if hard.any().item():
            # Keep the learned start and try a different projection trajectory only for difficult cases.
            retry = self.encode(
                self.optical_solve(amplitude[hard], self.retry_steps, pupil[hard])
            )
            keep = self._physical_error(retry, targets[hard]) < error[hard]
            mask[hard] = torch.where(keep[:, None, None], retry, mask[hard])
        return mask

    @torch.no_grad()
    def _refine_amplitude(self, mask, targets):
        """Refine a quantized pupil with a manual adjoint of amplitude error."""
        target = torch.fft.ifftshift(targets, dim=DIMS)
        power = target.square().mean(dim=DIMS, keepdim=True).clamp_min(EPS)
        shape_target = target / power.sqrt()
        phase = dequantize_phase_u8(torch.fft.ifftshift(mask, dim=DIMS))
        velocity, variance = torch.zeros_like(phase), torch.zeros_like(phase)
        best = mask
        best_error = torch.full((len(mask),), torch.inf, device=mask.device)
        for step in range(self.amplitude_steps + 1):
            encoded = self.encode(torch.exp(1j * phase))
            pupil = torch.exp(
                1j * dequantize_phase_u8(torch.fft.ifftshift(encoded, dim=DIMS))
            )
            field = self._propagate(pupil)
            amplitude = field.abs()
            peak = amplitude.amax(dim=DIMS, keepdim=True).clamp_min(EPS)
            normalized = amplitude / peak
            residual = normalized - target
            error = (residual.square() / power).mean(dim=(1, 2, 3))
            keep = error < best_error
            best = torch.where(keep[:, None, None], encoded, best)
            best_error = torch.minimum(best_error, error)
            if step == self.amplitude_steps:
                break
            ties = amplitude == peak
            peak_grad = (residual * normalized).sum(dim=DIMS, keepdim=True) * ties
            peak_grad = peak_grad / ties.sum(dim=DIMS, keepdim=True).clamp_min(1)
            gradient = (residual - peak_grad) / (peak * power) + 0.7 * (
                amplitude - shape_target
            )
            adjoint = self._back_project(gradient * field / amplitude.clamp_min(EPS))
            phase_gradient = (adjoint * pupil.conj()).imag
            velocity.lerp_(phase_gradient, 0.1)
            variance.lerp_(phase_gradient.square(), 0.01)
            rate = 0.035 * (1 - 0.8 * step / self.amplitude_steps)
            phase = phase - rate * velocity / (variance.sqrt() + 1e-7)
        return best

    @torch.no_grad()
    def candidates(self, targets):
        data = self.prepare(targets)
        _, amplitude, basis = data
        pupil, weights = self.weighted_pupil(data)
        return {
            "pilot": self.encode(unit(self._back_project(basis))),
            "weighted_pilot": self.encode(pupil),
            "optical_only": self.encode(
                self.optical_solve(amplitude, self.optical_steps)
            ),
            "weighted_refined": self._refined_mask(targets, amplitude, pupil),
        }, torch.fft.fftshift(weights, dim=DIMS)

    @torch.no_grad()
    def predict_weighted(self, targets):
        data = self.prepare(targets)
        pupil, weights = self.weighted_pupil(data)
        mask = self._refined_mask(targets, data[1], pupil)
        if self.amplitude_steps:
            mask = self._refine_amplitude(mask, targets)
        return mask, torch.fft.fftshift(weights, dim=DIMS)

    def forward(self, targets):
        # The project's existing prediction exporter expects phase in radians.
        mask, _ = self.predict_weighted(targets)
        return dequantize_phase_u8(mask)
