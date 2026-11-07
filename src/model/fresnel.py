"""Single-FFT Fresnel reconstruction for the four task distances."""
import math

import torch

N = 128
WAVELENGTH = 532e-9
PIXEL_SIZE = 10e-6
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
EPS = 1e-8


def fresnel(phase, z):
    """Return peak-normalized amplitude for a phase map in radians."""
    height, width = phase.shape[-2:]
    y = (torch.arange(height, device=phase.device) - height // 2) * PIXEL_SIZE
    x = (torch.arange(width, device=phase.device) - width // 2) * PIXEL_SIZE
    angle = math.pi * (y[:, None].square() + x.square()) / (WAVELENGTH * z)
    dims = (-2, -1)
    field = torch.exp(1j * (phase + angle))
    propagated = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field, dim=dims), dim=dims), dim=dims
    )
    amplitude = propagated.abs()
    return amplitude / amplitude.amax(dim=dims, keepdim=True).clamp_min(EPS)


def quantize_phase_u8(phase):
    return ((phase + math.pi) * (255 / (2 * math.pi))).round().clamp(0, 255).to(torch.uint8)


def dequantize_phase_u8(pixels):
    return pixels.float() / 255 * (2 * math.pi) - math.pi

def reconstruct(phase):
    """Peak-normalized amplitudes in the four observation planes."""
    return torch.stack([fresnel(phase, z) for z in DISTANCES], dim=-3)


DIMS = (-2, -1)


def input_chirp():
    axis = (torch.arange(N) - N // 2) * PIXEL_SIZE
    z = torch.tensor(DISTANCES)[:, None, None]
    angle = math.pi * (axis[:, None].square() + axis.square()) / (WAVELENGTH * z)
    return torch.fft.ifftshift(torch.exp(1j * angle), dim=DIMS)


def projection_amplitudes(targets):
    amplitude = torch.fft.ifftshift(targets, dim=DIMS)
    rms = amplitude.square().mean(dim=DIMS, keepdim=True).sqrt().clamp_min(EPS)
    return amplitude / rms


def project_planes(amplitude, field, chirp):
    observed = torch.fft.fft2(field[:, None] * chirp, dim=DIMS)
    desired = amplitude * (observed / observed.abs().clamp_min(1e-8))
    return torch.fft.ifft2(desired, dim=DIMS) * chirp.conj()
