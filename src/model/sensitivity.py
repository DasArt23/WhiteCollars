"""Optical sensitivity of the reconstruction metrics to a phase change."""
import torch

from .fresnel import DIMS, EPS


@torch.no_grad()
def metric_direction(phase, targets, chirp, pearson_weight=0.75):
    """One adjoint propagation. No differentiation through the network or FFT."""
    field = torch.exp(1j * torch.fft.ifftshift(phase, dim=DIMS))
    observed = torch.fft.fft2(field[:, None] * chirp, dim=DIMS, norm="ortho")
    amplitude = observed.abs()
    peak = amplitude.amax(dim=DIMS, keepdim=True).clamp_min(EPS)
    a = amplitude / peak
    b = torch.fft.ifftshift(targets, dim=DIMS)
    mu_a = a.mean(dim=DIMS, keepdim=True)
    mu_b = b.mean(dim=DIMS, keepdim=True)
    da, db = a - mu_a, b - mu_b
    va = da.square().mean(dim=DIMS, keepdim=True)
    vb = db.square().mean(dim=DIMS, keepdim=True)
    cov = (da * db).mean(dim=DIMS, keepdim=True)
    spread = (va * vb).sqrt().clamp_min(EPS)
    pearson_grad = db / spread - (cov / spread) * da / va.clamp_min(EPS)

    lum_num = 2 * mu_a * mu_b + 0.01 ** 2
    lum_den = mu_a.square() + mu_b.square() + 0.01 ** 2
    cov_num = 2 * cov + 0.03 ** 2
    cov_den = va + vb + 0.03 ** 2
    luminance = lum_num / lum_den
    contrast = cov_num / cov_den
    ssim_grad = (
        contrast * (2 * mu_b - 2 * mu_a * luminance) / lum_den
        + luminance * (2 * db - 2 * da * contrast) / cov_den
    )
    grad = pearson_weight * pearson_grad + (1 - pearson_weight) * ssim_grad
    peak_pixels = amplitude == peak
    peak_count = peak_pixels.sum(dim=DIMS, keepdim=True).clamp_min(1)
    peak_term = (grad * a).sum(dim=DIMS, keepdim=True) * peak_pixels / peak_count
    grad = (grad - peak_term) / peak

    back = torch.fft.ifft2(
        grad * observed / amplitude.clamp_min(EPS), dim=DIMS, norm="ortho",
    ) * chirp.conj()
    direction = (field.conj() * back.sum(dim=1)).imag
    direction = torch.fft.fftshift(direction, dim=DIMS)
    rms = direction.square().mean(dim=DIMS, keepdim=True).sqrt().clamp_min(EPS)
    return (direction / rms).clamp(-5, 5)


@torch.no_grad()
def plane_error_directions(phase, targets, chirp):
    """Return four phase-plane directions, one from each amplitude residual."""
    field = torch.exp(1j * torch.fft.ifftshift(phase, dim=DIMS))
    observed = torch.fft.fft2(field[:, None] * chirp, dim=DIMS, norm="ortho")
    amplitude = observed.abs()
    peak = amplitude.amax(dim=DIMS, keepdim=True).clamp_min(EPS)
    target = torch.fft.ifftshift(targets, dim=DIMS)
    residual = (target - amplitude / peak) * observed / amplitude.clamp_min(EPS)
    back = torch.fft.ifft2(residual, dim=DIMS, norm="ortho") * chirp.conj()
    direction = (field[:, None].conj() * back).imag
    scale = direction.square().mean(dim=DIMS, keepdim=True).sqrt().clamp_min(0.01)
    return torch.fft.fftshift((direction / scale).clamp(-5, 5), dim=DIMS)
