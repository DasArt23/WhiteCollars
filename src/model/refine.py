"""Optional phase-only optimization after neural inference."""
import torch
from torch import nn

from .fresnel import dequantize_phase_u8, quantize_phase_u8, reconstruct
from .metrics import compare_images


def quantized_phase(phase):
    wrapped = (phase + torch.pi).remainder(2 * torch.pi) - torch.pi
    return dequantize_phase_u8(quantize_phase_u8(wrapped))


class PhaseRefiner(nn.Module):
    """Keep a refined mask only if both quantized-mask metrics improve."""

    def __init__(self, model, steps):
        super().__init__()
        self.model = model
        self.steps = steps

    def forward(self, targets):
        with torch.no_grad():
            initial = quantized_phase(self.model(targets))
        with torch.inference_mode(False), torch.enable_grad():
            return self.refine(initial.clone(), targets.detach().clone())

    def refine(self, initial, targets):
        def scores(phase):
            return compare_images(reconstruct(phase), targets).mean(dim=1)

        with torch.no_grad():
            baseline = scores(initial)
            best = initial
            best_gain = initial.new_zeros(len(initial))

        for ssim_weight in (0.10, 0.25):
            phase = initial.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([phase], lr=0.08)
            for step in range(1, self.steps + 1):
                ssim, pearson = scores(phase).unbind(dim=1)
                loss = -(ssim_weight * ssim + (1 - ssim_weight) * pearson).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step in (self.steps // 3, 2 * self.steps // 3):
                    optimizer.param_groups[0]["lr"] *= 0.65
                if step % 10 == 0 or step == self.steps:
                    with torch.no_grad():
                        candidate = quantized_phase(phase)
                        gain = scores(candidate) - baseline
                        utility = gain.clamp_min(0).prod(dim=1)
                        improved = utility > best_gain
                        best_gain = torch.where(improved, utility, best_gain)
                        best = torch.where(improved[:, None, None], candidate, best)
        return best

