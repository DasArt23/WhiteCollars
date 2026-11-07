"""Global Pearson and SSIM as specified in the task."""
import torch


def compare_images(reconstructed, targets):
    a = reconstructed.flatten(start_dim=-2)
    b = targets.flatten(start_dim=-2)
    mu_a = a.mean(dim=-1, keepdim=True)
    mu_b = b.mean(dim=-1, keepdim=True)
    da, db = a - mu_a, b - mu_b
    var_a, var_b = da.square().mean(dim=-1), db.square().mean(dim=-1)
    covariance = (da * db).mean(dim=-1)
    spread = (var_a * var_b).sqrt()
    equal = (a - b).abs().amax(dim=-1) <= 1e-7
    pearson = torch.where(spread > 0, covariance / spread.clamp_min(1e-30), equal.float())
    mu_a, mu_b = mu_a.squeeze(-1), mu_b.squeeze(-1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * covariance + c2)) / (
        (mu_a.square() + mu_b.square() + c1) * (var_a + var_b + c2)
    )
    return torch.stack([ssim, pearson], dim=-1)

