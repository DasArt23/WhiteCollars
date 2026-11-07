"""Fit each target phase independently before training the neural network."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from .fresnel import reconstruct
from .metrics import compare_images


def make_supervised_loader(model, loader, device, *, steps, train_batch_size=32):
    examples = []
    total = len(loader)
    for index, (targets, _) in enumerate(loader, 1):
        targets = targets.to(device, non_blocking=True)
        features, initial, _ = model.optical_features(targets)
        phase = initial.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([phase], lr=0.08)
        best_phase = phase.detach().clone()
        best_ssim = phase.new_full((len(targets),), -torch.inf)
        for step in range(steps + 1):
            recon = reconstruct(phase)
            scores = compare_images(recon, targets).mean(dim=1)
            with torch.no_grad():
                better = scores[:, 0] > best_ssim
                best_ssim = torch.maximum(best_ssim, scores[:, 0])
                best_phase[better] = phase[better]
            if step < steps:
                optimizer.zero_grad(set_to_none=True)
                loss = 1 - (0.8 * scores[:, 0] + 0.2 * scores[:, 1]).mean()
                loss.backward()
                optimizer.step()
        # Оставляем тензоры сразу на устройстве (V100), чтобы не гонять по шине PCIe каждую эпоху
        examples.append((features.detach(), initial.detach(), best_phase.detach(), targets.detach()))
        if index % 5 == 0 or index == total:
            print(f"Phase targets: batch {index}/{total}", flush=True)
    dataset = TensorDataset(*(torch.cat(tensors) for tensors in zip(*examples, strict=True)))
    return DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
