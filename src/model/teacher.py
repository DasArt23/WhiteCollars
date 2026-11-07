"""Fit each target phase independently before training the neural network."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from .fresnel import quantize_phase_ste, reconstruct
from .metrics import balanced_optical_loss, compare_images


def make_supervised_loader(model, loader, device, *, steps, train_batch_size=32):
    examples = []
    total = len(loader)
    for index, (targets, _) in enumerate(loader, 1):
        targets = targets.to(device, non_blocking=True)
        features, initial, _ = model.optical_features(targets)
        correction = torch.zeros_like(initial, requires_grad=True)
        optimizer = torch.optim.Adam([correction], lr=0.08)
        best_phase = initial
        best_gain = initial.new_zeros(len(targets))
        for step in range(steps + 1):
            phase = initial + correction
            recon = reconstruct(quantize_phase_ste(phase))
            scores = compare_images(recon, targets).mean(dim=1)
            if step == 0:
                baseline = scores.detach()
            gains = scores - baseline
            with torch.no_grad():
                mean_gain = gains.mean(dim=1)
                better = (gains > 0).all(dim=1) & (mean_gain > best_gain)
                best_gain = torch.where(better, mean_gain, best_gain)
                best_phase = torch.where(better[:, None, None], phase, best_phase)
            if step < steps:
                optimizer.zero_grad(set_to_none=True)
                balanced_optical_loss(scores, baseline).backward()
                optimizer.step()
                with torch.no_grad():
                    correction.clamp_(-model.correction_limit, model.correction_limit)
        # Оставляем тензоры сразу на устройстве (V100), чтобы не гонять по шине PCIe каждую эпоху
        examples.append((features, initial, best_phase, targets, baseline))
        if index % 5 == 0 or index == total:
            print(f"Phase targets: batch {index}/{total}", flush=True)
    dataset = TensorDataset(*(torch.cat(tensors) for tensors in zip(*examples, strict=True)))
    return DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
