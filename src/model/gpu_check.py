"""Проверка CUDA, архитектуры GPU, FFT complex64 и backward с AMP."""
import torch

from doe_eval.config import N
from doe_eval.physics import fresnel_reconstruct


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна: проверьте драйвер и установку PyTorch для GPU")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    arch = f"sm_{capability[0]}{capability[1]}"
    print(f"GPU: {torch.cuda.get_device_name(device)}; {arch}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; arch list: {torch.cuda.get_arch_list()}")
    if capability == (7, 0) and arch not in torch.cuda.get_arch_list():
        raise SystemExit("Эта сборка не поддерживает V100. Установите uv sync --extra v100 --frozen")
    phase = torch.randn(1, N, N, device=device, requires_grad=True)
    amplitude = fresnel_reconstruct(phase)
    (amplitude / amplitude.amax()).square().mean().backward()
    assert phase.grad is not None and torch.isfinite(phase.grad).all() and phase.grad.abs().sum() > 0
    conv = torch.nn.Conv2d(4, 8, 3, padding=1).to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = conv(torch.rand(1, 4, N, N, device=device)).square().mean()
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(loss).backward()
    assert torch.isfinite(conv.weight.grad).all()
    torch.cuda.synchronize()
    print("[OK] CUDA, complex64 FFT, градиент фазы и AMP работают")


if __name__ == "__main__":
    main()
