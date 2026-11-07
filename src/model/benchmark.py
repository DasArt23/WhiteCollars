"""Сравнение random, GS и обученной сети на одних PNG и одинаковой физике."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from doe_eval.baseline import gerchberg_saxton_multiplane
from doe_eval.cli import evaluate_pair_arrays, reconstruct_from_doe_file, resolve_device
from doe_eval.config import DEFAULT_COMPARE, DEFAULT_MANIFEST, DEFAULT_NORM, N
from doe_eval.dataset import FourImageDataset
from doe_eval.io import save_comparison_png, save_doe_png
from main import load_trained_holonet
from model.holonet import hybrid_infer_phase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--split', default='validation', choices=['validation', 'test', 'diagnostic'])
    parser.add_argument('--checkpoint', help='Без checkpoint сравниваются только random и GS')
    parser.add_argument('--out', default='build/benchmark')
    parser.add_argument('--limit', type=int, default=10, help='0 = весь split; первые K примеров без отбора по качеству')
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--polish-steps', type=int, default=25)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.limit < 0 or args.iters < 1 or args.polish_steps < 0:
        parser.error('limit и polish-steps >= 0; iters >= 1')
    device = resolve_device(args.device)
    dataset = FourImageDataset(args.manifest, split=args.split)
    model = load_trained_holonet(args.checkpoint, str(device))[0] if args.checkpoint else None
    count = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(count):
        target, sid = dataset[i]
        target_t = target.to(device)
        rng = np.random.default_rng(3701 + i)
        methods = {
            'random': lambda rng=rng: torch.from_numpy(rng.uniform(-np.pi, np.pi, (N, N)).astype(np.float32)),
            'gs': lambda target_t=target_t: gerchberg_saxton_multiplane(target_t, iters=args.iters),
        }
        if model is not None:
            methods['network'] = lambda target_t=target_t: hybrid_infer_phase(model, target_t.unsqueeze(0), polish_steps=0)[0]
            methods['network_polish'] = lambda target_t=target_t: hybrid_infer_phase(model, target_t.unsqueeze(0), polish_steps=args.polish_steps)[0]
        for name, synthesize in methods.items():
            # Инициализация FFT/библиотек может входить во время первого примера;
            # это измерение конвейера синтеза, без PNG и финального расчёта метрик.
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            phase = synthesize().detach()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            path = output / name / f'{sid}.png'
            save_doe_png(phase.cpu().numpy(), path)
            recon = reconstruct_from_doe_file(path, device, DEFAULT_NORM, DEFAULT_COMPARE)
            report = evaluate_pair_arrays(target.numpy(), recon)
            save_comparison_png(target.numpy(), recon, output / name / 'preview' / f'{sid}_comparison.png')
            rows.append({'id': sid, 'method': name, 'synthesis_ms': elapsed, **report['mean']})
        print(f'[{i + 1}/{count}] {sid}', flush=True)
    with (output / 'results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        summary[method] = {}
        for key in ('pearson', 'ssim_global', 'ssim_skimage', 'synthesis_ms'):
            values = [r[key] for r in selected]
            summary[method][key] = {'mean': float(np.mean(values)), 'min': float(np.min(values)),
                                    'p10': float(np.quantile(values, .1)), 'median': float(np.median(values))}
    report = {'split': args.split, 'samples': count, 'iters': args.iters,
              'polish_steps': args.polish_steps, 'device': str(device), 'torch': torch.__version__,
              'checkpoint': args.checkpoint, 'norm': DEFAULT_NORM, 'compare': DEFAULT_COMPARE,
              'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'summary': summary}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    for method, metrics in summary.items():
        print(f"{method}: Pearson={metrics['pearson']['mean']:.4f}; global SSIM={metrics['ssim_global']['mean']:.4f}; skimage SSIM={metrics['ssim_skimage']['mean']:.4f}")
    print(f'Результаты: {output / "summary.json"}')


if __name__ == '__main__':
    main()
