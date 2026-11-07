"""Small TRAIN-only GS difficulty check; never a neural-model evaluation."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from dataset.generate_dataset import save_json
from dataset.generate_soft_dataset import reconstruct_u8
from doe_eval.baseline import gerchberg_saxton_multiplane
from doe_eval.metrics import pearson, ssim_global, ssim_skimage
from doe_eval.physics import quantize_phase_u8


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=12)
    p.add_argument('--iters', type=int, default=60)
    args = p.parse_args()
    if args.out.exists():
        p.error('Output exists')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    old = json.loads(args.old.read_text())
    new = json.loads(args.new.read_text())
    groups = [
        ('old', args.old, [s for s in old['samples'] if s['split'] == 'train']),
        ('old_blur2', args.old, [s for s in old['samples'] if s['split'] == 'train']),
        ('soft_independent', args.new, [s for s in new['samples'] if s['split'] == 'train' and s['params']['kind'] == 'soft_independent']),
        ('physical_forward', args.new, [s for s in new['samples'] if s['split'] == 'train' and s['params']['kind'] == 'physical_forward']),
    ]
    report = {'scope': 'first N TRAIN quartets per group; fixed before running; no test scores used',
              'warning': 'Small exploratory GS experiment, no trained network and no generalization claim. Groups are different tasks.',
              'protocol': {'method': 'existing_multiplane_GS', 'iterations': args.iters,
                           'seed': 42, 'device': 'CPU', 'threads': 2, 'phase': 'uint8 [-pi,pi]',
                           'target': 'amplitude PNG/255', 'reconstruction': 'per-plane max; uint8 round-trip',
                           'ssim_global': 'existing doe_eval.metrics.ssim_global, K1=.01 K2=.03 population moments',
                           'ssim_windowed': 'existing doe_eval.metrics.ssim_skimage default 7x7, data_range=1',
                           'timing': 'single exploratory mask wall time, not a latency benchmark'},
              'groups': {}}
    panel = Image.new('RGB', (660, len(groups)*290), (20,22,28))
    draw = ImageDraw.Draw(panel)
    for row, (name, path, samples) in enumerate(groups):
        results = []
        for idx, s in enumerate(samples[:args.samples]):
            arrays = []
            for rel in s['images']:
                with Image.open(path.parent/rel) as im:
                    if name == 'old_blur2':
                        im = im.filter(ImageFilter.GaussianBlur(2))
                    a = np.asarray(im, dtype=np.float32)
                    if name == 'old_blur2':
                        a = np.rint(a/a.max()*255)
                    arrays.append(a/255)
            targets = np.stack(arrays)
            start = time.perf_counter()
            phase = gerchberg_saxton_multiplane(torch.from_numpy(targets), iters=args.iters, seed=42)
            elapsed = time.perf_counter()-start
            phase_path = args.out/f'{name}_{idx+1:02d}_doe.png'
            Image.fromarray(quantize_phase_u8(phase).numpy()).save(phase_path)
            with Image.open(phase_path) as im:
                rec = reconstruct_u8(np.asarray(im))/255
            metrics = [{k: fn(a,b) for k,fn in [('pearson',pearson),('ssim_global',ssim_global),('ssim_windowed',ssim_skimage)]}
                       for a,b in zip(targets,rec)]
            results.append({'sample_id': s['sample_id'], 'mask_seconds': elapsed, 'planes': metrics,
                            **{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}})
            if idx == 0:
                draw.text((3,row*290+6), name, fill='white')
                draw.text((3,row*290+50), 'target', fill='white')
                draw.text((3,row*290+190), f'GS{args.iters}', fill='white')
                for col in range(4):
                    panel.paste(Image.fromarray(np.rint(targets[col]*255).astype(np.uint8)).convert('RGB'), (130+col*130,row*290+25))
                    panel.paste(Image.fromarray(np.rint(rec[col]*255).astype(np.uint8)).convert('RGB'), (130+col*130,row*290+155))
        summary = {k: float(np.mean([r[k] for r in results])) for k in ('pearson','ssim_global','ssim_windowed')}
        report['groups'][name] = {'quartets': len(results), 'mean': summary, 'results': results}
        print(name, summary, flush=True)
    save_json(args.out/'report.json', report)
    panel.save(args.out/'comparison.png')


if __name__ == '__main__':
    main()
