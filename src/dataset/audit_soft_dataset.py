"""Additional independent checks for soft-1: provenance, hashes, phase witnesses."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset.audit import inspect, pixel_hash, standardized
from dataset.generate_dataset import save_json
from dataset.generate_soft_dataset import reconstruct_u8
from doe_eval.physics import dequantize_phase_u8
from doe_eval.reference_numpy import fresnel_reconstruct_numpy


def quartet_screen(root, samples):
    """Aligned quartet comparison: top 5 at 32px, verified at full resolution."""
    rows = [s for s in samples if s['split'] != 'diagnostic']
    coarse = []
    for s in rows:
        planes = []
        for p in s['images']:
            with Image.open(root/p) as im:
                planes.append(np.asarray(im.resize((32,32), Image.Resampling.BOX)).reshape(-1))
        coarse.append(standardized(planes).reshape(-1)/2)
    coarse = np.asarray(coarse)
    train = np.array([i for i,s in enumerate(rows) if s['split'] == 'train'])
    cache = {}

    def full(i):
        if i not in cache:
            cache[i] = standardized(np.stack([np.asarray(Image.open(root/p)).reshape(-1)
                                              for p in rows[i]['images']]))
        return cache[i]

    result = {}
    for split in ('validation', 'test'):
        queries = [i for i,s in enumerate(rows) if s['split'] == split]
        maxima = []
        for i in queries:
            scores = coarse[train] @ coarse[i]
            candidates = train[np.argsort(scores)[-min(5,len(train)):]]
            best = max(float(np.mean(np.sum(full(i)*full(int(j)),axis=-1))) for j in candidates)
            maxima.append(best)
        result[split+'_train'] = {'queries': len(queries), 'maximum_mean_plane_pearson': max(maxima),
                                  'quartets_at_or_above_095': sum(v>=.95 for v in maxima)}
    return result


def inspect_soft(manifest_path, old_manifest=None):
    path = Path(manifest_path)
    root = path.parent
    manifest = json.loads(path.read_text())
    report = inspect(path)
    errors = report['errors']
    counts, origins, seeds, witness_hashes = Counter(), set(), set(), set()
    physical_errors, independent_checks, stats = [], [], defaultdict(list)
    torch.set_num_threads(2)
    for s in manifest['samples']:
        if s['split'] == 'diagnostic':
            continue
        params = s['params']
        kind = params.get('kind')
        counts[f"{s['split']}/{kind}"] += 1
        if kind not in ('soft_independent', 'physical_forward'):
            errors.append(f"Unknown kind: {s['sample_id']}")
            continue
        origin = params.get('origin_id')
        if not origin or origin in origins or s['seed'] in seeds:
            errors.append(f"Reused/missing origin or seed: {s['sample_id']}")
        origins.add(origin)
        seeds.add(s['seed'])
        try:
            arrays = np.stack([np.asarray(Image.open(root/p)).copy() for p in s['images']])
            for a in arrays:
                a = a.astype(float)/255
                stats[f"{s['split']}/{kind}"].append([
                    float((a >= 128/255).mean()), float(a.mean()),
                    float(np.sqrt((np.mean(np.diff(a, axis=0)**2)+np.mean(np.diff(a, axis=1)**2))/2)),
                ])
            if kind == 'physical_forward':
                rel = params.get('reference_phase', '')
                phase_path = (root / rel).resolve()
                if not rel or not phase_path.is_relative_to(root.resolve()):
                    raise ValueError('Unsafe/missing reference phase')
                with Image.open(phase_path) as im:
                    if im.mode != 'L' or im.size != (128,128) or im.format != 'PNG':
                        raise ValueError('Reference phase must be PNG L 128x128')
                    phase = np.array(im)
                digest = hashlib.sha256(phase.tobytes()).hexdigest()
                if digest in witness_hashes:
                    errors.append(f"Duplicate reference phase: {s['sample_id']}")
                witness_hashes.add(digest)
                expected = reconstruct_u8(phase)
                delta = np.abs(expected.astype(int) - arrays.astype(int))
                physical_errors.append(int(delta.max()))
                if delta.max() > 1:
                    errors.append(f"Phase/target mismatch: {s['sample_id']}, max LSB error {delta.max()}")
                # Independent float64 direct-DFT implementation on first 3 TRAIN witnesses.
                if s['split'] == 'train' and len(independent_checks) < 3:
                    independent = fresnel_reconstruct_numpy(dequantize_phase_u8(phase))
                    independent /= independent.max(axis=(-2,-1), keepdims=True)
                    independent = np.rint(independent * 255)
                    err = int(np.abs(independent - arrays).max())
                    independent_checks.append({'sample_id': s['sample_id'], 'max_lsb_error': err})
                    if err > 1:
                        errors.append(f"Independent DFT mismatch: {s['sample_id']}")
        except (ValueError, OSError) as exc:
            errors.append(f"{s['sample_id']}: {exc}")
    hashes_path = root / 'sha256_png_files.json'
    if not hashes_path.is_file():
        errors.append('Missing PNG checksum inventory')
    else:
        hashes = json.loads(hashes_path.read_text())
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*.png')}
        if actual != set(hashes):
            errors.append('PNG file inventory changed')
        for rel, expected in hashes.items():
            p = root/rel
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                errors.append(f'PNG checksum mismatch: {rel}')
    overlap = {'checked': False}
    if old_manifest:
        old_path = Path(old_manifest)
        old = json.loads(old_path.read_text())
        old_hashes = {pixel_hash(old_path.parent/p) for s in old['samples']
                      if s['split'] != 'diagnostic' for p in s['images']}
        collisions = [p for s in manifest['samples'] if s['split'] != 'diagnostic'
                      for p in s['images'] if pixel_hash(root/p) in old_hashes]
        overlap = {'checked': True, 'old_images': sum(len(s['images']) for s in old['samples']
                   if s['split'] != 'diagnostic'), 'exact_collisions': collisions,
                   'diagnostic': 'Shared A1 intentionally excluded; never train/evaluation'}
        if collisions:
            errors.append('New/old exact target collisions')
    report['soft_checks'] = {
        'counts_by_kind': dict(counts), 'unique_origins': len(origins),
        'unique_quartet_seeds': len(seeds), 'unique_reference_phases': len(witness_hashes),
        'witness_target_max_lsb_error': max(physical_errors, default=None),
        'independent_direct_dft': independent_checks, 'old_dataset_overlap': overlap,
        'statistics_by_kind': {k: {'bright_fill_mean': float(np.mean(a, axis=0)[0]),
                                   'bright_fill_min': float(np.min(a, axis=0)[0]),
                                   'mean_amplitude': float(np.mean(a, axis=0)[1]),
                                   'gradient_rms_mean': float(np.mean(a, axis=0)[2])}
                               for k, a in stats.items()},
        'interpretation': 'Witness agreement verifies generation, NOT model quality. Physical fields may reach image edges.',
    }
    if not errors:
        report['soft_checks']['quartet_near_screen'] = quartet_screen(root, manifest['samples'])
    report['status'] = 'failed' if errors else ('review_required' if report['warnings'] else 'passed')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--old-manifest', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Report exists; choose another output path')
    report = inspect_soft(a.manifest, a.old_manifest)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_json(a.out, report)
    print(json.dumps({k: report[k] for k in ('status', 'counts', 'errors', 'warnings', 'soft_checks')}, indent=2))
    raise SystemExit(2 if report['errors'] else 0)


if __name__ == '__main__':
    main()
