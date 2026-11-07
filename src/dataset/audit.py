"""Dataset audit: strict structure, actual pixel hashes, frozen split integrity,
statistics and approximate near-duplicate screening. No training or data edits.
Run from repository root: python -m dataset.audit --manifest ... --out ...
"""
import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw

from doe_eval.config import DISTANCES as DISTANCES_M
from doe_eval.config import IMAGE_FILENAMES, N

DISTANCES = [round(z * 1e3, 1) for z in DISTANCES_M]
SPLITS = ('diagnostic', 'train', 'validation', 'test')


def pixel_hash(path):
    with Image.open(path) as im:
        return hashlib.sha256(im.tobytes()).hexdigest()


def standardized(arr):
    x = np.asarray(arr, dtype=np.float32)
    x = x - x.mean(axis=-1, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def inspect(manifest, reference=None, near=0.95):
    manifest = Path(manifest).resolve()
    root = manifest.parent
    m = json.loads(manifest.read_text())
    errors, warnings, records, images, small = [], [], [], [], []
    ids, hashes = set(), {}
    counts, family_plane, appearances, objects = Counter(), Counter(), Counter(), Counter()
    stats = {s: [] for s in SPLITS}
    if m.get('format') != 'doe-manifest/1' or m.get('distances_mm') != DISTANCES:
        errors.append('Invalid top-level format or distances')
    for s in m.get('samples', []):
        sid, split = s.get('sample_id'), s.get('split')
        if not sid or sid in ids or split not in SPLITS:
            errors.append(f'Invalid/duplicate ID or split: {sid}')
            continue
        ids.add(sid)
        counts[split] += 1
        if len(s.get('images', [])) != 4 or s.get('distances_mm') != DISTANCES:
            errors.append(f'Invalid quartet: {sid}')
            continue
        for c, rel in enumerate(s['images']):
            path = (root / rel).resolve()
            if not path.is_relative_to(root) or Path(rel).parts[0] != split:
                errors.append(f'Unsafe/wrong-split path: {rel}')
                continue
            if Path(rel).name != IMAGE_FILENAMES[c]:
                errors.append(f'Wrong channel filename: {rel}')
            try:
                with Image.open(path) as im:
                    if im.format != 'PNG' or im.mode != 'L' or im.size != (N, N):
                        raise ValueError(f'Expected PNG L {N}x{N}')
                    arr = np.asarray(im).copy()
                    if arr.min() != 0 or arr.max() == 0 or arr.std() == 0:
                        raise ValueError('Missing foreground/background or constant image')
                    ys, xs = np.nonzero(arr > 2)
                    margin = int(min(xs.min(), ys.min(), N-1-xs.max(), N-1-ys.max()))
                    if margin < 4:
                        errors.append(f'Signal too close to edge: {rel}')
                    digest = hashlib.sha256(arr.tobytes()).hexdigest()
                    if digest in hashes:
                        errors.append(f'Exact duplicate: {hashes[digest]} / {rel}')
                    else:
                        hashes[digest] = rel
                    records.append({'path': rel, 'split': split, 'id': sid, 'channel': c,
                                    'pixel_sha256': digest})
                    images.append(arr.reshape(-1))
                    small.append(np.asarray(im.resize((32, 32), Image.Resampling.BOX)).reshape(-1))
                    stats[split].append((float((arr >= 128).mean()), float(arr.mean()/255), margin))
            except (OSError, ValueError) as exc:
                errors.append(f'{rel}: {exc}')
        for c, t in enumerate(s.get('params', {}).get('targets', [])):
            family_plane[f'{split}/z{c+1}/{t.get("family")}'] += 1
            appearances[f'{split}/{t.get("appearance")}'] += 1
            for obj in t.get('placements', []):
                objects[f'{split}/{obj.get("object")}'] += 1
    for split in SPLITS:
        if not counts[split]:
            errors.append(f'Missing split: {split}')

    frozen = None
    if reference:
        reference = Path(reference).resolve()
        ref = json.loads(reference.read_text())
        lookup = {s['sample_id']: s for s in m['samples']}
        frozen = {'checked_images': 0, 'mismatches': []}
        for s in ref['samples']:
            new = lookup.get(s['sample_id'])
            if not new or any(s[k] != new.get(k) for k in ('split', 'images', 'seed', 'params', 'distances_mm')):
                frozen['mismatches'].append(s['sample_id'])
                continue
            for rel in s['images']:
                try:
                    if pixel_hash(reference.parent/rel) != pixel_hash(root/rel):
                        frozen['mismatches'].append(rel)
                    frozen['checked_images'] += 1
                except OSError:
                    frozen['mismatches'].append(rel)
        # Frozen evaluation splits may not gain/lose records either.
        for split in ('validation', 'test', 'diagnostic'):
            if {s['sample_id'] for s in ref['samples'] if s['split'] == split} != {
                    s['sample_id'] for s in m['samples'] if s['split'] == split}:
                frozen['mismatches'].append(f'{split}: membership changed')
        if frozen['mismatches']:
            errors.append('Reference images/records or frozen split membership changed')

    similarity, pairs = {}, []
    if not errors:
        image_array = np.asarray(images, dtype=np.uint8)
        screen = standardized(small)
        train = np.array([i for i, r in enumerate(records) if r['split'] == 'train'])
        val = np.array([i for i, r in enumerate(records) if r['split'] == 'validation'])
        test = np.array([i for i, r in enumerate(records) if r['split'] == 'test'])
        sample = np.random.default_rng(3701).choice(train, min(300, len(train)), replace=False)
        for name, query, base in [('validation_train', val, train), ('test_train', test, train),
                                  ('test_validation', test, val), ('train_sample_train', sample, train)]:
            hits = 0
            top_scores = []
            for start in range(0, len(query), 64):
                qi = query[start:start+64]
                corr = screen[qi] @ screen[base].T
                corr[qi[:, None] == base[None, :]] = -2
                k = min(5, len(base))
                candidates = np.argpartition(corr, -k, axis=1)[:, -k:]
                for i, candidates_i in zip(qi, candidates):
                    cand = base[candidates_i]
                    cand = cand[cand != i]
                    if not len(cand):
                        continue
                    values = standardized(image_array[cand]) @ standardized(image_array[i])
                    j = int(np.argmax(values))
                    score = float(values[j])
                    top_scores.append(score)
                    if score >= near:
                        hits += 1
                        pairs.append({'comparison': name, 'a': records[i]['path'],
                                      'b': records[int(cand[j])]['path'], 'pearson': score})
            similarity[name] = {'queries': len(query), 'near_images_found': hits,
                                'share_found': hits/len(query),
                                'median_best_candidate_pearson': float(np.median(top_scores))}
            if hits:
                warnings.append(f'{name}: {hits}/{len(query)} near images require visual review')
    summary = {}
    for split, values in stats.items():
        if values:
            a = np.array(values)
            summary[split] = {'images': len(a), 'fill_mean': float(a[:, 0].mean()),
                              'fill_p05_p50_p95': np.quantile(a[:, 0], [.05, .5, .95]).tolist(),
                              'share_fill_under_3pct': float((a[:, 0] < .03).mean()),
                              'mean_brightness': float(a[:, 1].mean()), 'min_margin': int(a[:, 2].min())}
            if split != 'diagnostic' and summary[split]['share_fill_under_3pct'] > .25:
                warnings.append(f'{split}: over 25% of images have under 3% bright pixels; inspect foreground quality, not only background-dominated metrics')
    # Report observed family combinations, even when all single families are balanced.
    combos = Counter(tuple(sorted(t['family'] for t in s.get('params', {}).get('targets', [])))
                     for s in m['samples'] if s['split'] == 'train')
    if len(combos) < 6:
        warnings.append(f'Train uses only {len(combos)} distinct family combinations; diverse images do not imply arbitrary-scene coverage')
    return {'status': 'failed' if errors else ('review_required' if warnings else 'passed'),
            'dataset_version': m.get('dataset_version'), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__},
            'counts': dict(counts), 'checked_images': len(records), 'errors': errors, 'warnings': warnings,
            'statistics': summary, 'family_by_plane': dict(sorted(family_plane.items())),
            'appearance': dict(sorted(appearances.items())), 'objects': dict(sorted(objects.items())),
            'train_family_combinations': {' + '.join(k): v for k, v in combos.items()},
            'frozen_reference': frozen, 'near_threshold': near, 'near_similarity': similarity,
            'near_method': 'Approximate: top 5 candidates by 32x32 Pearson, verified at 128x128. All held-out queries; 300 seeded train queries. No alignment invariance; may miss similar pairs.',
            'near_pairs': sorted(pairs, key=lambda p: -p['pearson'])[:40],
            'pixel_hashes': {r['path']: r['pixel_sha256'] for r in records}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--near', type=float, default=.95)
    args = p.parse_args()
    if not 0 < args.near <= 1:
        p.error('--near must be in (0, 1]')
    if args.out.exists():
        p.error('Output folder exists; choose a new one')
    report = inspect(args.manifest, args.reference, args.near)
    args.out.mkdir(parents=True)
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if report['near_pairs']:
        pairs = report['near_pairs'][:12]
        panel = Image.new('RGB', (400, len(pairs)*150), 'white')
        draw = ImageDraw.Draw(panel)
        for row, pair in enumerate(pairs):
            for col, key in enumerate(('a', 'b')):
                with Image.open(args.manifest.parent/pair[key]) as im:
                    panel.paste(im.convert('RGB'), (col*135, row*150))
            draw.text((275, row*150+15), f"r={pair['pearson']:.4f}", fill='black')
            draw.text((0, row*150+130), pair['comparison'], fill='black')
        panel.save(args.out/'near_pairs.png')
    print(json.dumps({'status': report['status'], 'counts': report['counts'],
                      'checked_images': report['checked_images'], 'error_count': len(report['errors']),
                      'first_errors': report['errors'][:5], 'warnings': report['warnings']}, indent=2))
    raise SystemExit(2 if report['errors'] else 0)


if __name__ == '__main__':
    main()
