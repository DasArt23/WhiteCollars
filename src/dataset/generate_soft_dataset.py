"""Versioned soft/forward-simulated targets; keeps the original dataset intact."""
import argparse
import hashlib
import itertools
import platform
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image, ImageChops, ImageFilter

from dataset.generate_dataset import (
    FAMILIES,
    GEOMETRY,
    ICONS,
    Z,
    previews,
    primitive,
    save_json,
    stable_seed,
)
from doe_eval.baseline import gerchberg_saxton_multiplane
from doe_eval.config import PIXEL, WAVELENGTH
from doe_eval.physics import (
    dequantize_phase_u8,
    fresnel_reconstruct,
    normalize_reconstructions,
    quantize_phase_u8,
)

VERSION = 'soft-1.0.0'
KINDS = ('soft_independent', 'soft_independent', 'soft_independent',
         'physical_forward', 'physical_forward')
COMBINATIONS = list(itertools.combinations(FAMILIES, 4))


def names_for(family, rng):
    if family == 'geometry':
        return [rng.choice(GEOMETRY)]
    if family == 'digit':
        return ['glyph:' + rng.choice('0123456789')]
    if family == 'letter':
        return ['glyph:' + rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')]
    if family == 'icon':
        return [rng.choice(ICONS)]
    if family == 'strokes':
        return ['strokes']
    return rng.sample(GEOMETRY + ICONS, 2)


def soft_target(family, seed):
    """Draw large low-frequency targets. Defects change the desired amplitude.

    No camera-noise or denoising claim: each returned PNG is the actual target.
    Rejection uses image geometry only, never reconstruction/model scores.
    """
    for attempt in range(80):
        rng = random.Random(stable_seed(seed, 'draw', attempt))
        names = names_for(family, rng)
        canvas = Image.new('L', (512, 512))
        placements = []
        for name in names:
            tile = primitive(name, rng)
            # Thick strokes survive smoothing, including narrow glyphs and crosses.
            tile = tile.filter(ImageFilter.MaxFilter(11))
            angle = rng.uniform(-25, 25) if name.startswith('glyph:') else rng.uniform(-180, 180)
            tile = tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
            tile = tile.crop(tile.getbbox())
            extent = rng.randint(77, 97) if len(names) == 1 else rng.randint(48, 69)
            ratio = rng.uniform(.82, 1.2)
            tile = tile.resize((max(1, round(tile.width * ratio)), tile.height), Image.Resampling.LANCZOS)
            tile.thumbnail((extent * 4, extent * 4), Image.Resampling.LANCZOS)
            # thumbnail never enlarges: explicitly enlarge small original glyphs.
            scale = extent * 4 / max(tile.size)
            tile = tile.resize((round(tile.width * scale), round(tile.height * scale)), Image.Resampling.LANCZOS)
            x = rng.randint(60, 512 - 60 - tile.width)
            y = rng.randint(60, 512 - 60 - tile.height)
            layer = Image.new('L', canvas.size)
            layer.paste(tile, (x, y))
            canvas = ImageChops.lighter(canvas, layer)
            placements.append({'object': name, 'angle_deg': angle,
                               'bbox_highres': [x, y, x + tile.width, y + tile.height]})
        arr = np.array(canvas.resize((128, 128), Image.Resampling.LANCZOS), dtype=np.float64) / 255
        appearance = rng.choices(['soft_clean', 'soft_uneven', 'soft_defective'], [4, 3, 3])[0]
        yy, xx = np.mgrid[:128, :128]
        defect_params = []
        if appearance != 'soft_clean':
            texture = Image.new('L', (6, 6))
            texture.putdata([rng.randint(110, 255) for _ in range(36)])
            texture = texture.resize((128, 128), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(5))
            arr *= np.asarray(texture) / 255
        if appearance == 'soft_defective':
            coords = np.argwhere(arr > .25)
            for _ in range(rng.randint(1, 3)):
                cy, cx = coords[rng.randrange(len(coords))]
                sx, sy, depth = rng.uniform(3, 8), rng.uniform(4, 12), rng.uniform(.4, .9)
                attenuation = 1 - depth * np.exp(-.5 * (((xx-cx)/sx)**2 + ((yy-cy)/sy)**2))
                arr *= attenuation
                defect_params.append({'center_xy': [int(cx), int(cy)], 'sigma_xy_px': [sx, sy], 'depth': depth})
        sigma = rng.uniform(1.4, 3.2)
        im = Image.fromarray(np.rint(arr * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
        arr = np.asarray(im, dtype=np.float64)
        arr = np.rint(arr / arr.max() * 255).astype(np.uint8)
        ys, xs = np.nonzero(arr > 2)
        margin = min(xs.min(), ys.min(), 127-xs.max(), 127-ys.max())
        fill = float((arr >= 128).mean())
        if margin >= 4 and .06 <= fill <= .58:
            return arr, {'family': family, 'image_seed': seed, 'attempt': attempt,
                         'appearance': appearance, 'blur_sigma_px': sigma,
                         'defects': defect_params, 'placements': placements}
    raise RuntimeError(f'No acceptable target for {family}, seed {seed}')


def reconstruct_u8(phase_u8):
    """Same phase PNG decoding and max-normalized amplitude as the existing backend."""
    with torch.no_grad():
        phase = dequantize_phase_u8(torch.from_numpy(np.array(phase_u8, copy=True)))
        amp = normalize_reconstructions(fresnel_reconstruct(phase), norm='max', compare='amplitude')
        return np.rint(amp.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def generate(out, ntrain=600, nval=100, ntest=100, master_seed=737037, teacher_iters=40):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite frozen dataset: {out}')
    if min(ntrain, nval, ntest, teacher_iters) < 1:
        raise ValueError('Split sizes and teacher iterations must be positive')
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    samples = []
    source = Path(__file__).parent / 'diagnostic_source'
    diag = out / 'diagnostic' / '0000'
    diag.mkdir(parents=True)
    for c, name in enumerate(['01_ring.png', '02_cross.png', '03_triangle.png', '04_two_spots.png'], 1):
        shutil.copyfile(source / name, diag / f'{c:02d}.png')
    samples.append({'sample_id': 'diagnostic_0000', 'split': 'diagnostic',
                    'images': [f'diagnostic/0000/{c:02d}.png' for c in range(1, 5)],
                    'distances_mm': Z, 'seed': 0, 'generator_version': VERSION,
                    'params': {'kind': 'diagnostic', 'purpose': 'unchanged_A1_debug_only'}})
    for split, count in [('train', ntrain), ('validation', nval), ('test', ntest)]:
        for idx in range(1, count+1):
            seed = stable_seed(VERSION, master_seed, split, idx)
            rng = random.Random(seed)
            # 3 independent / 2 forward examples in each block; all 15 family sets.
            kinds = list(KINDS)
            random.Random(stable_seed(master_seed, split, 'kind-block', (idx-1)//5)).shuffle(kinds)
            kind = kinds[(idx-1) % 5]
            families = list(COMBINATIONS[(idx-1) % len(COMBINATIONS)])
            rng.shuffle(families)
            images, params = [], []
            for c, family in enumerate(families, 1):
                arr, p = soft_target(family, stable_seed(seed, c))
                images.append(arr)
                params.append(p)
            extra = {}
            if kind == 'physical_forward':
                target = torch.tensor(np.stack(images) / 255, dtype=torch.float32)
                teacher_seed = stable_seed(seed, 'GS') % (2**63-1)
                phase = gerchberg_saxton_multiplane(target, iters=teacher_iters, seed=teacher_seed)
                phase_u8 = quantize_phase_u8(phase).cpu().numpy()
                rel_phase = f'reference_phases/{split}/{idx:06d}.png'
                (out / rel_phase).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(phase_u8).save(out / rel_phase)
                # Re-read saved phase; all 4 targets come from precisely this one DOE.
                with Image.open(out / rel_phase) as saved:
                    images = reconstruct_u8(np.array(saved))
                extra = {'reference_phase': rel_phase, 'teacher_seed': teacher_seed,
                         'teacher_iters': teacher_iters, 'teacher': 'existing_multiplane_GS',
                         'parents_are_design_hints_not_final_targets': True}
                for p in params:
                    p['design_appearance'] = p['appearance']
                    p['appearance'] = 'physical_diffraction'
            folder = out / split / f'{idx:06d}'
            folder.mkdir(parents=True, exist_ok=True)
            paths = []
            for c, arr in enumerate(images, 1):
                rel = f'{split}/{idx:06d}/{c:02d}.png'
                Image.fromarray(arr).save(out / rel)
                paths.append(rel)
            samples.append({'sample_id': f'soft1_{split}_{idx:06d}', 'split': split,
                            'images': paths, 'distances_mm': Z, 'seed': seed,
                            'generator_version': VERSION,
                            'params': {'kind': kind, 'origin_id': f'soft1_{split}_{idx:06d}',
                                       'targets': params, **extra}})
            if idx % 100 == 0:
                print(f'{split}: {idx}/{count}', flush=True)
    manifest = {'format': 'doe-manifest/1', 'dataset_name': 'soft_and_forward_targets',
                'dataset_version': f'{VERSION}-seed{master_seed}-train{ntrain}-val{nval}-test{ntest}',
                'generator_version': VERSION, 'master_seed': master_seed,
                'distances_mm': Z, 'wavelength_m': WAVELENGTH, 'doe_pitch_m': PIXEL,
                'target_interpretation': 'normalized amplitude = png/255; per-plane max',
                'environment': {'python': platform.python_version(), 'pillow': PIL.__version__,
                                'numpy': np.__version__, 'torch': torch.__version__, 'device': 'cpu'},
                'code_sha256': {p: hashlib.sha256((Path(__file__).resolve().parents[1]/p).read_bytes()).hexdigest()
                                for p in ['dataset/generate_soft_dataset.py', 'dataset/generate_dataset.py',
                                          'doe_eval/physics.py', 'doe_eval/baseline.py', 'doe_eval/config.py']},
                'samples': samples}
    save_json(out / 'manifest.json', manifest)
    for kind in set(KINDS):
        sub = dict(manifest)
        sub['dataset_name'] = kind
        sub['samples'] = [s for s in samples if s['params']['kind'] in (kind, 'diagnostic')]
        save_json(out / f'manifest_{kind}.json', sub)
    previews(out, samples)
    hashes = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.rglob('*.png'))}
    save_json(out / 'sha256_png_files.json', hashes)
    print(dict(Counter((s['split'], s['params']['kind']) for s in samples)), flush=True)
    return out / 'manifest.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train', type=int, default=600)
    p.add_argument('--validation', type=int, default=100)
    p.add_argument('--test', type=int, default=100)
    p.add_argument('--seed', type=int, default=737037)
    p.add_argument('--teacher-iters', type=int, default=40)
    args = p.parse_args()
    generate(args.out, args.train, args.validation, args.test, args.seed, args.teacher_iters)


if __name__ == '__main__':
    main()
