"""Reproducible diagnostic targets and procedural amplitude dataset (Pillow only)."""
import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import PIL
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageStat

from doe_eval.config import DISTANCES, N

VERSION = '0.2.0'
SIZE, SCALE = N, 4
Z = [round(z * 1e3, 1) for z in DISTANCES]
FAMILIES = ['geometry', 'digit', 'letter', 'icon', 'strokes', 'composition']
GEOMETRY = ['circle', 'ring', 'rectangle', 'triangle', 'cross', 'ellipse',
            'diamond', 'star', 'pentagon', 'spots']
ICONS = ['house', 'tree', 'heart', 'arrow', 'cup', 'flag', 'sun', 'cloud', 'boat']


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def primitive(name, rng):
    """Draw in a padded local coordinate system, before placement in the frame."""
    im = Image.new('L', (256, 256))
    d = ImageDraw.Draw(im)
    width = rng.randint(15, 30)
    if name in ('circle', 'ring', 'ellipse'):
        b = (32, 60, 224, 196) if name == 'ellipse' else (32, 32, 224, 224)
        d.ellipse(b, fill=255 if name != 'ring' else None,
                  outline=255, width=width)
    elif name == 'rectangle':
        d.rectangle((35, 55, 221, 201), fill=255)
    elif name == 'triangle':
        d.polygon([(50, 30), (30, 223), (225, 223)], fill=255)
    elif name == 'cross':
        d.rectangle((111, 28, 145, 228), fill=255)
        d.rectangle((28, 111, 228, 145), fill=255)
    elif name in ('diamond', 'pentagon', 'star'):
        count = {'diamond': 4, 'pentagon': 5, 'star': 10}[name]
        points = []
        for i in range(count):
            a = -math.pi/2 + i * 2*math.pi/count
            radius = 45 if name == 'star' and i % 2 else 100
            points.append((128+radius*math.cos(a), 128+radius*math.sin(a)))
        d.polygon(points, fill=255)
    elif name == 'spots':
        for _ in range(rng.randint(2, 5)):
            x, y, r = rng.randint(45, 210), rng.randint(45, 210), rng.randint(15, 32)
            d.ellipse((x-r, y-r, x+r, y+r), fill=255)
    elif name == 'house':
        d.polygon([(25, 120), (128, 30), (231, 120)], fill=255)
        d.rectangle((55, 120, 201, 225), fill=255)
        d.rectangle((113, 165, 147, 225), fill=0)
    elif name == 'tree':
        d.rectangle((115, 155, 141, 230), fill=255)
        d.polygon([(128, 25), (30, 185), (226, 185)], fill=255)
    elif name == 'heart':
        d.ellipse((30, 38, 135, 143), fill=255)
        d.ellipse((121, 38, 226, 143), fill=255)
        d.polygon([(32, 103), (224, 103), (128, 225)], fill=255)
    elif name == 'arrow':
        d.rectangle((28, 108, 150, 148), fill=255)
        d.polygon([(140, 45), (230, 128), (140, 211)], fill=255)
    elif name == 'cup':
        d.ellipse((145, 70, 230, 170), outline=255, width=18)
        d.polygon([(35, 65), (175, 65), (160, 215), (50, 215)], fill=255)
    elif name == 'flag':
        d.rectangle((43, 25, 59, 230), fill=255)
        d.polygon([(59, 30), (220, 45), (181, 90), (218, 140), (59, 125)], fill=255)
    elif name == 'sun':
        d.ellipse((75, 75, 181, 181), fill=255)
        for i in range(8):
            a = i*math.pi/4
            d.line([(128+80*math.cos(a), 128+80*math.sin(a)),
                    (128+110*math.cos(a), 128+110*math.sin(a))], fill=255, width=16)
    elif name == 'cloud':
        d.ellipse((25, 105, 130, 200), fill=255)
        d.ellipse((65, 50, 182, 190), fill=255)
        d.ellipse((135, 95, 230, 200), fill=255)
        d.rectangle((65, 140, 185, 200), fill=255)
    elif name == 'boat':
        d.polygon([(25, 165), (230, 165), (193, 222), (65, 222)], fill=255)
        d.rectangle((124, 25, 136, 165), fill=255)
        d.polygon([(119, 35), (119, 150), (35, 150)], fill=255)
    elif name.startswith('glyph:'):
        glyph = name.split(':')[1]
        font = ImageFont.load_default(size=210)
        b = d.textbbox((0, 0), glyph, font=font)
        d.text(((256-(b[2]-b[0]))/2-b[0], (256-(b[3]-b[1]))/2-b[1]),
               glyph, fill=255, font=font)
    elif name == 'strokes':
        pts = [(rng.randint(35, 220), rng.randint(35, 220))
               for _ in range(rng.randint(3, 6))]
        d.line(pts, fill=255, width=width, joint='curve')
        for x, y in pts:
            r = width/2
            d.ellipse((x-r, y-r, x+r, y+r), fill=255)
    else:
        raise ValueError(name)
    return im


def target(family, seed):
    rng = random.Random(seed)
    if family == 'geometry':
        names = [rng.choice(GEOMETRY)]
    elif family == 'digit':
        names = ['glyph:' + rng.choice('0123456789')]
    elif family == 'letter':
        names = ['glyph:' + rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')]
    elif family == 'icon':
        names = [rng.choice(ICONS)]
    elif family == 'strokes':
        names = ['strokes']
    else:
        names = rng.sample(GEOMETRY + ICONS, rng.randint(2, 3))
    canvas = Image.new('L', (SIZE*SCALE, SIZE*SCALE))
    placements = []
    for name in names:
        tile = primitive(name, rng)
        angle = rng.uniform(-22, 22) if name.startswith('glyph:') else rng.uniform(-180, 180)
        tile = tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        tile = tile.crop(tile.getbbox())
        extent = rng.randint(52, 94) if len(names) == 1 else rng.randint(28, 53)
        ratio = rng.uniform(0.82, 1.18)
        w = max(1, round(extent*SCALE*min(1, ratio)))
        h = max(1, round(extent*SCALE*min(1, 1/ratio)))
        # Preserve native aspect ratio; random ratio adds a mild deformation.
        tile.thumbnail((w, h), Image.Resampling.LANCZOS)
        tile = tile.resize((max(1, round(tile.width*ratio)), tile.height), Image.Resampling.LANCZOS)
        margin = 12*SCALE
        x = rng.randint(margin, SIZE*SCALE-margin-tile.width)
        y = rng.randint(margin, SIZE*SCALE-margin-tile.height)
        layer = Image.new('L', canvas.size)
        layer.paste(tile, (x, y))
        canvas = ImageChops.lighter(canvas, layer)
        placements.append({'object': name, 'angle_deg': angle,
                           'bbox_highres': [x, y, x+tile.width, y+tile.height]})
    im = canvas.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    mode = rng.choices(['crisp', 'soft', 'soft_uneven'], [0.25, 0.50, 0.25])[0]
    sigma = 0.0 if mode == 'crisp' else rng.uniform(0.45, 1.15 if mode == 'soft' else 1.65)
    if sigma:
        im = im.filter(ImageFilter.GaussianBlur(sigma))
    # Smooth brightness variation only within the signal, retaining black background.
    if mode == 'soft_uneven':
        texture = Image.new('L', (8, 8))
        texture.putdata([rng.randint(150, 255) for _ in range(64)])
        texture = texture.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        texture = texture.filter(ImageFilter.GaussianBlur(3))
        im = ImageChops.multiply(im, texture)
    peak = im.getextrema()[1]
    im = im.point([round(v*255/peak) for v in range(256)])
    return im, {'family': family, 'image_seed': seed, 'appearance': mode,
                'blur_sigma_px': sigma, 'placements': placements}


def verify(manifest_path):
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert m['format'] == 'doe-manifest/1'
    assert m['distances_mm'] == Z
    seen_ids, seen_pixels = set(), {}
    counts, families, appearances = Counter(), Counter(), Counter()
    min_margin = SIZE
    hashes = {}
    for record in m['samples']:
        sid, split = record['sample_id'], record['split']
        assert sid not in seen_ids, f'Duplicate sample ID: {sid}'
        seen_ids.add(sid)
        assert split in ('train', 'validation', 'test', 'diagnostic')
        assert record['distances_mm'] == Z and len(record['images']) == 4
        counts[split] += 1
        for rel in record['images']:
            with Image.open(manifest_path.parent / rel) as im:
                assert im.format == 'PNG' and im.mode == 'L' and im.size == (SIZE, SIZE), rel
                assert im.getextrema()[0] == 0 and im.getextrema()[1] > 0, rel
                assert ImageStat.Stat(im).stddev[0] > 0, rel
                digest = hashlib.sha256(im.tobytes()).hexdigest()
                assert digest not in seen_pixels, f'Duplicate pixels: {rel} / {seen_pixels.get(digest)}'
                seen_pixels[digest] = rel
                # Bounding box of visible signal; includes blur down to 3/255.
                b = im.point(lambda v: 255 if v > 2 else 0).getbbox()
                margin = min(b[0], b[1], SIZE-b[2], SIZE-b[3])
                assert margin >= 4, f'Signal touches boundary: {rel}'
                min_margin = min(min_margin, margin)
                hashes[rel] = digest
        for p in record['params'].get('targets', []):
            families[f"{split}/{p['family']}"] += 1
            appearances[f"{split}/{p['appearance']}"] += 1
    return {'status': 'passed', 'counts_quartets': dict(counts),
            'total_target_images': len(hashes), 'exact_duplicate_images': 0,
            'minimum_visible_margin_px': min_margin,
            'families_per_split': dict(sorted(families.items())),
            'appearance_per_split': dict(sorted(appearances.items())),
            'pixel_hashes_sha256': hashes}


def previews(out, samples):
    chosen = [r for r in samples if r['split'] == 'train'][:10]
    chosen += [r for r in samples if r['split'] == 'validation'][:5]
    chosen += [r for r in samples if r['split'] == 'test'][:5]
    folder = out / 'previews'
    folder.mkdir()
    for page in range(4):
        sheet = Image.new('RGB', (650, 5*145), 'white')
        draw = ImageDraw.Draw(sheet)
        for row, record in enumerate(chosen[page*5:(page+1)*5]):
            draw.text((5, row*145+5), record['sample_id'], fill='black')
            for col, rel in enumerate(record['images']):
                with Image.open(out / rel) as im:
                    sheet.paste(im.convert('RGB'), (120+col*132, row*145+15))
        sheet.save(folder / f'contact_sheet_{page+1:02d}.png')


def generate(out, ntrain, nval, ntest, master_seed):
    if out.exists():
        raise SystemExit(f'Output already exists: {out}. Choose a new --out folder; frozen data are not overwritten.')
    out.mkdir(parents=True)
    samples = []
    source = Path(__file__).resolve().parent / 'diagnostic_source'
    files = ['01_ring.png', '02_cross.png', '03_triangle.png', '04_two_spots.png']
    diag = out / 'diagnostic' / '0000'
    diag.mkdir(parents=True)
    for i, name in enumerate(files, 1):
        shutil.copyfile(source / name, diag / f'{i:02d}.png')
    samples.append({'sample_id': 'diagnostic_0000', 'split': 'diagnostic',
                    'images': [f'diagnostic/0000/{i:02d}.png' for i in range(1, 5)],
                    'distances_mm': Z, 'seed': 0, 'generator_version': VERSION,
                    'params': {'purpose': 'fixed_A1_debug_only'}})
    for split, n in [('train', ntrain), ('validation', nval), ('test', ntest)]:
        for idx in range(1, n+1):
            seed = stable_seed(master_seed, split, idx)
            rng = random.Random(seed)
            # Balanced families, shuffled independently of the optical plane.
            families = [FAMILIES[((idx-1)*4+c) % len(FAMILIES)] for c in range(4)]
            rng.shuffle(families)
            folder = out / split / f'{idx:06d}'
            folder.mkdir(parents=True)
            paths, params = [], []
            for c, family in enumerate(families, 1):
                im, p = target(family, stable_seed(seed, c))
                rel = f'{split}/{idx:06d}/{c:02d}.png'
                im.save(out / rel)
                paths.append(rel)
                params.append(p)
            samples.append({'sample_id': f'{split}_{idx:06d}', 'split': split,
                            'images': paths, 'distances_mm': Z, 'seed': seed,
                            'generator_version': VERSION, 'params': {'targets': params}})
        print(f'{split}: {n} quartets', flush=True)
    manifest = {'format': 'doe-manifest/1', 'dataset_name': 'procedural_mixed_targets',
                'dataset_version': f'v2-seed{master_seed}-train{ntrain}-val{nval}-test{ntest}',
                'generator_version': VERSION, 'distances_mm': Z,
                'master_seed': master_seed, 'pillow_version': PIL.__version__,
                'target_interpretation': 'normalized amplitude = png/255', 'samples': samples}
    save_json(out / 'manifest.json', manifest)
    report = verify(out / 'manifest.json')
    save_json(out / 'validation_report.json', report)
    previews(out, samples)
    print(f"Verified: {report['counts_quartets']}; {report['total_target_images']} PNGs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data'))
    parser.add_argument('--train', type=int, default=1000)
    parser.add_argument('--validation', type=int, default=100)
    parser.add_argument('--test', type=int, default=100)
    parser.add_argument('--seed', type=int, default=3701)
    parser.add_argument('--verify', type=Path, help='Only verify an existing manifest')
    args = parser.parse_args()
    if args.verify:
        r = verify(args.verify)
        print(json.dumps({k: v for k, v in r.items() if k != 'pixel_hashes_sha256'}, indent=2))
    else:
        if min(args.train, args.validation, args.test) < 1:
            parser.error('All split sizes must be positive.')
        generate(args.out, args.train, args.validation, args.test, args.seed)


if __name__ == '__main__':
    main()
