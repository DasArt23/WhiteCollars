"""Версионированное расширение train полутонами; исходные сплиты неизменны.

Добавляет 6 семейств с независимым выбором для каждой из четырёх плоскостей.
Отдельный stress/manifest.json содержит новые validation-цели для проверки покрытия.
Генерация и проверка не обучают нейросеть и не используют интернет.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from doe_eval.config import DEFAULT_MANIFEST, DISTANCES, IMAGE_FILENAMES, N

VERSION = 'grayscale-1.0'
FAMILIES = ('gray_shapes', 'blobs', 'bands', 'gradient', 'texture', 'composition')


def target(family: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:N, :N] / (N - 1)
    background = float(rng.uniform(0, .32)) if rng.random() < .4 else 0.
    if family in ('gray_shapes', 'composition'):
        image = Image.new('L', (N, N), round(background * 255))
        draw = ImageDraw.Draw(image)
        for i in range(int(rng.integers(3, 10))):
            width, height = rng.integers(N // 8, N // 2, size=2)
            left = int(rng.integers(0, N - width))
            top = int(rng.integers(0, N - height))
            box = (left, top, left + int(width), top + int(height))
            tone = 255 if i == 0 else int(rng.integers(35, 245))
            if rng.random() < .5:
                draw.ellipse(box, fill=tone)
            else:
                draw.rectangle(box, fill=tone)
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0, 1.2))))
        values = np.asarray(image, dtype=np.float64) / 255
        if family == 'composition':
            values = .65 * values + .35 * (x * rng.random() + y * rng.random())
    elif family == 'blobs':
        values = np.zeros((N, N))
        for _ in range(int(rng.integers(3, 10))):
            cx, cy = rng.uniform(.1, .9, size=2)
            sx, sy = rng.uniform(.04, .25, size=2)
            values += rng.uniform(.2, 1.) * np.exp(-.5 * (((x - cx) / sx)**2 + ((y - cy) / sy)**2))
        values /= values.max()
        values = background + (1 - background) * values
    elif family == 'bands':
        angle = rng.uniform(0, 2 * np.pi)
        axis = x * np.cos(angle) + y * np.sin(angle)
        values = .5 + .5 * np.cos(2 * np.pi * rng.uniform(1, 9) * axis + rng.uniform(0, 2 * np.pi))
        values = background + (1 - background) * values
    elif family == 'gradient':
        angle = rng.uniform(0, 2 * np.pi)
        values = x * np.cos(angle) + y * np.sin(angle)
        if rng.random() < .5:
            cx, cy = rng.uniform(.2, .8, size=2)
            values = np.sqrt((x - cx)**2 + (y - cy)**2)
        values = (values - values.min()) / (values.max() - values.min())
        values = background + (1 - background) * values**rng.uniform(.5, 2.)
    elif family == 'texture':
        side = int(rng.integers(5, 24))
        coarse = rng.integers(0, 256, (side, side), dtype=np.uint8)
        image = Image.fromarray(coarse).resize((N, N), Image.Resampling.BICUBIC)
        values = np.asarray(image, dtype=np.float64) / 255
        values = background + (1 - background) * values
    else:
        raise ValueError(f'Неизвестное семейство: {family}')
    # Совпадает с max-нормировкой проверяющего конвейера; внутри остаются полутона.
    values = np.clip(values, 0, None)
    values /= max(float(values.max()), 1e-12)
    return Image.fromarray(np.rint(values * 255).astype(np.uint8))


def _records(root: Path, count: int, split: str, prefix: str, seed: int, seen: set[str]):
    records = []
    z_mm = [round(z * 1e3, 1) for z in DISTANCES]
    for i in range(count):
        sample_seed = seed + i * len(IMAGE_FILENAMES)
        rng = np.random.default_rng(sample_seed)
        families = rng.choice(FAMILIES, size=len(IMAGE_FILENAMES)).tolist()
        paths, params = [], []
        for channel, (family, name) in enumerate(zip(families, IMAGE_FILENAMES)):
            image_seed = sample_seed + channel
            # Не допускаем случайных точных дублей с исходными данными и stress.
            for attempt in range(100):
                image = target(family, image_seed + attempt * 1_000_000)
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest not in seen:
                    break
            else:
                raise RuntimeError('Не удалось сгенерировать уникальное изображение')
            seen.add(digest)
            rel = f'{split}/{prefix}{i + 1:06d}/{name}'
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            paths.append(rel)
            params.append({'family': family, 'image_seed': image_seed + attempt * 1_000_000})
        records.append({'sample_id': f'{split}_{prefix}{i + 1:06d}', 'split': split,
                        'images': paths, 'distances_mm': z_mm, 'seed': sample_seed,
                        'generator_version': VERSION, 'params': {'targets': params}})
    return records


def expand(source: Path, out: Path, extra_train: int = 1000, stress_count: int = 100, seed: int = 7421):
    source, out = Path(source), Path(out)
    if out.exists():
        raise ValueError('Выходная папка уже существует; выберите новую')
    if extra_train < 1 or stress_count < 1:
        raise ValueError('extra_train и stress_count должны быть положительными')
    manifest = json.loads(source.read_text())
    seen, hashes = set(), {}
    # Проверяем исходные PNG, затем копируем каждый файл из manifest без изменения пикселей.
    out.mkdir(parents=True)
    for record in manifest['samples']:
        for rel in record['images']:
            original = (source.parent / rel).resolve()
            if not original.is_relative_to(source.parent.resolve()):
                raise ValueError(f'Путь за пределами датасета: {rel}')
            with Image.open(original) as image:
                if image.format != 'PNG' or image.mode != 'L' or image.size != (N, N):
                    raise ValueError(f'Неверный входной PNG: {rel}')
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest in seen:
                    raise ValueError(f'Дубль в исходном наборе: {rel}')
                seen.add(digest)
                hashes[rel] = digest
            destination = out / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
    manifest['samples'].extend(_records(out, extra_train, 'train', 'gray', seed, seen))
    manifest['dataset_version'] = f"{manifest['dataset_version']}+{VERSION}-train{extra_train}"
    manifest['expansion'] = {'version': VERSION, 'seed': seed, 'extra_train': extra_train,
                             'source_manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                             'source_pixel_hashes': hashes,
                             'stress_manifest': 'stress/manifest.json'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    stress_root = out / 'stress'
    stress_records = _records(stress_root, stress_count, 'validation', 'gray', seed + 10_000_000, seen)
    stress = {'format': 'doe-manifest/1', 'dataset_name': 'grayscale_stress_validation',
              'dataset_version': VERSION, 'generator_version': VERSION,
              'distances_mm': [round(z * 1e3, 1) for z in DISTANCES],
              'master_seed': seed + 10_000_000, 'samples': stress_records}
    (stress_root / 'manifest.json').write_text(json.dumps(stress, indent=2, ensure_ascii=False) + '\n')
    report = verify(out / 'manifest.json')
    (out / 'grayscale_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report


def verify(manifest_path: Path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    stress = json.loads((manifest_path.parent / 'stress/manifest.json').read_text())
    hashes, counts, intermediate, levels, combos = {}, {}, [], [], set()
    for root, records in ((manifest_path.parent, manifest['samples']), (manifest_path.parent / 'stress', stress['samples'])):
        for record in records:
            label = record['split'] if root == manifest_path.parent else 'stress_validation'
            counts[label] = counts.get(label, 0) + 1
            if record['generator_version'] == VERSION:
                combos.add(tuple(t['family'] for t in record['params']['targets']))
            for rel in record['images']:
                with Image.open(root / rel) as image:
                    if image.format != 'PNG' or image.mode != 'L' or image.size != (N, N):
                        raise ValueError(f'Неверный PNG: {rel}')
                    values = np.asarray(image)
                    digest = hashlib.sha256(values.tobytes()).hexdigest()
                    if digest in hashes:
                        raise ValueError(f'Точный дубль: {rel} и {hashes[digest]}')
                    hashes[digest] = str(root / rel)
                    if (root == manifest_path.parent
                        and rel in manifest['expansion']['source_pixel_hashes']
                        and digest != manifest['expansion']['source_pixel_hashes'][rel]):
                        raise ValueError(f'Изменён исходный PNG: {rel}')
                    if record['generator_version'] == VERSION:
                        intermediate.append(float(((values > 0) & (values < 255)).mean()))
                        levels.append(len(np.unique(values)))
    return {'status': 'passed', 'counts': counts, 'checked_images': len(hashes),
            'exact_duplicates': 0, 'source_pixels_preserved': True,
            'new_family_sequences': len(combos),
            'new_intermediate_pixel_share_mean': float(np.mean(intermediate)),
            'new_unique_gray_levels_median': float(np.median(levels)),
            'limitations': ['Процедурные полутона не заменяют реальные фотографии.',
                            'Нет гарантии качества на неизвестных данных организаторов.',
                            'Новые stress-цели не участвуют в обучении или выборе checkpoint.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extra-train', type=int, default=1000)
    parser.add_argument('--stress-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7421)
    args = parser.parse_args()
    report = expand(args.source, args.out, args.extra_train, args.stress_count, args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
