"""Добавляет TRAIN из обменного ZIP к готовому набору без изменения held-out."""

import argparse
import hashlib
import io
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from doe_eval.config import DISTANCES, IMAGE_FILENAMES, PIXEL, WAVELENGTH, N


def merge(base: Path, archive: Path, out: Path) -> dict:
    base, archive, out = Path(base), Path(archive), Path(out)
    if out.exists():
        raise ValueError("Выходная папка уже существует; выберите новую")
    manifest = json.loads(base.read_text(encoding="utf-8"))
    distances = [round(z * 1000, 1) for z in DISTANCES]
    if (
        manifest.get("format") != "doe-manifest/1"
        or manifest.get("distances_mm") != distances
    ):
        raise ValueError("Неверный исходный manifest")
    known = set()
    for record in manifest["samples"]:
        for rel in record["images"]:
            path = (base.parent / rel).resolve()
            if not path.is_relative_to(base.parent.resolve()):
                raise ValueError("Путь за пределами исходного набора")
            with Image.open(path) as image:
                known.add(hashlib.sha256(image.tobytes()).hexdigest())
    stress_path = base.parent / "stress/manifest.json"
    if stress_path.exists():
        for record in json.loads(stress_path.read_text())["samples"]:
            for rel in record["images"]:
                path = (stress_path.parent / rel).resolve()
                if not path.is_relative_to(stress_path.parent.resolve()):
                    raise ValueError("Путь за пределами stress")
                with Image.open(path) as image:
                    known.add(hashlib.sha256(image.tobytes()).hexdigest())

    pending, keys, incoming = [], set(), set()
    with ZipFile(archive) as zipped:
        if len(zipped.namelist()) != len(set(zipped.namelist())):
            raise ValueError("Повторяющиеся пути в ZIP")
        metadata = json.loads(zipped.read("manifest_exchange.json"))
        if (
            metadata.get("schema") != "doe_train_targets_exchange_v1"
            or metadata.get("split") != "train"
            or metadata.get("plane_order") != ["z1", "z2", "z3", "z4"]
            or metadata.get("distances_mm") != distances
            or metadata.get("image_size_px") != [N, N]
            or metadata.get("target_quantity") != "normalized amplitude |U| (PNG/255)"
            or not math.isclose(metadata.get("wavelength_nm", 0) * 1e-9, WAVELENGTH)
            or not math.isclose(metadata.get("doe_pitch_um", 0) * 1e-6, PIXEL)
        ):
            raise ValueError(
                "Обменный архив не соответствует оптическим параметрам проекта"
            )
        samples = metadata["samples"]
        if (
            len(samples) != metadata["quartet_count"]
            or len(samples) * 4 != metadata["target_png_count"]
        ):
            raise ValueError("Число примеров не соответствует manifest")
        for record in samples:
            if record["split"] != "train" or record["key"] in keys:
                raise ValueError("Повтор ключа или импорт не из TRAIN")
            keys.add(record["key"])
            if len(record["paths"]) != 4 or len(record["sha256"]) != 4:
                raise ValueError("Неполная четвёрка")
            images = []
            for plane, (rel, expected) in enumerate(
                zip(record["paths"], record["sha256"]), 1
            ):
                path = Path(rel)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.name != f"z{plane}.png"
                ):
                    raise ValueError("Неверный путь или порядок плоскостей")
                data = zipped.read(rel)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Неверная SHA256: {rel}")
                with Image.open(io.BytesIO(data)) as image:
                    if (
                        image.format != "PNG"
                        or image.mode != "L"
                        or image.size != (N, N)
                    ):
                        raise ValueError(f"Неверный PNG: {rel}")
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest in known or digest in incoming:
                    raise ValueError(
                        f"Совпадение пикселей с имеющимися данными или внутри архива: {rel}"
                    )
                incoming.add(digest)
                images.append(data)
            pending.append((record, images))

    shutil.copytree(base.parent, out)
    for index, (record, images) in enumerate(pending, 1):
        sid = f"exchange{index:06d}"
        directory = out / "train" / sid
        directory.mkdir(exist_ok=False)
        paths = []
        for name, data in zip(IMAGE_FILENAMES, images):
            (directory / name).write_bytes(data)
            paths.append(f"train/{sid}/{name}")
        manifest["samples"].append(
            {
                "sample_id": f"train_{sid}",
                "split": "train",
                "images": paths,
                "distances_mm": distances,
                "seed": 0,
                "generator_version": "exchange-v1",
                "params": {
                    "exchange": record,
                    "preserve_plane_order": record["kind"] == "A",
                },
            }
        )
    report = {
        "counts": dict(Counter(r["split"] for r in manifest["samples"])),
        "imported_quartets": len(pending),
        "pixel_duplicates": 0,
        "held_out_preserved": True,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "kinds": dict(Counter(r["kind"] for r, _ in pending)),
    }
    manifest["dataset_version"] += f"+exchange{len(pending)}"
    manifest["exchange_import"] = report
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (out / "exchange_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    (out / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    old_report = out / "grayscale_report.json"
    if old_report.exists():
        old_report.rename(out / "grayscale_report_before_import.json")
    readme = out / "README.md"
    previous = readme.read_text() if readme.exists() else ""
    readme.write_text(
        "# Объединённый обучающий набор\n\n"
        + f"Текущие количества: {report['counts']}. Добавлено {len(pending)} TRAIN-четвёрок.\n"
        + "Validation/test/stress сохранены. Проверка: import_report.json.\n\n"
        + "## Описание исходного набора до импорта\n\n"
        + previous
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.base, args.archive, args.out), indent=2))


if __name__ == "__main__":
    main()
