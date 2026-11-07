"""Extract checksum-verified Soft600 and optionally add only its TRAIN to a base."""

import argparse
import hashlib
import json
import shutil
import stat
from collections import Counter
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from doe_eval.config import DISTANCES, IMAGE_FILENAMES, N


def extract(archive: Path, out: Path) -> int:
    out = Path(out).resolve()
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Repeated ZIP paths")
        hashes = json.loads(zipped.read("data_soft600/sha256_png_files.json"))
        checked = 0
        for item in zipped.infolist():
            if not item.filename.startswith("data_soft600/") or item.is_dir():
                continue
            relative = item.filename.removeprefix("data_soft600/")
            destination = (out / relative).resolve()
            if not destination.is_relative_to(out) or stat.S_ISLNK(
                item.external_attr >> 16
            ):
                raise ValueError("Unsafe ZIP path")
            data = zipped.read(item)
            if relative.endswith(".png"):
                if hashlib.sha256(data).hexdigest() != hashes.get(relative):
                    raise ValueError(f"PNG checksum mismatch: {relative}")
                checked += 1
            if destination.exists():
                if destination.read_bytes() != data:
                    raise ValueError(f"Existing file differs: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        if checked != len(hashes):
            raise ValueError("Incomplete PNG inventory")
    return checked


def merge(base: Path, source: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("Output already exists; choose a new directory")
    original = json.loads(base.read_text())
    soft = json.loads(source.read_text())
    expected = [round(z * 1000, 1) for z in DISTANCES]
    for manifest in (original, soft):
        if (
            manifest.get("format") != "doe-manifest/1"
            or manifest.get("distances_mm") != expected
        ):
            raise ValueError("Invalid manifest/plane ordering")
    existing = set()
    roots = [(base.parent, original)]
    stress = base.parent / "stress/manifest.json"
    if stress.exists():
        roots.append((stress.parent, json.loads(stress.read_text())))
    for root, manifest in roots:
        for record in manifest["samples"]:
            for relative in record["images"]:
                path = (root / relative).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("Target path outside dataset")
                with Image.open(path) as image:
                    existing.add(hashlib.sha256(image.tobytes()).hexdigest())
    incoming = set()
    for record in soft["samples"]:
        if record["split"] == "diagnostic":
            continue  # Deliberately shared diagnostic, never added to train.
        if (
            record["distances_mm"] != expected
            or tuple(Path(p).name for p in record["images"]) != IMAGE_FILENAMES
        ):
            raise ValueError("Invalid plane order")
        for relative in record["images"]:
            path = (source.parent / relative).resolve()
            if (
                not path.is_relative_to(source.parent.resolve())
                or Path(relative).parts[0] != record["split"]
            ):
                raise ValueError("Invalid image split/path")
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "L" or image.size != (N, N):
                    raise ValueError("Invalid target format")
                digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in existing or digest in incoming:
                raise ValueError(f"Duplicate target: {relative}")
            incoming.add(digest)
    shutil.copytree(base.parent, out)
    shutil.copytree(source.parent, out / "soft600")
    records = [r for r in soft["samples"] if r["split"] == "train"]
    for record in records:
        added = deepcopy(record)
        sid = f"soft600_{Path(record['images'][0]).parent.name}"
        destination = out / "train" / sid
        destination.mkdir(exist_ok=False)
        added["sample_id"] = f"train_{sid}"
        added["images"] = []
        for relative in record["images"]:
            name = Path(relative).name
            shutil.copy2(source.parent / relative, destination / name)
            added["images"].append(f"train/{sid}/{name}")
        added["params"]["source_sample_id"] = record["sample_id"]
        added["params"]["preserve_plane_order"] = True
        if "reference_phase" in added["params"]:
            added["params"]["reference_phase"] = (
                f"soft600/{added['params']['reference_phase']}"
            )
        original["samples"].append(added)
    report = {
        "counts": dict(Counter(r["split"] for r in original["samples"])),
        "added_train": len(records),
        "added_groups": dict(Counter(r["params"]["kind"] for r in records)),
        "base_held_out_preserved": True,
        "new_held_out": "soft600/manifest.json",
        "new_validation": sum(r["split"] == "validation" for r in soft["samples"]),
        "new_test": sum(r["split"] == "test" for r in soft["samples"]),
        "target_pixel_duplicates": 0,
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    original["dataset_version"] += f"+soft600-train{len(records)}"
    original["soft600_import"] = report
    (out / "manifest.json").write_text(
        json.dumps(original, indent=2, ensure_ascii=False) + "\n"
    )
    (out / "soft600_import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    readme = out / "README.md"
    previous = readme.read_text() if readme.exists() else ""
    readme.write_text(
        f"# Объединённый набор: {report['counts']}\n\n"
        "Soft600: добавлены только 600 TRAIN. Новые validation/test — soft600/manifest.json.\n"
        "Контрольные фазы не входят в изображения-входы. Отчёт: soft600_import_report.json.\n\n"
        "## Описание исходного набора до добавления Soft600\n\n" + previous
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("src/dataset/WhiteCollars_Soft600_dataset.zip"),
    )
    parser.add_argument(
        "--extract-to", type=Path, default=Path("src/dataset/data_soft600")
    )
    parser.add_argument("--merge-base", type=Path)
    parser.add_argument("--out", type=Path, default=Path("build/data_train5208"))
    args = parser.parse_args()
    print("Verified PNGs:", extract(args.archive, args.extract_to), flush=True)
    if args.merge_base:
        print(
            json.dumps(
                merge(args.merge_base, args.extract_to / "manifest.json", args.out),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
