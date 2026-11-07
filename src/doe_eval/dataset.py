from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import ALTERNATE_IMAGE_FILENAMES, DISTANCES, IMAGE_FILENAMES, NUM_PLANES
from .io import load_four_targets


def sample_image_paths(directory: str | Path, filenames: Sequence[str] | None = None) -> list[Path]:
    """Порядок файлов всегда соответствует DISTANCES; явные имена имеют приоритет."""
    directory = Path(directory)
    variants = (tuple(filenames),) if filenames is not None else (IMAGE_FILENAMES, ALTERNATE_IMAGE_FILENAMES)
    for names in variants:
        if len(names) != NUM_PLANES:
            raise ValueError(f"Ожидалось {NUM_PLANES} имён файлов")
        paths = [directory / name for name in names]
        if all(path.is_file() for path in paths):
            return paths
    raise FileNotFoundError(f"В {directory} нет полной четвёрки: {variants}")


class FourImageDataset(Dataset):
    """Манифест + split или папка <split>/<id> с 01..04.png / 1..4.png.

    При наличии родительского манифеста использует его ID и порядок каналов.
    Без манифеста ID равен имени папки примера, как в PROMPT.md.
    """

    def __init__(
        self,
        root: str | Path,
        filenames: Sequence[str] | None = None,
        split: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.items: list[tuple[list[Path], str]] = []
        manifest_file = None
        if self.root.is_file():
            manifest_file = self.root
        elif (self.root / "manifest.json").is_file():
            manifest_file = self.root / "manifest.json"
        elif (self.root.parent / "manifest.json").is_file():
            manifest_file = self.root.parent / "manifest.json"
            split = split or self.root.name

        if manifest_file is not None:
            raw = json.loads(manifest_file.read_text(encoding="utf-8"))
            expected_z = [round(z * 1e3, 1) for z in DISTANCES]
            if raw.get("format") != "doe-manifest/1" or raw.get("distances_mm") != expected_z:
                raise ValueError(f"{manifest_file}: неверный формат или порядок расстояний")
            base = manifest_file.parent.resolve()
            seen_ids = set()
            for rec in raw.get("samples", []):
                if split is not None and rec.get("split") != split:
                    continue
                sid = rec["sample_id"]
                if sid in seen_ids:
                    raise ValueError(f"Дублирующийся sample_id: {sid}")
                seen_ids.add(sid)
                if rec.get("distances_mm") != expected_z or len(rec.get("images", [])) != NUM_PLANES:
                    raise ValueError(f"{sid}: неверный порядок расстояний или число изображений")
                paths = [(base / rel).resolve() for rel in rec["images"]]
                if any(not p.is_relative_to(base) for p in paths):
                    raise ValueError(f"{sid}: путь изображения выходит за папку датасета")
                if tuple(p.name for p in paths) not in (IMAGE_FILENAMES, ALTERNATE_IMAGE_FILENAMES):
                    raise ValueError(f"{sid}: неверный порядок файлов плоскостей")
                if any(p.parent != paths[0].parent or p.relative_to(base).parts[0] != rec["split"] for p in paths):
                    raise ValueError(f"{sid}: изображения относятся к разным примерам или split")
                self.items.append((paths, sid))
        else:
            directory = self.root / split if split is not None else self.root
            if not directory.is_dir():
                raise FileNotFoundError(f"Папка датасета не найдена: {directory}")
            for d in sorted(directory.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    self.items.append((sample_image_paths(d, filenames), d.name))

        if not self.items:
            raise ValueError(f"Не найдено примеров в {self.root} (split={split})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        paths, sample_id = self.items[idx]
        return torch.from_numpy(load_four_targets(paths)), sample_id
