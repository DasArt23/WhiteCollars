from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .config import IMAGE_FILENAMES
from .io import load_four_targets


class FourImageDataset(Dataset):
    """
    Поддерживает:
    1) Путь к manifest.json (или папке с manifest.json) + фильтр по split
    2) Папку сплита data/<split>/<id>/01.png..04.png (или 1.png..4.png)
    """

    def __init__(
        self,
        root: str | Path,
        filenames: Sequence[str] = IMAGE_FILENAMES,
        split: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.filenames = tuple(filenames)
        self.items: list[tuple[ list[Path], str ]] = []

        manifest_file = (
            self.root if self.root.is_file() and self.root.name.endswith(".json")
            else (self.root / "manifest.json" if (self.root / "manifest.json").is_file() else None)
        )

        if manifest_file is not None:
            raw = json.loads(manifest_file.read_text(encoding="utf-8"))
            base = manifest_file.parent
            for rec in raw.get("samples", []):
                if split is not None and rec.get("split") != split:
                    continue
                paths = [base / rel for rel in rec["images"]]
                self.items.append((paths, rec["sample_id"]))
        else:
            if not self.root.is_dir():
                raise FileNotFoundError(f"Папка датасета не найдена: {self.root}")
            sample_dirs = sorted(
                [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")],
                key=lambda d: d.name,
            )
            for d in sample_dirs:
                if (d / "01.png").is_file():
                    fnames = ("01.png", "02.png", "03.png", "04.png")
                else:
                    fnames = ("1.png", "2.png", "3.png", "4.png")
                paths = [d / f for f in fnames]
                # Поддержка обоих вариантов ID: '000001' или 'validation_000001'
                sid = f"{self.root.name}_{d.name}" if self.root.name in ("train", "validation", "test", "diagnostic") else d.name
                self.items.append((paths, sid))

        if not self.items:
            raise ValueError(f"Не найдено примеров в {self.root} (split={split})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        paths, sample_id = self.items[idx]
        arr = load_four_targets(paths)
        return torch.from_numpy(arr), sample_id