from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .config import IMAGE_FILENAMES
from .io import load_four_targets


class FourImageDataset(Dataset):
    """
    Датасет четвёрок целевых картинок: <root>/<id>/1.png ... 4.png.
    Возвращает (tensor [4, N, N] float32 в [0, 1], sample_id: str).
    """

    def __init__(
        self,
        root: str | Path,
        filenames: Sequence[str] = IMAGE_FILENAMES,
    ) -> None:
        self.root = Path(root)
        self.filenames = tuple(filenames)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Папка датасета не найдена: {self.root}")

        self.sample_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.name,
        )
        if not self.sample_dirs:
            raise ValueError(f"В папке {self.root} не найдено подпапок с примерами")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        sample_dir = self.sample_dirs[idx]
        paths = [sample_dir / fname for fname in self.filenames]
        arr = load_four_targets(paths)
        return torch.from_numpy(arr), sample_dir.name