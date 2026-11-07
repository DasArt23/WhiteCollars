from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from doe_eval import (
    fresnel_reconstruct,
    normalize_reconstructions,
    pearson_torch,
    ssim_global_torch,
    ste_quantize,
)
from doe_eval.config import (
    DEFAULT_COMPARE,
    DEFAULT_MANIFEST,
    DEFAULT_NORM,
    DISTANCES,
    NUM_PLANES,
)
from doe_eval.io import load_target_png, save_doe_png

EXPECTED_FORMAT = "doe-manifest/1"
EXPECTED_DISTANCES_MM = [
    round(z * 1e3, 1) for z in DISTANCES
]  # [24.1, 25.3, 26.5, 27.7]
VALID_SPLITS = ("train", "validation", "test", "diagnostic")


@dataclass
class ManifestReport:
    counts: dict[str, int]
    total_images: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class ManifestSample:
    sample_id: str
    split: str
    images: tuple[str, str, str, str]
    distances_mm: tuple[float, float, float, float]
    seed: int
    generator_version: str
    params: dict[str, Any]


@dataclass
class ManifestData:
    manifest_path: Path
    root_dir: Path
    dataset_name: str
    dataset_version: str
    master_seed: int
    samples: list[ManifestSample]
    report: ManifestReport

    def by_split(self, split: str) -> list[ManifestSample]:
        if split not in VALID_SPLITS:
            raise ValueError(f"Неизвестный split '{split}'. Допустимые: {VALID_SPLITS}")
        return [s for s in self.samples if s.split == split]


def _validate_and_load_png(path: Path) -> np.ndarray:
    """Общий строгий контракт ТЗ: PNG L, N×N, float32 в [0,1]."""
    return load_target_png(path)


def load_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    verify_images: bool = True,
) -> ManifestData:
    """
    Загружает и проверяет data/manifest.json формата doe-manifest/1.
    Совместим с вызовом:
        d = load_manifest('data/manifest.json')
        print(d.report.counts, d.report.warnings[:5])
    """
    m_path = Path(manifest_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Манифест не найден: {m_path}")

    raw = json.loads(m_path.read_text(encoding="utf-8"))
    if raw.get("format") != EXPECTED_FORMAT:
        raise ValueError(
            f"Неподдерживаемый формат манифеста: {raw.get('format')} (ожидался {EXPECTED_FORMAT})"
        )

    if list(raw.get("distances_mm", [])) != EXPECTED_DISTANCES_MM:
        raise ValueError(
            f"Несовпадение плоскостей z: {raw.get('distances_mm')} != {EXPECTED_DISTANCES_MM}"
        )

    root_dir = m_path.parent
    warnings: list[str] = []
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    samples: list[ManifestSample] = []

    for rec in raw.get("samples", []):
        sid = rec["sample_id"]
        split = rec["split"]
        if sid in seen_ids:
            raise ValueError(f"Дублирующийся sample_id в манифесте: {sid}")
        seen_ids.add(sid)

        if split not in VALID_SPLITS:
            raise ValueError(f"Неизвестный split '{split}' у примера {sid}")

        imgs = rec.get("images", [])
        if len(imgs) != 4:
            raise ValueError(f"{sid}: ожидалось 4 изображения, получено {len(imgs)}")

        if list(rec.get("distances_mm", [])) != EXPECTED_DISTANCES_MM:
            raise ValueError(
                f"{sid}: расстояния {rec.get('distances_mm')} != {EXPECTED_DISTANCES_MM}"
            )

        for rel in imgs:
            if str(rel).startswith("previews/"):
                raise ValueError(
                    f"{sid}: файлы из previews/ не должны входить в samples ({rel})"
                )
            path = (root_dir / rel).resolve()
            if (
                not path.is_relative_to(root_dir.resolve())
                or Path(rel).parts[0] != split
            ):
                raise ValueError(
                    f"{sid}: путь выходит за датасет или относится к другому split: {rel}"
                )
        if [Path(rel).name for rel in imgs] not in (
            [f"{i}.png" for i in range(1, NUM_PLANES + 1)],
            [f"{i:02d}.png" for i in range(1, NUM_PLANES + 1)],
        ):
            raise ValueError(f"{sid}: неверный порядок файлов плоскостей")
        for rel in imgs:
            if verify_images:
                _validate_and_load_png(root_dir / rel)

        counts[split] += 1
        samples.append(
            ManifestSample(
                sample_id=sid,
                split=split,
                images=(imgs[0], imgs[1], imgs[2], imgs[3]),
                distances_mm=(
                    float(rec["distances_mm"][0]),
                    float(rec["distances_mm"][1]),
                    float(rec["distances_mm"][2]),
                    float(rec["distances_mm"][3]),
                ),
                seed=int(rec.get("seed", 0)),
                generator_version=str(rec.get("generator_version", "")),
                params=dict(rec.get("params", {})),
            )
        )

    ordered_counts = {
        k: counts[k]
        for k in ("diagnostic", "train", "validation", "test")
        if counts[k] > 0
    }
    report = ManifestReport(
        counts=ordered_counts,
        total_images=len(samples) * 4,
        warnings=warnings,
    )

    return ManifestData(
        manifest_path=m_path,
        root_dir=root_dir,
        dataset_name=str(raw.get("dataset_name", "procedural_mixed_targets")),
        dataset_version=str(raw.get("dataset_version", "")),
        master_seed=int(raw.get("master_seed", 0)),
        samples=samples,
        report=report,
    )


class ManifestQuartetDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | ManifestData = DEFAULT_MANIFEST,
        split: str = "train",
        smoke_limit: int | None = None,
        augment_channel_shuffle: bool = False,
        augment_dihedral: bool = False,
        seed: int = 3701,
        preload: bool = True,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            self.manifest_data = load_manifest(manifest, verify_images=False)
        else:
            self.manifest_data = manifest

        if split != "train" and (augment_channel_shuffle or augment_dihedral):
            raise ValueError("Аугментации разрешены ТОЛЬКО для split='train'!")

        self.split = split
        self.root_dir = self.manifest_data.root_dir
        items = self.manifest_data.by_split(split)
        if smoke_limit is not None:
            items = items[:smoke_limit]
        if not items:
            raise ValueError(f"Пустой список примеров для split='{split}'")

        self.samples = items
        self.augment_channel_shuffle = augment_channel_shuffle
        self.augment_dihedral = augment_dihedral
        self.rng = np.random.default_rng(seed)

        # Float32: около 256 КиБ на четвёрку; 4608 примеров занимают ~1,2 ГБ.
        self._cache: list[np.ndarray] | None = None
        if preload:
            self._cache = [
                np.stack(
                    [_validate_and_load_png(self.root_dir / rel) for rel in rec.images],
                    axis=0,
                )
                for rec in self.samples
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        worker = torch.utils.data.get_worker_info()
        if worker is not None and getattr(self, "_worker_seed", None) != worker.seed:
            self.rng = np.random.default_rng(worker.seed)
            self._worker_seed = worker.seed
        record = self.samples[idx]
        if self._cache is not None:
            arr = self._cache[idx].copy()
        else:
            arr = np.stack(
                [_validate_and_load_png(self.root_dir / rel) for rel in record.images],
                axis=0,
            )

        if self.split == "train":
            fixed_planes = record.params.get("preserve_plane_order", False) or (
                record.params.get("exchange", {}).get("kind") == "A"
            )
            if self.augment_channel_shuffle and not fixed_planes:
                arr = arr[self.rng.permutation(4)]
            if self.augment_dihedral:
                k = int(self.rng.integers(0, 4))
                if k:
                    arr = np.rot90(arr, k=k, axes=(1, 2))
                if self.rng.random() < 0.5:
                    arr = np.flip(arr, axis=2)
                arr = np.ascontiguousarray(arr)

        return torch.from_numpy(arr), record.sample_id


def build_dataloaders(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    batch_size: int = 16,
    num_workers: int = 4,
    mode: str = "v1",
    augment_train: bool = False,
    pin_memory: bool | None = None,
    seed: int = 3701,
) -> dict[str, DataLoader]:
    """
    Создаёт загрузчики для обучения и валидации:
    - mode="train_smoke": до 8 train / 2 validation / 1 diagnostic / 2 test
    - mode="v1": полные сплиты из manifest
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if mode not in ("v1", "train_smoke") or batch_size < 1 or num_workers < 0:
        raise ValueError("mode: v1 | train_smoke; batch_size >= 1; num_workers >= 0")
    m_data = load_manifest(manifest_path, verify_images=False)
    smoke_limit = 8 if mode == "train_smoke" else None

    train_ds = ManifestQuartetDataset(
        m_data,
        split="train",
        smoke_limit=smoke_limit,
        augment_channel_shuffle=augment_train,
        augment_dihedral=False,
        seed=seed,
    )
    val_ds = ManifestQuartetDataset(
        m_data, split="validation", smoke_limit=2 if mode == "train_smoke" else None
    )
    diag_ds = ManifestQuartetDataset(m_data, split="diagnostic")
    test_ds = ManifestQuartetDataset(
        m_data, split="test", smoke_limit=2 if mode == "train_smoke" else None
    )

    return {
        "train": DataLoader(
            train_ds,
            batch_size=min(batch_size, len(train_ds)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "validation": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "diagnostic": DataLoader(
            diag_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def forward_physics_loss(
    pred_phase: torch.Tensor,
    targets: torch.Tensor,
    w_pearson: float = 0.5,
    w_ssim: float = 0.5,
    use_ste_quantize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Соединяет выход модели pred_phase [B, 128, 128] в [-pi, pi] с мишенями [B, 4, 128, 128]
    через физический симулятор Френеля doe_eval.
    Возвращает: (loss, recon_01 [B, 4, 128, 128], metrics_dict).
    """
    with torch.amp.autocast("cuda", enabled=False):
        phase_f32 = pred_phase.float()
        targets_f32 = targets.float()

        phase_eff = ste_quantize(phase_f32) if use_ste_quantize else phase_f32
        recon_amp = fresnel_reconstruct(phase_eff)
        recon_01 = normalize_reconstructions(
            recon_amp, norm=DEFAULT_NORM, compare=DEFAULT_COMPARE
        )

        p_per_plane = pearson_torch(recon_01, targets_f32)  # [B, 4]
        s_per_plane = ssim_global_torch(recon_01, targets_f32)  # [B, 4]

        p_mean = p_per_plane.mean()
        s_mean = s_per_plane.mean()
        loss = w_pearson * (1.0 - p_mean) + w_ssim * (1.0 - s_mean)

    metrics = {
        "loss": loss.detach(),
        "pearson": p_mean.detach(),
        "ssim_global": s_mean.detach(),
    }
    return loss, recon_01.detach(), metrics


@torch.no_grad()
def export_split_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    out_doe_dir: str | Path,
    device: torch.device,
) -> list[Path]:
    """
    Прогоняет модель по сплиту и сохраняет квантованные 8-битные ДОЭ в <out_doe_dir>/<sample_id>.png
    (например, preds/validation_000001.png), готовые для doe-eval evaluate.
    """
    model.eval()
    out_dir = Path(out_doe_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for targets, sample_ids in loader:
        targets = targets.to(device)
        pred_phase = model(targets)
        if pred_phase.ndim == 4 and pred_phase.shape[1] == 1:
            pred_phase = pred_phase.squeeze(1)

        pred_np = pred_phase.detach().cpu().numpy()
        for phase_2d, sid in zip(pred_np, sample_ids):
            p = out_dir / f"{sid}.png"
            save_doe_png(phase_2d, p)
            saved_paths.append(p)

    return saved_paths
