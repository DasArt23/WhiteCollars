"""Terminal interface for the packaged RetrievalDOE model."""

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean

from torch.utils.data import DataLoader

from model.data_connector import ManifestQuartetDataset, export_split_predictions
from model.inference import DEFAULT_MANIFEST, DEFAULT_WEIGHTS, load_predictor
from model.prob import main as predict

from .web import serve


def export(argv):
    parser = argparse.ArgumentParser(
        prog="project export", description="Экспорт ДОЭ для одного сплита."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--out", type=Path, default=Path("preds/val"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optical-steps", type=int)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size должен быть положительным.")
    try:
        model, device = load_predictor(args.weights, args.device, args.optical_steps)
        dataset = ManifestQuartetDataset(args.manifest, split=args.split, preload=False)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    paths = export_split_predictions(
        model, DataLoader(dataset, batch_size=args.batch_size), args.out, device
    )
    print(f"Сохранено ДОЭ: {len(paths)} | {args.out.resolve()}")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--help"], ["-h"]):
        print(
            "WhiteCollars — синтез фазового ДОЭ\n"
            "  project serve            Веб-интерфейс\n"
            "  project predict --help   Один ДОЭ, четыре реконструкции и метрики\n"
            "  project export --help    Маски для validation или test\n"
            "  project                  Запуск на тестовом примере из комплекта"
        )
        return
    if args[:1] == ["serve"]:
        return serve(args[1:])
    if args[:1] == ["export"]:
        return export(args[1:])
    if args[:1] == ["predict"]:
        args = args[1:]

    out, _ = predict(args)
    report = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    print(f"{'z, мм':>6} {'SSIM (global)':>14} {'Pearson':>9}")
    for plane in report["planes"]:
        print(
            f"{plane['distance_mm']:>6.1f} {plane['ssim']:>14.4f} {plane['pearson']:>9.4f}"
        )
    mean_ssim = fmean(plane["ssim"] for plane in report["planes"])
    mean_pearson = fmean(plane["pearson"] for plane in report["planes"])
    print(f"{'Среднее':>6} {mean_ssim:>14.4f} {mean_pearson:>9.4f}")
    return out
