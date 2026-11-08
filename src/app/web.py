"""Local web adapter for the same prediction service used by the CLI."""

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from statistics import fmean
from threading import Lock

import numpy as np
import torch
from PIL import Image

from doe_eval.config import DISTANCES, NUM_PLANES, N
from doe_eval.dataset import sample_image_paths
from model.inference import DEFAULT_WEIGHTS, PredictionService, load_predictor

STATIC = Path(__file__).resolve().parent / "static"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_REQUEST_BYTES = 512 * 1024


def image_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def encode_png(pixels: torch.Tensor) -> str:
    stream = BytesIO()
    Image.fromarray(pixels.cpu().numpy()).save(stream, format="PNG")
    return image_url(stream.getvalue())


def decode_targets(payload: object) -> torch.Tensor:
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list) or len(images) != NUM_PLANES:
        raise ValueError("Загрузи ровно четыре PNG.")
    planes = []
    for value, distance in zip(images, DISTANCES, strict=True):
        try:
            if not isinstance(value, str):
                raise TypeError("Expected base64 text")
            data = base64.b64decode(value, validate=True)
            with Image.open(BytesIO(data), formats=["PNG"]) as image:
                if image.mode != "L" or image.size != (N, N):
                    raise ValueError("Unexpected image size or mode")
                planes.append(torch.from_numpy(np.array(image, dtype=np.float32)) / 255)
        except (OSError, TypeError, ValueError, Image.DecompressionBombError) as exc:
            distance_label = f"{distance * 1000:.1f}".replace(".", ",")
            raise ValueError(
                f"{distance_label} мм: нужен одноканальный PNG L8 128×128."
            ) from exc
    return torch.stack(planes)


class WebServer(ThreadingHTTPServer):
    def __init__(self, address, service: PredictionService, example_dir: Path) -> None:
        super().__init__(address, WebHandler)
        self.service = service
        self.example_dir = example_dir
        self.inference_lock = Lock()


class WebHandler(BaseHTTPRequestHandler):
    server: WebServer

    def send_body(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_body(status, "application/json; charset=utf-8", body)

    def do_GET(self) -> None:
        if self.path in ASSETS:
            name, content_type = ASSETS[self.path]
            self.send_body(HTTPStatus.OK, content_type, (STATIC / name).read_bytes())
        elif self.path == "/api/example":
            try:
                paths = sample_image_paths(self.server.example_dir)
                images = [image_url(path.read_bytes()) for path in paths]
            except (OSError, ValueError):
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Пример не найден."})
                return
            self.send_json(HTTPStatus.OK, {"images": images})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Ожидался JSON."}
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("Общий размер четырёх PNG должен быть до 380 КБ.")
            targets = decode_targets(json.loads(self.rfile.read(length)))
        except (json.JSONDecodeError, UnicodeError, RecursionError):
            self.send_json(
                HTTPStatus.BAD_REQUEST, {"error": "Не удалось прочитать запрос."}
            )
            return
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if not self.server.inference_lock.acquire(blocking=False):
            self.send_json(
                HTTPStatus.CONFLICT, {"error": "Уже идёт расчёт. Повтори чуть позже."}
            )
            return
        try:
            prediction = self.server.service.predict(targets)
        except (RuntimeError, ValueError) as exc:
            self.log_error("Prediction failed: %s", exc)
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Не удалось рассчитать ДОЭ."},
            )
            return
        finally:
            self.server.inference_lock.release()

        planes = [
            {
                "distance_mm": round(distance * 1000, 1),
                "ssim": ssim,
                "pearson": pearson,
                "image": encode_png((image.clamp(0, 1) * 255).round().byte()),
            }
            for distance, image, (ssim, pearson) in zip(
                DISTANCES,
                prediction.reconstructions,
                prediction.scores.tolist(),
                strict=True,
            )
        ]
        self.send_json(
            HTTPStatus.OK,
            {
                "doe": encode_png(prediction.mask),
                "planes": planes,
                "mean": {
                    key: fmean(plane[key] for plane in planes)
                    for key in ("ssim", "pearson")
                },
                "synthesis_ms": round(prediction.synthesis_ms, 2),
            },
        )


def serve(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="project serve", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--optical-steps", type=int)
    parser.add_argument(
        "--example", type=Path, default=Path("src/dataset/example/test/000001")
    )
    args = parser.parse_args(argv)
    try:
        model, device = load_predictor(args.weights, args.device, args.optical_steps)
        service = PredictionService(model, device)
        server = WebServer((args.host, args.port), service, args.example)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Не удалось запустить WhiteCollars: {exc}\n")
    with server:
        print(f"WhiteCollars | {args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
