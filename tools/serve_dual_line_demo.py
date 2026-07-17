#!/usr/bin/env python3
"""Serve a dependency-free, label-free single-image Dual-Line demo."""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.dual_line.runtime.single_image_inference import (
    SingleImageArtifacts,
    SingleImageInferenceEngine,
    SingleImageSettings,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "demo" / "dual_line_app"


def _data_url(image: Image.Image, *, format_name: str = "JPEG") -> str:
    buffer = io.BytesIO()
    save_image = image.convert("RGB")
    save_image.save(buffer, format=format_name, quality=90)
    mime = "image/jpeg" if format_name == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _visuals(image: Image.Image, bbox: list[float] | None) -> dict[str, str | None]:
    if bbox is None:
        return {"overlay": _data_url(image), "crop": None}
    width, height = image.size
    x0, y0, x1, y1 = bbox
    pixels = (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    line_width = max(3, round(min(width, height) / 120))
    draw.rectangle(pixels, outline=(146, 86, 178), width=line_width)
    crop = image.crop(pixels)
    return {"overlay": _data_url(overlay), "crop": _data_url(crop)}


def _handler(engine: SingleImageInferenceEngine, lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DualLineDemo/1.0"

        def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, data: dict[str, Any], status: int = 200) -> None:
            self._send(
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                self._json(
                    {
                        "ready": True,
                        "backbone": engine.settings.backbone,
                        "device": str(engine.device),
                        "parent_classes": list(engine.layout.parent_classes),
                        "fine_classes": list(engine.layout.fine_classes),
                        "label_free": True,
                    }
                )
                return
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            }
            if path not in files:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            filename, content_type = files[path]
            self._send((STATIC_DIR / filename).read_bytes(), content_type)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/predict":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 25 * 1024 * 1024:
                    raise ValueError("image request must be between 1 byte and 25 MB")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                data_url = str(request.get("image", ""))
                if "," not in data_url:
                    raise ValueError("missing image data")
                raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                filename = Path(str(request.get("filename") or "uploaded_image")).name
                start = time.perf_counter()
                with lock:
                    result = engine.predict(image, filename)
                result["runtime_ms"] = round((time.perf_counter() - start) * 1000, 1)
                result["visuals"] = _visuals(image, result["gate"]["chosen_bbox"])
                self._json(result)
            except Exception as exc:  # UI boundary: return a readable error packet.
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[HTTP] {self.address_string()} {format % args}", flush=True)

    return Handler


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no_browser", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone", default="resnet18")
    artifact_dir = "artifacts/demo_cls10"
    parser.add_argument("--train_cache", default=f"{artifact_dir}/backbone_spatial_cache_v232.npz")
    parser.add_argument("--normality_model", default=f"{artifact_dir}/habit_normality_v230.pkl")
    parser.add_argument("--projector_checkpoint", default=f"{artifact_dir}/trajectory_projector_v230.pt")
    parser.add_argument("--trajectory_npz", default=f"{artifact_dir}/learning_trajectory_v230.npz")
    parser.add_argument("--node_csv", default=f"{artifact_dir}/evidence_nodes.csv")
    parser.add_argument("--risk_model", default=f"{artifact_dir}/error_risk_ranker_v233.pkl")
    parser.add_argument("--correction_model", default=f"{artifact_dir}/multi_expert_correction_v234.pkl")
    args = parser.parse_args()

    artifacts = SingleImageArtifacts(
        train_cache=_path(args.train_cache),
        normality_model=_path(args.normality_model),
        projector_checkpoint=_path(args.projector_checkpoint),
        trajectory_npz=_path(args.trajectory_npz),
        node_csv=_path(args.node_csv),
        risk_model=_path(args.risk_model),
        correction_model=_path(args.correction_model),
    )
    missing = [str(path) for path in artifacts.__dict__.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing artifacts:\n" + "\n".join(missing))
    print("[STARTUP] loading Dual-Line artifacts...", flush=True)
    engine = SingleImageInferenceEngine(
        artifacts, SingleImageSettings(backbone=args.backbone, device=args.device)
    )
    lock = threading.Lock()
    server = ThreadingHTTPServer((args.host, args.port), _handler(engine, lock))
    url = f"http://{args.host}:{args.port}"
    print(f"[READY] {url}", flush=True)
    print("[MODE] raw image input, no truth labels", flush=True)
    if args.host in {"127.0.0.1", "localhost"}:
        print("[ACCESS] local computer only", flush=True)
    print("[STOP] press Ctrl+C", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
