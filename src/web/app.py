"""FastAPI backend: фронт шлёт кадры, бэкенд возвращает разметку."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .inference import InferenceService

log = logging.getLogger("web")


def _decode_jpeg(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def create_app(service: InferenceService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Прогрев модели при старте — не блокируя event loop.
        await asyncio.to_thread(service.load)
        yield

    app = FastAPI(title="Mask R-CNN live demo", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.websocket("/ws/infer")
    async def ws_infer(ws: WebSocket):
        await ws.accept()
        await ws.send_text(json.dumps({"type": "hello", "protocol": 2}))

        frames = 0
        t_start = asyncio.get_running_loop().time()
        try:
            while True:
                # Backpressure естественный: клиент не может прислать следующий
                # кадр, пока не получил ответ на предыдущий.
                data = await ws.receive_bytes()
                frame = _decode_jpeg(data)
                if frame is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "bad jpeg"}))
                    continue

                # Инференс — в отдельном потоке, чтобы не блокировать loop.
                jpeg, meta = await asyncio.to_thread(service.infer, frame)

                await ws.send_text(json.dumps(meta))
                await ws.send_bytes(jpeg)
                frames += 1
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.warning("ws error: %s", exc)
        finally:
            elapsed = asyncio.get_running_loop().time() - t_start
            if frames:
                log.info("client done: %d frames in %.1fs (%.1f fps)",
                         frames, elapsed, frames / max(elapsed, 1e-6))

    static_index = Path(__file__).parent / "static" / "index.html"
    if static_index.exists():
        @app.get("/")
        async def index():
            return FileResponse(static_index)

    return app
