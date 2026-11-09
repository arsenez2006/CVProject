"""Thread-safe shared state between the inference thread and WebSocket clients."""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class FramePayload:
    seq: int
    jpeg: bytes
    width: int
    height: int
    detections: list[dict]
    fps: float
    infer_ms: float


class FrameBus:
    """Holds only the latest annotated frame. Slow consumers drop old frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: FramePayload | None = None
        self._seq = 0

    def publish(
        self,
        jpeg: bytes,
        width: int,
        height: int,
        detections: list[dict],
        fps: float,
        infer_ms: float,
    ) -> FramePayload:
        with self._lock:
            self._seq += 1
            self._latest = FramePayload(
                seq=self._seq,
                jpeg=jpeg,
                width=width,
                height=height,
                detections=detections,
                fps=fps,
                infer_ms=infer_ms,
            )
            return self._latest

    def snapshot(self) -> FramePayload | None:
        with self._lock:
            return self._latest
