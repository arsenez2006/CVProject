"""Thread-safe Mask R-CNN inference service.

Модель загружается один раз, шарится между всеми WS-клиентами.
Forward сериализуется через Lock — torch-модели не потокобезопасны,
для 1-3 одновременных зрителей на демо этого достаточно.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from cvutils import draw_instances, load_model, put_label


class InferenceService:
    def __init__(
        self,
        weights: str,
        num_classes: int = 2,
        device: str = "auto",
        score_thr: float = 0.5,
        mask_thr: float = 0.5,
        draw_boxes: bool = True,
        jpeg_quality: int = 80,
    ) -> None:
        self.weights = weights
        self.num_classes = num_classes
        self.device = device
        self.score_thr = score_thr
        self.mask_thr = mask_thr
        self.draw_boxes = draw_boxes
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._predictor = None
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    def load(self) -> None:
        print(f"[infer] loading weights: {self.weights}")
        self._predictor = load_model(self.weights, self.num_classes, self.device)
        print("[infer] model ready")

    def infer(self, bgr_frame: np.ndarray) -> tuple[bytes, dict]:
        """Прогон модели на одном BGR-кадре. Возвращает (annotated_jpeg_bytes, meta)."""
        assert self._predictor is not None, "call load() first"

        t0 = time.perf_counter()
        # Весь forward + постпроцесс — под локом, чтобы не гонять torch из двух потоков.
        with self._lock:
            det = self._predictor(bgr_frame)
            d = det.filter(self.score_thr)
            masks = d.masks(self.mask_thr)
            scores = [float(d.scores[k]) for k in range(len(d))]
            boxes = [[float(v) for v in d.boxes[k]] for k in range(len(d))]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        vis = draw_instances(
            bgr_frame,
            masks,
            [b for b in boxes] if self.draw_boxes else None,
            scores if self.draw_boxes else None,
        )
        if infer_ms > 0:
            put_label(vis, f"{len(scores)} ped | {1000.0/infer_ms:.1f} FPS | {infer_ms:.0f} ms", (4, 22))

        ok, buf = cv2.imencode(".jpg", vis, self._encode_params)
        if not ok:
            raise RuntimeError("JPEG encode failed")

        h, w = vis.shape[:2]
        meta = {
            "type": "frame",
            "infer_ms": round(infer_ms, 1),
            "fps": round(1000.0 / infer_ms, 2) if infer_ms > 0 else 0.0,
            "width": w,
            "height": h,
            "detections": [
                {"score": s, "box_xyxy": b} for s, b in zip(scores, boxes)
            ],
        }
        return buf.tobytes(), meta
