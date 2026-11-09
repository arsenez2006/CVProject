"""Webcam capture + Mask R-CNN inference running off the asyncio event loop."""
from __future__ import annotations

import threading
import time

import cv2

from cvutils import draw_instances, load_model, put_label

from .state import FrameBus


class WebcamCapture:
    """Continuously reads frames from the camera, exposing only the latest one.

    Always gives the inference loop the freshest frame — critical for low
    latency when the model is slower than the camera FPS.
    """

    def __init__(self, camera_id: int, width: int = 0, height: int = 0) -> None:
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id!r}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Some backends honour this, some don't — the dedicated thread makes it
        # mostly irrelevant, but it doesn't hurt.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._latest = None
        self._version = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="webcam")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()

    def get_latest(self):
        with self._lock:
            return self._version, self._latest

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = frame
                self._version += 1


class InferencePipeline:
    def __init__(
        self,
        bus: FrameBus,
        camera_id: int,
        weights: str,
        num_classes: int = 2,
        device: str = "auto",
        score_thr: float = 0.5,
        mask_thr: float = 0.5,
        draw_boxes: bool = True,
        jpeg_quality: int = 80,
        width: int = 0,
        height: int = 0,
    ) -> None:
        self.bus = bus
        self.camera_id = camera_id
        self.weights = weights
        self.num_classes = num_classes
        self.device = device
        self.score_thr = score_thr
        self.mask_thr = mask_thr
        self.draw_boxes = draw_boxes
        self.jpeg_quality = jpeg_quality
        self.width = width
        self.height = height
        self._capture: WebcamCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="inference")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._capture:
            self._capture.stop()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        print(f"[pipeline] loading weights: {self.weights}")
        predictor = load_model(self.weights, self.num_classes, self.device)
        print("[pipeline] model loaded, opening camera")
        self._capture = WebcamCapture(self.camera_id, self.width, self.height)
        self._capture.start()

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        fps_ema = 0.0
        last_version = -1

        try:
            while not self._stop.is_set():
                version, frame = self._capture.get_latest()
                if frame is None or version == last_version:
                    time.sleep(0.005)
                    continue
                last_version = version

                t0 = time.perf_counter()
                det = predictor(frame)
                d = det.filter(self.score_thr)
                masks = d.masks(self.mask_thr)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                vis = draw_instances(
                    frame,
                    masks,
                    d.boxes if self.draw_boxes else None,
                    d.scores if self.draw_boxes else None,
                )

                if infer_ms > 0:
                    inst_fps = 1000.0 / infer_ms
                    fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps

                put_label(
                    vis,
                    f"{len(d)} ped | {fps_ema:.1f} FPS | {infer_ms:.0f} ms",
                    (4, 22),
                )

                ok, buf = cv2.imencode(".jpg", vis, encode_params)
                if not ok:
                    continue

                detections = [
                    {
                        "score": float(d.scores[k]),
                        "box_xyxy": [float(v) for v in d.boxes[k]],
                    }
                    for k in range(len(d))
                ]
                h, w = vis.shape[:2]
                self.bus.publish(
                    jpeg=buf.tobytes(),
                    width=w,
                    height=h,
                    detections=detections,
                    fps=fps_ema,
                    infer_ms=infer_ms,
                )
        finally:
            if self._capture:
                self._capture.stop()
            print("[pipeline] stopped")
