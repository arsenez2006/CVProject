"""PYTHONPATH=src uv run python -m web --weights artifacts/maskrcnn_baseline.pt"""
from __future__ import annotations

import argparse
import logging

import uvicorn

from .app import create_app
from .inference import InferenceService


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="artifacts/maskrcnn_baseline.pt")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--mask-thr", type=float, default=0.5)
    p.add_argument("--no-boxes", action="store_true")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    service = InferenceService(
        weights=args.weights,
        num_classes=args.num_classes,
        device=args.device,
        score_thr=args.score_thr,
        mask_thr=args.mask_thr,
        draw_boxes=not args.no_boxes,
        jpeg_quality=args.jpeg_quality,
    )
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
