"""OpenCV inference for the Mask R-CNN pedestrian baseline.

Input can be a single image, a folder of images, a video file or a webcam index.
OpenCV is used for all I/O and visualisation; torch is only used for the forward pass.

Examples
--------
  python src/infer.py --input data/processed/val --output outputs/infer
  python src/infer.py --input some_image.png --save-json --prob-heatmap
  python src/infer.py --input street.mp4 --output outputs/street_pred.mp4
  python src/infer.py --input 0 --show                      # webcam, press q to quit
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from cvutils import (
  IMG_EXTS,
  draw_instances,
  load_model,
  overlay_heatmap,
  put_label,
  read_image,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--weights", default="artifacts/maskrcnn_baseline.pt")
  p.add_argument("--input", required=True, help="image | folder | video | webcam index")
  p.add_argument("--output", default="outputs/infer", help="folder (images) or file (video)")
  p.add_argument("--score-thr", type=float, default=0.5)
  p.add_argument("--mask-thr", type=float, default=0.5)
  p.add_argument("--num-classes", type=int, default=2)
  p.add_argument("--device", default="auto")
  p.add_argument("--no-boxes", action="store_true", help="draw masks only")
  p.add_argument("--prob-heatmap", action="store_true",
                 help="also save a heatmap of the maximum mask probability per pixel")
  p.add_argument("--save-json", action="store_true",
                 help="write predictions.json (COCO-style RLE masks, needs pycocotools)")
  p.add_argument("--show", action="store_true", help="live preview window (q / Esc to quit)")
  return p.parse_args(argv)


def visualise(frame, det, args):
  d = det.filter(args.score_thr)
  masks = d.masks(args.mask_thr)
  vis = draw_instances(
    frame, masks, None if args.no_boxes else d.boxes, None if args.no_boxes else d.scores
  )
  return vis, d, masks


def encode_detections(d, masks):
  from pycocotools import mask as mu

  out = []
  for k in range(len(d)):
    rle = mu.encode(np.asfortranarray(masks[k].astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    out.append(
      {"score": float(d.scores[k]), "box_xyxy": [float(v) for v in d.boxes[k]], "segmentation": rle}
    )
  return out


def run_images(paths, predictor, args):
  out_dir = Path(args.output)
  out_dir.mkdir(parents=True, exist_ok=True)
  records, times = [], []
  for path in paths:
    img = read_image(path)
    t0 = time.perf_counter()
    det = predictor(img)
    times.append(time.perf_counter() - t0)
    vis, d, masks = visualise(img, det, args)
    put_label(vis, f"{len(d)} pedestrian(s)  {1000 * times[-1]:.0f} ms", (4, 18))
    cv2.imwrite(str(out_dir / f"{path.stem}_pred.png"), vis)
    if args.prob_heatmap:
      prob = det.probs.max(0) if len(det) else np.zeros(img.shape[:2], np.float32)
      cv2.imwrite(str(out_dir / f"{path.stem}_prob.png"), overlay_heatmap(img, prob, 0.85, 1.0))
    if args.save_json:
      records.append({"image": path.name, "detections": encode_detections(d, masks)})
    print(f"{path.name}: {len(d)} detection(s) >= {args.score_thr}  ({1000 * times[-1]:.0f} ms)")
    if args.show:
      cv2.imshow("infer", vis)
      if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
        break
  if args.save_json:
    (out_dir / "predictions.json").write_text(json.dumps(records), encoding="utf-8")
  if times:
    print(f"{len(times)} image(s), mean {1000 * np.mean(times):.0f} ms/img -> {out_dir}")


def run_video(source, predictor, args):
  cap = cv2.VideoCapture(source)
  if not cap.isOpened():
    raise RuntimeError(f"Cannot open video source {source!r}")
  fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
  w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  writer = None
  if not (isinstance(source, int) and args.output == "outputs/infer"):  # webcam: only write if asked
    out = Path(args.output)
    if out.suffix.lower() not in VIDEO_EXTS:
      out = out / "pred.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"writing {out}")
  n, t_start = 0, time.perf_counter()
  while True:
    ok, frame = cap.read()
    if not ok:
      break
    t0 = time.perf_counter()
    vis, d, _ = visualise(frame, predictor(frame), args)
    dt = time.perf_counter() - t0
    put_label(vis, f"{len(d)} ped | {1.0 / max(dt, 1e-6):.1f} FPS", (4, 18))
    if writer is not None:
      writer.write(vis)
    n += 1
    if args.show:
      cv2.imshow("infer", vis)
      if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break
  cap.release()
  if writer is not None:
    writer.release()
  print(f"{n} frame(s), {n / max(time.perf_counter() - t_start, 1e-6):.1f} FPS average")


def main(argv=None):
  args = parse_args(argv)
  predictor = load_model(args.weights, args.num_classes, args.device)
  if args.input.isdigit():
    run_video(int(args.input), predictor, args)
  else:
    src = Path(args.input)
    if src.is_dir():
      paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS and not p.stem.endswith("_mask")
                     and p.parent.name.lower() not in {"pedmasks", "masks"})
      if not paths:
        raise FileNotFoundError(f"No images in {src}")
      run_images(paths, predictor, args)
    elif src.suffix.lower() in VIDEO_EXTS:
      run_video(str(src), predictor, args)
    elif src.suffix.lower() in IMG_EXTS:
      run_images([src], predictor, args)
    else:
      raise ValueError(f"Unsupported input: {src}")
  if args.show:
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()