"""Shared helpers for the OpenCV inference / occlusion-analysis tools.

Conventions used everywhere in these tools
------------------------------------------
* Images are uint8 **BGR** arrays (OpenCV native). They are converted to RGB
  only right before being fed to the model.
* Instance masks are bool arrays of shape (N, H, W); boxes are xyxy in pixels.
* torch / torchvision are imported lazily so that everything except model
  loading can be used (and unit-tested) without them.
"""

from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours, one per instance
PALETTE = [
  (0, 114, 255), (255, 128, 0), (60, 200, 60), (220, 60, 220),
  (0, 200, 255), (255, 80, 80), (160, 160, 0), (128, 0, 255),
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def find_pairs(split_dir):
  """Return [(image_path, mask_path), ...] found anywhere below ``split_dir``.

  Works for both layouts PennFudanPed preprocessing might produce:
    * sub-folders  ``PNGImages/`` + ``PedMasks/`` (or ``images/`` + ``masks/``)
    * flat folder  ``FudanPed00001.png`` + ``FudanPed00001_mask.png``
  """
  split_dir = Path(split_dir)
  images, masks = {}, {}
  for p in sorted(split_dir.rglob("*")):
    if p.suffix.lower() not in IMG_EXTS:
      continue
    is_mask = p.stem.endswith("_mask") or p.parent.name.lower() in {"pedmasks", "masks"}
    key = p.stem[: -len("_mask")] if p.stem.endswith("_mask") else p.stem
    (masks if is_mask else images)[key] = p
  keys = sorted(set(images) & set(masks))
  if not keys:
    raise FileNotFoundError(f"No image/mask pairs found under {split_dir}")
  return [(images[k], masks[k]) for k in keys]


def read_image(path):
  img = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if img is None:
    raise FileNotFoundError(f"Could not read image: {path}")
  return img


def load_instance_masks(path):
  """PennFudan-style instance mask -> bool array (N, H, W).

  Background is 0 and every pedestrian has its own id (1..N). PIL is used on
  purpose: the masks are palette PNGs and ``cv2.imread`` would expand the
  palette to RGB and lose the indices.
  """
  from PIL import Image

  m = np.array(Image.open(path))
  if m.ndim == 3:  # RGB(A) fallback: one instance per unique non-black colour
    flat = m.reshape(-1, m.shape[-1])[:, :3]
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    inv = inv.reshape(m.shape[:2])
    out = [inv == i for i, c in enumerate(uniq) if c.any()]
  else:
    out = [m == v for v in np.unique(m) if v != 0]
  if not out:
    return np.zeros((0, *m.shape[:2]), dtype=bool)
  return np.stack(out)


def mask_to_box(mask):
  ys, xs = np.nonzero(mask)
  return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def masks_to_boxes(masks):
  boxes = [mask_to_box(m) for m in masks if m.any()]
  return np.stack(boxes) if boxes else np.zeros((0, 4), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Geometry / matching
# --------------------------------------------------------------------------- #
def box_iou_matrix(a, b):
  a = np.asarray(a, np.float32).reshape(-1, 4)
  b = np.asarray(b, np.float32).reshape(-1, 4)
  if len(a) == 0 or len(b) == 0:
    return np.zeros((len(a), len(b)), np.float32)
  ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
  iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
  ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
  iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
  inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
  area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
  area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
  return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-6)


def mask_overlap(a, b):
  """Pairwise intersection areas of two mask stacks.

  Returns (inter[Na, Nb], area_a[Na], area_b[Nb]); float32 is exact here
  because pixel counts stay far below 2**24.
  """
  af = a.reshape(len(a), -1).astype(np.float32)
  bf = b.reshape(len(b), -1).astype(np.float32)
  return af @ bf.T, af.sum(1), bf.sum(1)


def match_instances(iou, scores, thr=0.5):
  """COCO-style greedy matching. ``iou`` is (P, G).

  Predictions are visited in descending score order and each takes the best
  still-free ground truth with IoU >= thr.
  Returns (pairs[(p, g)], unmatched_pred[list], unmatched_gt[list]).
  """
  P, G = iou.shape
  taken = np.zeros(G, bool)
  pairs, used_pred = [], set()
  if G:
    for p in np.argsort(-np.asarray(scores)):
      cand = np.where(taken, -1.0, iou[p])
      g = int(cand.argmax())
      if cand[g] >= thr:
        taken[g] = True
        pairs.append((int(p), g))
        used_pred.add(int(p))
  return (
    pairs,
    [p for p in range(P) if p not in used_pred],
    [g for g in range(G) if not taken[g]],
  )


# --------------------------------------------------------------------------- #
# Model + inference
# --------------------------------------------------------------------------- #
@dataclass
class Detections:
  boxes: np.ndarray   # (N, 4) xyxy
  scores: np.ndarray  # (N,)
  probs: np.ndarray   # (N, H, W) float32 mask probabilities

  def __len__(self):
    return len(self.scores)

  def masks(self, thr=0.5):
    return self.probs >= thr

  def filter(self, score_thr):
    k = self.scores >= score_thr
    return Detections(self.boxes[k], self.scores[k], self.probs[k])

  @staticmethod
  def empty(h, w):
    return Detections(
      np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, h, w), np.float32)
    )


def resolve_device(name="auto"):
  import torch

  if name == "auto":
    name = "cuda" if torch.cuda.is_available() else "cpu"
  return torch.device(name)


def load_model(weights, num_classes=2, device="auto", box_score_thresh=0.05):
  """Rebuild the baseline Mask R-CNN (ResNet-50 FPN, 2 classes) and load weights.

  Accepts a plain state_dict, a dict wrapping one ("model"/"state_dict"/
  "model_state_dict") or a fully pickled nn.Module.
  """
  import torch
  from torchvision.models.detection import maskrcnn_resnet50_fpn
  from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
  from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

  device = resolve_device(device) if isinstance(device, str) else device
  model = maskrcnn_resnet50_fpn(
    weights=None, weights_backbone=None, box_score_thresh=box_score_thresh
  )
  in_box = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)
  in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
  model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)

  try:
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
  except Exception:  # e.g. a fully pickled module
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)

  if isinstance(ckpt, torch.nn.Module):
    model = ckpt
    model.roi_heads.score_thresh = box_score_thresh
  else:
    for key in ("model", "state_dict", "model_state_dict"):
      if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
        ckpt = ckpt[key]
        break
    try:
      model.load_state_dict(ckpt)
    except RuntimeError as e:
      raise RuntimeError(
        "Checkpoint does not match maskrcnn_resnet50_fpn with "
        f"{num_classes} classes. Check the model-building code in "
        "src/train_baseline.py (head sizes / --num-classes).\n" + str(e)
      ) from e
  return Predictor(model, device)


class Predictor:
  """Callable: BGR image -> Detections (all boxes >= model's box_score_thresh)."""

  def __init__(self, model, device):
    import torch

    self._torch = torch
    self.device = device
    self.model = model.to(device).eval()

  def __call__(self, image_bgr, score_thr=0.0):
    torch = self._torch
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div_(255.0).to(self.device)
    with torch.inference_mode():
      out = self.model([x])[0]
    keep = out["scores"] >= score_thr
    h, w = image_bgr.shape[:2]
    if int(keep.sum()) == 0:
      return Detections.empty(h, w)
    return Detections(
      out["boxes"][keep].cpu().numpy().astype(np.float32),
      out["scores"][keep].cpu().numpy().astype(np.float32),
      out["masks"][keep, 0].cpu().numpy().astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# Drawing (OpenCV)
# --------------------------------------------------------------------------- #
def put_label(img, text, org, color=(255, 255, 255), bg=(0, 0, 0), scale=0.5, thick=1):
  (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
  x, y = int(org[0]), max(int(org[1]), h + base)
  cv2.rectangle(img, (x, y - h - base), (x + w, y + base), bg, -1)
  cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_outlines(image, masks, color=(0, 255, 0), thickness=2):
  out = image.copy()
  for m in masks:
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, thickness)
  return out


def draw_instances(image, masks, boxes=None, scores=None, alpha=0.45, outline=2):
  """Translucent per-instance masks + contours (+ optional boxes/scores)."""
  out = image.copy()
  if len(masks):
    overlay = image.copy()
    for i, m in enumerate(masks):
      overlay[m] = PALETTE[i % len(PALETTE)]
    out = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    for i, m in enumerate(masks):
      cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cv2.drawContours(out, cnts, -1, PALETTE[i % len(PALETTE)], outline)
  if boxes is not None:
    for i, b in enumerate(boxes):
      x0, y0, x1, y1 = (int(round(v)) for v in b)
      col = PALETTE[i % len(PALETTE)]
      cv2.rectangle(out, (x0, y0), (x1, y1), col, 1)
      if scores is not None:
        put_label(out, f"{scores[i]:.2f}", (x0, y0 - 2), bg=col)
  return out


def colorize(arr, vmax=1.0, colormap=cv2.COLORMAP_INFERNO, scale=1):
  """float array -> BGR heatmap (optionally upscaled with cubic interpolation)."""
  u8 = np.clip(np.asarray(arr, np.float32) / max(vmax, 1e-8) * 255.0, 0, 255).astype(np.uint8)
  if scale != 1:
    u8 = cv2.resize(u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
  return cv2.applyColorMap(u8, colormap)


def overlay_heatmap(image, heat, strength=0.75, saturate_at=0.6):
  """Blend a [0, 1] error map over the image; zero error leaves pixels untouched."""
  u8 = np.clip(heat * 255, 0, 255).astype(np.uint8)
  col = cv2.applyColorMap(u8, cv2.COLORMAP_JET).astype(np.float32)
  w = (strength * np.clip(heat / saturate_at, 0, 1))[..., None]
  return (image.astype(np.float32) * (1 - w) + col * w).astype(np.uint8)


def titled(img, text, height=24):
  bar = np.full((height, img.shape[1], 3), 32, np.uint8)
  cv2.putText(bar, text, (6, height - 8), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
  return np.vstack([bar, img])


def hstack_same_height(imgs, gap=4):
  h = max(i.shape[0] for i in imgs)
  out = []
  for i in imgs:
    if i.shape[0] != h:
      i = cv2.resize(i, (int(round(i.shape[1] * h / i.shape[0])), h), interpolation=cv2.INTER_AREA)
    out += [i, np.full((h, gap, 3), 255, np.uint8)]
  return np.hstack(out[:-1])


def vstack_same_width(imgs, gap=4):
  w = max(i.shape[1] for i in imgs)
  out = []
  for i in imgs:
    if i.shape[1] != w:
      i = cv2.resize(i, (w, int(round(i.shape[0] * w / i.shape[1]))), interpolation=cv2.INTER_AREA)
    out += [i, np.full((gap, w, 3), 255, np.uint8)]
  return np.vstack(out[:-1])
