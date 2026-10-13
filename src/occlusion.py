"""Controlled synthetic occlusion for PennFudanPed-style instance masks.

PennFudanPed has no occlusion labels, so we *create* them. For every pedestrian
we paste an opaque occluder that hides (approximately) ``level`` of that
pedestrian's mask area. The realised fraction is measured and returned, so the
analysis can use the real number rather than the nominal one.

Scenarios (shape / position of the occluder, all clipped to the instance box so
the occluder stays local to the pedestrian it is meant to hide):
  random : rectangle centred on a random body pixel, random aspect ratio
  lower  : band covering the bottom of the person (cars, bushes, benches)
  side   : vertical strip entering from the left or right (poles, walls)
  mixed  : one of the above chosen at random per pedestrian

Ground truth after occlusion:
  modal   (default) : visible part only  -> "segment what you can see"
  amodal            : original full mask -> "also hallucinate the hidden part"
Instances with fewer than ``min_visible_px`` visible pixels are dropped from GT.
"""

from dataclasses import dataclass

import cv2
import numpy as np

SCENARIOS = ("random", "lower", "side")
FILLS = ("solid", "noise", "bg")


@dataclass
class OccludedSample:
  image: np.ndarray      # (H, W, 3) BGR with occluders painted in
  gt_masks: np.ndarray   # (K, H, W) bool, instances kept after occlusion
  occluder: np.ndarray   # (H, W) bool union of all occluders
  occ_ratio: np.ndarray  # (K,) realised hidden fraction of each kept instance
  kept: np.ndarray       # (K,) indices into the original instance list


def _bbox(mask):
  ys, xs = np.nonzero(mask)
  return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _rect_random(mask, level, rng):
  """Rectangle around a random mask pixel, scaled (bisection) to hit ``level``."""
  x0, y0, x1, y1 = _bbox(mask)
  crop = mask[y0:y1, x0:x1]
  hc, wc = crop.shape
  ys, xs = np.nonzero(crop)
  k = rng.integers(len(xs))
  cx, cy = xs[k] + 0.5, ys[k] + 0.5
  aspect = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))  # width / height
  ii = cv2.integral(crop.astype(np.uint8))  # (hc+1, wc+1)
  target = level * float(crop.sum())

  def rect(s):
    w, h = aspect * s, s
    l = int(np.clip(round(cx - w / 2), 0, wc))
    r = int(np.clip(round(cx + w / 2), 0, wc))
    t = int(np.clip(round(cy - h / 2), 0, hc))
    b = int(np.clip(round(cy + h / 2), 0, hc))
    return l, t, r, b

  def covered(s):
    l, t, r, b = rect(s)
    return float(ii[b, r] - ii[t, r] - ii[b, l] + ii[t, l])

  lo, hi = 0.0, 10.0 * (hc + wc)
  for _ in range(24):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if covered(mid) < target else (lo, mid)
  l, t, r, b = rect(hi)
  return x0 + l, y0 + t, x0 + r, y0 + b


def _rect_lower(mask, level, rng):
  x0, y0, x1, y1 = _bbox(mask)
  crop = mask[y0:y1, x0:x1]
  cum = np.cumsum(crop.sum(1)[::-1])  # mask pixels from the bottom up
  n = int(np.searchsorted(cum, level * crop.sum())) + 1
  n = int(np.clip(n, 1, crop.shape[0]))
  return x0, y1 - n, x1, y1


def _rect_side(mask, level, rng):
  x0, y0, x1, y1 = _bbox(mask)
  crop = mask[y0:y1, x0:x1]
  from_right = bool(rng.integers(2))
  cols = crop.sum(0)
  cum = np.cumsum(cols[::-1] if from_right else cols)
  n = int(np.searchsorted(cum, level * crop.sum())) + 1
  n = int(np.clip(n, 1, crop.shape[1]))
  return (x1 - n, y0, x1, y1) if from_right else (x0, y0, x0 + n, y1)


_RECTS = {"random": _rect_random, "lower": _rect_lower, "side": _rect_side}


def _paint(image, occ, fill, rng, bg_mask):
  out = image.copy()
  n = int(occ.sum())
  if n == 0:
    return out
  if fill == "solid":
    out[occ] = (114, 114, 114)
  elif fill == "noise":
    out[occ] = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
  elif fill == "bg":  # random pixels from the pedestrian-free background
    pool = image[bg_mask] if bg_mask.sum() > 100 else image.reshape(-1, 3)
    out[occ] = pool[rng.integers(0, len(pool), size=n)]
  else:
    raise ValueError(f"unknown fill {fill!r}")
  return out


def occlude(
  image,
  masks,
  level,
  scenario="random",
  fill="solid",
  gt_mode="modal",
  rng=None,
  min_visible_px=64,
):
  """Hide ~``level`` of every pedestrian. ``level == 0`` returns the input unchanged."""
  rng = rng if rng is not None else np.random.default_rng(0)
  H, W = image.shape[:2]
  occ = np.zeros((H, W), bool)
  if level > 0:
    for m in masks:
      if m.sum() < 2:
        continue
      sc = scenario if scenario != "mixed" else SCENARIOS[rng.integers(len(SCENARIOS))]
      x0, y0, x1, y1 = _RECTS[sc](m, level, rng)
      occ[y0:y1, x0:x1] = True

  out = _paint(image, occ, fill, rng, ~masks.any(0)) if len(masks) else image.copy()
  visible = masks & ~occ[None]
  orig_area = np.maximum(masks.reshape(len(masks), -1).sum(1), 1)
  vis_area = visible.reshape(len(masks), -1).sum(1)
  keep = vis_area >= min_visible_px
  gt = (visible if gt_mode == "modal" else masks)[keep]
  return OccludedSample(
    image=out,
    gt_masks=gt,
    occluder=occ,
    occ_ratio=(1.0 - vis_area / orig_area)[keep],
    kept=np.nonzero(keep)[0],
  )


def natural_occlusion(masks):
  """Occlusion proxy for the *clean* annotations, per instance.

  Fraction of an instance's bounding box covered by the boxes of the other
  pedestrians in the same image (0 = isolated, 1 = box fully overlapped).
  Crude, but PennFudan has no occlusion labels and it correlates with people
  standing in front of / behind each other.
  """
  boxes = [_bbox(m) for m in masks if m.any()]
  out = np.zeros(len(boxes), np.float32)
  for i, (x0, y0, x1, y1) in enumerate(boxes):
    canvas = np.zeros((y1 - y0, x1 - x0), bool)
    for j, (a0, b0, a1, b1) in enumerate(boxes):
      if i == j:
        continue
      ix0, iy0, ix1, iy1 = max(x0, a0), max(y0, b0), min(x1, a1), min(y1, b1)
      if ix1 > ix0 and iy1 > iy0:
        canvas[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = True
    out[i] = canvas.mean()
  return out
