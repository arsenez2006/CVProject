"""Error maps and failure-type bookkeeping for instance segmentation.

Per image (``compute_errors``)
------------------------------
Every GT instance is compared with the prediction that overlaps it most
(highest mask IoU). Spurious predictions that are nobody's best match are
compared with "nothing".

  err : soft error heatmap in [0, 1]
          matched pair  -> |mask_probability - gt_mask|
          missed GT     -> 1 inside the GT mask
          spurious pred -> its mask probability
        (pixelwise max over all contributions). Using the *probabilities*
        instead of thresholded masks also shows where the model is unsure.
  fn  : bool, GT pixels not covered by the matched prediction  (misses)
  fp  : bool, predicted pixels outside the matched GT          (leakage)
        If one prediction swallows two pedestrians ("merge"), the swallowed
        pedestrian shows up as FP: the pixels are right but the identity is not.

Per GT instance outcome (IoU threshold 0.5, COCO-style greedy matching):
  ok         mask IoU >= 0.5 with a prediction
  poor_mask  box IoU >= 0.5 with some prediction but its mask IoU < 0.5
             -> detection succeeded, segmentation failed (the "Mask AP < Box AP" case)
  missed     no prediction with box IoU >= 0.5

Per unmatched prediction:
  fp_duplicate  >= 50 % of its area lies inside a GT mask (duplicate / fragment)
  fp_background otherwise
  merged_preds  predictions covering >= 30 % of two or more GT instances

Aggregation (``HeatmapAccumulator``)
------------------------------------
  scene    maps: every image resized to a fixed canvas -> where in the *frame*
  instance maps: every GT box resized to a fixed canvas -> where on the *body*
                 (head / torso / legs / limbs) masks fail.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from cvutils import (
  Detections,
  box_iou_matrix,
  mask_overlap,
  mask_to_box,
  match_instances,
)


@dataclass
class GtOutcome:
  index: int
  outcome: str      # "ok" | "poor_mask" | "missed"
  mask_iou: float   # best mask IoU with any prediction
  box: tuple        # (x0, y0, x1, y1)


@dataclass
class ImageErrors:
  err: np.ndarray
  fn: np.ndarray
  fp: np.ndarray
  gt_outcomes: list = field(default_factory=list)
  fp_background: int = 0
  fp_duplicate: int = 0
  merged_preds: int = 0
  per_gt: list = field(default_factory=list)  # dicts of box-cropped maps
  quality: float = float("nan")               # mean best mask IoU over GT


def compute_errors(gt, det, hw, score_thr=0.5, iou_thr=0.5, mask_thr=0.5):
  H, W = hw
  det = det.filter(score_thr) if len(det) else Detections.empty(H, W)
  pm = det.masks(mask_thr)
  G, P = len(gt), len(det)

  err = np.zeros((H, W), np.float32)
  fn = np.zeros((H, W), bool)
  fp = np.zeros((H, W), bool)
  res = ImageErrors(err, fn, fp)

  if G and P:
    inter, g_area, p_area = mask_overlap(gt, pm)
    iou = inter / np.maximum(g_area[:, None] + p_area[None] - inter, 1.0)  # (G, P)
    gt_boxes = np.stack([mask_to_box(m) for m in gt])
    b_iou = box_iou_matrix(gt_boxes, det.boxes)
    pairs, unmatched_p, _ = match_instances(iou.T, det.scores, iou_thr)
  else:
    inter = iou = b_iou = None
    gt_boxes = np.stack([mask_to_box(m) for m in gt]) if G else np.zeros((0, 4), np.float32)
    pairs, unmatched_p = [], list(range(P))
  matched_g = {g for _, g in pairs}

  used = set()
  qualities = []
  for g in range(G):
    gm = gt[g]
    best = int(iou[g].argmax()) if P and iou[g].max() > 0 else None
    q = float(iou[g].max()) if P else 0.0
    qualities.append(q)
    if best is not None:
      used.add(best)
      e = np.abs(det.probs[best] - gm.astype(np.float32))
      f_n, f_p = gm & ~pm[best], pm[best] & ~gm
    else:
      e, f_n, f_p = gm.astype(np.float32), gm, np.zeros_like(gm)
    np.maximum(err, e, out=err)
    fn |= f_n
    fp |= f_p

    if g in matched_g:
      outcome = "ok"
    elif P and b_iou[g].max() >= iou_thr:
      outcome = "poor_mask"
    else:
      outcome = "missed"
    x0, y0, x1, y1 = (int(v) for v in gt_boxes[g])
    res.gt_outcomes.append(GtOutcome(g, outcome, q, (x0, y0, x1, y1)))
    if x1 > x0 and y1 > y0:
      res.per_gt.append(
        {
          "gt": gm[y0:y1, x0:x1].astype(np.float32),
          "fn": f_n[y0:y1, x0:x1].astype(np.float32),
          "fp": f_p[y0:y1, x0:x1].astype(np.float32),
          "err": e[y0:y1, x0:x1].astype(np.float32),
        }
      )

  for p in range(P):  # predictions that are nobody's best match
    if p not in used:
      np.maximum(err, det.probs[p], out=err)
      fp |= pm[p]

  for p in unmatched_p:
    if G and (inter[:, p] / max(float(p_area[p]), 1.0)).max() >= 0.5:
      res.fp_duplicate += 1
    else:
      res.fp_background += 1
  if G >= 2 and P:
    cover = inter / np.maximum(g_area[:, None], 1.0)  # share of each GT inside each pred
    res.merged_preds = int(((cover >= 0.3).sum(0) >= 2).sum())

  res.quality = float(np.mean(qualities)) if qualities else float("nan")
  return res


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _resize(a, hw):
  h, w = hw
  interp = cv2.INTER_AREA if (a.shape[0] >= h and a.shape[1] >= w) else cv2.INTER_LINEAR
  return cv2.resize(np.asarray(a, np.float32), (w, h), interpolation=interp)


class HeatmapAccumulator:
  def __init__(self, scene_hw=(128, 128), inst_hw=(128, 64)):
    self.scene_hw, self.inst_hw = scene_hw, inst_hw
    self.scene = {k: np.zeros(scene_hw, np.float64) for k in ("fn", "fp", "err")}
    self.inst = {k: np.zeros(inst_hw, np.float64) for k in ("gt", "fn", "fp", "err")}
    self.n_images = 0
    self.n_inst = 0

  def add(self, e):
    self.n_images += 1
    for k, a in (("fn", e.fn), ("fp", e.fp), ("err", e.err)):
      self.scene[k] += _resize(a.astype(np.float32), self.scene_hw)
    for g in e.per_gt:
      self.n_inst += 1
      for k in ("gt", "fn", "fp", "err"):
        self.inst[k] += _resize(g[k], self.inst_hw)

  # ---- read-out ---------------------------------------------------------- #
  def scene_maps(self):
    n = max(self.n_images, 1)
    return {k: v / n for k, v in self.scene.items()}

  def instance_maps(self, min_support=0.05):
    """fn_rate: P(pixel missed | pixel belongs to a pedestrian at this body position).
    fp_freq / err_mean: averaged over all instances."""
    n = max(self.n_inst, 1)
    support = self.inst["gt"] / n
    with np.errstate(divide="ignore", invalid="ignore"):
      fn_rate = np.where(support >= min_support, self.inst["fn"] / self.inst["gt"], 0.0)
    return {
      "fn_rate": fn_rate,
      "fp_freq": self.inst["fp"] / n,
      "err_mean": self.inst["err"] / n,
      "support": support,
    }
