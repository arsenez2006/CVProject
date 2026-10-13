"""Occlusion robustness study for the Mask R-CNN baseline.

For every scenario (random / lower / side occluder) and occlusion level
(fraction of each pedestrian that is hidden) this script
  1. runs the model on the occluded validation images,
  2. computes COCO Mask AP and Box AP (pycocotools) -> "AP drop" curves,
  3. classifies every GT pedestrian as ok / poor_mask / missed (+ false positives),
  4. aggregates error heatmaps (scene-level and body-relative),
  5. mines the images whose quality drops most and renders filmstrips,
  6. also evaluates the *clean* images split by natural overlap between people,
  7. writes a markdown report with the numbers filled in.

Everything is seeded and deterministic. The validation split is tiny
(~34 images), so each condition is repeated ``--repeats`` times with different
occluder placements and reported as mean +- std.

Example
-------
  python src/occlusion_eval.py --weights artifacts/maskrcnn_baseline.pt \
      --data-dir data/processed --output-dir artifacts/occlusion
"""

import argparse
import contextlib
import copy
import io
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cvutils import (
  draw_instances,
  draw_outlines,
  find_pairs,
  hstack_same_height,
  load_instance_masks,
  load_model,
  overlay_heatmap,
  put_label,
  read_image,
  titled,
  vstack_same_width,
  colorize,
  FONT,
)
from error_maps import HeatmapAccumulator, compute_errors
from occlusion import FILLS, SCENARIOS, natural_occlusion, occlude

METRIC_KEYS = ("AP", "AP50", "AP75", "AR100")


# --------------------------------------------------------------------------- #
# COCO evaluation
# --------------------------------------------------------------------------- #
def _rle(mask):
  from pycocotools import mask as mu

  r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
  r["counts"] = r["counts"].decode("ascii")
  return r


class CocoCollector:
  """Collects GT + detections of one run and evaluates them with pycocotools."""

  def __init__(self):
    self.images, self.anns, self.segm, self.bbox = [], [], [], []

  def add(self, image_id, hw, gt_masks, det):
    from pycocotools import mask as mu

    self.images.append({"id": image_id, "height": int(hw[0]), "width": int(hw[1])})
    for m in gt_masks:
      rle = _rle(m)
      self.anns.append(
        {
          "id": len(self.anns) + 1,
          "image_id": image_id,
          "category_id": 1,
          "segmentation": rle,
          "area": float(mu.area(rle)),
          "bbox": [float(v) for v in mu.toBbox(rle)],
          "iscrowd": 0,
        }
      )
    masks = det.masks()
    for k in range(len(det)):
      s = float(det.scores[k])
      x0, y0, x1, y1 = (float(v) for v in det.boxes[k])
      self.segm.append(
        {"image_id": image_id, "category_id": 1, "segmentation": _rle(masks[k]), "score": s}
      )
      # NB: segm and bbox results must be separate lists. COCO.loadRes() replaces
      # 'segmentation' by the box polygon whenever a 'bbox' key is present.
      self.bbox.append(
        {"image_id": image_id, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": s}
      )

  def evaluate(self, img_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    nan = {k: float("nan") for k in METRIC_KEYS}
    out = {"segm": dict(nan), "bbox": dict(nan)}
    if not self.anns:
      return out
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
      gt = COCO()
      gt.dataset = {
        "images": self.images,
        "annotations": self.anns,
        "categories": [{"id": 1, "name": "pedestrian"}],
      }
      gt.createIndex()
    for iou_type, dets in (("segm", self.segm), ("bbox", self.bbox)):
      if not dets:
        out[iou_type] = {k: 0.0 for k in METRIC_KEYS}
        continue
      with contextlib.redirect_stdout(quiet):
        dt = gt.loadRes(copy.deepcopy(dets))
        ev = COCOeval(gt, dt, iou_type)
        if img_ids is not None:
          ev.params.imgIds = sorted(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
      s = ev.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ...]
      vals = (s[0], s[1], s[2], s[8])
      out[iou_type] = {k: (100.0 * v if v >= 0 else float("nan")) for k, v in zip(METRIC_KEYS, vals)}
    return out


# --------------------------------------------------------------------------- #
# One condition = (scenario, level)
# --------------------------------------------------------------------------- #
@dataclass
class Condition:
  scenario: str
  level: float
  ap: dict = field(default_factory=dict)         # {"segm": {"AP": mean, "AP_std": ..}, ...}
  occ_realised: float = 0.0
  outcomes: Counter = field(default_factory=Counter)
  quality: np.ndarray = None                     # per image, repeat 0
  heat: HeatmapAccumulator = None
  collector: CocoCollector = None                # kept for the clean condition only
  n_images: int = 0

  @property
  def n_gt(self):
    return sum(self.outcomes[k] for k in ("ok", "poor_mask", "missed"))

  def frac(self, key):
    return self.outcomes[key] / max(self.n_gt, 1)

  def per_image(self, key):
    return self.outcomes[key] / max(self.n_images, 1)


def process_image(args, predict, image, gt, rng, level, scenario):
  sample = occlude(
    image, gt, level, scenario=scenario, fill=args.fill, gt_mode=args.gt,
    rng=rng, min_visible_px=args.min_visible_px,
  )
  det = predict(sample.image)
  errs = compute_errors(sample.gt_masks, det, sample.image.shape[:2], score_thr=args.score_thr)
  return sample, det, errs


def rng_for(args, s_idx, l_idx, repeat, i):
  return np.random.default_rng([args.seed, s_idx, l_idx, repeat, i])


def run_condition(args, predict, images, gts, scenario, s_idx, level, l_idx, repeats, keep_collector=False):
  cond = Condition(scenario, level, heat=HeatmapAccumulator())
  per_repeat = {"segm": [], "bbox": []}
  occ_vals = []
  for r in range(repeats):
    coll = CocoCollector()
    q = np.full(len(images), np.nan)
    for i, (img, gt) in enumerate(zip(images, gts)):
      sample, det, errs = process_image(
        args, predict, img, gt, rng_for(args, s_idx, l_idx, r, i), level, scenario
      )
      coll.add(i, img.shape[:2], sample.gt_masks, det)
      cond.heat.add(errs)
      for o in errs.gt_outcomes:
        cond.outcomes[o.outcome] += 1
      cond.outcomes["fp_background"] += errs.fp_background
      cond.outcomes["fp_duplicate"] += errs.fp_duplicate
      cond.outcomes["merged_preds"] += errs.merged_preds
      cond.n_images += 1
      occ_vals.extend(sample.occ_ratio.tolist())
      q[i] = errs.quality
    res = coll.evaluate()
    for t in per_repeat:
      per_repeat[t].append(res[t])
    if r == 0:
      cond.quality = q
      if keep_collector:
        cond.collector = coll
  for t, runs in per_repeat.items():
    cond.ap[t] = {}
    for k in METRIC_KEYS:
      v = np.array([x[k] for x in runs], float)
      cond.ap[t][k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
      cond.ap[t][k + "_std"] = float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")
  cond.occ_realised = float(np.mean(occ_vals)) if occ_vals else 0.0
  return cond


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def render_row(sample, det, errs, level, args):
  occl = draw_outlines(sample.image, sample.gt_masks, (0, 255, 0), 2)
  # show the occluder footprint with a thin red outline
  occl = draw_outlines(occl, sample.occluder[None], (0, 0, 255), 1) if sample.occluder.any() else occl
  d = det.filter(args.score_thr)
  pred = draw_instances(sample.image, d.masks(), d.boxes, d.scores)
  heat = overlay_heatmap(sample.image, errs.err)
  c = Counter(o.outcome for o in errs.gt_outcomes)
  txt = f"ok {c['ok']} | poor_mask {c['poor_mask']} | missed {c['missed']} | FP {errs.fp_background + errs.fp_duplicate}"
  if errs.merged_preds:
    txt += f" | merged {errs.merged_preds}"
  put_label(heat, txt, (4, heat.shape[0] - 6), scale=0.45)
  occ = float(sample.occ_ratio.mean()) if len(sample.occ_ratio) else 0.0
  return hstack_same_height(
    [
      titled(occl, f"L={level:.2f} (realised {occ:.2f})  input + GT (green)"),
      titled(pred, f"prediction (score>={args.score_thr})"),
      titled(heat, "error heatmap"),
    ]
  )


def render_filmstrip(args, predict, image, gt, i, s_idx, scenario, levels, panel_levels):
  rows = []
  for level in panel_levels:
    l_idx = levels.index(level)
    sample, det, errs = process_image(
      args, predict, image, gt, rng_for(args, s_idx, l_idx, 0, i), level, scenario
    )
    rows.append(render_row(sample, det, errs, level, args))
  return vstack_same_width(rows)


def render_heat_montage(conds, path):
  """rows = occlusion levels, columns = scene FN / scene FP / body FN-rate / body FP / body error."""
  scene = [c.heat.scene_maps() for c in conds]
  inst = [c.heat.instance_maps() for c in conds]
  vmax = {
    "s_fn": max(max(s["fn"].max() for s in scene), 0.05),
    "s_fp": max(max(s["fp"].max() for s in scene), 0.05),
    "i_fn": 1.0,
    "i_fp": max(max(i["fp_freq"].max() for i in inst), 0.05),
    "i_err": max(max(i["err_mean"].max() for i in inst), 0.05),
  }
  heads = [
    "scene: missed (FN)", "scene: leakage (FP)",
    "body: FN rate", "body: FP freq", "body: mean error",
  ]
  cell_h = 256
  rows = []
  for c, s, i in zip(conds, scene, inst):
    cells = [
      colorize(s["fn"], vmax["s_fn"]), colorize(s["fp"], vmax["s_fp"]),
      colorize(i["fn_rate"], vmax["i_fn"]), colorize(i["fp_freq"], vmax["i_fp"]),
      colorize(i["err_mean"], vmax["i_err"]),
    ]
    cells = [cv2.resize(x, (int(x.shape[1] * cell_h / x.shape[0]), cell_h), interpolation=cv2.INTER_CUBIC) for x in cells]
    row = hstack_same_height(cells)
    rows.append(titled(row, f"occlusion {c.level:.2f}  ({c.n_images} images, {c.heat.n_inst} pedestrians)"))
  head = np.full((26, rows[0].shape[1], 3), 32, np.uint8)
  x = 0
  widths = [int(round(cell_h * w)) for w in (1, 1, 0.5, 0.5, 0.5)]
  for t, w in zip(heads, widths):
    cv2.putText(head, t, (x + 4, 18), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    x += w + 4
  cv2.imwrite(str(path), np.vstack([head, vstack_same_width(rows)]))


def plot_curves(results, levels, path):
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("[warn] matplotlib not installed -> skipping ap_vs_occlusion.png (uv add matplotlib)")
    return
  fig, ax = plt.subplots(1, 3, figsize=(15, 4))
  for sc, conds in results.items():
    x = [c.level for c in conds]
    for a, t, title in ((ax[0], "segm", "Mask AP"), (ax[1], "bbox", "Box AP")):
      y = np.array([c.ap[t]["AP"] for c in conds])
      e = np.array([c.ap[t]["AP_std"] for c in conds])
      a.errorbar(x, y, yerr=e, marker="o", capsize=3, label=sc)
      a.set_title(title)
    gap = [c.ap["bbox"]["AP"] - c.ap["segm"]["AP"] for c in conds]
    ax[2].plot(x, gap, marker="o", label=sc)
  ax[2].set_title("Box AP - Mask AP (gap)")
  for a in ax:
    a.set_xlabel("fraction of each pedestrian occluded")
    a.set_ylabel("AP (%)")
    a.grid(alpha=0.3)
  ax[0].legend()
  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _f(v, nd=1):
  return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _pm(d, k):
  return f"{_f(d[k])} ± {_f(d[k + '_std'])}"


def clean_json(o):
  if isinstance(o, dict):
    return {k: clean_json(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [clean_json(v) for v in o]
  if isinstance(o, (float, np.floating)):
    return None if math.isnan(o) else float(o)
  if isinstance(o, np.integer):
    return int(o)
  return o


def build_report(args, results, clean, natural, n_images):
  L = []
  L.append("# Occlusion analysis: Mask AP under occlusion\n")
  L.append(
    f"*Auto-generated by `src/occlusion_eval.py`. Split: `{args.split}` ({n_images} images), "
    f"occluder fill: `{args.fill}`, GT: `{args.gt}`, repeats: {args.repeats}, seed: {args.seed}. "
    "AP values are COCO-style (IoU 0.50:0.95), in %. ± is the std over repeats "
    "(different occluder placements), not over data.*\n"
  )
  L.append(
    f"Clean baseline: **Mask AP {_f(clean.ap['segm']['AP'])}**, Box AP {_f(clean.ap['bbox']['AP'])}, "
    f"Mask AP50 {_f(clean.ap['segm']['AP50'])}, Mask AP75 {_f(clean.ap['segm']['AP75'])}.\n"
  )
  for sc, conds in results.items():
    L.append(f"## Scenario `{sc}`\n")
    L.append(
      "| level | realised occ. | Mask AP | Δ Mask AP | rel. drop | Box AP | Δ Box AP | Box−Mask gap |"
      " ok % | poor_mask % | missed % | FP/img | merged/img |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    base_s, base_b = clean.ap["segm"]["AP"], clean.ap["bbox"]["AP"]
    for c in conds:
      s, b = c.ap["segm"], c.ap["bbox"]
      rel = 100 * (base_s - s["AP"]) / base_s if base_s else float("nan")
      L.append(
        f"| {c.level:.2f} | {c.occ_realised:.2f} | {_pm(s, 'AP')} | {_f(s['AP'] - base_s)} | {_f(rel)}% | "
        f"{_pm(b, 'AP')} | {_f(b['AP'] - base_b)} | {_f(b['AP'] - s['AP'])} | "
        f"{100 * c.frac('ok'):.0f} | {100 * c.frac('poor_mask'):.0f} | {100 * c.frac('missed'):.0f} | "
        f"{c.per_image('fp_background') + c.per_image('fp_duplicate'):.2f} | {c.per_image('merged_preds'):.2f} |"
      )
    L.append("")
    L.append(f"![heatmaps]({sc}/error_heatmaps.png)\n")
    L.append(f"Worst-case filmstrips: `failures/{sc}/` (rows = increasing occlusion).\n")

  if natural:
    L.append("## Natural occlusion (clean images, grouped by overlap between people)\n")
    L.append(
      "Proxy = max over pedestrians of the fraction of their box covered by other pedestrians' boxes. "
      "Small groups ⇒ noisy numbers; treat as indicative.\n"
    )
    L.append("| group | images | pedestrians | Mask AP | Box AP |")
    L.append("|---|---|---|---|---|")
    for n in natural:
      L.append(f"| {n['name']} | {n['images']} | {n['gt']} | {_f(n['segm'])} | {_f(n['bbox'])} |")
    L.append("")

  L.append("## Figures\n")
  L.append("![AP vs occlusion](ap_vs_occlusion.png)\n")
  L.append(
    "## Qualitative observations (fill in after looking at `failures/` and the heatmaps)\n\n"
    "- **Where does Mask AP fall first, AP50 or AP75?** (boundary quality vs. detection)\n"
    "- **Does the Box−Mask gap widen with occlusion?** (are masks hurt more than boxes?)\n"
    "- **Body-relative heatmaps:** which parts (head / torso / legs / arms) are lost first? Compare scenarios.\n"
    "- **Failure modes seen in filmstrips:** merged people, fragmented masks, mask bleeding into the occluder, "
    "hallucinated hidden parts, lost detections.\n"
    "- **Caveats:** synthetic occluders are opaque rectangles; ~34 validation images; one trained baseline (3 epochs).\n"
  )
  return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--weights", default="artifacts/maskrcnn_baseline.pt")
  p.add_argument("--data-dir", default="data/processed")
  p.add_argument("--split", default="val")
  p.add_argument("--output-dir", default="artifacts/occlusion")
  p.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
  p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS) + ["mixed"])
  p.add_argument("--fill", default="solid", choices=FILLS)
  p.add_argument("--gt", default="modal", choices=["modal", "amodal"],
                 help="modal: score against the visible part; amodal: against the full original mask")
  p.add_argument("--repeats", type=int, default=3, help="occluder placements per condition")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--score-thr", type=float, default=0.5, help="threshold for outcomes/heatmaps/figures (AP uses all)")
  p.add_argument("--min-visible-px", type=int, default=64)
  p.add_argument("--num-classes", type=int, default=2)
  p.add_argument("--device", default="auto")
  p.add_argument("--mine-level", type=float, default=0.4, help="level used to rank worst-case images")
  p.add_argument("--panel-levels", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6])
  p.add_argument("--num-failures", type=int, default=6)
  p.add_argument("--natural-thresholds", type=float, nargs=2, default=[0.05, 0.25])
  p.add_argument("--max-images", type=int, default=0, help="debug: only use the first N images")
  return p.parse_args(argv)


def main(args=None, predict=None):
  args = args or parse_args()
  out = Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)

  levels = sorted(set([0.0] + [float(x) for x in args.levels]))
  pairs = find_pairs(Path(args.data_dir) / args.split)
  if args.max_images:
    pairs = pairs[: args.max_images]
  images = [read_image(p) for p, _ in pairs]
  gts = [load_instance_masks(m) for _, m in pairs]
  stems = [p.stem for p, _ in pairs]
  print(f"{len(images)} images, {sum(len(g) for g in gts)} pedestrians from {args.data_dir}/{args.split}")

  if predict is None:
    predictor = load_model(args.weights, args.num_classes, args.device)
    predict = lambda img: predictor(img)  # noqa: E731  (keeps all boxes >= 0.05)

  t0 = time.time()
  clean = run_condition(args, predict, images, gts, "clean", 0, 0.0, 0, repeats=1, keep_collector=True)
  print(f"clean: Mask AP {clean.ap['segm']['AP']:.1f} | Box AP {clean.ap['bbox']['AP']:.1f}  [{time.time() - t0:.0f}s]")

  results = {}
  for s_idx, sc in enumerate(args.scenarios):
    conds = []
    for l_idx, level in enumerate(levels):
      if level == 0.0:
        conds.append(clean)
        continue
      c = run_condition(args, predict, images, gts, sc, s_idx, level, l_idx, args.repeats)
      conds.append(c)
      print(
        f"{sc:7s} L={level:.2f} occ={c.occ_realised:.2f} | Mask AP {c.ap['segm']['AP']:5.1f} "
        f"(d {c.ap['segm']['AP'] - clean.ap['segm']['AP']:+.1f}) | Box AP {c.ap['bbox']['AP']:5.1f} "
        f"| ok {100 * c.frac('ok'):.0f}% poor_mask {100 * c.frac('poor_mask'):.0f}% "
        f"missed {100 * c.frac('missed'):.0f}%  [{time.time() - t0:.0f}s]"
      )
    results[sc] = conds

    # heatmap montage + worst-case filmstrips
    (out / sc).mkdir(exist_ok=True)
    render_heat_montage(conds, out / sc / "error_heatmaps.png")
    mine = min(levels, key=lambda v: abs(v - args.mine_level))
    drop = clean.quality - conds[levels.index(mine)].quality
    order = [i for i in np.argsort(-np.nan_to_num(drop, nan=-np.inf)) if np.isfinite(drop[i])]
    fdir = out / "failures" / sc
    fdir.mkdir(parents=True, exist_ok=True)
    plevels = [v for v in levels if any(abs(v - q) < 1e-9 for q in args.panel_levels)]
    for rank, i in enumerate(order[: args.num_failures], 1):
      strip = render_filmstrip(args, predict, images[i], gts[i], int(i), s_idx, sc, levels, plevels)
      cv2.imwrite(str(fdir / f"rank{rank:02d}_{stems[i]}_drop{drop[i]:.2f}.png"), strip)

  # natural occlusion on the clean images
  natural = []
  prox = np.array([natural_occlusion(g).max() if len(g) else 0.0 for g in gts])
  lo, hi = args.natural_thresholds
  groups = {
    f"isolated (<{lo:.2f})": np.nonzero(prox < lo)[0],
    f"light ({lo:.2f}-{hi:.2f})": np.nonzero((prox >= lo) & (prox < hi))[0],
    f"heavy (>={hi:.2f})": np.nonzero(prox >= hi)[0],
  }
  for name, ids in groups.items():
    if len(ids) == 0:
      continue
    r = clean.collector.evaluate(img_ids=[int(i) for i in ids])
    natural.append(
      {"name": name, "images": int(len(ids)), "gt": int(sum(len(gts[i]) for i in ids)),
       "segm": r["segm"]["AP"], "bbox": r["bbox"]["AP"]}
    )

  # write outputs
  plot_curves(results, levels, out / "ap_vs_occlusion.png")
  rows, flat = [], {"clean_segm_AP": clean.ap["segm"]["AP"], "clean_bbox_AP": clean.ap["bbox"]["AP"]}
  full = {"meta": {k: v for k, v in vars(args).items()}, "levels": levels, "natural": natural, "scenarios": {}}
  for sc, conds in results.items():
    full["scenarios"][sc] = {}
    for c in conds:
      key = f"{c.level:.2f}"
      full["scenarios"][sc][key] = {
        "ap": c.ap, "realised_occlusion": c.occ_realised, "n_gt": c.n_gt,
        "outcomes": dict(c.outcomes),
      }
      flat[f"{sc}_{key}_segm_AP"] = c.ap["segm"]["AP"]
      flat[f"{sc}_{key}_bbox_AP"] = c.ap["bbox"]["AP"]
      rows.append(
        f"{sc},{c.level:.2f},{c.occ_realised:.4f},{c.ap['segm']['AP']:.3f},{c.ap['segm']['AP_std']:.3f},"
        f"{c.ap['segm']['AP50']:.3f},{c.ap['segm']['AP75']:.3f},{c.ap['bbox']['AP']:.3f},"
        f"{c.ap['bbox']['AP_std']:.3f},{c.frac('ok'):.4f},{c.frac('poor_mask'):.4f},{c.frac('missed'):.4f}"
      )
  (out / "ap_vs_occlusion.csv").write_text(
    "scenario,level,realised_occ,segm_AP,segm_AP_std,segm_AP50,segm_AP75,bbox_AP,bbox_AP_std,"
    "frac_ok,frac_poor_mask,frac_missed\n" + "\n".join(rows) + "\n",
    encoding="utf-8",
  )
  (out / "results.json").write_text(json.dumps(clean_json(full), indent=2), encoding="utf-8")
  (out / "metrics.json").write_text(json.dumps(clean_json(flat), indent=2), encoding="utf-8")
  (out / "report.md").write_text(
    build_report(args, results, clean, natural, len(images)), encoding="utf-8"
  )
  print(f"Done in {time.time() - t0:.0f}s -> {out}/ (report.md, results.json, metrics.json, figures)")


if __name__ == "__main__":
  main()