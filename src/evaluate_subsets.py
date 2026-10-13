import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import PedestrianDataset
from train_baseline import evaluate_model


def box_iou(box_a, box_b):
    """Return intersection-over-union of two [x1,y1,x2,y2] boxes."""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def has_overlapping_boxes(boxes, threshold):
    """Approximate crowded/occluded scenes using ground-truth box overlap."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iou(boxes[i], boxes[j]) >= threshold:
                return True
    return False


def select_subsets(base_dataset, overlap_threshold):
    groups = {
        "all": list(range(len(base_dataset))),
        "isolated": [],
        "multi_nonoverlap": [],
        "overlap": [],
        "multi_person": [],
    }
    for dataset_index in range(len(base_dataset)):
        _, target = base_dataset[dataset_index]
        boxes = target["boxes"].tolist()
        if len(boxes) == 1:
            groups["isolated"].append(dataset_index)
        elif len(boxes) >= 2:
            groups["multi_person"].append(dataset_index)
            if has_overlapping_boxes(boxes, overlap_threshold):
                groups["overlap"].append(dataset_index)
            else:
                groups["multi_nonoverlap"].append(dataset_index)
    return groups


class ReindexedSubset(Dataset):
    def __init__(self, base_dataset, original_indices):
        self.base_dataset = base_dataset
        self.original_indices = list(original_indices)

    def __len__(self):
        return len(self.original_indices)

    def __getitem__(self, subset_index):
        image, target = self.base_dataset[self.original_indices[subset_index]]
        target = dict(target)
        target["image_id"] = torch.tensor([subset_index], dtype=torch.int64)
        return image, target


def load_trained_model(checkpoint_path, device):
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, 2)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def collate_fn(batch):
    return tuple(zip(*batch))


def format_threshold_tag(value: float) -> str:
    """0.05 -> 'iou005', 0.10 -> 'iou010', 0.20 -> 'iou020'."""
    return f"iou{int(round(value * 100)):03d}"


def evaluate_groups(model, base_dataset, groups, device, overlap_threshold):
    rows = []
    tag = format_threshold_tag(overlap_threshold)
    for name, original_indices in groups.items():
        if not original_indices:
            print(f"SKIP [{tag}] {name}: no validation images in this subset")
            continue
        subset = ReindexedSubset(base_dataset, original_indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
        print(f"\n--- [{tag}] Evaluating {name} ({len(subset)} images) ---")
        metrics = evaluate_model(model, loader, device)
        gt_people = sum(len(subset[i][1]["boxes"]) for i in range(len(subset)))
        row = {
            "subset": name,
            "images": len(subset),
            "ground_truth_pedestrians": gt_people,
            **metrics,
            "gap": float(metrics["bbox_ap"] - metrics["mask_ap"]),
        }
        rows.append(row)
        print(f"{name}: BBox AP={row['bbox_ap']:.4f}; "
              f"Mask AP={row['mask_ap']:.4f}; Gap={row['gap']:.4f}")
    return rows


def write_results(rows, groups, output_dir, overlap_threshold,
                  checkpoint, val_dir, tag=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    csv_path = output_dir / f"subset_metrics{suffix}.csv"
    json_path = output_dir / f"subset_metrics{suffix}.json"
    fieldnames = ["subset", "images", "ground_truth_pedestrians", "bbox_ap",
                  "bbox_ap50", "bbox_ap75", "mask_ap", "mask_ap50", "mask_ap75", "gap"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({
            "checkpoint": str(checkpoint),
            "validation_directory": str(val_dir),
            "overlap_definition": f"GT bounding-box IoU >= {overlap_threshold}",
            "notes": [
                "Overlap is a bounding-box heuristic, not verified human occlusion labels.",
                "The multi_person group includes both multi_nonoverlap and overlap groups.",
                "All subsets were evaluated with the same fixed checkpoint.",
            ],
            "subset_counts": {name: len(indices) for name, indices in groups.items()},
            "results": rows,
        }, file, ensure_ascii=False, indent=2)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="COCO metrics by pedestrian subset")
    parser.add_argument("--val-dir", type=Path, default=Path("data/processed/val"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/maskrcnn_baseline.pt"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/subset_evaluation"))
    parser.add_argument("--overlap-iou", type=float, default=0.05,
                        help="Primary overlap threshold for the top-level result.")
    parser.add_argument("--sensitivity-thresholds", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20],
                        help="IoU thresholds evaluated into threshold_sensitivity/.")
    args = parser.parse_args()

    if not 0.0 < args.overlap_iou <= 1.0:
        parser.error("--overlap-iou must be in (0, 1].")
    for t in args.sensitivity_thresholds:
        if not 0.0 < t <= 1.0:
            parser.error(f"--sensitivity-thresholds value {t} must be in (0, 1].")
    if not args.val_dir.is_dir():
        parser.error(f"Validation directory not found: {args.val_dir}. "
                     "Run preprocessing first.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. "
                     "Obtain/train the model first.")

    base_dataset = PedestrianDataset(str(args.val_dir))
    if len(base_dataset) == 0:
        parser.error("Validation dataset is empty.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = load_trained_model(args.checkpoint, device)
    print(f"Using device: {device}; validation images: {len(base_dataset)}")

    # --- Primary evaluation ---
    primary_groups = select_subsets(base_dataset, args.overlap_iou)
    print("Primary subset sizes:",
          {name: len(indices) for name, indices in primary_groups.items()})
    primary_rows = evaluate_groups(model, base_dataset, primary_groups,
                                   device, args.overlap_iou)
    csv_path, json_path = write_results(
        primary_rows, primary_groups, args.output_dir, args.overlap_iou,
        args.checkpoint, args.val_dir, tag=None,
    )
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    # --- Sensitivity sweep ---
    sens_dir = args.output_dir / "threshold_sensitivity"
    for t in args.sensitivity_thresholds:
        groups = select_subsets(base_dataset, t)
        rows = evaluate_groups(model, base_dataset, groups, device, t)
        tag = format_threshold_tag(t)
        csv_path, json_path = write_results(
            rows, groups, sens_dir, t,
            args.checkpoint, args.val_dir, tag=tag,
        )
        print(f"Saved: {csv_path}")
        print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
