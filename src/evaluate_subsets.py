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


def main():
    parser = argparse.ArgumentParser(description="COCO metrics by pedestrian subset")
    parser.add_argument("--val-dir", type=Path, default=Path("data/processed/val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/maskrcnn_baseline.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/subset_evaluation"))
    parser.add_argument("--overlap-iou", type=float, default=0.05)
    args = parser.parse_args()

    if not 0.0 < args.overlap_iou <= 1.0:
        parser.error("--overlap-iou must be in (0, 1].")
    if not args.val_dir.is_dir():
        parser.error(f"Validation directory not found: {args.val_dir}. Run preprocessing first.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Obtain/train the model first.")

    base_dataset = PedestrianDataset(str(args.val_dir))
    if len(base_dataset) == 0:
        parser.error("Validation dataset is empty.")
    groups = select_subsets(base_dataset, args.overlap_iou)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.checkpoint, device)
    print(f"Using device: {device}; validation images: {len(base_dataset)}")
    print("Subset sizes:", {name: len(indices) for name, indices in groups.items()})

    rows = []
    for name, original_indices in groups.items():
        if not original_indices:
            print(f"SKIP {name}: no validation images in this subset")
            continue
        subset = ReindexedSubset(base_dataset, original_indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
        print(f"\n--- Evaluating {name} ({len(subset)} images) ---")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "subset_metrics.csv"
    json_path = args.output_dir / "subset_metrics.json"
    fieldnames = ["subset", "images", "ground_truth_pedestrians", "bbox_ap",
                  "bbox_ap50", "bbox_ap75", "mask_ap", "mask_ap50", "mask_ap75", "gap"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "validation_directory": str(args.val_dir),
            "overlap_definition": f"GT bounding-box IoU >= {args.overlap_iou}",
            "notes": [
                "Overlap is a bounding-box heuristic, not verified human occlusion labels.",
                "The multi_person group includes both multi_nonoverlap and overlap groups.",
                "All subsets were evaluated with the same fixed checkpoint.",
            ],
            "subset_counts": {name: len(indices) for name, indices in groups.items()},
            "results": rows,
        }, file, ensure_ascii=False, indent=2)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()