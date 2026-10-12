import argparse
import json
from pathlib import Path
import ssl
import os
import certifi

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import PedestrianDataset

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_mask(mask: np.ndarray) -> dict:
    mask = np.asfortranarray(mask.astype(np.uint8))
    encoded = mask_utils.encode(mask)
    counts = encoded["counts"]
    if isinstance(counts, (bytes, bytearray)):
        counts = counts.decode("utf-8")
    return {"size": list(encoded["size"]), "counts": counts}


def build_coco_ground_truth(dataset):
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pedestrian"}],
    }

    for image_id in range(len(dataset)):
        image, target = dataset[image_id]

        if hasattr(image, "size") and len(image.size) == 2:
            width, height = image.size
        else:
            _, height, width = image.shape

        coco_dataset["images"].append(
            {"id": image_id, "width": int(width), "height": int(height)}
        )

        boxes = target["boxes"]
        masks = target["masks"]
        for instance_id, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            width_box = max(0.0, x2 - x1)
            height_box = max(0.0, y2 - y1)
            if width_box <= 0.0 or height_box <= 0.0:
                continue

            mask = masks[instance_id].cpu().numpy().astype(np.uint8)
            segmentation = encode_mask(mask)
            coco_dataset["annotations"].append(
                {
                    "id": len(coco_dataset["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(width_box), float(height_box)],
                    "area": float(mask.sum()),
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )

    coco = COCO()
    coco.dataset = coco_dataset
    coco.createIndex()
    return coco


def build_model(num_classes=2):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    return model


def run_training(train_loader, val_loader, device, epochs=5, lr=0.005):
    model = build_model(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    coco_gt = build_coco_ground_truth(val_loader.dataset)

    history = []
    best_scores = {"bbox_ap": -1.0, "mask_ap": -1.0}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [F.to_tensor(image).to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            running_loss += float(losses.item())

        avg_train_loss = running_loss / max(1, len(train_loader))
        metrics = evaluate_model(model, val_loader, device, coco_gt)
        history.append({"epoch": epoch, "train_loss": avg_train_loss, **metrics})
        print(
            f"Epoch {epoch}/{epochs}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"bbox_ap={metrics['bbox_ap']:.4f}, "
            f"mask_ap={metrics['mask_ap']:.4f}"
        )

        if metrics["bbox_ap"] > best_scores["bbox_ap"]:
            best_scores["bbox_ap"] = metrics["bbox_ap"]
        if metrics["mask_ap"] > best_scores["mask_ap"]:
            best_scores["mask_ap"] = metrics["mask_ap"]

    return model, history, best_scores


def evaluate_model(model, val_loader, device, coco_gt=None):
    model.eval()
    predictions = []

    for images, targets in val_loader:
        images = [F.to_tensor(image).to(device) for image in images]
        with torch.no_grad():
            outputs = model(images)

        for image_index, output in enumerate(outputs):
            target = targets[image_index]
            image_id = int(target["image_id"][0].item())

            for box, score, label, mask in zip(
                output["boxes"], output["scores"], output["labels"], output["masks"]
            ):
                if score.item() < 0.05:
                    continue

                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 0.0 or height <= 0.0:
                    continue

                mask_array = (mask[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                segmentation = encode_mask(mask_array)
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label.item()),
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "area": float(mask_array.sum()),
                        "score": float(score.item()),
                        "segmentation": segmentation,
                    }
                )

    if not predictions:
        return {
            "bbox_ap": 0.0,
            "bbox_ap50": 0.0,
            "bbox_ap75": 0.0,
            "mask_ap": 0.0,
            "mask_ap50": 0.0,
            "mask_ap75": 0.0,
        }

    val_dataset = val_loader.dataset
    if coco_gt is None:
        coco_gt = build_coco_ground_truth(val_dataset)

    coco_dt = coco_gt.loadRes(predictions)

    bbox_eval = COCOeval(coco_gt, coco_dt, "bbox")
    bbox_eval.params.imgIds = list(range(len(val_dataset)))
    bbox_eval.evaluate()
    bbox_eval.accumulate()
    bbox_eval.summarize()

    segm_eval = COCOeval(coco_gt, coco_dt, "segm")
    segm_eval.params.imgIds = list(range(len(val_dataset)))
    segm_eval.evaluate()
    segm_eval.accumulate()
    segm_eval.summarize()

    return {
        "bbox_ap": float(bbox_eval.stats[0]),
        "bbox_ap50": float(bbox_eval.stats[1]),
        "bbox_ap75": float(bbox_eval.stats[2]),
        "mask_ap": float(segm_eval.stats[0]),
        "mask_ap50": float(segm_eval.stats[1]),
        "mask_ap75": float(segm_eval.stats[2]),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a first Mask R-CNN baseline on PennFudanPed."
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        type=str,
        help="Path to processed train/val directory.",
    )
    parser.add_argument(
        "--epochs", default=3, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        default=2,
        type=int,
        help="Batch size used for training and validation.",
    )
    parser.add_argument(
        "--lr", default=0.005, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        type=str,
        help="Directory to save outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    data_dir = Path(args.data_dir)
    print(f"Data dir: {data_dir.resolve()}")
    print(f"Train path exists: {(data_dir / 'train').exists()}")
    print(f"Val path exists: {(data_dir / 'val').exists()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PedestrianDataset(str(data_dir / "train"))
    val_dataset = PedestrianDataset(str(data_dir / "val"))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    try:
        model, history, best_scores = run_training(
            train_loader, val_loader, device, epochs=args.epochs, lr=args.lr
        )
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    torch.save(model.state_dict(), output_dir / "maskrcnn_baseline.pt")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({"best_scores": best_scores, "history": history}, file, indent=2)

    print("Best scores:")
    print(json.dumps(best_scores, indent=2))
    print(f"Saved model weights to {output_dir / 'maskrcnn_baseline.pt'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
