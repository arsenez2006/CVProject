from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PedestrianDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = Path(root_dir)
        self.transforms = transforms

        self.imgs = sorted((self.root_dir / "PNGImages").glob("*"))
        masks_by_stem = {
            path.stem.removesuffix("_mask"): path
            for path in (self.root_dir / "PedMasks").glob("*")
        }
        missing_masks = [path.name for path in self.imgs if path.stem not in masks_by_stem]
        if missing_masks:
            raise FileNotFoundError(
                f"Missing masks for {len(missing_masks)} image(s), first: {missing_masks[0]}"
            )
        self.masks = [masks_by_stem[path.stem] for path in self.imgs]

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        mask_np = np.array(mask)
        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[1:]  # Убираем фон

        masks = mask_np == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1]) + 1
            ymin = np.min(pos[0])
            ymax = np.max(pos[0]) + 1
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # Все классы = 1 (пешеход)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = masks.flatten(1).sum(dim=1).to(torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
