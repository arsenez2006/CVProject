from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RandomOcclusion:
    def __init__(self, probability=0.5, max_fraction=0.5, fill_value=114):
        self.probability = probability
        self.max_fraction = max_fraction
        self.fill_value = fill_value

    def __call__(self, image, target):
        if np.random.random() >= self.probability or not len(target["masks"]):
            return image, target

        image_array = np.asarray(image).copy()
        masks = target["masks"].numpy().astype(bool)
        instance_index = np.random.randint(len(masks))
        mask = masks[instance_index]
        ys, xs = np.where(mask)
        if not len(xs):
            return image, target

        fraction = np.random.uniform(0.2, self.max_fraction)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        height = max(1, int((y1 - y0) * np.sqrt(fraction)))
        width = max(1, int((x1 - x0) * np.sqrt(fraction)))
        left = np.random.randint(x0, max(x0 + 1, x1 - width + 1))
        top = np.random.randint(y0, max(y0 + 1, y1 - height + 1))
        occluder = np.zeros(mask.shape, dtype=bool)
        occluder[top : top + height, left : left + width] = True
        occluder &= mask

        image_array[occluder] = self.fill_value
        visible_masks = masks & ~occluder[None]
        keep = visible_masks.reshape(len(visible_masks), -1).sum(axis=1) > 0
        visible_masks = visible_masks[keep]
        if not keep.any():
            return image, target

        boxes = []
        for visible_mask in visible_masks:
            visible_y, visible_x = np.where(visible_mask)
            boxes.append(
                [
                    visible_x.min(),
                    visible_y.min(),
                    visible_x.max() + 1,
                    visible_y.max() + 1,
                ]
            )

        target = dict(target)
        target["masks"] = torch.as_tensor(visible_masks, dtype=torch.uint8)
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = target["labels"][keep]
        target["area"] = target["masks"].flatten(1).sum(dim=1).to(torch.float32)
        target["iscrowd"] = target["iscrowd"][keep]
        return Image.fromarray(image_array), target


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
