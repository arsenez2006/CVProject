import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PedestrianDataset(Dataset):
    """
    Класс датасета для загрузки изображений пешеходов и их масок сегментации.
    Подходит под требования Mask R-CNN (возвращает boxes, masks, labels и др.).
    """
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        
        # Сортируем файлы, чтобы изображения и маски строго совпадали по индексам
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "PedMasks"))))

    def __getitem__(self, idx):
        # Загрузка изображения и соответствующей маски
        img_path = os.path.join(self.root_dir, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "PedMasks", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        mask_np = np.array(mask)
        # Идетификаторы объектов (0 — фон, остальные числа — разные пешеходы)
        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[1:]  # Убираем фон

        # Превращаем цветовую маску в набор бинарных масок для каждого экземпляра
        masks = mask_np == obj_ids[:, None, None]

        # Вычисление bounding boxes [xmin, ymin, xmax, ymax] на основе масок
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # Конвертация в тензоры PyTorch, необходимые для Mask R-CNN
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # Все классы = 1 (пешеход)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
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
