import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_data_splits(raw_dir="data/raw/PennFudanPed", processed_dir="data/processed"):
    """
    Проверяет наличие папок датасета и создает структуру для train/val сплитов.
    """
    images_dir = os.path.join(raw_dir, "PNGImages")
    masks_dir = os.path.join(raw_dir, "PedMasks")
    
    if not os.path.exists(images_dir):
        print(f"Error: Dataset directory {images_dir} not found. Please download PennFudanPed to data/raw/")
        return

    images = sorted(os.listdir(images_dir))
    
    # Делаем train/val сплит (80% / 20%)
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    print(f"Total images: {len(images)}")
    print(f"Training split: {len(train_imgs)}")
    print(f"Validation split: {len(val_imgs)}")
    
    # Создаем директории для обработанных данных
    os.makedirs(os.path.join(processed_dir, "train", "PNGImages"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "train", "PedMasks"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "val", "PNGImages"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "val", "PedMasks"), exist_ok=True)

    def copy_subset(img_list, subset_name):
        for img_name in img_list:
            # Копируем картинку
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(processed_dir, subset_name, "PNGImages", img_name)
            shutil.copyfile(src_img, dst_img)
            
            # Копируем соответствующую маску (имя файла совпадает)
            mask_name = img_name.replace(".png", "_mask.png") # или проверьте расширение масок
            # Для PennFudanPed маски обычно имеют ту же базовую часть имени
            base_name = img_name.split('.')[0]
            # Находим маску по префиксу
            mask_files = [f for f in os.listdir(masks_dir) if f.startswith(base_name)]
            if mask_files:
                src_mask = os.path.join(masks_dir, mask_files[0])
                dst_mask = os.path.join(processed_dir, subset_name, "PedMasks", mask_files[0])
                shutil.copyfile(src_mask, dst_mask)

    copy_subset(train_imgs, "train")
    copy_subset(val_imgs, "val")
    print("Preprocessing and dataset splitting completed successfully!")

if __name__ == "__main__":
    prepare_data_splits()
