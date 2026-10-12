import argparse
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split


def resolve_dataset_root(raw_input: str) -> Path:
    input_path = Path(raw_input)
    if input_path.is_dir() and (input_path / "PNGImages").exists() and (input_path / "PedMasks").exists():
        return input_path

    if input_path.is_dir():
        candidates = [
            path for path in input_path.iterdir()
            if path.is_dir() and (path / "PNGImages").exists() and (path / "PedMasks").exists()
        ]
        if len(candidates) == 1:
            return candidates[0]

    raise FileNotFoundError(
        f"Could not find PennFudanPed-style dataset under '{raw_input}'. "
        "Expected a directory containing PNGImages/ and PedMasks/."
    )


def prepare_data_splits(raw_dir="data/raw/PennFudanPed", processed_dir="data/processed", target_size=(320, 320)):
    """Create a fixed train/validation split for the PennFudanPed dataset and resize images to a modest size."""
    dataset_root = resolve_dataset_root(raw_dir)
    images_dir = dataset_root / "PNGImages"
    masks_dir = dataset_root / "PedMasks"

    images = sorted(p.name for p in images_dir.iterdir() if p.is_file())
    masks_by_stem = {
        path.stem.removesuffix("_mask"): path.name
        for path in masks_dir.iterdir()
        if path.is_file() and path.stem.endswith("_mask")
    }
    missing_masks = [name for name in images if Path(name).stem not in masks_by_stem]
    if missing_masks:
        raise FileNotFoundError(
            f"Missing masks for {len(missing_masks)} image(s), first: {missing_masks[0]}"
        )
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    print(f"Total images: {len(images)}")
    print(f"Training split: {len(train_imgs)}")
    print(f"Validation split: {len(val_imgs)}")
    print(f"Resizing processed samples to {target_size}")

    output_root = Path(processed_dir)
    for subset in ("train", "val"):
        (output_root / subset / "PNGImages").mkdir(parents=True, exist_ok=True)
        (output_root / subset / "PedMasks").mkdir(parents=True, exist_ok=True)

    def copy_subset(img_list, subset_name):
        for img_name in img_list:
            src_img = images_dir / img_name
            img = Image.open(src_img).convert("RGB")
            resized_img = img.resize(target_size, Image.Resampling.BILINEAR)
            dst_img = output_root / subset_name / "PNGImages" / img_name
            resized_img.save(dst_img)

            base_name = Path(img_name).stem
            mask_name = masks_by_stem[base_name]
            src_mask = masks_dir / mask_name
            mask_img = Image.open(src_mask)
            resized_mask = mask_img.resize(target_size, Image.Resampling.NEAREST)
            dst_mask = output_root / subset_name / "PedMasks" / mask_name
            resized_mask.save(dst_mask)

    copy_subset(train_imgs, "train")
    copy_subset(val_imgs, "val")
    print("Preprocessing and dataset splitting completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/validation splits for PennFudanPed.")
    parser.add_argument("--input", default="data/raw", help="Dataset root or PennFudanPed parent directory.")
    parser.add_argument("--output", default="data/processed", help="Directory for processed train/val splits.")
    parser.add_argument("--size", type=int, nargs=2, default=(320, 320), metavar=("WIDTH", "HEIGHT"), help="Target size used for resized images and masks.")
    args = parser.parse_args()

    prepare_data_splits(raw_dir=args.input, processed_dir=args.output, target_size=tuple(args.size))
