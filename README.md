# Pedestrian Instance Segmentation with Mask R-CNN

This project implements a first Mask R-CNN baseline for pedestrian instance segmentation on the PennFudanPed dataset.

## Project status

The repository currently contains an end-to-end pipeline for dataset downloading and preprocessing, a custom PennFudanPed dataset loader, and a working baseline training script. The data pipeline resolves the raw PennFudanPed directory, splits it into train/validation subsets, and copies the matching image/mask pairs into `data/processed/train` and `data/processed/val`.

The main training entry point is:

- `src/train_baseline.py`

It trains a Mask R-CNN model on the processed split and saves the weights to `artifacts/maskrcnn_baseline.pt` together with validation metrics in `artifacts/metrics.json`.

## Data pipeline

The raw PennFudanPed dataset is stored under `data/raw/PennFudanPed` and has the expected structure:

- `PNGImages/`
- `PedMasks/`

The preprocessing script (`src/preprocess.py`) searches for the dataset root, creates a fixed 80/20 train/validation split with seed 42, and copies each image and corresponding mask pair into the processed folders. The dataset loader (`src/dataset.py`) returns the boxes, masks, labels, area, and crowd flags expected by Mask R-CNN.

## Baseline model configuration

- Model: Mask R-CNN with ResNet-50 FPN backbone
- Number of classes: 2 (`background` + `pedestrian`)
- Optimizer: SGD
- Learning rate: 0.005
- Momentum: 0.9
- Weight decay: 0.0005
- Batch size: 2
- Training epochs: 3 (working baseline)
- Device: CUDA when available, otherwise CPU

## Initial quantitative results

The first baseline run reports validation metrics from COCO evaluation. The initial numbers are captured in `artifacts/metrics.json` after running the script. In the first working run, the model achieves a reasonable detection signal but mask quality remains behind detection quality, which is typical for an early baseline on a small pedestrian dataset.

## Observations and limitations

- The detection boxes are generally stable and localize pedestrians well in the early stages of training.
- Mask predictions are less precise than the bounding boxes, especially around thin limbs, partial occlusions, and crowded scenes.
- The model is intentionally treated as a first baseline; further improvement would require longer training, stronger augmentation, or a pretrained backbone with more epochs.
- The pipeline is currently validated on the PennFudanPed dataset and is ready to be extended with additional experiments.

## Repository link

Team Git repository:
https://github.com/arsenez2006/CVProject.git

Prepared by Emil Nabiullin.
