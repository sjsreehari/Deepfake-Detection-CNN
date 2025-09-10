# Deepfake Detector

## Dataset Setup
1. Download the dataset using Kaggle API:
	- Place your `kaggle.json` in `~/.kaggle/` or set environment variables.
	- Run: `python -m src.data.split_dataset`
2. The script will download, copy, and split images into train/val/test folders under `data/`.


## Training
Run the training script:
```bash
python src/train.py
```
Checkpoints and logs will be saved in `outputs/checkpoints/`.

## Evaluation
Run the evaluation script:
```bash
python src/evaluate.py
```
Metrics and plots will be saved in `outputs/plots/`.
# Deepfake Detector

## Dataset Setup
1. Download the dataset using Kaggle API:
	- Place your `kaggle.json` in `~/.kaggle/` or set environment variables.
	- Run: `python -m src.data.split_dataset`
2. The script will download, copy, and split images into train/val/test folders under `data/`.
