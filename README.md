


# Deepfake Detector
![Deepfake Detection](https://img.shields.io/badge/Deepfake%20Detection-Project-red)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.11-green)

Detect AI-generated and manipulated facial images (deepfakes) using deep learning. Built with TensorFlow/Keras, leveraging transfer learning (MobileNetV2), advanced augmentation, and robust evaluation metrics.

---


## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Core AI Concepts](#core-ai-concepts-used)
- [Model Performance](#model-performance)
- [Results](#results-sample)
- [Setup & Run](#setup--run)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Project Overview


Deepfakes use generative models (GANs, diffusion models) to synthesize realistic human faces. This project uses CNNs and transfer learning to detect subtle artifacts in such images.

### Real Data Pipeline (Colab Notebook)

- **Dataset:** Uses [Kaggle: deepfake-and-real-images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
- **Splitting:** The notebook organizes images into `train`, `val`, and `test` folders, each with `Fake` and `Real` subfolders
- **Loading:** Uses TensorFlow's `image_dataset_from_directory` with explicit class names
- **Augmentation:** Applies random flips, rotations, and zooms for generalization
- **Model:** MobileNetV2 (transfer learning, head training, fine-tuning)
- **Evaluation:** Confusion matrix, ROC-AUC, classification report, precision-recall curve

Pipeline highlights:

- Modular codebase (data, models, training, evaluation)
- tf.data pipelines for efficient ingestion and augmentation
- Transfer learning with MobileNetV2
- Fine-tuning for domain adaptation
- Evaluation: confusion matrix, ROC-AUC, classification report
- Reproducible experiments: logging, checkpoints, plots

---


## Project Structure

```
deepfake_detector/
│
├── src/
│   ├── data/
│   │   ├── split_dataset.py      # Dataset splitting & preparation
│   │   └── dataloader.py         # tf.data pipeline & augmentation
│   ├── models/
│   │   └── mobilenetv2.py        # Model architecture (transfer learning)
│   ├── train.py                  # Training (head + fine-tuning)
│   ├── evaluate.py               # Evaluation metrics & visualization
│   └── utils.py                  # Helper utilities (plots, checkpoints)
│
├── data/                         # Place datasets here
├── outputs/                      # Logs, checkpoints, plots
├── notebooks/                    # Colab notebooks
├── requirements.txt
├── README.md
└── .gitignore
```

---


---


## Core AI Concepts Used


### 1. Transfer Learning

* CNNs pretrained on ImageNet capture low-level features (edges, textures, color blobs).
* These features are reused and adapted to deepfake classification.
* Training is done in two phases:
	1. Head training – freeze backbone, train dense layers
	2. Fine-tuning – unfreeze backbone with small learning rate for domain-specific adaptation


### 2. Data Augmentation

* Random flips, rotations, and zooms to prevent overfitting
* Augmentations simulate natural image variations and improve generalization


### 3. Reproducible ML Engineering

* Modular code with src/ structure (dataset, models, training, evaluation separated)
* tf.data pipelines with prefetching (AUTOTUNE) for GPU efficiency
* Logging, checkpoints, and plots for experiment tracking


### 4. Evaluation Metrics

* Accuracy is not sufficient in imbalanced datasets
* Metrics used:
	* Confusion Matrix – per-class errors
	* ROC Curve and AUC – threshold-independent separability
	* Classification Report (precision, recall, F1)


### 5. Conceptual AI Foundation

* CNNs as feature extractors: local receptive fields, parameter sharing
* Fine-tuning as domain adaptation
* Deepfake detection as a distribution shift problem (detecting artifacts not seen in natural photos)
* Project design emphasizes explainability and scalability (future: ResNet, EfficientNet, Vision Transformers)

---


---




## Model Performance


### Classification Metrics
<img src="https://github.com/sjsreehari/deepfake-detection-cnn/blob/main/result/Heatmap%20for%20Confusion%20Matrix.png" alt="Metrics" width="400" height="300" />

### ROC Curve
<img src="https://github.com/sjsreehari/deepfake-detection-cnn/blob/main/result/ROC%20Curve.png" alt="ROC Curve" width="400" height="300" />

### Confusion Matrix
<img src="https://github.com/sjsreehari/deepfake-detection-cnn/blob/main/result/Heatmap%20for%20Confusion%20Matrix.png" alt="Confusion Matrix" width="400" height="300" />

- **Classification Metrics:** Shows precision, recall, and F1-score for each class.
- **ROC Curve:** Area under curve (AUC = 0.977) shows excellent class separation.
- **Confusion Matrix:** Shows number of correct and incorrect predictions for each class.

---

## Results (Sample)

- **Training Accuracy:** ~92%
- **Validation Accuracy:** ~90%
- **ROC-AUC:** ~0.95
- **Classification Report:**
	- Precision, recall, F1-score for each class (see notebook output)
- **Confusion Matrix:**
	- Shows per-class errors (see notebook heatmap)
- **ROC Curve:**
	- Plots true positive rate vs. false positive rate
- **Precision-Recall Curve:**
	- Plots precision vs. recall, with average precision (AP) score

All metrics and plots are generated using scikit-learn and matplotlib, as shown in the notebook's evaluation cells.

---


---



## Setup & Run


1. **Install requirements:**
	```bash
	pip install -r requirements.txt
	```


2. **Download and split dataset:**
		- For Colab: Use the notebook cells to download from Kaggle and split into `train_data`, `val_data`, and `test_data` folders with `Fake` and `Real` subfolders.
		- For local: Run
			```bash
			python -m src.data.split_dataset
			```

3. **Train model:**
	```bash
	python src/train.py
	```

4. **Evaluate model:**
	```bash
	python src/evaluate.py
	```

Outputs are saved in:

```
outputs/
 ├── checkpoints/
 ├── logs/
 └── plots/
```

---

---


---


## Future Work

* Extend to video deepfake detection (temporal inconsistencies)
* Experiment with Vision Transformers (ViT) for long-range feature capture
* Add explainability tools (Grad-CAM, saliency maps)
* Deploy as an API service with FastAPI or Flask

---


---



## Documentation

- Source code: [`src/`](src/)
- Example notebook: [`notebooks/optimized_deepfake_detector_colab.ipynb`](notebooks/optimized_deepfake_detector_colab.ipynb) — shows real data workflow, splitting, augmentation, training, and evaluation
- Configuration: [`configs/train.yaml`](configs/train.yaml)

---

## Acknowledgements

* Dataset: [Kaggle – Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
* Backbone: MobileNetV2 – Google AI Research
* Libraries: TensorFlow, scikit-learn, matplotlib, seaborn

---
