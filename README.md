# Lightweight CNN-Based Disaster Detection Framework for Resource-Constrained Environments

A lightweight convolutional neural network (CNN) framework that classifies aerial and ground-level disaster images into five categories. The architecture uses **depthwise separable convolutions** (MobileNet-style) to minimise the parameter count (~320 K parameters at the default width), making it suitable for deployment on edge devices, embedded systems, and other resource-constrained environments.

---

## Disaster Classes

| Index | Class |
|-------|-------|
| 0 | Cyclone |
| 1 | Earthquake |
| 2 | Flood |
| 3 | Wildfire |
| 4 | No Disaster |

---

## Repository Structure

```
.
├── src/
│   ├── model.py       # Lightweight CNN architecture
│   ├── dataset.py     # Data-loading utilities (ImageFolder-compatible)
│   ├── train.py       # Training script
│   ├── test.py        # Evaluation / testing script
│   └── inference.py   # Single-image inference script
├── tests/
│   └── test_framework.py   # Unit & integration tests (pytest)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/khushiraj29/Lightweight-CNN-Based-Disaster-Detection-Framework-for-Resource-Constrained-Environments.git
cd Lightweight-CNN-Based-Disaster-Detection-Framework-for-Resource-Constrained-Environments

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

Organise your dataset in the standard **ImageFolder** layout:

```
data/
├── train/
│   ├── Cyclone/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── Earthquake/
│   ├── Flood/
│   ├── Wildfire/
│   └── No_Disaster/
├── val/
│   ├── Cyclone/
│   └── ...
└── test/
    ├── Cyclone/
    └── ...
```

Each subdirectory name becomes the class label. The names must be consistent across all splits.

> **Public datasets** you can use:
> - [AIDER – Aerial Image Dataset for Emergency Response](https://github.com/ckyrkou/AIDER)
> - [MEDIC – Disaster Image Dataset](https://crisisnlp.qcri.org/medic/index.html)
> - [Disaster Response (Kaggle)](https://www.kaggle.com/datasets/mikolajbabula/disaster-response-dataset)

---

## Training

```bash
python src/train.py \
    --data_dir data/ \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3
```

Key options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data` | Root directory with `train/` and `val/` sub-folders |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate (cosine annealing schedule) |
| `--width_multiplier` | `1.0` | Channel scaling factor – use `0.5` or `0.25` for smaller models |
| `--output_dir` | `checkpoints` | Where to save `.pth` checkpoints |

The best model (highest validation accuracy) is saved to `checkpoints/best_model.pth`.

---

## Testing a Trained Model

### Evaluate on the test set

```bash
python src/test.py \
    --data_dir data/ \
    --checkpoint checkpoints/best_model.pth
```

This prints a full **classification report** (precision, recall, F1-score per class) and overall accuracy.

#### Save a confusion matrix

```bash
python src/test.py \
    --data_dir data/ \
    --checkpoint checkpoints/best_model.pth \
    --save_confusion_matrix results/confusion_matrix.png
```

#### Save the report as JSON

```bash
python src/test.py \
    --data_dir data/ \
    --checkpoint checkpoints/best_model.pth \
    --save_report results/report.json
```

All `test.py` options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data` | Root data directory |
| `--checkpoint` | *(required)* | Path to `.pth` checkpoint |
| `--split` | `test` | Which split to evaluate (`train`, `val`, `test`) |
| `--batch_size` | `32` | Batch size |
| `--width_multiplier` | `1.0` | Must match the value used during training |
| `--save_confusion_matrix` | *(empty)* | Path to save confusion matrix PNG |
| `--save_report` | *(empty)* | Path to save JSON classification report |
| `--device` | *(auto)* | Force a specific device, e.g. `cpu` or `cuda` |

### Single-image inference

```bash
python src/inference.py \
    --image path/to/flood_image.jpg \
    --checkpoint checkpoints/best_model.pth
```

Example output:

```
Image: flood_image.jpg
Predicted class : Flood
Confidence      : 94.73%

Class probabilities:
  Flood           94.73%  ██████████████████
  Cyclone          2.41%
  No Disaster      1.87%
  Wildfire         0.64%
  Earthquake       0.35%
```

Save the result as JSON:

```bash
python src/inference.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output_json results/prediction.json
```

---

## Running the Unit Tests

The test suite validates the model architecture, data transforms, checkpoint loading, and inference pipeline using randomly generated data (no real images required).

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_framework.py::TestDepthwiseSeparableConv::test_output_shape_stride1 PASSED
tests/test_framework.py::TestDepthwiseSeparableConv::test_output_shape_stride2 PASSED
...
24 passed in 3.02s
```

---

## Model Architecture

`LightweightDisasterCNN` is a fully custom lightweight CNN inspired by MobileNetV1:

```
Input (3 × 224 × 224)
  └── Stem: Conv2d(3→32, k=3, s=2) + BN + ReLU6
  └── DS-Conv 32→64 (stride 1)
  └── DS-Conv 64→128 (stride 2)
  └── DS-Conv 128→128 (stride 1)
  └── DS-Conv 128→256 (stride 2)
  └── DS-Conv 256→256 (stride 1)
  └── DS-Conv 256→512 (stride 2)
  └── DS-Conv 512→512 (stride 1) × 2
  └── DS-Conv 512→256 (stride 2)
  └── AdaptiveAvgPool2d(1×1)
  └── Dropout + Linear(256 → num_classes)
```

`DS-Conv` = Depthwise Separable Convolution (depthwise + pointwise + BN + ReLU6).

Use the `width_multiplier` argument (e.g. `0.5`) to proportionally reduce all channel counts for an even smaller model:

| Width | Parameters |
|-------|-----------|
| 1.0 | ~320 K |
| 0.75 | ~182 K |
| 0.5 | ~83 K |
| 0.25 | ~22 K |

---

## License

This project is released under the MIT License.