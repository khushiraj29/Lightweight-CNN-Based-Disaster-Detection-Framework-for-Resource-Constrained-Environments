# Lightweight CNN-Based Disaster Detection Framework for Resource-Constrained Environments

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project develops an efficient deep learning model for **detecting disasters from images**, optimized for deployment on resource-constrained devices such as Raspberry Pi, drones, and mobile phones. Using **MobileNetV2** as a lightweight backbone, the framework classifies scenes into five categories: fire, flood, landslide, earthquake damage, and normal scenes, enabling rapid disaster response in low-power environments.

## Project Structure

```
├── data/
│   └── .gitkeep          # Placeholder for dataset directory
├── notebooks/
│   └── training.ipynb    # End-to-end model training notebook
├── models/
│   └── .gitkeep          # Placeholder for saved model files
├── app/
│   └── app.py            # Streamlit web application
├── requirements.txt      # Python dependencies
└── README.md
```

## Classes

| # | Class | Emoji | Description |
|---|-------|-------|-------------|
| 0 | earthquake_damage | 🏚️ | Collapsed buildings, rubble, structural damage |
| 1 | fire | 🔥 | Active fires, smoke, burn areas |
| 2 | flood | 🌊 | Flooded streets, submerged areas, water overflow |
| 3 | landslide | ⛰️ | Mudslides, debris flows, slope failures |
| 4 | normal | 🌳 | Unaffected scenes, baseline conditions |

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/khushiraj29/Lightweight-CNN-Based-Disaster-Detection-Framework-for-Resource-Constrained-Environments.git
cd Lightweight-CNN-Based-Disaster-Detection-Framework-for-Resource-Constrained-Environments
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Organize your images into the following subfolder structure inside the `data/` directory:

```
data/
├── earthquake_damage/
│   ├── img1.jpg
│   └── ...
├── fire/
│   ├── img1.jpg
│   └── ...
├── flood/
│   ├── img1.jpg
│   └── ...
├── landslide/
│   ├── img1.jpg
│   └── ...
└── normal/
    ├── img1.jpg
    └── ...
```

### 4. Train the model

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/training.ipynb
```

### 5. Run the Streamlit app

```bash
cd app
streamlit run app.py
```

## Model Architecture

The model uses **MobileNetV2** as a frozen feature extractor with a custom classification head optimized for the 5-class disaster detection task.

```
Input (224×224×3)
        │
MobileNetV2 (imagenet weights, frozen, no top)
        │
GlobalAveragePooling2D
        │
Dense(128, ReLU)
        │
Dropout(0.3)
        │
Dense(64, ReLU)
        │
Dropout(0.2)
        │
Dense(5, Softmax)
        │
Output (5 classes)
```

**Rationale:** MobileNetV2's depthwise separable convolutions reduce computation by up to 8–9× compared to standard convolutions, making it ideal for edge devices while maintaining strong representational power via transfer learning from ImageNet.

## Evaluation Metrics

The model is evaluated using the following metrics on the held-out validation set:

- **Accuracy** – Overall percentage of correctly classified images
- **Precision** (weighted) – Fraction of positive predictions that are correct
- **Recall** (weighted) – Fraction of actual positives correctly identified
- **F1-Score** (weighted) – Harmonic mean of precision and recall
- **Confusion Matrix** – Per-class prediction breakdown visualized as a heatmap

## Model Optimization

| Format | File | Typical Size | Avg Inference Time | Suitability |
|--------|------|-------------|-------------------|-------------|
| Keras (.h5) | `disaster_model.h5` | ~14 MB | ~45 ms | Development, GPU servers |
| TFLite (.tflite) | `disaster_model.tflite` | ~3.5 MB | ~12 ms | Raspberry Pi, mobile, drones |

TFLite quantization (post-training dynamic range quantization via `tf.lite.Optimize.DEFAULT`) reduces model size by ~75% and inference latency by ~3–4× with minimal accuracy loss.

## Deployment

The **Streamlit web app** (`app/app.py`) provides:

- Upload any JPEG/PNG image for real-time classification
- Choice between Keras (.h5) and TFLite (.tflite) inference engines
- Per-class confidence bar charts
- Inference time and model size metrics

**Example output:**

```
🔥 Fire detected – Confidence: 93.2%
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [MobileNetV2](https://arxiv.org/abs/1801.04381) – Sandler et al., Google Inc.
- [TensorFlow Lite](https://www.tensorflow.org/lite) – On-device ML framework
- [Streamlit](https://streamlit.io/) – Web app framework for ML
- Dataset images sourced from publicly available disaster image repositories
