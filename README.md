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

---

## 🖥️ Running in VS Code – Step-by-Step

This section covers everything you need to open, train, and run the web app entirely inside **Visual Studio Code**.

### Prerequisites

| Requirement | Download / Install |
|-------------|-------------------|
| Python 3.8–3.11 | https://www.python.org/downloads/ (TF 2.15 does not support 3.12+) |
| Visual Studio Code | https://code.visualstudio.com/ |
| Git | https://git-scm.com/downloads |

**Recommended VS Code extensions** (install from the Extensions panel `Ctrl+Shift+X`):

- **Python** (`ms-python.python`) – Python language support, IntelliSense, linting
- **Jupyter** (`ms-toolsai.jupyter`) – Run `.ipynb` notebooks directly inside VS Code
- **Pylance** (`ms-python.vscode-pylance`) – Fast type checking and auto-complete

---

### Step 1 – Clone and Open the Project

Open a terminal on your machine and run:

```bash
git clone https://github.com/khushiraj29/Lightweight-CNN-Based-Disaster-Detection-Framework-for-Resource-Constrained-Environments.git
cd Lightweight-CNN-Based-Disaster-Detection-Framework-for-Resource-Constrained-Environments
code .
```

> **Tip:** The `code .` command opens the current folder directly in VS Code.  
> If `code` is not recognised, open VS Code manually and choose **File → Open Folder**, then select the cloned folder.

---

### Step 2 – Create a Virtual Environment

Open the **integrated terminal** in VS Code with `` Ctrl+` `` (backtick) and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

After activation you should see `(venv)` at the start of your terminal prompt.

> **Select the interpreter in VS Code:** Press `Ctrl+Shift+P`, type  
> `Python: Select Interpreter`, and choose the one that shows `venv`.

---

### Step 3 – Install Dependencies

With the virtual environment active, run:

```bash
pip install -r requirements.txt
```

This installs TensorFlow, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Pillow, and Streamlit.  
Installation may take a few minutes depending on your internet speed.

---

### Step 4 – Prepare Your Dataset

Place your disaster images inside the `data/` folder, one sub-folder per class:

```
data/
├── earthquake_damage/   ← put earthquake images here
├── fire/                ← put fire images here
├── flood/               ← put flood images here
├── landslide/           ← put landslide images here
└── normal/              ← put normal scene images here
```

---

### Step 5 – Train the Model (Jupyter Notebook)

1. In the **Explorer** panel on the left, open `notebooks/training.ipynb`.
2. VS Code will open the notebook in its built-in Jupyter editor.
3. Click **"Select Kernel"** (top-right of the notebook) → choose your `venv` Python interpreter.
4. Run all cells in order:
   - Click **"Run All"** at the top of the notebook, **or**
   - Press `Shift+Enter` to run one cell at a time.

The trained model is saved automatically to:
- `models/disaster_model.h5` (Keras)
- `models/disaster_model.tflite` (TFLite, quantized)

> **Alternative:** If you prefer the classic Jupyter browser UI, run in the terminal:
> ```bash
> jupyter notebook notebooks/training.ipynb
> ```

---

### Step 6 – Run the Streamlit Web App

In the VS Code integrated terminal, run:

```bash
cd app
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).  
Hold `Ctrl` and click the link, or open it in any browser.

You should see the **Disaster Detection AI** app where you can:
- Upload a JPEG or PNG image
- Switch between Keras and TFLite inference
- View the predicted class, confidence score, and per-class probability bars

To stop the app press `Ctrl+C` in the terminal.

---

### Quick-Reference Command Cheatsheet

```bash
# 1. Clone
git clone https://github.com/khushiraj29/Lightweight-CNN-Based-Disaster-Detection-Framework-for-Resource-Constrained-Environments.git
cd Lightweight-CNN-Based-Disaster-Detection-Framework-for-Resource-Constrained-Environments

# 2. Open in VS Code
code .

# 3. Create & activate virtual environment
python -m venv venv          # Windows
python3 -m venv venv         # macOS/Linux

venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the model (browser notebook)
jupyter notebook notebooks/training.ipynb

# 6. Run the web app
cd app
streamlit run app.py
```

---

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `code .` not found | Open VS Code → `Ctrl+Shift+P` → `Shell Command: Install 'code' command in PATH` |
| `python` not found on Windows | Use `python3` or add Python to your system PATH during installation |
| `streamlit` not found | Make sure the venv is activated (`(venv)` visible in terminal) and re-run `pip install -r requirements.txt` |
| Notebook kernel not found | Press `Ctrl+Shift+P` → `Python: Select Interpreter` → choose the `venv` entry |
| TensorFlow install fails | Ensure Python 3.8–3.11; TF 2.15 does not support Python 3.12+ |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Wrong interpreter on Windows | In `.vscode/settings.json` change `venv/bin/python` → `venv/Scripts/python.exe` |

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
