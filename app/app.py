"""
Disaster Detection AI – Streamlit Web Application
===================================================
This app classifies uploaded images into one of five disaster categories
(earthquake_damage, fire, flood, landslide, normal) using a MobileNetV2-based
CNN model.  Two inference back-ends are supported:

* Keras (.h5)  – standard TensorFlow/Keras model, suitable for development.
* TFLite (.tflite) – quantized model for fast, low-memory inference.

Run from the app/ directory:
    streamlit run app.py
"""

import os
import time

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLASS_NAMES = ["earthquake_damage", "fire", "flood", "landslide", "normal"]

CLASS_EMOJIS = {
    "earthquake_damage": "🏚️",
    "fire": "🔥",
    "flood": "🌊",
    "landslide": "⛰️",
    "normal": "🌳",
}

CLASS_DISPLAY_NAMES = {
    "earthquake_damage": "Earthquake Damage",
    "fire": "Fire",
    "flood": "Flood",
    "landslide": "Landslide",
    "normal": "Normal",
}

CLASS_COLORS = {
    "earthquake_damage": "#8B4513",
    "fire": "#FF4500",
    "flood": "#1E90FF",
    "landslide": "#8B8000",
    "normal": "#228B22",
}

IMG_SIZE = (224, 224)

# Paths relative to the app/ directory
KERAS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/disaster_model.h5")
TFLITE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../models/disaster_model.tflite"
)

# ---------------------------------------------------------------------------
# Model loading (cached so the model is loaded only once per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_keras_model():
    """Load the saved Keras (.h5) model from disk.

    The model is cached with @st.cache_resource so it is instantiated only
    once per Streamlit session, avoiding repeated disk I/O and model
    compilation overhead.

    Returns
    -------
    tf.keras.Model
        The loaded Keras model ready for inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at KERAS_MODEL_PATH.
    """
    # Import TensorFlow lazily to keep start-up time low when TFLite is used
    import tensorflow as tf  # noqa: PLC0415

    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(
            f"Keras model not found at '{KERAS_MODEL_PATH}'. "
            "Please train the model first by running notebooks/training.ipynb."
        )
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    return model


@st.cache_resource
def load_tflite_model():
    """Load the TFLite (.tflite) model and allocate tensors.

    TFLite models are loaded via the TFLiteInterpreter which is optimised for
    low-memory, low-latency inference on edge devices.  Tensor allocation
    is performed once here and the interpreter is reused across calls.

    Returns
    -------
    tuple[tf.lite.Interpreter, dict, dict]
        * interpreter – the allocated TFLite interpreter
        * input_details – dict returned by get_input_details()[0]
        * output_details – dict returned by get_output_details()[0]

    Raises
    ------
    FileNotFoundError
        If the TFLite model file does not exist at TFLITE_MODEL_PATH.
    """
    import tensorflow as tf  # noqa: PLC0415

    if not os.path.exists(TFLITE_MODEL_PATH):
        raise FileNotFoundError(
            f"TFLite model not found at '{TFLITE_MODEL_PATH}'. "
            "Please train and convert the model first by running "
            "notebooks/training.ipynb."
        )
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def preprocess_image(image: Image.Image, target_size: tuple = IMG_SIZE) -> np.ndarray:
    """Pre-process a PIL image for model inference.

    Steps
    -----
    1. Convert to RGB (handles RGBA / grayscale uploads).
    2. Resize to ``target_size`` using high-quality Lanczos resampling.
    3. Normalise pixel values from [0, 255] → [0.0, 1.0] (float32).
    4. Add a batch dimension so the shape is (1, H, W, 3).

    Parameters
    ----------
    image : PIL.Image.Image
        Raw image uploaded by the user.
    target_size : tuple[int, int]
        (height, width) expected by the model.  Defaults to IMG_SIZE=(224,224).

    Returns
    -------
    np.ndarray
        Float32 array of shape (1, 224, 224, 3) in range [0, 1].
    """
    image = image.convert("RGB")  # ensure 3-channel RGB
    image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
    img_array = np.array(image, dtype=np.float32) / 255.0  # normalise
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim → (1,224,224,3)
    return img_array


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def predict_keras(model, img_array: np.ndarray):
    """Run a single-image inference pass with the Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded Keras model.
    img_array : np.ndarray
        Pre-processed image array of shape (1, 224, 224, 3).

    Returns
    -------
    tuple[np.ndarray, float]
        * predictions – softmax probability array of shape (5,)
        * inference_ms – wall-clock inference time in milliseconds
    """
    start = time.perf_counter()
    predictions = model.predict(img_array, verbose=0)[0]  # shape: (5,)
    inference_ms = (time.perf_counter() - start) * 1000.0
    return predictions, inference_ms


def predict_tflite(interpreter, input_details: dict, output_details: dict, img_array: np.ndarray):
    """Run a single-image inference pass with the TFLite interpreter.

    The TFLite interpreter is stateful: we set the input tensor, invoke the
    interpreter, then read the output tensor.

    Parameters
    ----------
    interpreter : tf.lite.Interpreter
        Allocated TFLite interpreter (from load_tflite_model()).
    input_details : dict
        Metadata for the model's input tensor.
    output_details : dict
        Metadata for the model's output tensor.
    img_array : np.ndarray
        Pre-processed image array of shape (1, 224, 224, 3).

    Returns
    -------
    tuple[np.ndarray, float]
        * predictions – softmax probability array of shape (5,)
        * inference_ms – wall-clock inference time in milliseconds
    """
    # Cast input to the dtype expected by the model (usually float32)
    input_data = img_array.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], input_data)

    start = time.perf_counter()
    interpreter.invoke()
    inference_ms = (time.perf_counter() - start) * 1000.0

    predictions = interpreter.get_tensor(output_details["index"])[0]  # shape: (5,)
    return predictions, inference_ms


# ---------------------------------------------------------------------------
# Main Streamlit UI
# ---------------------------------------------------------------------------


def main():
    """Entry point for the Streamlit application.

    Renders the full UI:
    * Page configuration and custom CSS
    * App header and subtitle
    * Sidebar: model selector, supported classes, model info
    * Main area: image uploader (left) + prediction results (right)
    """
    # ------------------------------------------------------------------
    # Page configuration
    # ------------------------------------------------------------------
    st.set_page_config(
        page_title="Disaster Detection AI",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ------------------------------------------------------------------
    # Custom CSS
    # ------------------------------------------------------------------
    st.markdown(
        """
        <style>
        /* Prediction result box */
        .prediction-box {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            border-left: 5px solid #e94560;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .prediction-title {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 4px;
        }
        .prediction-subtitle {
            font-size: 1rem;
            color: #a0aec0;
        }
        /* Metric cards */
        .metric-card {
            background-color: #1e293b;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("🌍 Disaster Detection AI")
    st.markdown(
        "Upload an image to classify it as **earthquake damage**, **fire**, "
        "**flood**, **landslide**, or **normal** using a lightweight MobileNetV2 CNN."
    )
    st.divider()

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Settings")

        model_type = st.radio(
            "Inference engine",
            options=["Keras (.h5)", "TFLite (.tflite)"],
            index=0,
            help=(
                "Keras: standard TF model, higher accuracy. "
                "TFLite: quantized, faster & smaller – ideal for edge devices."
            ),
        )

        st.divider()
        st.subheader("📋 Supported Classes")
        for cls in CLASS_NAMES:
            st.markdown(
                f"{CLASS_EMOJIS[cls]} **{CLASS_DISPLAY_NAMES[cls]}**"
            )

        st.divider()
        st.subheader("🧠 Model Info")
        st.markdown(
            """
            - **Backbone:** MobileNetV2 (ImageNet)
            - **Input size:** 224 × 224 px
            - **Output classes:** 5
            - **Head:** GAP → Dense(128) → Dense(64) → Dense(5)
            - **Optimisation:** TFLite dynamic quantization
            """
        )

    # ------------------------------------------------------------------
    # Main columns: upload (left) | results (right)
    # ------------------------------------------------------------------
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a JPG or PNG image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPEG, PNG.  Images are resized to 224×224 internally.",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True, output_format="auto")
            st.markdown(
                f"**File:** `{uploaded_file.name}`  \n"
                f"**Size:** {image.size[0]} × {image.size[1]} px  \n"
                f"**Mode:** {image.mode}"
            )

    with col_right:
        st.subheader("🎯 Prediction Results")

        if uploaded_file is None:
            # Placeholder when no image has been uploaded yet
            st.info(
                "Upload an image on the left to see the prediction results here.\n\n"
                "**Example output:**\n\n> 🔥 Fire detected – Confidence: 93%"
            )
        else:
            # ----------------------------------------------------------------
            # Load model and run inference
            # ----------------------------------------------------------------
            img_array = preprocess_image(image)

            try:
                with st.spinner("Running inference…"):
                    if model_type == "Keras (.h5)":
                        model = load_keras_model()
                        predictions, inference_ms = predict_keras(model, img_array)
                        model_label = "Keras (.h5)"
                    else:
                        interpreter, input_details, output_details = load_tflite_model()
                        predictions, inference_ms = predict_tflite(
                            interpreter, input_details, output_details, img_array
                        )
                        model_label = "TFLite (.tflite)"
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()

            # ----------------------------------------------------------------
            # Parse results
            # ----------------------------------------------------------------
            predicted_idx = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(predictions[predicted_idx]) * 100.0
            emoji = CLASS_EMOJIS[predicted_class]
            display_name = CLASS_DISPLAY_NAMES[predicted_class]
            color = CLASS_COLORS[predicted_class]

            # ----------------------------------------------------------------
            # Styled prediction box
            # ----------------------------------------------------------------
            st.markdown(
                f"""
                <div class="prediction-box" style="border-left-color: {color};">
                    <div class="prediction-title">{emoji} {display_name}</div>
                    <div class="prediction-subtitle">Confidence: {confidence:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Success banner
            st.success(
                f"{emoji} **{display_name} detected** – Confidence: {confidence:.1f}%"
            )

            # Three metric cards
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{confidence:.1f}%")
            m2.metric("Inference time", f"{inference_ms:.1f} ms")
            m3.metric("Model", model_label)

            # ----------------------------------------------------------------
            # Per-class probability bars
            # ----------------------------------------------------------------
            st.markdown("#### Class probabilities")
            for idx, cls in enumerate(CLASS_NAMES):
                prob = float(predictions[idx])
                bar_color = CLASS_COLORS[cls]
                label = f"{CLASS_EMOJIS[cls]} {CLASS_DISPLAY_NAMES[cls]}"
                st.markdown(
                    f"<span style='color:{bar_color}; font-weight:600'>{label}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(prob, text=f"{prob * 100:.1f}%")


if __name__ == "__main__":
    main()
