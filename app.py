
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from skimage.feature import local_binary_pattern

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
IMAGE_SIZE = (128, 128)

# Avocado branding
st.set_page_config(
    page_title="Avocado Disease Classifier",
    page_icon="🥑",
    layout="wide",
)

st.title("🥑 Avocado Disease Classification")
st.write(
    "Upload an avocado image and compare predictions from "
    "classical machine learning and deep learning."
)

st.info(
    "This application classifies avocado images into the "
    "four dataset classes: Blackspot, Canker, Fresh, and Greening."
)


@st.cache_resource
def load_models():
    required = [
        MODEL_DIR / "classical_model.joblib",
        MODEL_DIR / "cnn_model.keras",
        MODEL_DIR / "class_names.json",
        MODEL_DIR / "metrics.json",
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
        )

    classical = joblib.load(required[0])
    cnn = tf.keras.models.load_model(
        required[1], compile=False
    )

    with open(required[2], encoding="utf-8") as file:
        classes = json.load(file)

    with open(required[3], encoding="utf-8") as file:
        metrics = json.load(file)

    return classical, cnn, classes, metrics


def extract_features(image_rgb):
    """Extract the same color and texture features used in training."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, IMAGE_SIZE)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    features = []

    for channel in range(3):
        hist = cv2.calcHist(
            [hsv], [channel], None, [16], [0, 256]
        )
        features.extend(
            cv2.normalize(hist, hist).flatten()
        )

    hist_a = cv2.calcHist(
        [lab], [1], None, [16], [0, 256]
    )
    features.extend(
        cv2.normalize(hist_a, hist_a).flatten()
    )

    lbp = local_binary_pattern(
        gray, P=8, R=1, method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 12),
        range=(0, 11),
    )

    hist = hist.astype(np.float32)
    hist /= max(hist.sum(), 1)
    features.extend(hist)

    return np.asarray(features, dtype=np.float32)


def show_probabilities(classes, probabilities):
    df = pd.DataFrame({
        "Class": classes,
        "Model probability (%)": np.asarray(probabilities) * 100,
    }).sort_values(
        "Model probability (%)", ascending=False
    )

    st.dataframe(
        df.style.format({"Model probability (%)": "{:.2f}%"}),
        hide_index=True,
        use_container_width=True,
    )


try:
    classical, cnn, classes, metrics = load_models()
except Exception as exc:
    st.error(f"Unable to load the trained models: {exc}")
    st.warning(
        "Train and save both models using train_models.py, "
        "then restart this application."
    )
    st.stop()


st.sidebar.header("🥑 About this application")
st.sidebar.write(
    "Avocado disease classification using computer vision "
    "and a convolutional neural network."
)
st.sidebar.write("Classes: " + ", ".join(classes))

uploaded = st.file_uploader(
    "Upload an avocado image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Choose an image from your computer.",
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        image_rgb = np.asarray(image)

        st.image(
            image,
            caption="Uploaded avocado image",
            width=400,
        )

        if st.button(
            "🥑 Classify Avocado",
            type="primary",
        ):
            with st.spinner("Analyzing avocado image..."):
                # Classical ML prediction
                features = extract_features(
                    image_rgb
                ).reshape(1, -1)

                classical_probs = classical.predict_proba(
                    features
                )[0]

                classical_classes = classical.classes_.tolist()
                classical_label = classical_classes[
                    int(np.argmax(classical_probs))
                ]

                # CNN prediction
                cnn_image = cv2.resize(
                    image_rgb, IMAGE_SIZE
                ).astype(np.float32) / 255.0

                cnn_probs = cnn.predict(
                    np.expand_dims(cnn_image, axis=0),
                    verbose=0,
                )[0]

                cnn_label = classes[
                    int(np.argmax(cnn_probs))
                ]

            st.divider()
            st.header("🥑 Avocado Classification Results")

            left, right = st.columns(2)

            with left:
                st.subheader("Classical ML")
                st.success(f"Prediction: {classical_label}")
                show_probabilities(
                    classical_classes, classical_probs
                )

            with right:
                st.subheader("Deep Learning CNN")
                st.success(f"Prediction: {cnn_label}")
                show_probabilities(classes, cnn_probs)

            if classical_label == cnn_label:
                st.success(
                    "Both models predict the same class."
                )
            else:
                st.warning(
                    "The models predict different classes. "
                    "Review the image and test-set metrics."
                )

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")


st.divider()
st.header("📊 Model Performance Comparison")

st.caption(
    "Metrics below are from the held-out test dataset, "
    "not from the currently uploaded image."
)

comparison = pd.DataFrame([
    {
        "Model": "Classical CV + Random Forest",
        "Accuracy": metrics["classical"]["accuracy"] * 100,
        "Precision": metrics["classical"]["precision_macro"] * 100,
        "Recall": metrics["classical"]["recall_macro"] * 100,
        "F1-score": metrics["classical"]["f1_macro"] * 100,
    },
    {
        "Model": "CNN",
        "Accuracy": metrics["cnn"]["accuracy"] * 100,
        "Precision": metrics["cnn"]["precision_macro"] * 100,
        "Recall": metrics["cnn"]["recall_macro"] * 100,
        "F1-score": metrics["cnn"]["f1_macro"] * 100,
    },
])

st.dataframe(
    comparison.style.format({
        "Accuracy": "{:.2f}%",
        "Precision": "{:.2f}%",
        "Recall": "{:.2f}%",
        "F1-score": "{:.2f}%",
    }),
    hide_index=True,
    use_container_width=True,
)

st.bar_chart(
    comparison.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1-score"]
    ]
)

st.subheader("Confusion Matrices")

left, right = st.columns(2)

with left:
    st.write("**Classical ML**")
    st.dataframe(
        pd.DataFrame(
            metrics["classical"]["confusion_matrix"],
            index=classes,
            columns=classes,
        ),
        use_container_width=True,
    )

with right:
    st.write("**CNN**")
    st.dataframe(
        pd.DataFrame(
            metrics["cnn"]["confusion_matrix"],
            index=classes,
            columns=classes,
        ),
        use_container_width=True,
    )

st.caption(
    "For educational purposes only. Predictions depend on "
    "the training dataset and are not a substitute for "
    "expert agricultural diagnosis."
)
