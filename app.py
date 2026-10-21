
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

st.set_page_config(
    page_title="Orange Disease Classifier",
    page_icon="🍊",
    layout="wide",
)

st.title("🍊 Orange Disease Classification")
st.write(
    "Compare classical computer vision and deep learning "
    "for classifying orange images."
)

st.info(
    "Classes: Blackspot, Canker, Fresh, and Greening. "
    "Predictions are experimental and should be verified."
)


@st.cache_resource
def load_models():
    required = [
        MODEL_DIR / "classical_model.joblib",
        MODEL_DIR / "cnn_model.keras",
        MODEL_DIR / "class_names.json",
        MODEL_DIR / "metrics.json",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model files are missing: " + ", ".join(missing)
        )

    classical = joblib.load(required[0])
    cnn = tf.keras.models.load_model(required[1], compile=False)

    with open(required[2]) as file:
        classes = json.load(file)

    with open(required[3]) as file:
        metrics = json.load(file)

    return classical, cnn, classes, metrics


def extract_features(image_rgb):
    # OpenCV uses BGR ordering.
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
    features.extend(cv2.normalize(hist_a, hist_a).flatten())

    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, 12), range=(0, 11)
    )
    hist = hist.astype(np.float32)
    hist /= max(hist.sum(), 1)
    features.extend(hist)

    return np.asarray(features, dtype=np.float32)


def show_probabilities(classes, probabilities):
    df = pd.DataFrame({
        "Class": classes,
        "Probability (%)": np.asarray(probabilities) * 100,
    })
    df = df.sort_values("Probability (%)", ascending=False)
    st.dataframe(
        df.style.format({"Probability (%)": "{:.2f}%"}),
        hide_index=True,
        use_container_width=True,
    )


def show_metrics(metrics):
    cols = st.columns(4)

    fields = [
        ("Accuracy", "accuracy"),
        ("Macro precision", "precision_macro"),
        ("Macro recall", "recall_macro"),
        ("Macro F1-score", "f1_macro"),
    ]

    for col, (label, key) in zip(cols, fields):
        col.metric(label, f"{metrics[key] * 100:.2f}%")


try:
    classical, cnn, classes, metrics = load_models()
except Exception as exc:
    st.error(f"Could not load the trained models: {exc}")
    st.warning(
        "Run `python train_models.py` first, then restart the app."
    )
    st.stop()


uploaded = st.file_uploader(
    "Upload an orange image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_rgb = np.asarray(image)

    st.image(
        image,
        caption="Uploaded orange image",
        width=400,
    )

    if st.button("Classify and Compare", type="primary"):
        with st.spinner("Analyzing the image..."):
            features = extract_features(image_rgb).reshape(1, -1)

            classical_probs = classical.predict_proba(features)[0]
            classical_classes = classical.classes_.tolist()
            classical_label = classical_classes[
                int(np.argmax(classical_probs))
            ]

            cnn_input = cv2.resize(image_rgb, IMAGE_SIZE)
            cnn_input = cnn_input.astype(np.float32) / 255.0
            cnn_probs = cnn.predict(
                np.expand_dims(cnn_input, axis=0),
                verbose=0,
            )[0]
            cnn_label = classes[int(np.argmax(cnn_probs))]

        st.subheader("Prediction comparison")

        left, right = st.columns(2)

        with left:
            st.markdown("### Classical CV + Random Forest")
            st.success(f"Predicted class: **{classical_label}**")
            show_probabilities(classical_classes, classical_probs)

        with right:
            st.markdown("### Deep Learning CNN")
            st.success(f"Predicted class: **{cnn_label}**")
            show_probabilities(classes, cnn_probs)

        if classical_label == cnn_label:
            st.success("Both models predict the same class.")
        else:
            st.warning(
                "The models disagree. Review the image and "
                "compare their test-set evaluation metrics."
            )

st.divider()
st.header("Model performance on the test dataset")

st.caption(
    "These metrics are measured on the held-out test dataset, "
    "not on the image uploaded above."
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

st.subheader("Confusion matrices")

left, right = st.columns(2)

with left:
    st.markdown("**Classical model**")
    st.dataframe(
        pd.DataFrame(
            metrics["classical"]["confusion_matrix"],
            index=classes,
            columns=classes,
        ),
        use_container_width=True,
    )

with right:
    st.markdown("**CNN**")
    st.dataframe(
        pd.DataFrame(
            metrics["cnn"]["confusion_matrix"],
            index=classes,
            columns=classes,
        ),
        use_container_width=True,
    )
