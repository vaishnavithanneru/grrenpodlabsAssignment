
import json
import os
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf

from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Orange Dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = BASE_DIR / "models"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def image_paths(directory):
    """Read image paths from class subfolders."""
    if not directory.exists():
        raise FileNotFoundError(f"Folder not found: {directory}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = []

    for class_dir in sorted(directory.iterdir()):
        if not class_dir.is_dir():
            continue

        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in extensions:
                records.append((str(path), class_dir.name))

    if not records:
        raise ValueError(f"No images found in {directory}")

    return records


# --------------------------------------------------
# Approach 1: Classical CV + Random Forest
# --------------------------------------------------
def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    # Color histograms: HSV channels
    for channel in range(3):
        hist = cv2.calcHist(
            [hsv], [channel], None, [16], [0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # Lab a-channel: green-to-red color information
    hist_a = cv2.calcHist(
        [lab], [1], None, [16], [0, 256]
    )
    hist_a = cv2.normalize(hist_a, hist_a).flatten()
    features.extend(hist_a)

    # LBP texture histogram
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    texture_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 12),
        range=(0, 11),
    )
    texture_hist = texture_hist.astype(np.float32)
    texture_hist /= max(texture_hist.sum(), 1)
    features.extend(texture_hist)

    return np.asarray(features, dtype=np.float32)


def train_classical(train_records, test_records, class_names):
    print("\nTraining classical model...")

    X_train = np.array([
        extract_features(path) for path, _ in train_records
    ])
    y_train = np.array([
        label for _, label in train_records
    ])

    X_test = np.array([
        extract_features(path) for path, _ in test_records
    ])
    y_test = np.array([
        label for _, label in test_records
    ])

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    joblib.dump(model, MODEL_DIR / "classical_model.joblib")

    return evaluate_model(
        "Classical CV + Random Forest",
        y_test,
        predictions,
        class_names,
        model.predict_proba(X_test),
        model.classes_.tolist(),
    )


# --------------------------------------------------
# Approach 2: Deep learning CNN
# --------------------------------------------------
def train_cnn(class_names):
    print("\nLoading CNN datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=class_names,
        shuffle=True,
        seed=SEED,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=class_names,
        shuffle=False,
    )

    # Normalize pixels and improve input throughput.
    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(
        lambda x, y: (normalization(x), y)
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (normalization(x), y)
    ).prefetch(tf.data.AUTOTUNE)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        augmentation,

        tf.keras.layers.Conv2D(
            32, 3, activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64, 3, activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128, 3, activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(
            len(class_names), activation="softmax"
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "cnn_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # Reserve 20% of the training images for validation.
    # The separate test set is used only for final evaluation.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=class_names,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        shuffle=True,
    ).map(lambda x, y: (normalization(x), y)).prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=class_names,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        shuffle=True,
    ).map(lambda x, y: (normalization(x), y)).prefetch(
        tf.data.AUTOTUNE
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save the best validation checkpoint.
    best_model = tf.keras.models.load_model(
        MODEL_DIR / "cnn_best.keras",
        compile=False,
    )
    best_model.save(MODEL_DIR / "cnn_model.keras")

    probabilities = best_model.predict(test_ds, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    y_test = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    return evaluate_model(
        "CNN",
        np.array(class_names)[y_test],
        np.array(class_names)[predictions],
        class_names,
        probabilities,
        class_names,
    )


# --------------------------------------------------
# Shared evaluation
# --------------------------------------------------
def evaluate_model(name, y_true, y_pred, class_names,
                   probabilities, probability_classes):
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=class_names
        ).tolist(),
    }

    print(f"\n{name}")
    print(classification_report(
        y_true, y_pred, labels=class_names, zero_division=0
    ))
    print("Confusion matrix:")
    print(result["confusion_matrix"])

    return result


def main():
    train_records = image_paths(TRAIN_DIR)
    test_records = image_paths(TEST_DIR)

    class_names = sorted({
        label for _, label in train_records
    })
    test_classes = {label for _, label in test_records}

    if len(class_names) < 2:
        raise ValueError("At least two training classes are required.")

    if set(class_names) != test_classes:
        raise ValueError(
            f"Training classes {class_names} do not match "
            f"test classes {sorted(test_classes)}"
        )

    # Ensure every class has enough training images for validation.
    for name in class_names:
        count = sum(label == name for _, label in train_records)
        if count < 5:
            raise ValueError(
                f"Class {name} has only {count} training images. "
                "Add more images before using the 80/20 split."
            )

    with open(MODEL_DIR / "class_names.json", "w") as file:
        json.dump(class_names, file, indent=2)

    classical_metrics = train_classical(
        train_records, test_records, class_names
    )
    cnn_metrics = train_cnn(class_names)

    metrics = {
        "classes": class_names,
        "test_image_count": len(test_records),
        "classical": classical_metrics,
        "cnn": cnn_metrics,
    }

    with open(MODEL_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print("\nTraining and evaluation completed.")
    print(f"Saved models and metrics in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
