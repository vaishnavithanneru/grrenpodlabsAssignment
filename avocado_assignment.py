import argparse
import json
import random
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from skimage.feature import local_binary_pattern

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Project directory is the directory containing this Python file.
PROJECT_DIR = Path(__file__).resolve().parent

# YOUR DATASET PATH
DATA_DIR = PROJECT_DIR / "Orange Dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Output directories
OUTPUT_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Keep these class names consistent with your dataset folders.
CLASS_NAMES = [
    "blackspot",
    "canker",
    "fresh",
    "grenning",
]

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001
RANDOM_STATE = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


# ============================================================
# 2. REPRODUCIBILITY AND FOLDER SETUP
# ============================================================

def set_seed(seed=RANDOM_STATE):
    """Set random seeds to make experiments more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_directories():
    """Create model and results folders if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. DATASET DISCOVERY
# ============================================================

def collect_images(folder):
    """
    Read images from:
        folder/class_name/image.jpg

    Returns a list of (image_path, class_label) pairs.
    """
    records = []

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            "Check your Orange Dataset folder location."
        )

    for class_name in CLASS_NAMES:
        class_folder = folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_folder}"
            )

        image_paths = sorted(
            path for path in class_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            raise ValueError(
                f"No images found in: {class_folder}"
            )

        for image_path in image_paths:
            records.append((str(image_path), class_name))

    return records


def validate_images(records):
    """Check whether image files can be opened."""
    valid_records = []
    invalid_count = 0

    for image_path, label in records:
        try:
            with Image.open(image_path) as image:
                image.verify()
            valid_records.append((image_path, label))
        except Exception:
            print(f"Warning: unreadable image skipped: {image_path}")
            invalid_count += 1

    print(f"Unreadable images skipped: {invalid_count}")
    return valid_records


def load_dataset_records():
    """Load the supplied train and test folders."""
    print("\nReading dataset...")
    print(f"Training folder: {TRAIN_DIR}")
    print(f"Testing folder:  {TEST_DIR}")

    train_records = validate_images(collect_images(TRAIN_DIR))
    test_records = validate_images(collect_images(TEST_DIR))

    if not train_records or not test_records:
        raise ValueError("Training or testing dataset is empty.")

    train_df = pd.DataFrame(train_records, columns=["path", "label"])
    test_df = pd.DataFrame(test_records, columns=["path", "label"])

    print("\nTraining image counts:")
    print(train_df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0))

    print("\nTesting image counts:")
    print(test_df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0))

    return train_df, test_df


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def perform_eda(train_df, test_df):
    """Generate dataset summaries and sample-image visualizations."""
    print("\n========== DATASET SUMMARY ==========")
    print(f"Total training images: {len(train_df)}")
    print(f"Total testing images:  {len(test_df)}")
    print(f"Number of classes:     {len(CLASS_NAMES)}")
    print(f"Classes: {CLASS_NAMES}")

    counts = train_df["label"].value_counts().reindex(
        CLASS_NAMES, fill_value=0
    )

    summary = pd.DataFrame({
        "class": CLASS_NAMES,
        "training_images": counts.values,
        "testing_images": (
            test_df["label"].value_counts()
            .reindex(CLASS_NAMES, fill_value=0).values
        ),
    })

    summary.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    print("\nDataset summary:")
    print(summary.to_string(index=False))

    # Class distribution plot
    plt.figure(figsize=(9, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Training Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Sample image grid
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(16, 4))

    for index, class_name in enumerate(CLASS_NAMES):
        samples = train_df[train_df["label"] == class_name]

        if len(samples) == 0:
            axes[index].set_title(f"{class_name}\nNo images")
            axes[index].axis("off")
            continue

        image_path = samples.iloc[0]["path"]

        with Image.open(image_path) as image:
            axes[index].imshow(image.convert("RGB"))

        axes[index].set_title(class_name)
        axes[index].axis("off")

    plt.suptitle("Sample Images from Each Class")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_images.png", dpi=150)
    plt.close()

    print(f"\nEDA outputs saved in: {RESULTS_DIR}")


# ============================================================
# 5. TRAINING / VALIDATION SPLIT
# ============================================================

def split_training_data(train_df):
    """
    Split the provided training set into:
        80% training
        20% validation

    The supplied test folder is kept separate.
    """
    class_counts = train_df["label"].value_counts()

    if (class_counts < 2).any():
        raise ValueError(
            "Every class needs at least two training images "
            "for a stratified train/validation split."
        )

    training_df, validation_df = train_test_split(
        train_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train_df["label"],
    )

    training_df = training_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)

    print("\n========== DATA SPLIT ==========")
    print(f"Training subset:   {len(training_df)}")
    print(f"Validation subset: {len(validation_df)}")
    print("Test set:          unchanged; loaded from Orange Dataset/test")

    return training_df, validation_df


# ============================================================
# 6. CLASSICAL APPROACH: COLOR + LBP + SVM
# ============================================================

def extract_classical_features(image_path):
    """
    Extract:
      1. CIELAB color histograms
      2. Local Binary Pattern (LBP) texture histogram

    Color and texture features are combined into one feature vector.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (128, 128))

    # Convert BGR to CIELAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    color_features = []

    # Extract histogram from each Lab channel
    for channel_index in range(3):
        histogram = cv2.calcHist(
            [lab_image],
            [channel_index],
            None,
            [32],
            [0, 256],
        )

        histogram = cv2.normalize(histogram, histogram).flatten()
        color_features.extend(histogram)

    # Convert to grayscale for texture analysis
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # LBP texture features
    radius = 2
    number_of_points = 8 * radius

    lbp_image = local_binary_pattern(
        gray_image,
        number_of_points,
        radius,
        method="uniform",
    )

    lbp_histogram, _ = np.histogram(
        lbp_image.ravel(),
        bins=np.arange(0, number_of_points + 3),
        range=(0, number_of_points + 2),
    )

    lbp_histogram = lbp_histogram.astype("float32")
    lbp_histogram /= (lbp_histogram.sum() + 1e-7)

    return np.concatenate([
        np.asarray(color_features, dtype=np.float32),
        lbp_histogram,
    ])


def create_feature_matrix(dataframe):
    """Convert a dataframe of image paths into feature vectors."""
    features = []
    labels = []

    for index, row in dataframe.iterrows():
        try:
            feature_vector = extract_classical_features(row["path"])
            features.append(feature_vector)
            labels.append(row["label"])
        except Exception as error:
            print(f"Feature extraction failed for {row['path']}: {error}")

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(dataframe)} images")

    if not features:
        raise ValueError("No classical features could be extracted.")

    return np.asarray(features), np.asarray(labels)


def train_classical_model(training_df, validation_df, test_df):
    """Train and evaluate the SVM model."""
    print("\n========== APPROACH 1: SVM ==========")
    print("Extracting training features...")

    x_train, y_train = create_feature_matrix(training_df)

    print("\nExtracting validation features...")
    x_validation, y_validation = create_feature_matrix(validation_df)

    print("\nExtracting test features...")
    x_test, y_test = create_feature_matrix(test_df)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    print("\nTraining SVM...")
    model.fit(x_train, y_train)

    validation_predictions = model.predict(x_validation)
    test_predictions = model.predict(x_test)

    print("\nSVM validation results:")
    print(classification_report(
        y_validation,
        validation_predictions,
        labels=CLASS_NAMES,
        zero_division=0,
    ))

    metrics = evaluate_model(
        "SVM",
        y_test,
        test_predictions,
    )

    joblib.dump(model, OUTPUT_DIR / "classical_svm.joblib")

    print(f"SVM model saved: {OUTPUT_DIR / 'classical_svm.joblib'}")

    return metrics


# ============================================================
# 7. DEEP LEARNING DATASET
# ============================================================

class OrangeImageDataset(Dataset):
    """Dataset class that loads image files and their labels."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_to_index = {
            class_name: index
            for index, class_name in enumerate(CLASS_NAMES)
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = row["path"]
        label = row["label"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_index = self.label_to_index[label]

        return image, label_index


def get_image_transforms():
    """Define training augmentation and inference transforms."""
    weights = MobileNet_V2_Weights.DEFAULT
    normalization = weights.transforms()

    training_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return training_transform, evaluation_transform


# ============================================================
# 8. MOBILENETV2 MODEL
# ============================================================

def build_mobilenet(pretrained=True):
    """
    Create MobileNetV2 for four-class classification.

    During training, pretrained ImageNet weights are used.
    During inference, the saved state dictionary is loaded
    into the same model architecture.
    """
    if pretrained:
        model = models.mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT
        )
    else:
        model = models.mobilenet_v2(weights=None)

    # Freeze the pretrained feature extractor
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace final classification layer
    input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, len(CLASS_NAMES)),
    )

    return model


# ============================================================
# 9. DEEP LEARNING TRAINING
# ============================================================

def train_deep_learning_model(training_df, validation_df, test_df):
    """Train MobileNetV2 and evaluate it on the held-out test set."""
    print("\n========== APPROACH 2: MOBILENETV2 ==========")
    print(f"Using device: {DEVICE}")

    train_transform, eval_transform = get_image_transforms()

    train_dataset = OrangeImageDataset(
        training_df, transform=train_transform
    )

    validation_dataset = OrangeImageDataset(
        validation_df, transform=eval_transform
    )

    test_dataset = OrangeImageDataset(
        test_df, transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Download ImageNet weights on the first run if not cached.
    model = build_mobilenet(pretrained=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")
    patience = 3
    patience_counter = 0

    best_model_path = OUTPUT_DIR / "mobilenetv2.pth"
    history = []

    for epoch in range(EPOCHS):
        # -------------------- Training --------------------
        model.train()
        training_loss_total = 0.0
        training_correct = 0
        training_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            training_correct += (predictions == labels).sum().item()
            training_total += labels.size(0)

        training_loss = training_loss_total / max(training_total, 1)
        training_accuracy = training_correct / max(training_total, 1)

        # -------------------- Validation --------------------
        model.eval()
        validation_loss_total = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                validation_loss_total += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)
                validation_correct += (
                    predictions == labels
                ).sum().item()

                validation_total += labels.size(0)

        validation_loss = (
            validation_loss_total / max(validation_total, 1)
        )

        validation_accuracy = (
            validation_correct / max(validation_total, 1)
        )

        history.append({
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "training_accuracy": training_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {training_loss:.4f} | "
            f"Train Acc: {training_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        # Save the model with the lowest validation loss.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_counter = 0

            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping: validation loss did not improve.")
            break

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / "mobilenet_training_history.csv",
        index=False,
    )

    # Load the best checkpoint before testing.
    state_dict = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(predictions.tolist())
            y_true.extend(labels.numpy().tolist())

    y_true_names = [CLASS_NAMES[index] for index in y_true]
    y_pred_names = [CLASS_NAMES[index] for index in y_pred]

    print("\nMobileNetV2 test results:")
    print(classification_report(
        y_true_names,
        y_pred_names,
        labels=CLASS_NAMES,
        zero_division=0,
    ))

    metrics = evaluate_model(
        "MobileNetV2",
        y_true_names,
        y_pred_names,
    )

    with open(OUTPUT_DIR / "class_names.json", "w") as file:
        json.dump(CLASS_NAMES, file, indent=4)

    print(f"MobileNetV2 model saved: {best_model_path}")

    return metrics


# ============================================================
# 10. EVALUATION AND CONFUSION MATRIX
# ============================================================

def evaluate_model(model_name, y_true, y_pred):
    """Calculate test metrics and save a confusion matrix."""
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0,
    )

    print(f"\n{model_name} test metrics:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1-score:    {f1:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        zero_division=0,
    ))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{model_name} Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    threshold = cm.max() / 2 if cm.size else 0

    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            ax.text(
                column_index,
                row_index,
                str(cm[row_index, column_index]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[row_index, column_index] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(
        RESULTS_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


# ============================================================
# 11. COMPARE BOTH APPROACHES
# ============================================================

def save_model_comparison(metrics_list):
    """Save model metrics in a CSV and print a comparison."""
    comparison = pd.DataFrame(metrics_list)

    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    # Select by macro F1, a useful metric for multi-class data.
    best_model = comparison.loc[
        comparison["macro_f1"].idxmax()
    ]

    print(
        "\nModel with the highest test macro F1-score: "
        f"{best_model['model']}"
    )

    print(f"Comparison saved to: {RESULTS_DIR / 'model_comparison.csv'}")


# ============================================================
# 12. PREDICT A NEW IMAGE USING BOTH MODELS
# ============================================================

def predict_new_image(image_path):
    """Predict the class of a new image using both trained models."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\nPredicting image: {image_path}")

    # -------------------- SVM prediction --------------------
    svm_path = OUTPUT_DIR / "classical_svm.joblib"

    if not svm_path.exists():
        raise FileNotFoundError(
            "SVM model not found. Train the models first."
        )

    svm_model = joblib.load(svm_path)

    feature_vector = extract_classical_features(image_path)
    svm_prediction = svm_model.predict(
        feature_vector.reshape(1, -1)
    )[0]

    print(f"SVM prediction: {svm_prediction}")

    if hasattr(svm_model, "predict_proba"):
        svm_probabilities = svm_model.predict_proba(
            feature_vector.reshape(1, -1)
        )[0]

        print("SVM probabilities:")
        for class_name, probability in zip(
            svm_model.classes_, svm_probabilities
        ):
            print(f"  {class_name}: {probability:.4f}")

    # -------------------- MobileNetV2 prediction --------------------
    deep_model_path = OUTPUT_DIR / "mobilenetv2.pth"

    if not deep_model_path.exists():
        raise FileNotFoundError(
            "MobileNetV2 model not found. Train the models first."
        )

    deep_model = build_mobilenet(pretrained=False).to(DEVICE)

    state_dict = torch.load(
        deep_model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    deep_model.load_state_dict(state_dict)
    deep_model.eval()

    _, eval_transform = get_image_transforms()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = deep_model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = int(probabilities.argmax().item())

    print(f"MobileNetV2 prediction: {CLASS_NAMES[predicted_index]}")
    print("MobileNetV2 probabilities:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {probabilities[index].item():.4f}")


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orange image classification: SVM vs MobileNetV2"
    )

    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to a new image to classify using both trained models",
    )

    args = parser.parse_args()

    set_seed()
    create_output_directories()

    print("========================================")
    print("ORANGE IMAGE CLASSIFICATION PROJECT")
    print("========================================")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_DIR.resolve()}")
    print(f"Classes: {CLASS_NAMES}")

    if args.predict:
        predict_new_image(args.predict)
        return

    # Load the supplied train and test data.
    train_df, test_df = load_dataset_records()

    # Perform dataset analysis.
    perform_eda(train_df, test_df)

    # Create training and validation subsets from train only.
    training_df, validation_df = split_training_data(train_df)

    # Train and evaluate the classical ML approach.
    svm_metrics = train_classical_model(
        training_df,
        validation_df,
        test_df,
    )

    # Train and evaluate the deep learning approach.
    mobilenet_metrics = train_deep_learning_model(
        training_df,
        validation_df,
        test_df,
    )

    # Compare both models.
    save_model_comparison([
        svm_metrics,
        mobilenet_metrics,
    ])

    print("\n========================================")
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("========================================")
    print(f"Trained models: {OUTPUT_DIR.resolve()}")
    print(f"Results:       {RESULTS_DIR.resolve()}")
    print("\nTo predict a new image, run:")
    print('python3 avocado_assignment.py --predict "path/to/image.jpg"')


if __name__ == "__main__":
    main()