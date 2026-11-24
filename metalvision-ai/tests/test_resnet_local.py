import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from backend.resnet_inference import ResNetModel

# Dataset and checkpoint locations
DATA_DIR = ROOT / "resnet_local_test"
CSV_PATH = DATA_DIR / "test.csv"

# Initialize and load the actual integrated ResNet model
print("\nLoading ResNet model...")

model = ResNetModel(ROOT / "models", config.Config.CHECKPOINT_FALLBACKS["resnet"])
model.ensure_loaded()

class_names = model.meta["class_names"]

print("Model loaded successfully!")
print("Classes:", class_names)
print("Image size:", model.meta["img_size"])

# Read test CSV
df = pd.read_csv(CSV_PATH)

required_columns = {"crop_path", "label"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_columns}")

print("\nTotal test samples:", len(df))

all_preds = []
all_labels = []
failed_images = []
total_time = 0.0

print("\nRunning local predictions...\n")

for index, row in df.iterrows():

    image_path = DATA_DIR / row["crop_path"]
    true_label = row["label"]

    # Read image using OpenCV
    image = cv2.imread(str(image_path))

    if image is None:
        failed_images.append(str(image_path))
        continue

    # Actual integrated ResNet crop prediction
    start = time.perf_counter()
    result = model.predict_crop(image)
    elapsed = (time.perf_counter() - start) * 1000

    total_time += elapsed

    predicted_label = result["predicted_class"]

    all_labels.append(true_label)
    all_preds.append(predicted_label)

    if index < 5:
        print(
            f"{index}: Actual={true_label}, "
            f"Predicted={predicted_label}, "
            f"Confidence={result['confidence']:.4f}"
        )

# Check evaluation completion
if not all_labels:
    raise RuntimeError("No images were successfully evaluated.")

if failed_images:
    print("\nMissing/unreadable images:", len(failed_images))
    for path in failed_images[:10]:
        print(path)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)

macro_f1 = f1_score(
    all_labels,
    all_preds,
    labels=class_names,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 55)
print("LOCAL RESNET EVALUATION RESULTS")
print("=" * 55)

print("Total CSV samples:", len(df))
print("Successfully evaluated:", len(all_labels))
print("Failed images:", len(failed_images))

print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro-F1: {macro_f1:.6f}")

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        labels=class_names,
        zero_division=0
    )
)

print("Confusion Matrix:")
print("Class order:", class_names)
print(confusion_matrix(all_labels, all_preds, labels=class_names))

if all_labels:
    print(f"\nTotal inference time: {total_time:.2f} ms")
    print(f"Average time per crop: {total_time / len(all_labels):.2f} ms")

print("\nEvaluation completed.")
