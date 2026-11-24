
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(r"C:\Users\Saurabh\Downloads\resnet_test_crops")
TEST_CSV = DATASET_ROOT / "test.csv"
CHECKPOINT = PROJECT_ROOT / "models" / "resnet" / "resnet50_best.pth"

# Import the existing ResNet implementation
sys.path.insert(0, str(PROJECT_ROOT))

from backend.resnet_inference import ResNetModel


def main():
    print("========== RESNET REAL CROP TEST ==========")

    # Verify files
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    # Load test manifest
    df = pd.read_csv(TEST_CSV)
    print("Test crops:", len(df))

    # Load ResNet
    model = ResNetModel(PROJECT_ROOT / "models")
    model.load(CHECKPOINT)

    predictions = []
    actual_labels = []
    inference_times = []

    print("\nRunning predictions...")

    for index, row in df.iterrows():
        crop_path = Path(row["crop_path"])

        # ZIP manifest contains original Colab paths.
        # Rebuild the path using the extracted local folder.
        relative_path = Path("crops") / row["label"] / crop_path.name
        local_crop_path = DATASET_ROOT / relative_path

        if not local_crop_path.exists():
            raise FileNotFoundError(
                f"Crop not found: {local_crop_path}"
            )

        # Read crop as OpenCV BGR
        crop = cv2.imread(str(local_crop_path))

        if crop is None:
            raise ValueError(f"Could not read crop: {local_crop_path}")

        # Measure prediction time only
        start = time.perf_counter()
        result = model.predict_crop(crop)
        elapsed = time.perf_counter() - start

        inference_times.append(elapsed)
        predictions.append(result["predicted_class"])
        actual_labels.append(row["label"])

        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{len(df)} crops")

    # Evaluation
    accuracy = accuracy_score(actual_labels, predictions)

    print("\n========== RESULTS ==========")
    print("Total crops:", len(df))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy * 100:.2f}%")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(
        actual_labels,
        predictions,
        labels=model.meta["class_names"],
        zero_division=0
    ))

    print("\n========== CONFUSION MATRIX ==========")
    print(confusion_matrix(
        actual_labels,
        predictions,
        labels=model.meta["class_names"]
    ))

    # Inference speed
    times_ms = [t * 1000 for t in inference_times]

    print("\n========== INFERENCE SPEED ==========")
    print(f"Average: {sum(times_ms) / len(times_ms):.2f} ms/crop")
    print(f"Median: {sorted(times_ms)[len(times_ms) // 2]:.2f} ms/crop")
    print(f"Total prediction time: {sum(times_ms):.2f} ms")
    print(f"Throughput: {len(times_ms) / sum(inference_times):.2f} crops/sec")

    # Save results
    df["predicted_class"] = predictions
    df["correct"] = df["label"] == df["predicted_class"]
    df["inference_time_ms"] = times_ms

    output_file = PROJECT_ROOT / "reports" / "resnet_crop_test_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()
