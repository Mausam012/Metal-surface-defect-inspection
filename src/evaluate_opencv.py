
import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from predict_cnn import (
    DefectCNN,
    CLASS_NAMES,
    MODEL_PATH,
    DEVICE
)

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    ROOT / "dataset" / "processed"
    / "cnn_split_manifest.csv"
)

IMAGE_SIZE = 128


# Load test data
df = pd.read_csv(MANIFEST)
test_df = df[df["split"] == "test"].copy()

# Load trained CNN
model = DefectCNN().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("CNN model loaded.")
print("Test images:", len(test_df))


# OpenCV preprocessing methods
def preprocess(image, method):

    if method == "gaussian":
        image = cv2.GaussianBlur(
            image, (3, 3), 0
        )

    elif method == "clahe":

        lab = cv2.cvtColor(
            image, cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        image = cv2.cvtColor(
            lab, cv2.COLOR_LAB2BGR
        )

    # BGR to RGB
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    image = image.astype(np.float32) / 255.0

    image = (image - 0.5) / 0.5

    # HWC to CHW
    image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(
        np.ascontiguousarray(image)
    ).float().unsqueeze(0)


# Run evaluation
methods = ["original", "gaussian", "clahe"]

results = []

for method in methods:

    y_true = []
    y_pred = []

    print("\nEvaluating:", method)

    for _, row in test_df.iterrows():

        image_path = ROOT / row["crop_path"]

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        tensor = preprocess(image, method).to(DEVICE)

        with torch.no_grad():

            output = model(tensor)

            prediction = torch.argmax(
                output, dim=1
            ).item()

        y_true.append(int(row["cnn_label"]))
        y_pred.append(prediction)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "method": method,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    })

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(f1, 4))


# Save comparison
results_df = pd.DataFrame(results)

output_path = ROOT / "results" / "opencv_comparison.csv"

results_df.to_csv(output_path, index=False)

print("\n========== FINAL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nSaved:", output_path)
print("\nOPENCV EVALUATION COMPLETED")
