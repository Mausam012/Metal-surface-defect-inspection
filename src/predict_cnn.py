
import torch
import torch.nn as nn
import cv2
import numpy as np

from pathlib import Path
from torchvision import transforms
from PIL import Image


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "cnn_four_class.pth"

IMAGE_SIZE = 128

CLASS_NAMES = [
    "silk_spot",
    "inclusion",
    "waist_folding",
    "rolled_pit"
]

DEVICE = torch.device("cpu")


# ==========================================
# 2. RECREATE THE TRAINED CNN ARCHITECTURE
# ==========================================

class DefectCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

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

    print("CNN model loaded successfully!")

    return model


# ==========================================
# 4. OPENCV IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path):

    image_path = str(image_path)

    # Read image using OpenCV
    image = cv2.imread(image_path)

    if image is None:

        raise ValueError(
            "Unable to load image. Check image path."
        )

    # Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize to CNN input size
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Convert NumPy image to PIL
    image = Image.fromarray(image)

    # Same transformation used during training
    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    ])

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# ==========================================
# 5. PREDICT DEFECT CLASS
# ==========================================

def predict_defect(image_path):

    model = load_model()

    image_tensor = preprocess_image(image_path)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    class_id = predicted_class.item()

    predicted_label = CLASS_NAMES[class_id]

    confidence_value = confidence.item() * 100

    print("\n========== CNN PREDICTION ==========")

    print("Image:", image_path)

    print("Predicted defect:", predicted_label)

    print(
        f"Prediction confidence: {confidence_value:.2f}%"
    )

    print("\nAll class probabilities:")

    for index, probability in enumerate(
        probabilities[0]
    ):

        print(
            f"{CLASS_NAMES[index]}: "
            f"{probability.item() * 100:.2f}%"
        )

    print("\nPREDICTION COMPLETED")

    return predicted_label, confidence_value


# ==========================================
# 6. RUN PREDICTION
# ==========================================

if __name__ == "__main__":

    # Select one existing defect crop for testing

    test_image = (
        ROOT / "dataset" / "cnn_crops"
        / "silk_spot" / "crop_00000.jpg"
    )

    if not test_image.exists():

        print("Test image not found:", test_image)

    else:

        predict_defect(test_image)
