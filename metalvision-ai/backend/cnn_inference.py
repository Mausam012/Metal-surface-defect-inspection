"""CNN inference matching the teammate's trained DefectCNN checkpoint."""
import cv2
import numpy as np

from .base_model import BaseModel, ModelNotAvailable
from .torch_utils import load_checkpoint, require_torch


def build_cnn(num_classes=4):
    require_torch()
    import torch.nn as nn

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
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return DefectCNN()


class CNNModel(BaseModel):
    name = "CNN"
    dir_name = "cnn"
    architecture = "Convolutional Neural Network"
    description = "Four-class CNN trained to classify metal surface defects."
    purpose = "Image-level defect classification."
    expected_output = "One of four trained defect classes."
    output_type = "Image classification"
    checkpoint_hint = "models/cnn/cnn_four_class.pth"

    def load(self, checkpoint_path):
        torch = require_torch()
        ck = load_checkpoint(checkpoint_path, required=("model_state_dict", "class_names"))

        if "model_state_dict" not in ck or "class_names" not in ck:
            raise ModelNotAvailable("CNN checkpoint is missing model_state_dict or class_names.")

        names = list(ck["class_names"])
        expected = ["silk_spot", "inclusion", "waist_folding", "rolled_pit"]
        if names != expected:
            raise ModelNotAvailable(f"Unexpected CNN class order: {names}")

        image_size = int(ck.get("image_size", 128))
        if image_size != 128:
            raise ModelNotAvailable(f"Unsupported CNN input size: {image_size}")

        model = build_cnn(len(names))
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.eval()

        self._model = model
        self.meta = {
            "class_names": names,
            "img_size": image_size,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "version": f"trained_cnn / {checkpoint_path.name}",
        }

    def _predict(self, gray8):
        torch = require_torch()
        m = self.meta

        # The app currently passes its processed grayscale X-ray.
        # Repeat it into 3 channels to match the CNN's RGB input shape.
        gray = cv2.resize(
            gray8, (m["img_size"], m["img_size"]),
            interpolation=cv2.INTER_LINEAR
        )
        rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 255.0

        mean = np.asarray(m["mean"], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(m["std"], dtype=np.float32).reshape(1, 1, 3)
        rgb = (rgb - mean) / std

        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        return {
            "status": "Requires Manual Review",
            "defect_category": m["class_names"][idx],
            "confidence": confidence,
            "notes": [
                "CNN returned its highest-probability class among four defect classes.",
                "This checkpoint has no Normal class; manual inspection is required.",
                "Input was converted to three identical channels from the processed grayscale image.",
                "Softmax confidence is not calibrated.",
            ],
        }

