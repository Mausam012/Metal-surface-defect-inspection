
"""ResNet module - Five-class metal defect crop classification.

ARCHITECTURE:
    ResNet50 is a deep convolutional neural network that uses residual
    (skip) connections to improve information and gradient flow.

    A residual block learns:
        y = F(x) + x

    The shortcut connection helps deep networks train effectively.

TRANSFER LEARNING:
    The model uses ImageNet-pretrained ResNet50 weights as its starting
    point. Its final fully connected layer is replaced with a five-class
    classification layer and fine-tuned on metal defect crops.

INPUT:
    OpenCV BGR colour image containing a metal defect crop.
    The image is converted to RGB, resized to 224 x 224 while preserving
    aspect ratio, padded with grey, and ImageNet-normalized.

OUTPUT CLASSES:
    1. silk_spot
    2. inclusion
    3. waist_folding
    4. rolled_pit
    5. water_spot

PREDICTION:
    Use predict_crop(bgr_crop) to obtain the predicted class, confidence,
    and probabilities for all five classes.

GRAD-CAM:
    Grad-CAM provides a visual indication of image regions influencing
    the prediction. It is an evidence map, not an exact defect boundary.

CHECKPOINT:
    models/resnet/resnet50_best.pth

NOTE:
    This module supports five-class RGB crop classification. It does not
    implement normal-versus-defect classification on grayscale images.
"""

import cv2
import numpy as np

from .base_model import BaseModel, ModelNotAvailable
from .torch_utils import require_torch

SUPPORTED_ARCHS = ("resnet18", "resnet34", "resnet50")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def bgr_crop_to_tensor(bgr_crop, img_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Preprocess an OpenCV BGR defect crop for ResNet50.

    Steps:
    1. Convert BGR to RGB.
    2. Resize while preserving aspect ratio.
    3. Apply grey padding.
    4. Convert to PyTorch tensor.
    5. Apply ImageNet normalization.
    """
    import torch

    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("Input crop is empty.")

    if bgr_crop.ndim != 3 or bgr_crop.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    # Convert OpenCV BGR image to RGB
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]

    # Calculate dimensions while preserving aspect ratio
    # Preserve aspect ratio without enlarging small crops
    scale = min(1.0, img_size / w, img_size / h)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    # Resize using OpenCV Lanczos interpolation
    resized = cv2.resize(
        rgb,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    # Create grey canvas
    canvas = np.full(
        (img_size, img_size, 3),
        128,
        dtype=np.uint8
    )

    # Place resized image in the centre
    x = (img_size - new_w) // 2
    y = (img_size - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    # Convert HWC to CHW and scale pixels to [0, 1]
    tensor = torch.from_numpy(
        canvas.transpose(2, 0, 1).copy()
    ).float() / 255.0

    # ImageNet normalization
    mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    tensor = (tensor - mean) / std

    # Add batch dimension
    return tensor.unsqueeze(0)


def build_resnet(arch="resnet18", num_classes=2, pretrained=False, in_channels=3):
    """Create a ResNet-18/34/50 with a new final layer (shared with the training script)."""
    require_torch()
    import torch
    import torch.nn as nn
    from torchvision import models

    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"arch must be one of {SUPPORTED_ARCHS}")
    ctor = getattr(models, arch)
    if pretrained:
        weights = getattr(models, f"ResNet{arch[6:]}_Weights").DEFAULT   # ImageNet weights
        model = ctor(weights=weights)
    else:
        model = ctor(weights=None)

    if in_channels == 1:                                # option B: single-channel input
        old = model.conv1
        new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        model.conv1 = new

    model.fc = nn.Linear(model.fc.in_features, num_classes)   # replace the classifier head
    return model


class ResNetModel(BaseModel):
   
    name = "ResNet"
    dir_name = "resnet"
    architecture = "ResNet-50"
    description = (
        "Deep convolutional neural network using residual skip connections. "
        "Uses ImageNet transfer learning and a fine-tuned classification head "
        "to identify five types of metal surface defects."
    )
    purpose = "Five-class classification of RGB metal defect crops."
    expected_output = (
        "Predicted defect class, confidence, and probabilities "
        "for all five defect classes."
    )
    output_type = "Five-class image classification"
    checkpoint_hint = "models/resnet/resnet50_best.pth"
    extras = [
        "Architecture: ResNet-50",
        "Transfer learning from ImageNet",
        "Input: RGB defect crops resized to 224 x 224",
        "Classes: silk_spot, inclusion, waist_folding, rolled_pit, water_spot",
        "Grad-CAM can provide visual evidence, not exact defect boundaries"
    ]

    
    def load(self, checkpoint_path):
        """Load and validate the trained five-class ResNet checkpoint."""

        torch = require_torch()

        try:
            ck = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=True
            )
        except Exception as exc:
            raise ModelNotAvailable(
                f"Could not load ResNet checkpoint: {type(exc).__name__}"
            )

        if not isinstance(ck, dict):
            raise ModelNotAvailable("Invalid checkpoint format.")

        required_keys = ["arch", "class_names", "model_state_dict"]
        missing = [key for key in required_keys if key not in ck]

        if missing:
            raise ModelNotAvailable(
                f"Checkpoint is missing keys: {missing}"
            )

        arch = ck["arch"]
        names = list(ck["class_names"])

        expected_classes = [
            "silk_spot",
            "inclusion",
            "waist_folding",
            "rolled_pit",
            "water_spot"
        ]

        if arch not in SUPPORTED_ARCHS:
            raise ModelNotAvailable(
                f"Unsupported architecture: {arch}"
            )

        if names != expected_classes:
            raise ModelNotAvailable(
                f"Unexpected class order: {names}"
            )

        img_size = int(ck.get("img_size", 224))
        mean = list(ck.get("mean", IMAGENET_MEAN))
        std = list(ck.get("std", IMAGENET_STD))

        if img_size <= 0:
            raise ModelNotAvailable("Invalid checkpoint image size.")

        if len(mean) != 3 or len(std) != 3:
            raise ModelNotAvailable("Invalid RGB normalization values.")

        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ModelNotAvailable("Normalization values must be finite.")

        if any(value <= 0 for value in std):
            raise ModelNotAvailable("Standard deviation values must be positive.")

        in_ch = 3

        model = build_resnet(
            arch=arch,
            num_classes=len(names),
            pretrained=False,
            in_channels=in_ch
        )

        try:
            model.load_state_dict(
                ck["model_state_dict"],
                strict=True
            )
        except Exception as exc:
            raise ModelNotAvailable(
                f"Checkpoint weights are incompatible: {type(exc).__name__}"
            )

        model.eval()
        self._model = model

        self.meta = {
            "arch": arch,
            "class_names": names,
            "in_channels": in_ch,
            "img_size": img_size,
            "mean": mean,
            "std": std,
            "version": f"{arch} / {checkpoint_path.name}"
        }

        print("ResNet checkpoint loaded successfully.")
        print("Architecture:", arch)
        print("Classes:", names)
        print("Input size:", self.meta["img_size"])

    
    def predict_crop(self, bgr_crop):
        """
        Classify a defect crop using the trained five-class ResNet.
        Input: OpenCV BGR image.
        """

        torch = require_torch()

        # Check whether the model is loaded
        if self._model is None or not self.meta:
            raise ModelNotAvailable("ResNet model is not loaded.")

        # Preprocess the input crop
        x = bgr_crop_to_tensor(
            bgr_crop,
            self.meta["img_size"],
            self.meta["mean"],
            self.meta["std"]
        )

        # Run prediction
        self._model.eval()

        with torch.inference_mode():
            logits = self._model(x)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Convert probabilities into a Python list
        probs = probabilities.cpu().tolist()

        class_names = self.meta["class_names"]

        # Find the class with the highest probability
        predicted_index = int(probabilities.argmax().item())

        predicted_class = class_names[predicted_index]
        confidence = float(probs[predicted_index])

        # Store probabilities for all five classes
        class_probabilities = {
            name: float(prob)
            for name, prob in zip(class_names, probs)
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities
        }
  


    def _gradcam(self, x, class_idx):
        """Grad-CAM on the last residual stage. Evidence map only, not a segmentation."""
        torch = require_torch()
        acts, grads = {}, {}

        def fwd_hook(_module, _inp, out):
            acts["a"] = out
            out.register_hook(lambda g: grads.__setitem__("g", g))

        handle = self._model.layer4.register_forward_hook(fwd_hook)
        try:
            self._model.zero_grad()
            logits = self._model(x.clone().requires_grad_(True))
            logits[0, class_idx].backward()
        finally:
            handle.remove()
        weights = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts["a"]).sum(dim=1))[0].detach().cpu().numpy()
        cam -= cam.min()
        return (cam / cam.max()).astype(np.float32) if cam.max() > 0 else cam.astype(np.float32)

    
    def _predict(self, gray8):
        """The five-class ResNet accepts RGB crops through predict_crop()."""
        raise ModelNotAvailable(
            "This ResNet model supports five-class RGB crop classification. "
            "Use predict_crop(bgr_crop) instead of predict(gray8)."
        )
