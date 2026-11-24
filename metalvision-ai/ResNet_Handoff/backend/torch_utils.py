"""Small PyTorch helpers shared by the classification models (ResNet, CNN).

PyTorch is imported lazily, so the web app still starts on a laptop that does
not have it installed (the model then simply reports 'Not Integrated').
"""
import cv2
import numpy as np

from .base_model import ModelNotAvailable


def require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        raise ModelNotAvailable("PyTorch / Torchvision are not installed (pip install torch torchvision).")
    return torch


def load_checkpoint(path, required=("arch", "state_dict", "class_names")) -> dict:
    """Load a checkpoint written by our training scripts.

    weights_only=True means torch refuses to unpickle arbitrary Python objects,
    so a malicious .pt file cannot execute code while loading.
    """
    torch = require_torch()
    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or any(k not in data for k in required):
        raise ModelNotAvailable(f"Checkpoint is not in the expected format (needs keys: {', '.join(required)}).")
    return data


def gray_to_tensor(gray8, size, in_channels, mean, std):
    """uint8 gray image -> normalized (1, C, size, size) tensor.

    A pretrained ResNet expects 3 channels. A grayscale X-ray is handled by
    repeating the single channel 3 times (default) -- or the training script can
    build a 1-channel network by summing the pretrained first-layer filters.
    """
    import torch
    img = cv2.resize(gray8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    t = torch.from_numpy(img).unsqueeze(0)
    if in_channels == 3:
        t = t.repeat(3, 1, 1)
    m = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    return ((t - m) / s).unsqueeze(0)


def split_probs(probs, class_names):
    """Return (p_defect, category) from softmax probabilities.

    Requires a class literally named 'normal'. p_defect = 1 - p(normal).
    """
    names = [str(n).lower() for n in class_names]
    if "normal" not in names:
        raise ModelNotAvailable("class_names must contain a 'normal' class.")
    normal_idx = names.index("normal")
    p_defect = float(1.0 - probs[normal_idx])
    others = [i for i in range(len(names)) if i != normal_idx]
    if len(others) == 1:
        return p_defect, others[0], "Unspecified (binary model)"
    best = max(others, key=lambda i: probs[i])
    return p_defect, best, str(class_names[best])
