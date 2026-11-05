"""Single entry point for inference:  predict(image, model_name).

Returns the common result dict from base_model.make_result(). When the model is
not available the result has status 'Model Not Available' and
is_real_prediction=False -- it never contains invented scores.
"""
from pathlib import Path

import cv2
import numpy as np

from .base_model import make_result
from .model_manager import ModelManager

_default_manager = None


def default_manager() -> ModelManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelManager(Path(__file__).resolve().parent.parent / "models")
    return _default_manager


def to_gray8(image) -> np.ndarray:
    """Accept any ndarray and return a 2-D uint8 image."""
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] >= 3 else img[:, :, 0]
    if img.dtype != np.uint8:
        f = img.astype(np.float32)
        lo, hi = f.min(), f.max()
        img = ((f - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros(img.shape, np.uint8)
    return img


def predict(image, model_name: str, manager: ModelManager = None) -> dict:
    manager = manager or default_manager()
    try:
        model = manager.get(model_name)
    except ValueError as exc:
        return make_result(str(model_name), error=str(exc))
    try:
        gray8 = to_gray8(image)
    except Exception:
        return make_result(model.name, error="The input image could not be converted for inference.")
    return model.predict(gray8)
