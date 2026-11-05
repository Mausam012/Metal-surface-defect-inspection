"""Registry of the four architectures. Add a new model = add one line to _CLASSES."""
from collections import OrderedDict

from .cnn_inference import CNNModel
from .resnet_inference import ResNetModel
from .unet_inference import UNetModel
from .yolo_inference import YOLOModel

_CLASSES = [CNNModel, YOLOModel, UNetModel, ResNetModel]
_ALIASES = {"unet": "U-Net", "u_net": "U-Net", "u-net": "U-Net", "cnn": "CNN", "yolo": "YOLO", "resnet": "ResNet"}


class ModelManager:
    def __init__(self, models_dir):
        self._models = OrderedDict((cls.name, cls(models_dir)) for cls in _CLASSES)

    def names(self):
        return list(self._models)

    def get(self, name: str):
        key = _ALIASES.get(str(name).strip().lower(), name)
        if key not in self._models:
            raise ValueError(f"Unknown model '{name}'. Available: {', '.join(self._models)}.")
        return self._models[key]

    def describe_all(self):
        """Info for the Models page: metadata + live status + real metrics (or None)."""
        rows = []
        for m in self._models.values():
            rows.append({
                "name": m.name, "dir": m.dir_name, "architecture": m.architecture,
                "description": m.description, "purpose": m.purpose,
                "expected_output": m.expected_output, "output_type": m.output_type,
                "checkpoint_hint": m.checkpoint_hint, "extras": m.extras,
                "status": m.status(), "metrics": m.metrics(),
            })
        return rows

    def integrated_count(self):
        return sum(1 for m in self._models.values() if m.status()["state"] == "Integrated")
