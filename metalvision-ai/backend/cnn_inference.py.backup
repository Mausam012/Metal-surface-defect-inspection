"""CNN module (owner: CNN team member).

A compact custom CNN for normal-vs-suspicious classification. Replace
`build_simple_cnn` with your own design: just keep the checkpoint format
(keys: arch, state_dict, class_names, in_channels, img_size, mean, std).

Checkpoint location: models/cnn/*.pt
"""
import numpy as np

from .base_model import BaseModel, ModelNotAvailable, score_to_status
from .torch_utils import gray_to_tensor, load_checkpoint, require_torch, split_probs


def build_simple_cnn(num_classes=2, in_channels=1):
    """4 conv blocks -> global average pooling -> linear classifier."""
    require_torch()
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True), nn.MaxPool2d(2))

    return nn.Sequential(block(in_channels, 16), block(16, 32), block(32, 64), block(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                         nn.Linear(128, num_classes))


class CNNModel(BaseModel):
    name = "CNN"
    dir_name = "cnn"
    architecture = "Convolutional Neural Network"
    description = ("Stacked convolution + pooling layers learn local texture and edge features, "
                   "followed by a classifier head.")
    purpose = "Feature extraction and image-level classification."
    expected_output = "Class probabilities -> defect score and status."
    output_type = "Image classification"
    checkpoint_hint = "models/cnn/cnn_best.pt"
    extras = ["Custom architecture defined in backend/cnn_inference.py"]

    def load(self, checkpoint_path):
        require_torch()
        ck = load_checkpoint(checkpoint_path)
        if ck["arch"] != "simple_cnn":
            raise ModelNotAvailable(f"Unsupported architecture '{ck['arch']}' (expected 'simple_cnn').")
        names, in_ch = list(ck["class_names"]), int(ck.get("in_channels", 1))
        model = build_simple_cnn(len(names), in_ch)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        self._model = model
        self.meta = {"class_names": names, "in_channels": in_ch, "img_size": int(ck.get("img_size", 224)),
                     "mean": list(ck.get("mean", [0.5] * in_ch)), "std": list(ck.get("std", [0.5] * in_ch)),
                     "version": f"simple_cnn / {checkpoint_path.name}"}

    def _predict(self, gray8):
        torch = require_torch()
        m = self.meta
        x = gray_to_tensor(gray8, m["img_size"], m["in_channels"], m["mean"], m["std"])
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1)[0].tolist()
        p_defect, _idx, category = split_probs(probs, m["class_names"])
        status = score_to_status(p_defect)
        return {"status": status, "defect_category": category if status != "Normal" else None,
                "score": p_defect, "confidence": float(max(probs)),
                "notes": ["Softmax probabilities are not calibrated."]}
