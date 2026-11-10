"""ResNet module  (owner: ResNet team member).

WHAT IS A RESNET?
    A plain deep CNN gets *harder* to train as layers are added (vanishing
    gradients / degradation). ResNet adds a "skip connection" around a few layers:

            y = F(x) + x

    x  : input of the block
    F(x): what the stacked conv layers learn (the *residual*)
    y  : block output.   If the best mapping is "do nothing", the block only has
         to learn F(x) = 0, which is easy -- and gradients flow straight through
         the "+ x" path. That is why 50+ layer networks train well.
    (If x and F(x) differ in shape, a 1x1 convolution on the shortcut matches them.)
    See models/resnet/residual_block_demo.py for a ~30-line implementation.

TRANSFER LEARNING
    Start from weights pretrained on ImageNet (generic edges/textures), replace
    the final fully-connected layer `fc` with a new one for our classes, then
    fine-tune on X-ray data. Needs far fewer labelled images than training from scratch.

GRAYSCALE X-RAYS
    Pretrained ResNets expect 3 channels. Two valid options:
      A) (default) repeat the gray channel 3x            -> in_channels=3
      B) sum the 3 pretrained conv1 filters into one      -> in_channels=1
    The chosen option is stored in the checkpoint so inference matches training.

STATUS: this module is a TEMPLATE. Until a genuine checkpoint produced by
backend/resnet_training.py is placed in models/resnet/, the model is reported
as "Not Integrated" and NO prediction is produced.
"""
import cv2
import numpy as np

from .base_model import BaseModel, ModelNotAvailable, score_to_status
from .torch_utils import gray_to_tensor, load_checkpoint, require_torch, split_probs

SUPPORTED_ARCHS = ("resnet18", "resnet34", "resnet50")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    architecture = "Residual Neural Network (ResNet-18 / 34 / 50)"
    description = ("Deep CNN with skip connections, y = F(x) + x. The shortcut lets gradients "
                   "flow through very deep networks, so pretrained ResNets are strong starting "
                   "points for transfer learning.")
    purpose = "Image-level classification: normal vs. suspicious (optionally multi-class defect type)."
    expected_output = "Class probabilities -> defect score, status and (optionally) a Grad-CAM heatmap."
    output_type = "Image classification (+ Grad-CAM heatmap)"
    checkpoint_hint = "models/resnet/resnet18_best.pt  (created by backend/resnet_training.py)"
    extras = ["Variants: ResNet-18, ResNet-34, ResNet-50",
              "Transfer learning from ImageNet weights, new final layer",
              "Grad-CAM shows evidence regions, NOT exact defect boundaries"]

    def load(self, checkpoint_path):
        require_torch()
        ck = load_checkpoint(checkpoint_path)
        arch = ck["arch"]
        if arch not in SUPPORTED_ARCHS:
            raise ModelNotAvailable(f"Unsupported architecture '{arch}' in checkpoint.")
        names = list(ck["class_names"])
        in_ch = int(ck.get("in_channels", 3))
        model = build_resnet(arch, len(names), pretrained=False, in_channels=in_ch)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        self._model = model
        self.meta = {
            "arch": arch, "class_names": names, "in_channels": in_ch,
            "img_size": int(ck.get("img_size", 224)),
            "mean": list(ck.get("mean", IMAGENET_MEAN if in_ch == 3 else [0.45])),
            "std": list(ck.get("std", IMAGENET_STD if in_ch == 3 else [0.225])),
            "version": f"{arch} / {checkpoint_path.name}",
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
        torch = require_torch()
        m = self.meta
        x = gray_to_tensor(gray8, m["img_size"], m["in_channels"], m["mean"], m["std"])
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1)[0].tolist()
        p_defect, defect_idx, category = split_probs(probs, m["class_names"])
        status = score_to_status(p_defect)
        heat = self._gradcam(x, defect_idx)
        heat = cv2.resize(heat, (gray8.shape[1], gray8.shape[0]), interpolation=cv2.INTER_LINEAR)
        return {
            "status": status,
            "defect_category": category if status != "Normal" else None,
            "score": p_defect,
            "confidence": float(max(probs)),
            "heatmap": heat,
            "notes": ["Softmax probabilities are not calibrated.",
                      "Grad-CAM is an evidence map, not a defect boundary."],
        }
