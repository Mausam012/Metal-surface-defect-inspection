"""U-Net module (owner: U-Net team member).

Encoder-decoder with skip connections: the encoder compresses the image into
features, the decoder upsamples back to full resolution and concatenates the
matching encoder features, producing a PIXEL-LEVEL mask.

Checkpoint format: keys arch='unet', state_dict, class_names, in_channels, base, img_size.
Checkpoint location: models/unet/*.pt
"""
import cv2
import numpy as np

from .base_model import BaseModel, ModelNotAvailable
from .torch_utils import load_checkpoint, require_torch

MIN_MASK_FRACTION = 0.0005   # mask must cover >= 0.05 % of the image to call it Suspicious


def build_unet(in_channels=1, out_channels=1, base=16):
    require_torch()
    import torch
    import torch.nn as nn

    def dc(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = base
            self.d1, self.d2, self.d3, self.d4 = dc(in_channels, b), dc(b, 2 * b), dc(2 * b, 4 * b), dc(4 * b, 8 * b)
            self.mid = dc(8 * b, 16 * b)
            self.pool = nn.MaxPool2d(2)
            self.up4, self.u4 = nn.ConvTranspose2d(16 * b, 8 * b, 2, stride=2), dc(16 * b, 8 * b)
            self.up3, self.u3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2), dc(8 * b, 4 * b)
            self.up2, self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2), dc(4 * b, 2 * b)
            self.up1, self.u1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2), dc(2 * b, b)
            self.out = nn.Conv2d(b, out_channels, 1)

        def forward(self, x):                       # input H, W must be divisible by 16
            e1 = self.d1(x)
            e2 = self.d2(self.pool(e1))
            e3 = self.d3(self.pool(e2))
            e4 = self.d4(self.pool(e3))
            m = self.mid(self.pool(e4))
            x = self.u4(torch.cat([self.up4(m), e4], 1))
            x = self.u3(torch.cat([self.up3(x), e3], 1))
            x = self.u2(torch.cat([self.up2(x), e2], 1))
            x = self.u1(torch.cat([self.up1(x), e1], 1))
            return self.out(x)                      # logits; apply sigmoid for probabilities

    return UNet()


class UNetModel(BaseModel):
    name = "U-Net"
    dir_name = "unet"
    architecture = "U-Net (encoder-decoder with skip connections)"
    description = ("Encoder downsamples to capture context, decoder upsamples to recover detail; "
                   "skip connections carry fine spatial information across.")
    purpose = "Pixel-level defect segmentation."
    expected_output = "Binary segmentation mask of suspicious pixels."
    output_type = "Segmentation mask"
    checkpoint_hint = "models/unet/unet_best.pt  (needs pixel-level masks to train)"
    extras = ["Requires pixel-wise annotations (data/annotations/)"]

    def load(self, checkpoint_path):
        require_torch()
        ck = load_checkpoint(checkpoint_path, required=("arch", "state_dict"))
        if ck["arch"] != "unet":
            raise ModelNotAvailable(f"Unsupported architecture '{ck['arch']}' (expected 'unet').")
        base, in_ch = int(ck.get("base", 16)), int(ck.get("in_channels", 1))
        model = build_unet(in_ch, 1, base)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        self._model = model
        self.meta = {"img_size": int(ck.get("img_size", 256)), "version": f"unet / {checkpoint_path.name}"}

    def _predict(self, gray8):
        torch = require_torch()
        size = self.meta["img_size"]
        small = cv2.resize(gray8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = torch.from_numpy(small)[None, None]
        with torch.no_grad():
            prob = torch.sigmoid(self._model(x))[0, 0].numpy()
        prob = cv2.resize(prob, (gray8.shape[1], gray8.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = (prob > 0.5).astype(np.uint8)
        suspicious = float(mask.mean()) >= MIN_MASK_FRACTION
        return {"status": "Suspicious" if suspicious else "Normal",
                "defect_category": "Unspecified (binary segmentation)" if suspicious else None,
                "score": float(prob.max()), "segmentation_mask": mask,
                "notes": ["Score = highest pixel probability (uncalibrated).",
                          f"Mask covers {mask.mean() * 100:.2f}% of the image."]}
