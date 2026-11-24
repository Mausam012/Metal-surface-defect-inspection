"""U-Net module (owner: U-Net team member).

Encoder-decoder with skip connections: the encoder compresses the image into
features, the decoder upsamples back to full resolution and concatenates the
matching encoder features, producing a PIXEL-LEVEL mask.

Two checkpoint formats are accepted:

1. The team's trained model (notebooks/Metal_Defect_UNet_Training*.ipynb,
   architecture in src/unet.py): a raw state_dict saved with
   torch.save(model.state_dict(), "best_unet_opencv_improved.pth").
   3-level encoder (16-32-64, bottleneck 128), RGB input, 256 x 256,
   OpenCV CLAHE on the L channel, pixels scaled to [0, 1].
2. The template format: {"arch": "unet", "state_dict", "base", "in_channels", "img_size"}.

Note: the team's masks were generated from bounding boxes, so the U-Net
predicts box-shaped regions, not exact defect outlines.
"""
import cv2
import numpy as np

from .base_model import BaseModel, ModelNotAvailable
from .torch_utils import require_torch

MIN_MASK_FRACTION = 0.0005   # mask must cover >= 0.05 % of the image to call it Suspicious
THRESHOLD = 0.5


def build_team_unet():
    """Exact copy of src/unet.py (the architecture the team trained in Colab)."""
    require_torch()
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))

        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1, self.enc2, self.enc3 = DoubleConv(3, 16), DoubleConv(16, 32), DoubleConv(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(64, 128)
            self.up3, self.dec3 = nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64)
            self.up2, self.dec2 = nn.ConvTranspose2d(64, 32, 2, stride=2), DoubleConv(64, 32)
            self.up1, self.dec1 = nn.ConvTranspose2d(32, 16, 2, stride=2), DoubleConv(32, 16)
            self.out = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)

    return UNet()


def build_unet(in_channels=1, out_channels=1, base=16):
    """Template 4-level U-Net with batch norm (format 2)."""
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
            return self.out(x)

    return UNet()


class UNetModel(BaseModel):
    name = "U-Net"
    dir_name = "unet"
    architecture = "U-Net (encoder-decoder with skip connections)"
    description = ("Encoder downsamples to capture context, decoder upsamples to recover detail; "
                   "skip connections carry fine spatial information across.")
    purpose = "Pixel-level map of suspicious regions over the whole image."
    expected_output = "Binary segmentation mask of suspicious pixels."
    output_type = "Segmentation mask"
    checkpoint_hint = "models/unet/best_unet_opencv_improved.pth  (from the team's Google Drive)"
    extras = ["Input: RGB image resized to 256 x 256 with OpenCV CLAHE",
              "Trained on masks generated from bounding boxes (box-shaped output)"]

    def load(self, checkpoint_path):
        torch = require_torch()
        ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        if not isinstance(ck, dict):
            raise ModelNotAvailable("U-Net checkpoint is not a state_dict.")

        if "enc1.conv.0.weight" in ck:                          # format 1: team notebook
            model = build_team_unet()
            model.load_state_dict(ck, strict=True)
            self.meta = {"format": "team", "img_size": 256, "version": f"unet / {checkpoint_path.name}"}
        elif ck.get("arch") == "unet" and "state_dict" in ck:   # format 2: template
            base, in_ch = int(ck.get("base", 16)), int(ck.get("in_channels", 1))
            model = build_unet(in_ch, 1, base)
            model.load_state_dict(ck["state_dict"], strict=True)
            self.meta = {"format": "template", "in_channels": in_ch, "img_size": int(ck.get("img_size", 256)),
                         "version": f"unet / {checkpoint_path.name}"}
        else:
            raise ModelNotAvailable("Unrecognized U-Net checkpoint (expected the team's state_dict or arch='unet').")
        model.eval()
        self._model = model

    def segment(self, bgr):
        """8-bit BGR image -> (probability map, binary mask) at the input resolution."""
        torch = require_torch()
        if self._model is None:
            raise ModelNotAvailable("U-Net model is not loaded.")
        size = self.meta["img_size"]
        h, w = bgr.shape[:2]
        if self.meta["format"] == "team":
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size))   # same as the notebook
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            x = torch.from_numpy(small)[None, None]
            if self.meta.get("in_channels", 1) == 3:
                x = x.repeat(1, 3, 1, 1)
        with torch.inference_mode():
            prob = torch.sigmoid(self._model(x))[0, 0].numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return prob, (prob > THRESHOLD).astype(np.uint8)

    def _predict(self, gray8):
        prob, mask = self.segment(cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR))
        suspicious = float(mask.mean()) >= MIN_MASK_FRACTION
        return {"status": "Suspicious" if suspicious else "Normal",
                "defect_category": "Unspecified (binary segmentation)" if suspicious else None,
                "score": float(prob.max()), "segmentation_mask": mask,
                "notes": ["Score = highest pixel probability (uncalibrated).",
                          f"Mask covers {mask.mean() * 100:.2f}% of the image."]}
