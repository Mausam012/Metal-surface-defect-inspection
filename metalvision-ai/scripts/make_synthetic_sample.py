"""Create a SYNTHETIC test image (NOT an X-ray, contains NO defects) to try the upload flow.

    python scripts/make_synthetic_sample.py
Writes data/sample/synthetic_test_image_NOT_XRAY.png (8-bit) and ..._16bit.tif (16-bit).
Use real, authorized radiographs (see README, Dataset section) for any meaningful test.
"""
from pathlib import Path

import cv2
import numpy as np

out = Path(__file__).resolve().parent.parent / "data" / "sample"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
h, w = 600, 900
base = np.linspace(0.35, 0.65, w)[None, :].repeat(h, 0) + rng.normal(0, 0.03, (h, w))
img = np.clip(cv2.GaussianBlur(base.astype(np.float32), (0, 0), 3), 0, 1)
cv2.imwrite(str(out / "synthetic_test_image_NOT_XRAY.png"), (img * 255).astype(np.uint8))
cv2.imwrite(str(out / "synthetic_test_image_NOT_XRAY_16bit.tif"), (img * 60000).astype(np.uint16))
print("Wrote synthetic images to", out)
