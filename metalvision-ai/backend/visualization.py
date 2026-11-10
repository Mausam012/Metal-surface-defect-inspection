"""Overlay rendering: heatmap, bounding boxes and segmentation mask.

Real model output  -> drawn exactly as the model returned it.
No real model      -> a SIMULATED overlay with FIXED, arbitrary geometry that is NOT
                      derived from the image, and a banner burned into the pixels so
                      it cannot be cropped out or mistaken for a prediction.
"""
from pathlib import Path

import cv2
import numpy as np

WATERMARK = "SIMULATED VISUALIZATION \u2014 NOT AN AI PREDICTION"   # for HTML / PDF
_WATERMARK_ASCII = "SIMULATED VISUALIZATION - NOT AN AI PREDICTION"  # OpenCV fonts are ASCII only


def demo_geometry(w: int, h: int):
    """Fixed placeholder shapes (same for every image on purpose)."""
    boxes = [{"x": int(.38 * w), "y": int(.40 * h), "w": int(.20 * w), "h": int(.14 * h),
              "label": "simulated", "confidence": None}]
    cx, cy = int(.48 * w), int(.47 * h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sigma = .08 * min(w, h)
    heat = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * sigma ** 2)).astype(np.float32)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (max(2, int(.09 * w)), max(2, int(.05 * h))), 0, 0, 360, 1, -1)
    return boxes, heat, mask


def _banner(img):
    h, w = img.shape[:2]
    bar = max(22, int(h * 0.05))
    cv2.rectangle(img, (0, 0), (w, bar), (0, 0, 0), -1)
    (tw, th), _ = cv2.getTextSize(_WATERMARK_ASCII, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    scale = max(0.3, min(1.0, 0.6 * (w * 0.96) / tw))
    (tw, th), _ = cv2.getTextSize(_WATERMARK_ASCII, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(img, _WATERMARK_ASCII, (max(2, (w - tw) // 2), (bar + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 215, 255), 2, cv2.LINE_AA)


def render_overlays(gray8: np.ndarray, result: dict, out_dir, simulated=False) -> dict:
    """Write overlay_<kind>.png files and return {kind: filename} for the kinds available."""
    out_dir = Path(out_dir)
    for old in out_dir.glob("overlay_*.png"):
        old.unlink()
    h, w = gray8.shape
    base = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    if simulated:
        boxes, heat, mask = demo_geometry(w, h)
    else:
        boxes, heat, mask = result.get("bounding_boxes") or [], result.get("heatmap"), result.get("segmentation_mask")

    produced = {}

    def save(kind, img):
        if simulated:
            _banner(img)
        cv2.imwrite(str(out_dir / f"overlay_{kind}.png"), img)
        produced[kind] = f"overlay_{kind}.png"

    if heat is not None:
        color = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        save("heatmap", cv2.addWeighted(base, 0.55, color, 0.45, 0))
    if boxes:
        img = base.copy()
        for b in boxes:
            cv2.rectangle(img, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (214, 195, 25), 2)
            label = b["label"] if b.get("confidence") is None else f'{b["label"]} {b["confidence"]:.2f}'
            cv2.putText(img, label, (b["x"], max(12, b["y"] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (214, 195, 25), 1, cv2.LINE_AA)
        save("boxes", img)
    if mask is not None:
        img = base.copy()
        red = np.zeros_like(img)
        red[:, :] = (60, 60, 230)
        m = mask.astype(bool)
        img[m] = cv2.addWeighted(base, 0.5, red, 0.5, 0)[m]
        save("mask", img)
    return produced
