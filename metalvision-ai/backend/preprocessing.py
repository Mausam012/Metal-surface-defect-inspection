"""OpenCV preprocessing pipeline.

Design rules (to avoid inventing or hiding defect-like patterns):
  * The original upload is NEVER modified; every output is a new PNG.
  * The original bit depth is read as-is (cv2.IMREAD_UNCHANGED).
  * Normalization is a linear intensity stretch only (min-max or percentile).
  * Resizing happens only if the longest side exceeds `max_side` (area interpolation).
  * CLAHE is OFF by default because local contrast enhancement can create or
    hide defect-like texture. Enable it only deliberately.
Outputs in <out_dir>: original_preview.png, gray.png, normalized.png, processed.png
"""
from pathlib import Path

import cv2
import numpy as np


class PreprocessingError(RuntimeError):
    pass


DEFAULT_SETTINGS = {
    "normalization": "percentile",  # 'percentile' or 'minmax'
    "low_percentile": 1.0,
    "high_percentile": 99.0,
    "max_side": 1024,               # resize only if longest side is larger
    "clahe": False,
    "clahe_clip": 2.0,
}

_BIT_DEPTH = {"uint8": 8, "uint16": 16, "int16": 16, "float32": 32, "float64": 64}


def sanitize_settings(raw=None) -> dict:
    """Merge user settings with defaults and clamp them to safe ranges."""
    s = dict(DEFAULT_SETTINGS)
    raw = raw or {}
    if raw.get("normalization") in ("percentile", "minmax"):
        s["normalization"] = raw["normalization"]
    try:
        lo = float(raw.get("low_percentile", s["low_percentile"]))
        hi = float(raw.get("high_percentile", s["high_percentile"]))
        lo, hi = max(0.0, min(lo, 49.0)), min(100.0, max(hi, 51.0))
        s["low_percentile"], s["high_percentile"] = lo, hi
        s["max_side"] = int(max(256, min(int(raw.get("max_side", s["max_side"])), 4096)))
        s["clahe_clip"] = float(max(1.0, min(float(raw.get("clahe_clip", s["clahe_clip"])), 10.0)))
    except (TypeError, ValueError):
        pass
    s["clahe"] = bool(raw.get("clahe", s["clahe"]))
    return s


def _minmax_uint8(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    a, b = float(f.min()), float(f.max())
    if b <= a:
        return np.zeros(img.shape, np.uint8)
    return ((f - a) / (b - a) * 255.0).round().astype(np.uint8)


def _native_uint8(img: np.ndarray) -> np.ndarray:
    """Scale by the data type's full range (no stretching) for the 'grayscale' view."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 65535.0 * 255.0).round().astype(np.uint8)
    if img.dtype in (np.float32, np.float64):
        return (np.clip(img, 0, 1) * 255.0).round().astype(np.uint8)
    return _minmax_uint8(img)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.dtype not in (np.uint8, np.uint16, np.float32):
        img = img.astype(np.float32)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_float(gray: np.ndarray, method: str, lo: float, hi: float) -> np.ndarray:
    """Linear stretch to [0, 1] float32."""
    g = gray.astype(np.float32)
    if method == "minmax":
        a, b = float(g.min()), float(g.max())
    else:
        a, b = (float(v) for v in np.percentile(g, [lo, hi]))
    if b <= a:
        return np.zeros_like(g)
    return np.clip((g - a) / (b - a), 0.0, 1.0)


def preprocess_image(src_path, out_dir, settings=None) -> dict:
    """Run the pipeline and return metadata (also saved as the reproducibility record)."""
    s = sanitize_settings(settings)
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise PreprocessingError("The image could not be read.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = 1 if img.ndim == 2 else img.shape[2]

    # 1) Original preview (browser-displayable copy; the uploaded file is untouched)
    if img.dtype == np.uint8:
        preview = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if channels == 4 else img
    else:
        preview = _minmax_uint8(_to_gray(img))
    cv2.imwrite(str(out_dir / "original_preview.png"), preview)

    # 2) Grayscale (native range, no stretching)
    gray = _to_gray(img)
    cv2.imwrite(str(out_dir / "gray.png"), _native_uint8(gray))

    # 3) Normalized (linear intensity stretch, full resolution)
    norm = normalize_float(gray, s["normalization"], s["low_percentile"], s["high_percentile"])
    cv2.imwrite(str(out_dir / "normalized.png"), (norm * 255.0).round().astype(np.uint8))

    # 4) Processed = normalized (+ resize only when needed) (+ optional CLAHE)
    h, w = norm.shape
    proc, resized = norm, False
    if max(h, w) > s["max_side"]:
        scale = s["max_side"] / max(h, w)
        proc = cv2.resize(norm, (max(1, round(w * scale)), max(1, round(h * scale))),
                          interpolation=cv2.INTER_AREA)
        resized = True
    proc8 = (proc * 255.0).round().astype(np.uint8)
    if s["clahe"]:
        proc8 = cv2.createCLAHE(clipLimit=s["clahe_clip"], tileGridSize=(8, 8)).apply(proc8)
    cv2.imwrite(str(out_dir / "processed.png"), proc8)

    return {
        "original_shape": [int(h), int(w)],
        "channels": int(channels),
        "dtype": str(img.dtype),
        "bit_depth": _BIT_DEPTH.get(str(img.dtype), 8),
        "value_min": float(gray.min()),
        "value_max": float(gray.max()),
        "processed_shape": [int(proc8.shape[0]), int(proc8.shape[1])],
        "resized": resized,
        "settings": s,
    }
