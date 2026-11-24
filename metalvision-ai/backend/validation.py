"""Upload validation: extension, magic bytes, decodability and pixel limits."""
import io

import cv2
import numpy as np
from PIL import Image


class ImageValidationError(ValueError):
    """Raised with a user-safe message when an upload is rejected."""


# Magic bytes: the first bytes of a real PNG / JPEG / TIFF file.
_MAGIC = {
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpg": [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    "tif": [b"II*\x00", b"MM\x00*"],
    "tiff": [b"II*\x00", b"MM\x00*"],
}


def validate_image_bytes(data: bytes, filename: str, allowed: set, max_pixels: int) -> dict:
    """Return {'extension','format','width','height'} or raise ImageValidationError."""
    if not filename or "." not in filename:
        raise ImageValidationError("The file has no extension. Use PNG, JPG, JPEG or TIFF.")
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in allowed:
        raise ImageValidationError(
            f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(allowed)).upper()}."
        )
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if not any(data.startswith(m) for m in _MAGIC[ext]):
        raise ImageValidationError("The file content does not match its extension.")

    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            fmt = (im.format or ext).upper()
    except Exception:
        raise ImageValidationError("The image is corrupted or cannot be read.")

    if width * height > max_pixels:
        raise ImageValidationError(
            f"The image is too large ({width}x{height}). Maximum is {max_pixels // 1_000_000} megapixels."
        )
    # Full decode with OpenCV (same reader the pipeline will use later).
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ImageValidationError("The image could not be decoded. Try re-exporting it as PNG or TIFF.")

    return {"extension": "jpg" if ext == "jpeg" else ext, "format": fmt, "width": width, "height": height}
