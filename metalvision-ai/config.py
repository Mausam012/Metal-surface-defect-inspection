"""Central configuration for METALVISION AI.

Every path is resolved relative to this file, so the project runs from any
working directory. Change values here, not inside the route handlers.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent          # Metal-surface-defect-inspection/ (team repository root)

APP_NAME = "METALVISION AI"
APP_TAGLINE = "Intelligent X-Ray Based Metal Inspection System"
APP_VERSION = "0.1.0-prototype"
PROJECT_TITLE = "AI-Based Internal Anomaly Detection in Metal Sheets Using X-Ray Images"


class Config:
    # Used only for flash messages / sessions. Override with an env variable.
    SECRET_KEY = os.environ.get("METALVISION_SECRET_KEY", "dev-only-change-me")

    # Hard upload limit enforced by Flask (returns HTTP 413 when exceeded).
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}
    MAX_PIXELS = 50_000_000  # guards against decompression bombs

    # Folders
    UPLOAD_DIR = BASE_DIR / "uploads"        # untouched original files (uuid names)
    PROCESSED_DIR = BASE_DIR / "processed"   # one sub-folder per inspection
    REPORT_DIR = BASE_DIR / "reports"
    MODELS_DIR = BASE_DIR / "models"
    DB_PATH = BASE_DIR / "database" / "inspection.db"

    # Where the team's trained checkpoints already live in the repository.
    # models/<name>/ is always searched first; these are used only if it is empty,
    # so nobody has to copy 100 MB files around to run the app.
    CHECKPOINT_FALLBACKS = {
        "cnn": [REPO_DIR / "models" / "cnn_four_class.pth"],
        "resnet": [BASE_DIR / "ResNet_Handoff" / "models" / "resnet" / "resnet50_best.pth"],
        "yolo": [REPO_DIR / "yolo_part" / "models" / "balanced_v3_best.pt"],
        "unet": [],   # U-Net checkpoint was saved to Google Drive; place it in models/unet/
    }
