"""Central configuration for METALVISION AI.

Every path is resolved relative to this file, so the project runs from any
working directory. Change values here, not inside the route handlers.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

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
