"""Optional: download Bootstrap and Chart.js once so the app works fully OFFLINE (e.g. in a lab without internet).

    python scripts/download_vendor.py
Files go to frontend/static/vendor/. The templates use them automatically when present.
"""
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "frontend" / "static" / "vendor"
FILES = {
    "bootstrap.min.css": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
    "bootstrap.bundle.min.js": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js",
    "chart.umd.min.js": "https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js",
}
DEST.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    print("downloading", name)
    urllib.request.urlretrieve(url, DEST / name)
print("Done. Restart the app.")
