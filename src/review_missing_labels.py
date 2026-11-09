
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
IMAGE_DIR = RAW_DIR / "images"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Index images only from the primary image directory
image_index = {}

for path in IMAGE_DIR.rglob("*"):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        image_index.setdefault(path.name.lower(), []).append(path)

records = []

for xml_path in (RAW_DIR / "label").rglob("*.xml"):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        continue

    filename = root.findtext("filename", "").strip()

    matches = image_index.get(filename.lower(), []) if filename else []

    # Fallback: match using XML filename stem
    if not matches:
        matches = [
            p for name, paths in image_index.items()
            if Path(name).stem.lower() == xml_path.stem.lower()
            for p in paths
        ]

    for obj in root.findall("object"):
        original_label = (obj.findtext("name") or "").strip()

        if original_label:
            continue

        if len(matches) == 1:
            inferred_label = matches[0].parent.name
            status = "REVIEW_FOLDER_LABEL"
            image_path = str(matches[0].relative_to(PROJECT_ROOT))
        elif len(matches) == 0:
            inferred_label = ""
            status = "IMAGE_NOT_FOUND"
            image_path = ""
        else:
            inferred_label = ""
            status = "AMBIGUOUS_IMAGE_MATCH"
            image_path = ""

        box = obj.find("bndbox")

        records.append({
            "xml_file": xml_path.name,
            "image_path": image_path,
            "original_label": original_label,
            "proposed_label": inferred_label,
            "review_status": status,
            "has_bounding_box": box is not None
        })

output = PROJECT_ROOT / "dataset" / "processed" / "missing_label_review.csv"

pd.DataFrame(records).to_csv(output, index=False)

print("Missing-label objects:", len(records))
print("\nReview status:")
if records:
    print(pd.Series(
        [r["review_status"] for r in records]
    ).value_counts().to_string())

print("\nReport saved:", output)
