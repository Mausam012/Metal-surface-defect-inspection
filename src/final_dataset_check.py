
from pathlib import Path
from collections import Counter
import hashlib
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_ROOT / "dataset" / "processed" / "annotations_cleaned.csv"
REPORT_DIR = PROJECT_ROOT / "dataset" / "processed"

df = pd.read_csv(CSV_FILE)

issues = []
folder_labels = Counter()
image_hashes = {}

# Cache image information so each image is checked once
image_info = {}

for index, row in df.iterrows():
    image_path = PROJECT_ROOT / str(row["image_path"])
    xml_path = PROJECT_ROOT / str(row["annotation_file"])

    folder = image_path.parent.name
    label = str(row["defect_class"]).strip()

    folder_labels[(folder, label)] += 1

    if not image_path.is_file():
        issues.append({
            "row": index,
            "image": row["image_name"],
            "issue": "IMAGE_NOT_FOUND"
        })
        continue

    if not xml_path.is_file():
        issues.append({
            "row": index,
            "image": row["image_name"],
            "issue": "XML_NOT_FOUND"
        })

    if image_path not in image_info:
        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                width, height = img.size

            image_info[image_path] = (width, height)

            digest = hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest()

            image_hashes.setdefault(digest, []).append(
                str(image_path)
            )

        except Exception:
            issues.append({
                "row": index,
                "image": row["image_name"],
                "issue": "CORRUPTED_IMAGE"
            })
            continue

    width, height = image_info[image_path]

    try:
        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])

        if (
            xmin < 0 or ymin < 0 or
            xmax > width or ymax > height or
            xmin >= xmax or ymin >= ymax
        ):
            issues.append({
                "row": index,
                "image": row["image_name"],
                "issue": "INVALID_BOUNDING_BOX"
            })

    except (TypeError, ValueError):
        issues.append({
            "row": index,
            "image": row["image_name"],
            "issue": "INVALID_COORDINATE"
        })

# Save reports
issue_df = pd.DataFrame(issues)

issue_df.to_csv(
    REPORT_DIR / "final_validation_issues.csv",
    index=False
)

label_df = pd.DataFrame(
    [
        {"folder_class": folder, "annotation_class": label,
         "count": count}
        for (folder, label), count in folder_labels.items()
    ]
)

label_df.to_csv(
    REPORT_DIR / "label_folder_review_final.csv",
    index=False
)

duplicate_groups = [
    paths for paths in image_hashes.values()
    if len(paths) > 1
]

print("\n========== FINAL DATASET VALIDATION ==========")
print("Annotation records:", len(df))
print("Unique image paths:", df["image_path"].nunique())
print("Images successfully checked:", len(image_info))
print("Validation issues:", len(issues))
print("Exact duplicate image groups:", len(duplicate_groups))

print("\nIssue distribution:")
if issues:
    print(issue_df["issue"].value_counts().to_string())
else:
    print("No validation issues found.")

print("\nReports saved:")
print("final_validation_issues.csv")
print("label_folder_review_final.csv")

print("\n========== VALIDATION FINISHED ==========")
