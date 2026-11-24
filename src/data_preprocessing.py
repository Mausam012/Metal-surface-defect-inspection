
import pandas as pd
from pathlib import Path
from PIL import Image
import math

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT / "dataset" / "processed"
    / "annotations_training_ready.csv"
)

IMAGE_FOLDER = ROOT / "dataset" / "raw" / "images"

OUTPUT_FOLDER = ROOT / "dataset" / "cnn_crops"

MANIFEST_PATH = (
    ROOT / "dataset" / "processed"
    / "cnn_crops_manifest.csv"
)

# ==========================================
# 2. SELECT FOUR MAJOR CLASSES
# ==========================================

SELECTED_CLASSES = [
    "silk_spot",
    "inclusion",
    "waist_folding",
    "rolled_pit"
]

# New labels for our four-class CNN
CLASS_TO_LABEL = {
    "silk_spot": 0,
    "inclusion": 1,
    "waist_folding": 2,
    "rolled_pit": 3
}

# ==========================================
# 3. LOAD TRAINING ANNOTATIONS
# ==========================================

print("\n========== CNN DATA PREPROCESSING ==========")

df = pd.read_csv(CSV_PATH)

print("Original annotation records:", len(df))

df = df[df["class_name"].isin(SELECTED_CLASSES)].copy()

print("Selected annotation records:", len(df))

print("\nSelected class distribution:")
print(df["class_name"].value_counts())

# ==========================================
# 4. CREATE CLASS FOLDERS
# ==========================================

for class_name in SELECTED_CLASSES:

    class_folder = OUTPUT_FOLDER / class_name

    class_folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# 5. EXTRACT DEFECT CROPS
# ==========================================

manifest = []

skipped = 0

for row_number, row in df.iterrows():

    image_name = str(row["image_name"])

    image_path = Path(
        str(row["image_path"]).replace("\\", "/")
    )

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    class_name = row["class_name"]

    try:

        # Load original image
        with Image.open(image_path) as original:

            image = original.convert("RGB")

            image_width, image_height = image.size

            # Read bounding box
            xmin = math.floor(float(row["xmin"]))
            ymin = math.floor(float(row["ymin"]))

            xmax = math.ceil(float(row["xmax"])) + 1
            ymax = math.ceil(float(row["ymax"])) + 1

            # Keep coordinates inside image boundaries
            xmin = max(0, min(xmin, image_width))
            ymin = max(0, min(ymin, image_height))

            xmax = max(0, min(xmax, image_width))
            ymax = max(0, min(ymax, image_height))

            # Check bounding box validity
            if xmax <= xmin or ymax <= ymin:

                skipped += 1
                continue

            # Extract defect region
            crop = image.crop(
                (xmin, ymin, xmax, ymax)
            )

            # Save crop
            crop_name = (
                f"crop_{row_number:05d}.jpg"
            )

            crop_path = (
                OUTPUT_FOLDER
                / class_name
                / crop_name
            )

            crop.save(crop_path, quality=95)

            # Store information for future training
            manifest.append({

                "crop_path": crop_path.relative_to(ROOT).as_posix(),

                "original_image": image_name,

                "image_path": str(
                    image_path.relative_to(ROOT).as_posix()
                ),

                "class_name": class_name,

                "class_id": int(row["class_id"]),

                "cnn_label": CLASS_TO_LABEL[class_name],

                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax

            })

    except Exception as error:

        skipped += 1

        print(
            f"Skipped {image_name}: {error}"
        )

# ==========================================
# 6. SAVE CROP MANIFEST
# ==========================================

manifest_df = pd.DataFrame(manifest)

manifest_df.to_csv(
    MANIFEST_PATH,
    index=False
)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n========== PREPROCESSING RESULTS ==========")

print("Successfully extracted crops:", len(manifest_df))

print("Skipped annotations:", skipped)

print("\nCrops per class:")

if len(manifest_df) > 0:

    print(
        manifest_df["class_name"].value_counts()
    )

    print("\nUnique original images per class:")

    print(
        manifest_df.groupby("class_name")[
            "original_image"
        ].nunique()
    )

print("\nSaved crop folder:", OUTPUT_FOLDER)

print("Saved manifest:", MANIFEST_PATH)

print("\nCNN DATA PREPROCESSING COMPLETED")
