
import pandas as pd
import shutil
import random
from pathlib import Path
from PIL import Image

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(
    r"C:\Users\sm423\Downloads\dataset\dataset\raw"
)

IMAGE_ROOT = DATASET_ROOT / "images"

CSV_PATH = (
    PROJECT_ROOT / "dataset" / "processed" / "annotations.csv"
)

OUTPUT = PROJECT_ROOT / "dataset" / "yolo_dataset"

# Reproducible split
random.seed(42)
TRAIN_RATIO = 0.80

# Load CSV
df = pd.read_csv(CSV_PATH)

# Create class IDs automatically
classes = sorted(df["defect_class"].dropna().unique())
class_map = {name: i for i, name in enumerate(classes)}

print("Class mapping:")
for name, idx in class_map.items():
    print(idx, ":", name)

# Create output folders
for split in ["train", "val"]:
    (OUTPUT / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

# Find image files
image_index = {}

for p in IMAGE_ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in [
        ".jpg", ".jpeg", ".png"
    ]:
        image_index[p.name.lower()] = p

# Match images
df["source_image"] = df["image_name"].apply(
    lambda name: image_index.get(str(name).lower())
)

df = df.dropna(subset=["source_image"]).copy()

# Group annotations by image
groups = {
    str(path): group
    for path, group in df.groupby("source_image")
}

image_paths = list(groups.keys())
random.shuffle(image_paths)

split_index = int(len(image_paths) * TRAIN_RATIO)

train_images = set(image_paths[:split_index])
val_images = set(image_paths[split_index:])

# Keep classes represented by only one image in training
class_image_counts = df.groupby("defect_class")[
    "source_image"
].nunique()

for class_name, count in class_image_counts.items():
    if count == 1:
        rare_paths = df.loc[
            df["defect_class"] == class_name,
            "source_image"
        ].unique()

        for rare_path in rare_paths:
            key = str(rare_path)
            if key in val_images:
                val_images.remove(key)
                train_images.add(key)

# Convert annotations
converted = 0
skipped = 0

for split, selected in [
    ("train", train_images),
    ("val", val_images)
]:
    for image_key in selected:

        image_path = Path(image_key)
        rows = groups[image_key]

        # Read actual image dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        labels = []

        for _, row in rows.iterrows():

            class_name = row["defect_class"]

            if class_name not in class_map:
                skipped += 1
                continue

            xmin = max(0, float(row["xmin"]))
            ymin = max(0, float(row["ymin"]))
            xmax = min(width, float(row["xmax"]))
            ymax = min(height, float(row["ymax"]))

            if xmax <= xmin or ymax <= ymin:
                skipped += 1
                continue

            xc = ((xmin + xmax) / 2) / width
            yc = ((ymin + ymax) / 2) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height

            labels.append(
                f"{class_map[class_name]} "
                f"{xc:.6f} {yc:.6f} "
                f"{bw:.6f} {bh:.6f}"
            )

        if not labels:
            skipped += 1
            continue

        # Copy image
        destination_image = (
            OUTPUT / "images" / split / image_path.name
        )

        shutil.copy2(image_path, destination_image)

        # Save YOLO label
        label_path = (
            OUTPUT / "labels" / split /
            f"{image_path.stem}.txt"
        )

        label_path.write_text(
            "\n".join(labels),
            encoding="utf-8"
        )

        converted += 1

# Generate data.yaml
yaml_path = OUTPUT / "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {OUTPUT.as_posix()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"nc: {len(classes)}\n")
    f.write("names:\n")

    for idx, name in enumerate(classes):
        f.write(f"  {idx}: '{name}'\n")

print("\n========== YOLO DATASET CREATED ==========")
print("Classes:", len(classes))
print("Training images:", len(train_images))
print("Validation images:", len(val_images))
print("Successfully converted:", converted)
print("Skipped annotations:", skipped)
print("Output folder:", OUTPUT)
print("YAML file:", yaml_path)
