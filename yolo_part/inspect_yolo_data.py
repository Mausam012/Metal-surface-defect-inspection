
import pandas as pd
from pathlib import Path

# Actual dataset location
DATASET_ROOT = Path(
    r"C:\Users\sm423\Downloads\dataset\dataset\raw"
)

IMAGE_ROOT = DATASET_ROOT / "images"

# CSV location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "dataset" / "processed" / "annotations.csv"

df = pd.read_csv(CSV_PATH)

# Index images using their filenames
image_index = {}

for image in IMAGE_ROOT.rglob("*"):
    if image.is_file() and image.suffix.lower() in [
        ".jpg", ".jpeg", ".png"
    ]:
        image_index.setdefault(image.name.lower(), []).append(image)

# Resolve actual image paths
df["resolved_image"] = df["image_name"].apply(
    lambda name: image_index.get(str(name).lower(), [None])[0]
)

existing = df["resolved_image"].notna()

print("Total records:", len(df))
print("Existing images:", existing.sum())
print("Missing images:", (~existing).sum())

print("\nActual classes:")
print(df["defect_class"].value_counts())

print("\nDuplicate image filenames:")
print(sum(len(v) > 1 for v in image_index.values()))

print("\nMissing image examples:")
print(df.loc[~existing, "image_name"].head(10).to_string(index=False))
