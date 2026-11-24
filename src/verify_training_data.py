
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

csv_path = (
    ROOT / "dataset" / "processed"
    / "annotations_training_ready.csv"
)

df = pd.read_csv(csv_path)

print("\n========== TRAINING DATA VERIFICATION ==========")

print("Total annotations:", len(df))

print("Unique images:", df["image_name"].nunique())

print("\nStandardized classes:")
print(df["class_name"].value_counts())

print("\nClass IDs:")
print(
    df.groupby("class_name")["class_id"]
    .unique()
)

print("\nMissing values:")
print(df[
    ["image_name", "image_path", "class_name",
     "class_id", "xmin", "ymin", "xmax", "ymax"]
].isnull().sum())

print("\n========== VERIFICATION COMPLETE ==========")
