
import pandas as pd
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Load cleaned CSV
csv_path = ROOT / "dataset" / "processed" / "annotations_cleaned.csv"

df = pd.read_csv(csv_path)

print("\n========== DATASET INFORMATION ==========")

print("CSV location:", csv_path)
print("Total annotation records:", len(df))
print("Total unique images:", df["image_name"].nunique())

print("\nAvailable columns:")
print(df.columns.tolist())

# Correct class column
class_col = "defect_class"

print("\nClass distribution:")
print(df[class_col].value_counts(dropna=False))

# Check image paths
print("\n========== IMAGE PATH CHECK ==========")

missing = 0
existing = 0

for path in df["image_path"].dropna().unique():

    full_path = ROOT / str(path)

    if full_path.exists():
        existing += 1
    else:
        missing += 1

print("Existing image paths:", existing)
print("Missing image paths:", missing)

print("\n========== INSPECTION COMPLETE ==========")
