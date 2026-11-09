
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "dataset" / "processed" / "annotations.csv"
OUTPUT_FILE = PROJECT_ROOT / "dataset" / "processed" / "annotations_cleaned.csv"

df = pd.read_csv(INPUT_FILE)

original = df.copy()

# Clip coordinates to image boundaries
df["xmin"] = df["xmin"].clip(lower=0)
df["ymin"] = df["ymin"].clip(lower=0)
df["xmax"] = df[["xmax", "width"]].min(axis=1)
df["ymax"] = df[["ymax", "height"]].min(axis=1)

# Identify invalid boxes after clipping
invalid = (
    (df["xmin"] >= df["xmax"]) |
    (df["ymin"] >= df["ymax"])
)

print("Total annotations:", len(df))
print("Coordinates changed:",
      (df[["xmin", "ymin", "xmax", "ymax"]].values !=
       original[["xmin", "ymin", "xmax", "ymax"]].values).any(axis=1).sum())

print("Invalid boxes remaining:", invalid.sum())

# Save only if every box is valid
if invalid.any():
    print("\nSome boxes still need manual inspection:")
    print(df.loc[invalid].to_string(index=False))
else:
    df.to_csv(OUTPUT_FILE, index=False)
    print("\nCleaned annotations saved:", OUTPUT_FILE)
