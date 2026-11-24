
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILE = (
    ROOT / "dataset" / "processed"
    / "cnn_split_manifest.csv"
)

df = pd.read_csv(FILE)

print("\n========== DATA LEAKAGE CHECK ==========")

print("\nCrops in each split:")
print(df["split"].value_counts())

# Unique original images in each split
for split in ["train", "validation", "test"]:

    subset = df[df["split"] == split]

    print(
        f"\n{split}: "
        f"{subset['original_image'].nunique()} unique original images"
    )

# Check overlap between original images
train = set(
    df[df["split"] == "train"]["original_image"]
)

validation = set(
    df[df["split"] == "validation"]["original_image"]
)

test = set(
    df[df["split"] == "test"]["original_image"]
)

train_val = train & validation
train_test = train & test
val_test = validation & test

print("\nOriginal image overlap:")

print("Train vs Validation:", len(train_val))
print("Train vs Test:", len(train_test))
print("Validation vs Test:", len(val_test))

if not train_val and not train_test and not val_test:
    print("\nPASS: No original-image leakage detected.")
else:
    print("\nWARNING: Original images overlap between splits.")

print("\nCHECK COMPLETED")
