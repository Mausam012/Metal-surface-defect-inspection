
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

files = [
    ROOT / "dataset" / "processed" / "cnn_crops_manifest.csv",
    ROOT / "dataset" / "processed" / "cnn_split_manifest.csv"
]

for file in files:

    print("\n" + "=" * 50)
    print("FILE:", file.name)
    print("=" * 50)

    if not file.exists():
        print("File not found!")
        continue

    df = pd.read_csv(file)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head().to_string(index=False))

    print("\nTotal records:", len(df))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    if "split" in df.columns:
        print("\nSplit distribution:")
        print(df["split"].value_counts().to_string())

    if "class_name" in df.columns:
        print("\nClass distribution:")
        print(df["class_name"].value_counts().to_string())

print("\nMANIFEST INSPECTION COMPLETED")
