
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "dataset" / "processed"
    / "annotations_cleaned.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "dataset" / "processed"
    / "annotations_training_ready.csv"
)

EXCLUDED_FILE = (
    PROJECT_ROOT / "dataset" / "processed"
    / "annotations_excluded_for_review.csv"
)

# Official GC10-DET classes
CLASS_MAP = {
    "1_chongkong": ("punching_hole", 0),
    "2_hanfeng": ("welding_line", 1),
    "3_yueyawan": ("crescent_gap", 2),
    "4_shuiban": ("water_spot", 3),
    "5_youban": ("oil_spot", 4),
    "6_siban": ("silk_spot", 5),
    "7_yiwu": ("inclusion", 6),
    "8_yahen": ("rolled_pit", 7),
    "9_zhehen": ("crease", 8),
    "10_yaozhe": ("waist_folding", 9),
}

# Known spelling variations
ALIASES = {
    "4_shuibian": "4_shuiban",
    "10_yaozhed": "10_yaozhe",
}

df = pd.read_csv(INPUT_FILE)

df["original_label"] = df["defect_class"].astype(str).str.strip()

df["standard_label"] = df["original_label"].replace(ALIASES)

df["class_name"] = df["standard_label"].map(
    lambda x: CLASS_MAP[x][0] if x in CLASS_MAP else None
)

df["class_id"] = df["standard_label"].map(
    lambda x: CLASS_MAP[x][1] if x in CLASS_MAP else None
)

# Keep only verified GC10-DET classes
training_df = df[df["class_name"].notna()].copy()

# Preserve other records for manual review
excluded_df = df[df["class_name"].isna()].copy()

training_df["class_id"] = training_df["class_id"].astype(int)

training_df.to_csv(OUTPUT_FILE, index=False)
excluded_df.to_csv(EXCLUDED_FILE, index=False)

print("\n========== LABEL STANDARDIZATION ==========")

print("Original annotation records:", len(df))
print("Training-ready annotations:", len(training_df))
print("Excluded for review:", len(excluded_df))

print("\nStandardized defect distribution:")
print(training_df["class_name"].value_counts().to_string())

print("\nOriginal label corrections:")
print(df["original_label"].value_counts().to_string())

print("\nSaved files:")
print(OUTPUT_FILE)
print(EXCLUDED_FILE)

print("\nLABEL STANDARDIZATION COMPLETED")
