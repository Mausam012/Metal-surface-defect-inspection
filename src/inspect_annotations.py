
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "dataset" / "processed" / "annotations.csv"

# Verified starting translations for GC10 labels
LABEL_MAP = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "welding_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "inclusion",
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding",
    "10_yaozhed": "waist_folding_variant_review",
    "deburring": "deburring_review",
}

def main():
    if not CSV_PATH.exists():
        print("CSV file not found:", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)

    print("\n========== DATASET SUMMARY ==========")
    print("Total annotation rows:", len(df))
    print("Columns:", df.columns.tolist())

    class_col = "defect_class"

    if class_col not in df.columns:
        print("Defect class column not found.")
        print("Please check the CSV column names.")
        return

    df[class_col] = df[class_col].astype("string").str.strip()

    print("\n========== ORIGINAL LABEL COUNTS ==========")
    print(df[class_col].value_counts(dropna=False).to_string())

    # Add English labels without changing original labels
    df["english_label"] = df[class_col].map(LABEL_MAP)

    # Preserve all unmapped labels for manual review
    unknown = df.loc[
        df["english_label"].isna(), class_col
    ].dropna().unique()

    print("\n========== LABELS NEEDING REVIEW ==========")
    if len(unknown):
        for label in sorted(unknown):
            print("-", label)
    else:
        print("No unknown labels.")

    print("\n========== ENGLISH LABEL COUNTS ==========")
    print(df["english_label"].value_counts(dropna=False).to_string())

    output_path = (
        PROJECT_ROOT / "dataset" / "processed"
        / "annotations_review.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nReview file saved:", output_path)
    print("\nInspection completed.")

if __name__ == "__main__":
    main()
