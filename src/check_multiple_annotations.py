
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

csv_path = ROOT / "dataset" / "processed" / "annotations_cleaned.csv"

df = pd.read_csv(csv_path)

# Count distinct defect classes per image
image_classes = df.groupby("image_name")["defect_class"].nunique()

# Images having more than one distinct defect class
multiple_classes = image_classes[image_classes > 1]

# Images having only one distinct defect class
single_class = image_classes[image_classes == 1]

print("\n========== IMAGE ANNOTATION ANALYSIS ==========")

print("Total unique images:", df["image_name"].nunique())

print("Images with one defect class:", len(single_class))

print("Images with multiple defect classes:", len(multiple_classes))

print("\nImages containing multiple classes:")

for image in multiple_classes.index[:20]:
    labels = df[df["image_name"] == image]["defect_class"].unique()
    print(image, ":", labels)

print("\n========== ANALYSIS COMPLETE ==========")
