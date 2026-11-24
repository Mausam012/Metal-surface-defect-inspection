
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

# Project paths
CSV_PATH = "dataset/processed/annotations.csv"
IMAGE_DIR = "dataset/raw/images"
MASK_DIR = "dataset/masks"

os.makedirs(MASK_DIR, exist_ok=True)

# Read annotations
df = pd.read_csv(CSV_PATH)

created = 0
missing = 0

for image_name, group in df.groupby("image_name"):

    image_path = group.iloc[0]["image_path"]

    if not os.path.exists(image_path):
        print("Missing image:", image_path)
        missing += 1
        continue

    # Read actual image dimensions
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Create a black mask
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Draw each defect bounding box in white
    for _, row in group.iterrows():

        xmin = max(0, min(width - 1, int(row["xmin"])))
        ymin = max(0, min(height - 1, int(row["ymin"])))
        xmax = max(0, min(width - 1, int(row["xmax"])))
        ymax = max(0, min(height - 1, int(row["ymax"])))

        if xmax >= xmin and ymax >= ymin:
            draw.rectangle(
                [xmin, ymin, xmax, ymax],
                fill=255
            )

    # Save mask using image filename
    mask_name = os.path.splitext(image_name)[0] + ".png"
    mask_path = os.path.join(MASK_DIR, mask_name)
    mask.save(mask_path)

    created += 1

print("\nMask generation completed!")
print("Masks created:", created)
print("Missing images:", missing)
print("Saved in:", MASK_DIR)
