
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
IMAGE_DIR = RAW_DIR / "images"
XML_DIR = RAW_DIR / "label"

xml_classes = Counter()
folder_classes = Counter()
missing_xml_names = 0
examples = []

for xml_path in sorted(XML_DIR.rglob("*.xml")):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        continue

    filename = root.findtext("filename", "").strip()
    candidates = list(IMAGE_DIR.rglob(filename)) if filename else []

    if len(candidates) != 1:
        candidates = list(IMAGE_DIR.rglob(xml_path.stem + ".jpg"))

    if len(candidates) != 1:
        continue

    folder_class = candidates[0].parent.name
    objects = root.findall("object")

    for obj in objects:
        xml_name = obj.findtext("name", "").strip()

        if xml_name:
            xml_classes[xml_name] += 1
        else:
            missing_xml_names += 1

        folder_classes[folder_class] += 1

        if len(examples) < 15:
            examples.append(
                (xml_path.name, folder_class, xml_name or "(missing)")
            )

print("===== XML LABEL COUNTS =====")
for name, count in xml_classes.most_common():
    print(f"{name}: {count}")

print("\n===== IMAGE FOLDER LABEL COUNTS =====")
for name, count in folder_classes.most_common():
    print(f"{name}: {count}")

print("\nObjects missing XML class names:", missing_xml_names)

print("\n===== EXAMPLES =====")
for filename, folder, xml_name in examples:
    print(f"{filename} | folder={folder} | XML class={xml_name}")

from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from collections import Counter

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

image_files = [
    p for p in DATASET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

xml_files = list(DATASET_DIR.rglob("*.xml"))

print("\n========== DATASET AUDIT ==========")
print("Total images:", len(image_files))
print("Total XML files:", len(xml_files))

# Check image readability
corrupted_images = []

for image_path in image_files:
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception:
        corrupted_images.append(str(image_path))

print("\nCorrupted images:", len(corrupted_images))

# Check duplicate filenames

from collections import defaultdict
import hashlib

# Group images by filename
filename_groups = defaultdict(list)

for path in image_files:
    filename_groups[path.name].append(path)

duplicates = {
    name: paths
    for name, paths in filename_groups.items()
    if len(paths) > 1
}

print("\nDuplicate filenames:", len(duplicates))

# Compare actual image contents using SHA256
def calculate_hash(path):
    sha = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha.update(chunk)

    return sha.hexdigest()

identical_groups = []

for name, paths in duplicates.items():
    hashes = {}

    for path in paths:
        hashes.setdefault(calculate_hash(path), []).append(path)

    for group in hashes.values():
        if len(group) > 1:
            identical_groups.append(group)

print("Groups with identical image content:",
      len(identical_groups))

print("\nSample duplicate filenames:")

for name, paths in list(duplicates.items())[:5]:
    print(f"\n{name}")

    for path in paths:
        print("  ", path.relative_to(DATASET_DIR))

print("\nIdentical image examples:")

for group in identical_groups[:5]:
    for path in group:
        print("  ", path.relative_to(DATASET_DIR))

# Check XML annotations
classes = Counter()
invalid_boxes = []
missing_fields = []
xml_errors = []

for xml_path in xml_files:
    try:
        root = ET.parse(xml_path).getroot()

        for obj in root.findall("object"):
            name = obj.findtext("name")

            if not name:
                missing_fields.append(str(xml_path))
                continue

            classes[name] += 1

            box = obj.find("bndbox")

            if box is None:
                invalid_boxes.append(str(xml_path))
                continue

            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))

                if xmin >= xmax or ymin >= ymax:
                    invalid_boxes.append(str(xml_path))

            except (TypeError, ValueError):
                invalid_boxes.append(str(xml_path))

    except ET.ParseError:
        xml_errors.append(str(xml_path))

print("\n========== ANNOTATION RESULTS ==========")
print("XML parsing errors:", len(xml_errors))
print("Missing class names:", len(missing_fields))
print("Invalid bounding boxes:", len(invalid_boxes))

print("\n========== DEFECT CLASSES ==========")

for class_name, count in classes.items():
    print(f"{class_name}: {count}")

print("\n========== AUDIT COMPLETED ==========")
