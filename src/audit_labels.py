
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
