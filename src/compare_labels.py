
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
IMAGE_DIR = RAW_DIR / "images"
XML_DIR = RAW_DIR / "label"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    image_lookup = defaultdict(list)

    for path in IMAGE_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_lookup[path.stem.lower()].append(path)

    folder_vs_xml = Counter()
    missing_names_by_folder = Counter()
    unmatched_xml = []
    examples = defaultdict(list)

    for xml_path in sorted(XML_DIR.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue

        filename = root.findtext("filename", "").strip()
        stem = Path(filename).stem.lower() if filename else xml_path.stem.lower()

        candidates = image_lookup.get(stem, [])

        if len(candidates) != 1:
            unmatched_xml.append((xml_path.name, len(candidates)))
            continue

        image_path = candidates[0]
        folder_label = image_path.parent.name

        objects = root.findall("object")

        for obj in objects:
            xml_label = (obj.findtext("name") or "").strip()

            if not xml_label:
                missing_names_by_folder[folder_label] += 1
                xml_label = "(missing)"

            folder_vs_xml[(folder_label, xml_label)] += 1

            if len(examples[(folder_label, xml_label)]) < 3:
                examples[(folder_label, xml_label)].append(xml_path.name)

    print("===== FOLDER LABEL -> XML LABEL =====")

    for (folder, xml_label), count in sorted(folder_vs_xml.items()):
        print(f"{folder:20} -> {xml_label:20} : {count}")

    print("\n===== MISSING XML LABELS BY FOLDER =====")
    for folder, count in missing_names_by_folder.items():
        print(f"{folder}: {count}")

    print("\n===== UNMATCHED XML FILES =====")
    print("Count:", len(unmatched_xml))
    for item in unmatched_xml[:10]:
        print(item)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "label_comparison.txt"

    with report_path.open("w", encoding="utf-8") as file:
        for (folder, xml_label), count in sorted(folder_vs_xml.items()):
            file.write(f"{folder} -> {xml_label}: {count}\n")

    print("\nReport saved to:", report_path)


if __name__ == "__main__":
    main()
