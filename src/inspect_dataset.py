
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def main():
    images = [
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    xml_files = [
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() == ".xml"
    ]

    print("===== DATASET REPORT =====")
    print("Images found:", len(images))
    print("XML annotations found:", len(xml_files))

    print("\nImage folders:")
    for folder in sorted({p.parent for p in images}):
        print(f"{folder.relative_to(RAW_DIR)}: "
              f"{sum(1 for p in images if p.parent == folder)} images")

    print("\nSample image files:")
    for p in images[:5]:
        print(p.relative_to(RAW_DIR))

    print("\nSample XML files:")
    for p in xml_files[:5]:
        print(p.relative_to(RAW_DIR))

    if xml_files:
        print("\n===== FIRST XML STRUCTURE =====")
        try:
            root = ET.parse(xml_files[0]).getroot()
            print(ET.tostring(root, encoding="unicode")[:3000])
        except ET.ParseError as error:
            print("XML parsing error:", error)


if __name__ == "__main__":
    main()
