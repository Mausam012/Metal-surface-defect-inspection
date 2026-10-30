
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use the primary image and annotation folders.
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
IMAGE_DIR = RAW_DIR / "images" 
XML_DIR = RAW_DIR / "label" 

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "annotations.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_text(parent, tag, default=""):
    element = parent.find(tag)

    if element is None or element.text is None:
        return default

    return element.text.strip()


def main():
    if not IMAGE_DIR.exists():
        print("ERROR: Image folder not found:", IMAGE_DIR)
        return

    if not XML_DIR.exists():
        print("ERROR: XML folder not found:", XML_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Index images by their exact filename.
    image_lookup = {}

    for image_path in IMAGE_DIR.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_lookup.setdefault(image_path.name.lower(), []).append(image_path)

    rows = []
    missing_images = []
    invalid_xml = []
    missing_boxes = 0

    for xml_path in sorted(XML_DIR.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as error:
            invalid_xml.append((xml_path.name, str(error)))
            continue

        filename = get_text(root, "filename")

        # Match XML to the image using its filename.
        if filename:
            candidates = image_lookup.get(Path(filename).name.lower(), [])
        else:
            candidates = image_lookup.get(
                (xml_path.stem + ".jpg").lower(), []
            )

        if len(candidates) != 1:
            missing_images.append({
                "xml": xml_path.name,
                "matches": len(candidates)
            })
            continue

        image_path = candidates[0]

        size = root.find("size")
        width = get_text(size, "width") if size is not None else ""
        height = get_text(size, "height") if size is not None else ""

        objects = root.findall("object")

        if not objects:
            rows.append({
                "image_name": image_path.name,
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "defect_class": "unknown_or_unannotated",
                "width": width,
                "height": height,
                "xmin": "",
                "ymin": "",
                "xmax": "",
                "ymax": "",
                "annotation_file": xml_path.relative_to(PROJECT_ROOT).as_posix()
            })
            continue

        for obj in objects:
            box = obj.find("bndbox")

            if box is None:
                xmin = ymin = xmax = ymax = ""
                missing_boxes += 1
            else:
                xmin = get_text(box, "xmin")
                ymin = get_text(box, "ymin")
                xmax = get_text(box, "xmax")
                ymax = get_text(box, "ymax")

            rows.append({
                "image_name": image_path.name,
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "defect_class": (
                     get_text(obj, "name")
                     or image_path.parent.name
                ),
                "width": width,
                "height": height,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "annotation_file": xml_path.relative_to(PROJECT_ROOT).as_posix()
            })

    columns = [
        "image_name", "image_path", "defect_class",
        "width", "height", "xmin", "ymin", "xmax", "ymax",
        "annotation_file"
    ]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== CSV GENERATION REPORT =====")
    print("XML files processed:", len(list(XML_DIR.rglob("*.xml"))))
    print("CSV rows created:", len(df))
    print("XML files with invalid syntax:", len(invalid_xml))
    print("XML files with no unique matching image:", len(missing_images))
    print("Objects without bounding boxes:", missing_boxes)
    print("CSV saved to:", OUTPUT_FILE)

    if not df.empty:
        print("\nDefect classes found in XML:")
        print(df["defect_class"].value_counts().to_string())

        print("\nFirst five CSV rows:")
        print(df.head().to_string(index=False))

    if missing_images:
        print("\nFirst unmatched XML files:")
        for item in missing_images[:10]:
            print(item)

    if invalid_xml:
        print("\nFirst XML parsing errors:")
        for item in invalid_xml[:5]:
            print(item)


if __name__ == "__main__":
    main()
