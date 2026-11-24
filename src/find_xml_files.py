
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

print("\n========== SEARCHING XML FILES ==========")

xml_files = list(DATASET.rglob("*.xml"))

print("Total XML files found:", len(xml_files))

print("\nXML folder locations:")

folders = sorted(set(file.parent for file in xml_files))

for folder in folders:
    print(folder)
    print("Files:", len(list(folder.glob("*.xml"))))

print("\nFirst 15 XML files:")

for file in xml_files[:15]:
    print(file.relative_to(ROOT))

print("\n========== SEARCH COMPLETE ==========")
