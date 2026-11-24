
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

folders = [
    ROOT / "dataset" / "raw" / "images",
    ROOT / "dataset" / "raw" / "defects"
]

for folder in folders:

    print("\n==========", folder.name, "==========")

    if not folder.exists():
        print("Folder not found:", folder)
        continue

    files = [f for f in folder.rglob("*") if f.is_file()]

    print("Total files:", len(files))

    extensions = Counter(
        f.suffix.lower() for f in files
    )

    print("File types:", extensions)

    print("\nFirst 10 files:")

    for file in files[:10]:
        print(file.name)

print("\n========== CHECK COMPLETE ==========")
