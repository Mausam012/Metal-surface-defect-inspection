# data/annotations

Placeholder for **expert-reviewed** defect annotations of flat-metal-sheet radiographs:
- YOLO: one `.txt` per image (`class x_center y_center width height`, normalized)
- U-Net: one binary mask PNG per image
- ResNet / CNN: `manifest.csv` with columns `path,label,group[,split]`

Record for every file: source, material, thickness, geometry, annotator, date, and usage license.
