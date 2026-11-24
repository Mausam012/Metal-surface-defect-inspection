"""YOLO module (owner: YOLO team member).

Uses the `ultralytics` package. The team's trained weights live in
yolo_part/models/balanced_v3_best.pt (picked up automatically) or models/yolo/.

The checkpoint was trained on the raw GC10-DET folder labels (pinyin codes such
as '7_yiwu'); they are translated to the English names used by the CNN and
ResNet so that all models speak the same vocabulary.

SECURITY NOTE: Ultralytics .pt files are pickle based. Only load files you
trained yourself or fully trust.
"""
import cv2

from .base_model import BaseModel, ModelNotAvailable

CONF_THRESHOLD = 0.25      # minimum confidence for a box to be reported
STRONG_CONF = 0.50         # best box below this -> "Requires Manual Review"

# GC10-DET folder codes -> English class names (same as src/standardize_labels.py)
GC10_NAMES = {
    "1_chongkong": "punching_hole", "2_hanfeng": "welding_line", "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot", "4_shuibian": "water_spot", "5_youban": "oil_spot", "6_siban": "silk_spot",
    "7_yiwu": "inclusion", "8_yahen": "rolled_pit", "9_zhehen": "crease",
    "10_yaozhe": "waist_folding", "10_yaozhed": "waist_folding", "deburring": "deburring",
}


def english_label(raw: str) -> str:
    return GC10_NAMES.get(str(raw).strip(), str(raw).strip())


class YOLOModel(BaseModel):
    name = "YOLO"
    dir_name = "yolo"
    architecture = "YOLO (Ultralytics single-stage object detector)"
    description = "Predicts bounding boxes and class labels for every defect in one forward pass over the full image."
    purpose = "Finds and localizes defects; its boxes feed the CNN and ResNet classifiers."
    expected_output = "Bounding boxes with class label and confidence."
    output_type = "Detection (bounding boxes)"
    checkpoint_hint = "models/yolo/best.pt  (falls back to ../yolo_part/models/balanced_v3_best.pt)"
    extras = ["Trained on GC10-DET box labels (12 raw classes, mapped to English names)",
              "No detections does NOT prove a part is defect-free"]

    def load(self, checkpoint_path):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ModelNotAvailable("The 'ultralytics' package is not installed (pip install ultralytics).")
        self._model = YOLO(str(checkpoint_path))
        self.meta = {"version": f"yolo / {checkpoint_path.name}",
                     "classes": sorted({english_label(n) for n in self._model.names.values()})}

    def detect(self, bgr, conf=CONF_THRESHOLD):
        """Run on an 8-bit BGR image; boxes are returned in that image's pixel coordinates."""
        if self._model is None:
            raise ModelNotAvailable("YOLO model is not loaded.")
        res = self._model.predict(bgr, conf=conf, verbose=False)[0]
        h, w = bgr.shape[:2]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            raw = str(res.names[int(b.cls[0])])
            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "label": english_label(raw),
                          "raw_label": raw, "confidence": float(b.conf[0])})
        return sorted(boxes, key=lambda d: d["confidence"], reverse=True)

    def _predict(self, gray8):
        boxes = self.detect(cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR))
        if not boxes:
            return {"status": "Normal", "notes": ["No detections above threshold; this is not proof of absence."]}
        best = boxes[0]
        status = "Suspicious" if best["confidence"] >= STRONG_CONF else "Requires Manual Review"
        return {"status": status, "defect_category": best["label"], "score": best["confidence"],
                "confidence": best["confidence"], "bounding_boxes": boxes}
