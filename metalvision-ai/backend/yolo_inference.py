"""YOLO module (owner: YOLO team member).

Uses the `ultralytics` package (pip install ultralytics). Train with your own
labelled boxes (YOLO txt format in data/annotations/) and copy the resulting
`best.pt` to models/yolo/.

SECURITY NOTE: Ultralytics .pt files are pickle based. Only load files you
trained yourself or fully trust.
"""
import cv2

from .base_model import BaseModel, ModelNotAvailable

CONF_THRESHOLD = 0.25      # minimum confidence for a box to be reported
STRONG_CONF = 0.50         # best box below this -> "Requires Manual Review"


class YOLOModel(BaseModel):
    name = "YOLO"
    dir_name = "yolo"
    architecture = "You Only Look Once (single-stage object detector)"
    description = "Predicts bounding boxes and class labels in a single forward pass over the image."
    purpose = "Defect detection and localization."
    expected_output = "Bounding boxes with class label and confidence."
    output_type = "Bounding boxes"
    checkpoint_hint = "models/yolo/best.pt  (Ultralytics weights)"
    extras = ["Needs box annotations (YOLO txt format)", "No detections does NOT prove a part is defect-free"]

    def load(self, checkpoint_path):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ModelNotAvailable("The 'ultralytics' package is not installed (pip install ultralytics).")
        self._model = YOLO(str(checkpoint_path))
        self.meta = {"version": f"yolo / {checkpoint_path.name}"}

    def _predict(self, gray8):
        frame = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        res = self._model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        names = res.names
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            boxes.append({"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1),
                          "label": str(names[int(b.cls[0])]), "confidence": float(b.conf[0])})
        if not boxes:
            return {"status": "Normal", "notes": ["No detections above threshold; this is not proof of absence."]}
        best = max(boxes, key=lambda d: d["confidence"])
        status = "Suspicious" if best["confidence"] >= STRONG_CONF else "Requires Manual Review"
        return {"status": status, "defect_category": best["label"], "score": best["confidence"],
                "confidence": best["confidence"], "bounding_boxes": boxes}
