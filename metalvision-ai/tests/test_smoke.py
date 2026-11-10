"""End-to-end smoke test of the demo flow (no PyTorch required).  Run:  python -m pytest -q   or   python tests/test_smoke.py"""
import io
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import create_app  # noqa: E402


def make_client():
    tmp = Path(tempfile.mkdtemp())
    app = create_app({"TESTING": True, "UPLOAD_DIR": tmp / "up", "PROCESSED_DIR": tmp / "proc", "REPORT_DIR": tmp / "rep",
                      "DB_PATH": tmp / "db" / "t.db", "MODELS_DIR": tmp / "models"})
    return app.test_client()


def png_bytes(dtype=np.uint8, w=300, h=200):
    arr = (np.random.default_rng(1).random((h, w)) * (255 if dtype == np.uint8 else 65535)).astype(dtype)
    return cv2.imencode(".png", arr)[1].tobytes()


def test_full_flow():
    c = make_client()
    for page in ("/", "/upload", "/models", "/history", "/about"):
        assert c.get(page).status_code == 200, page

    r = c.post("/api/upload", data={"image": (io.BytesIO(png_bytes(np.uint16)), "sheet 1.png"), "analyze_with": "ResNet"})
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    iid = out["id"]
    assert c.get(out["redirect"]).status_code == 200

    rec = c.get(f"/api/inspections/{iid}").get_json()
    assert rec["status"] == "Model Not Available" and rec["is_real_prediction"] == 0 and rec["mode"] == "demonstration"
    assert rec["score"] is None and rec["bit_depth"] == 16
    assert "stored_filename" not in rec                              # internal names not exposed

    for name in ("original_preview.png", "gray.png", "normalized.png", "processed.png", "overlay_heatmap.png", "overlay_boxes.png", "overlay_mask.png"):
        assert c.get(f"/files/{iid}/{name}").status_code == 200, name
    assert c.get(f"/files/{iid}/../../app.py").status_code == 404
    assert c.get(f"/files/{iid}/secret.txt").status_code == 404

    assert c.post(f"/api/preprocess/{iid}", json={"normalization": "minmax", "max_side": 256}).status_code == 200
    assert c.post(f"/api/analyze/{iid}", json={"model": "YOLO"}).status_code == 200
    assert c.post(f"/api/analyze/{iid}", json={"model": "Nope"}).status_code == 400
    assert c.post(f"/api/inspections/{iid}/review", json={"review_status": "False Alarm", "note": "ok"}).status_code == 200

    pdf = c.get(f"/report/{iid}.pdf")
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF")

    assert len(c.get("/api/inspections?q=sheet").get_json()["inspections"]) == 1
    assert len(c.get("/api/inspections?status=Normal").get_json()["inspections"]) == 0
    s = c.get("/api/stats").get_json()
    assert s["total_inspected"] == 1 and s["real_predictions"] == 0 and s["normal"] == 0

    assert c.delete(f"/api/inspections/{iid}").status_code == 200
    assert c.get(f"/api/inspections/{iid}").status_code == 404


def test_validation():
    c = make_client()
    bad = [(io.BytesIO(b"not an image"), "x.png"), (io.BytesIO(b""), "x.png"), (io.BytesIO(png_bytes()), "x.exe"),
           (io.BytesIO(b"\x89PNG\r\n\x1a\ngarbage"), "x.png"), (io.BytesIO(png_bytes()), "x.jpg")]
    for data, name in bad:
        r = c.post("/api/upload", data={"image": (data, name)})
        assert r.status_code == 400 and "error" in r.get_json(), name
    assert c.post("/api/upload", data={}).status_code == 400


def test_demo_records():
    c = make_client()
    assert c.post("/api/history/seed-demo").get_json()["added"] == 6
    assert c.get("/api/stats").get_json()["total_uploaded"] == 0       # demo excluded by default
    assert c.get("/api/stats?demo=1").get_json()["total_uploaded"] == 6
    assert c.get("/results/1").status_code == 200
    assert c.get("/report/1.pdf").status_code == 200
    assert c.delete("/api/history/demo").get_json()["removed"] == 6


def test_real_prediction_path():
    """A fake model proves that real results are stored/labelled as real and never as demonstration."""
    from backend.base_model import BaseModel

    class Fake(BaseModel):
        name, dir_name = "ResNet", "resnet"
        def find_checkpoint(self): return Path("fake.pt")
        def ensure_loaded(self): pass
        def _predict(self, gray8):
            h, w = gray8.shape
            return {"status": "Suspicious", "defect_category": "porosity", "score": 0.9, "confidence": 0.9,
                    "heatmap": np.ones((h, w), np.float32) * 0.5, "segmentation_mask": np.ones((h, w), np.uint8),
                    "bounding_boxes": [{"x": 5, "y": 5, "w": 20, "h": 20, "label": "porosity", "confidence": 0.9}]}

    tmp = Path(tempfile.mkdtemp())
    app = create_app({"TESTING": True, "UPLOAD_DIR": tmp / "u", "PROCESSED_DIR": tmp / "p", "REPORT_DIR": tmp / "r",
                      "DB_PATH": tmp / "d" / "t.db", "MODELS_DIR": tmp / "m"})
    c = app.test_client()
    mgr = app.extensions["mv_manager"]
    mgr._models["ResNet"] = Fake(tmp / "m")
    out = c.post("/api/upload", data={"image": (io.BytesIO(png_bytes()), "a.png"), "analyze_with": "ResNet"}).get_json()
    rec = c.get(f"/api/inspections/{out['id']}").get_json()
    assert rec["is_real_prediction"] == 1 and rec["mode"] == "real" and rec["status"] == "Suspicious"
    assert rec["score"] == 0.9 and rec["inference_time_ms"] is not None
    assert set(rec["overlays"]) == {"heatmap", "boxes", "mask"}
    assert b"DEMONSTRATION" not in c.get(out["redirect"]).data
    assert c.get(f"/report/{out['id']}.pdf").status_code == 200


if __name__ == "__main__":
    test_full_flow(); test_validation(); test_demo_records(); test_real_prediction_path(); print("all smoke tests passed")
