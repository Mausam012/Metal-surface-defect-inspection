"""SQLite persistence for inspections (plain sqlite3, no ORM, easy to read)."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

STATUSES = ["Uploaded", "Normal", "Suspicious", "Requires Manual Review", "Model Not Available"]
REVIEW_STATUSES = ["Pending Review", "Confirmed Defect", "False Alarm", "Needs Re-scan"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_filename TEXT NOT NULL,
    stored_filename TEXT,
    uploaded_at TEXT NOT NULL,
    analyzed_at TEXT,
    file_size INTEGER, image_format TEXT, width INTEGER, height INTEGER, bit_depth INTEGER,
    preprocess_settings TEXT, preprocess_meta TEXT,
    model_used TEXT, model_version TEXT,
    status TEXT NOT NULL DEFAULT 'Uploaded',
    defect_category TEXT, score REAL, confidence REAL, inference_time_ms REAL,
    is_real_prediction INTEGER NOT NULL DEFAULT 0,
    mode TEXT,                       -- 'real' | 'demonstration' | NULL (not analyzed)
    overlays TEXT, result_json TEXT,
    review_status TEXT NOT NULL DEFAULT 'Pending Review', review_note TEXT,
    is_demo_data INTEGER NOT NULL DEFAULT 0
);
"""
_JSON_COLS = {"preprocess_settings", "preprocess_meta", "overlays", "result_json"}
_COLS = {"original_filename", "stored_filename", "uploaded_at", "analyzed_at", "file_size", "image_format",
         "width", "height", "bit_depth", "preprocess_settings", "preprocess_meta", "model_used", "model_version",
         "status", "defect_category", "score", "confidence", "inference_time_ms", "is_real_prediction", "mode",
         "overlays", "result_json", "review_status", "review_note", "is_demo_data"}


@contextmanager
def _conn(db_path):
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _now(offset_days=0):
    return (datetime.now() - timedelta(days=offset_days)).isoformat(timespec="seconds")


def _row(r):
    if r is None:
        return None
    d = dict(r)
    for c in _JSON_COLS:
        if d.get(c):
            try:
                d[c] = json.loads(d[c])
            except ValueError:
                d[c] = None
    return d


def init_db(db_path):
    with _conn(db_path) as c:
        c.executescript(SCHEMA)


def create_inspection(db_path, **fields) -> int:
    fields.setdefault("uploaded_at", _now())
    return _insert(db_path, fields)


def _insert(db_path, fields) -> int:
    bad = set(fields) - _COLS
    if bad:
        raise ValueError(f"Unknown columns: {bad}")
    fields = {k: (json.dumps(v) if k in _JSON_COLS and v is not None else v) for k, v in fields.items()}
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    with _conn(db_path) as c:
        return c.execute(f"INSERT INTO inspections ({cols}) VALUES ({marks})", list(fields.values())).lastrowid


def update_inspection(db_path, insp_id, **fields):
    bad = set(fields) - _COLS
    if bad:
        raise ValueError(f"Unknown columns: {bad}")
    fields = {k: (json.dumps(v) if k in _JSON_COLS and v is not None else v) for k, v in fields.items()}
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _conn(db_path) as c:
        c.execute(f"UPDATE inspections SET {sets} WHERE id = ?", [*fields.values(), insp_id])


def get_inspection(db_path, insp_id):
    with _conn(db_path) as c:
        return _row(c.execute("SELECT * FROM inspections WHERE id = ?", (insp_id,)).fetchone())


def latest_id(db_path):
    with _conn(db_path) as c:
        r = c.execute("SELECT id FROM inspections WHERE is_demo_data = 0 ORDER BY id DESC LIMIT 1").fetchone()
        return r["id"] if r else None


def list_inspections(db_path, q="", status="", include_demo=True, limit=500):
    sql, args = "SELECT * FROM inspections WHERE 1=1", []
    if q:
        sql += " AND (original_filename LIKE ? OR model_used LIKE ? OR defect_category LIKE ? OR ('MV-' || printf('%06d', id)) LIKE ?)"
        args += [f"%{q}%"] * 4
    if status:
        sql += " AND status = ?"
        args.append(status)
    if not include_demo:
        sql += " AND is_demo_data = 0"
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _conn(db_path) as c:
        return [_row(r) for r in c.execute(sql, args).fetchall()]


def delete_inspection(db_path, insp_id) -> bool:
    with _conn(db_path) as c:
        return c.execute("DELETE FROM inspections WHERE id = ?", (insp_id,)).rowcount > 0


def stats(db_path, include_demo=False) -> dict:
    where = "" if include_demo else "WHERE is_demo_data = 0"
    with _conn(db_path) as c:
        by_status = {r["status"]: r["n"] for r in c.execute(
            f"SELECT status, COUNT(*) n FROM inspections {where} GROUP BY status")}
        real = c.execute(f"SELECT COUNT(*) n FROM inspections {where}{' AND' if where else 'WHERE'} is_real_prediction = 1").fetchone()["n"]
        demo_rows = c.execute("SELECT COUNT(*) n FROM inspections WHERE is_demo_data = 1").fetchone()["n"]
        days = [r for r in c.execute(
            f"SELECT substr(uploaded_at,1,10) d, COUNT(*) n FROM inspections {where} GROUP BY d ORDER BY d DESC LIMIT 14")]
    total = sum(by_status.values())
    return {
        "total_uploaded": total,
        "total_inspected": total - by_status.get("Uploaded", 0),
        "normal": by_status.get("Normal", 0),
        "suspicious": by_status.get("Suspicious", 0),
        "manual_review": by_status.get("Requires Manual Review", 0),
        "model_not_available": by_status.get("Model Not Available", 0),
        "awaiting_analysis": by_status.get("Uploaded", 0),
        "real_predictions": real,
        "demo_records_in_db": demo_rows,
        "by_status": by_status,
        "by_day": [{"date": r["d"], "count": r["n"]} for r in reversed(days)],
    }


def seed_demo(db_path) -> int:
    """Insert clearly flagged DEMO records (no images, no scores). Real data is never touched."""
    samples = [("DEMO_sheet_A01.png", "Normal", "ResNet", "Pending Review"),
               ("DEMO_sheet_A02.png", "Suspicious", "YOLO", "Confirmed Defect"),
               ("DEMO_sheet_B01.png", "Requires Manual Review", "U-Net", "Pending Review"),
               ("DEMO_sheet_B02.png", "Normal", "CNN", "Pending Review"),
               ("DEMO_sheet_C01.png", "Suspicious", "ResNet", "False Alarm"),
               ("DEMO_sheet_C02.png", "Model Not Available", "ResNet", "Pending Review")]
    for i, (name, status, model, review) in enumerate(samples):
        _insert(db_path, {"original_filename": name, "uploaded_at": _now(len(samples) - i), "analyzed_at": _now(len(samples) - i),
                          "model_used": model, "status": status, "review_status": review, "mode": "demo-seed",
                          "is_demo_data": 1, "image_format": "PNG"})
    return len(samples)


def delete_demo(db_path) -> int:
    with _conn(db_path) as c:
        return c.execute("DELETE FROM inspections WHERE is_demo_data = 1").rowcount
