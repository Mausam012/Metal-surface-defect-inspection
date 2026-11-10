"""PDF inspection report (ReportLab). Clearly separates real inference from demonstration mode."""
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .visualization import WATERMARK

NAVY, CYAN = colors.HexColor("#0F2540"), colors.HexColor("#0E8FA0")


def _val(v, suffix=""):
    return "Not available" if v is None or v == "" else f"{v}{suffix}"


def _img(path: Path, max_w, max_h):
    with PILImage.open(path) as im:
        w, h = im.size
    k = min(max_w / w, max_h / h)
    return Image(str(path), width=w * k, height=h * k)


def build_report(rec: dict, insp_dir: Path, out_path: Path, app_name: str, version: str) -> Path:
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=NAVY, fontSize=20, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=12, spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#4A5568"))

    real = bool(rec.get("is_real_prediction"))
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title=f"{app_name} inspection report")
    el = [Paragraph(app_name, h1),
          Paragraph(f"Inspection report \u2014 MV-{rec['id']:06d} \u2014 prototype v{escape(version)}", small), Spacer(1, 6)]

    if real:
        banner, bg = ("REAL MODEL INFERENCE. Result produced by a loaded model checkpoint. Uncalibrated; "
                      "all highlighted regions require verification by a qualified human inspector."), colors.HexColor("#E6F4EA")
    else:
        banner, bg = ("DEMONSTRATION MODE \u2014 NO AI PREDICTION. No trained model produced this result. "
                      "Any highlighted region is a " + WATERMARK.lower() + "."), colors.HexColor("#FFF4D6")
    t = Table([[Paragraph(f"<b>{escape(banner)}</b>", body)]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.8, NAVY), ("PADDING", (0, 0), (-1, -1), 6)]))
    el += [t, Spacer(1, 6)]

    boxes = (rec.get("result_json") or {}).get("bounding_boxes") or []
    location = f"{len(boxes)} bounding box(es)" if boxes else None
    rows = [("Inspection ID", f"MV-{rec['id']:06d}"), ("Image filename", escape(rec["original_filename"])),
            ("Uploaded", _val(rec.get("uploaded_at"))), ("Analyzed", _val(rec.get("analyzed_at"))),
            ("Image", f"{_val(rec.get('image_format'))}, {_val(rec.get('width'))} x {_val(rec.get('height'))} px, "
                      f"{_val(rec.get('bit_depth'))}-bit"),
            ("Model used", _val(rec.get("model_used"))), ("Model version", _val(rec.get("model_version"))),
            ("Mode", "Real inference" if real else "Demonstration (no trained model)"),
            ("Inspection status", rec["status"]), ("Defect category", _val(rec.get("defect_category"))),
            ("Anomaly / defect score", _val(None if rec.get("score") is None else round(rec["score"], 4))),
            ("Defect location", _val(location)),
            ("Inference time", _val(rec.get("inference_time_ms"), " ms")),
            ("Human review status", rec.get("review_status", "Pending Review")),
            ("Reviewer note", escape(rec.get("review_note") or "\u2014"))]
    tbl = Table([[Paragraph(f"<b>{k}</b>", body), Paragraph(str(v), body)] for k, v in rows], colWidths=[48 * mm, 126 * mm])
    tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
                             ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F4F6F9")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    el += [tbl, Paragraph("Preprocessing settings (reproducibility)", h2),
           Paragraph(escape(str(rec.get("preprocess_settings") or "not recorded")), small)]

    el.append(Paragraph("Images", h2))
    pics = []
    for title, name in (("Original (display copy)", "original_preview.png"), ("Processed", "processed.png")):
        p = insp_dir / name
        if p.exists():
            pics.append([Paragraph(f"<b>{title}</b>", small), _img(p, 82 * mm, 70 * mm)])
    if pics:
        t2 = Table([[c[0] for c in pics], [c[1] for c in pics]])
        el.append(KeepTogether(t2))
    overlays = rec.get("overlays") or {}
    for kind, fname in overlays.items():
        p = insp_dir / fname
        if p.exists():
            cap = f"{kind.capitalize()} overlay" + ("" if real else f" \u2014 {WATERMARK}")
            el += [Spacer(1, 6), KeepTogether([Paragraph(f"<b>{escape(cap)}</b>", small), _img(p, 120 * mm, 80 * mm)])]
    el += [Spacer(1, 10), Paragraph(
        "Academic prototype. Heatmaps (e.g. Grad-CAM) show evidence regions, not exact defect boundaries. "
        "Outputs must be verified by a human. Not for safety-critical use.", small)]
    doc.build(el)
    return out_path
