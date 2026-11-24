"""Multi-model inspection pipeline: the piece that turns four separate models into one product.

    original image (8-bit BGR, exactly what the models were trained on)
        |
        +-- YOLO ----------> defect regions (boxes + label + confidence)
        |                        |
        |   manual region? ------+   (optional rectangle drawn by the user)
        |                        v
        |                  each region is cropped
        |                        +-- ResNet-50 -> 5-class label + probabilities
        |                        +-- CNN       -> 4-class label + probabilities
        |                        +-- consensus label (majority vote of the models that saw it)
        |
        +-- U-Net ---------> pixel mask of suspicious areas over the whole image

Any subset of models can be selected. A model that is not installed is
reported as unavailable; it never contributes invented output.
"""
import time
from collections import Counter

from .base_model import STATUSES, ModelNotAvailable, make_result  # noqa: F401  (STATUSES re-exported)

PIPELINE_NAME = "Full Inspection"
ALL_MODELS = ["YOLO", "ResNet", "CNN", "U-Net"]
CLASSIFIERS = ["ResNet", "CNN"]
MAX_REGIONS = 10               # classify at most this many YOLO boxes (highest confidence first)
STRONG_CONF = 0.50             # a YOLO box at/above this makes the image "Suspicious"
MIN_MASK_FRACTION = 0.0005     # U-Net mask must cover >= 0.05 % of the image


def selected_models(choice: str):
    """'Full Inspection' -> all four, otherwise the single model name."""
    return list(ALL_MODELS) if choice == PIPELINE_NAME else [choice]


def _timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    return out, round((time.perf_counter() - t) * 1000.0, 2)


def _consensus(region):
    votes = [v["label"] for v in (region.get("yolo"), region.get("resnet"), region.get("cnn")) if v]
    if not votes:
        return None, ""
    label, n = Counter(votes).most_common(1)[0]
    return label, f"{n}/{len(votes)}"


def run_inspection(manager, bgr, models, manual_crop=None):
    """Run the selected models on one image and return a make_result()-style dict.

    bgr          8-bit BGR ndarray of the ORIGINAL upload (models were trained on raw images)
    models       list of model names (subset of ALL_MODELS)
    manual_crop  optional {"x","y","width","height"} in original-image pixels
    """
    h, w = bgr.shape[:2]
    status_by_model = {}       # name -> {"state": ran|unavailable|failed|skipped, "message", "time_ms"}
    regions = []
    mask = prob = None
    versions = {}

    def get_loaded(name):
        m = manager.get(name)
        m.ensure_loaded()
        versions[name] = m.version()
        return m

    # 1) YOLO: find regions --------------------------------------------------
    if "YOLO" in models:
        try:
            yolo = get_loaded("YOLO")
            boxes, ms = _timed(yolo.detect, bgr)
            status_by_model["YOLO"] = {"state": "ran", "time_ms": ms,
                                       "message": f"{len(boxes)} region(s) detected."}
            for b in boxes[:MAX_REGIONS]:
                regions.append({"source": "YOLO", "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                                "yolo": {"label": b["label"], "confidence": b["confidence"]}})
            if len(boxes) > MAX_REGIONS:
                status_by_model["YOLO"]["message"] += f" Top {MAX_REGIONS} kept for classification."
        except ModelNotAvailable as exc:
            status_by_model["YOLO"] = {"state": "unavailable", "message": str(exc)}
        except Exception as exc:
            status_by_model["YOLO"] = {"state": "failed", "message": f"Detection failed ({type(exc).__name__})."}

    if manual_crop:
        regions.insert(0, {"source": "Manual selection", "x": manual_crop["x"], "y": manual_crop["y"],
                           "w": manual_crop["width"], "h": manual_crop["height"]})

    # 2) Classifiers on every region ----------------------------------------
    for name in CLASSIFIERS:
        if name not in models:
            continue
        if not regions:
            hint = ("YOLO found no defect regions." if status_by_model.get("YOLO", {}).get("state") == "ran"
                    else "Draw a region on the upload page, or include YOLO to find regions automatically.")
            status_by_model[name] = {"state": "skipped", "message": f"No region to classify. {hint}"}
            continue
        try:
            clf = get_loaded(name)
            total = 0.0
            for r in regions:
                crop = bgr[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]]
                pred, ms = _timed(clf.predict_crop, crop)
                total += ms
                r[name.lower()] = {"label": pred["predicted_class"], "confidence": pred["confidence"],
                                   "probabilities": pred["class_probabilities"]}
            status_by_model[name] = {"state": "ran", "time_ms": round(total, 2),
                                     "message": f"Classified {len(regions)} region(s)."}
        except ModelNotAvailable as exc:
            status_by_model[name] = {"state": "unavailable", "message": str(exc)}
        except Exception as exc:
            status_by_model[name] = {"state": "failed", "message": f"Classification failed ({type(exc).__name__})."}

    # 3) U-Net over the whole image -----------------------------------------
    coverage = None
    if "U-Net" in models:
        try:
            unet = get_loaded("U-Net")
            (prob, mask), ms = _timed(unet.segment, bgr)
            coverage = float(mask.mean())
            status_by_model["U-Net"] = {"state": "ran", "time_ms": ms,
                                        "message": f"Mask covers {coverage * 100:.2f}% of the image."}
        except ModelNotAvailable as exc:
            status_by_model["U-Net"] = {"state": "unavailable", "message": str(exc)}
        except Exception as exc:
            status_by_model["U-Net"] = {"state": "failed", "message": f"Segmentation failed ({type(exc).__name__})."}

    # 4) Combine -------------------------------------------------------------
    for i, r in enumerate(regions, 1):
        r["id"] = i
        r["consensus"], r["agreement"] = _consensus(r)

    ran = [n for n, s in status_by_model.items() if s["state"] == "ran"]
    model_label = PIPELINE_NAME if len(models) > 1 else models[0]
    if not ran:
        reasons = "; ".join(f"{n}: {s['message']}" for n, s in status_by_model.items())
        return make_result(model_label, error=reasons or "No model could run.", models=status_by_model)

    yolo_ran = status_by_model.get("YOLO", {}).get("state") == "ran"
    yolo_regions = [r for r in regions if r["source"] == "YOLO"]
    unet_hit = coverage is not None and coverage >= MIN_MASK_FRACTION
    classified = [r for r in regions if r.get("consensus")]

    if yolo_regions:
        best = yolo_regions[0]
        status = "Suspicious" if best["yolo"]["confidence"] >= STRONG_CONF else "Requires Manual Review"
    elif classified:
        best = classified[0]
        status = "Requires Manual Review"      # classifiers have no 'normal' class
    elif unet_hit:
        best, status = None, "Suspicious" if not yolo_ran else "Requires Manual Review"
    else:
        best, status = None, "Normal"

    notes = []
    if status == "Normal":
        notes.append("No model flagged a defect. This is not proof that the part is defect-free.")
    if any(r.get("resnet") or r.get("cnn") for r in regions):
        notes.append("CNN and ResNet have no 'normal' class: they name the most likely defect type of a region.")
    if unet_hit and not yolo_regions and yolo_ran:
        notes.append("U-Net marked pixels that YOLO did not detect; inspect the mask manually.")
    notes.append("Confidences are softmax/detector scores and are not calibrated.")

    if best is not None:
        category = best["consensus"]
        confidence = (best.get("yolo") or best.get("resnet") or best.get("cnn"))["confidence"]
    else:
        category = "Unspecified (segmentation)" if unet_hit else None
        confidence = None
    score = (yolo_regions[0]["yolo"]["confidence"] if yolo_regions
             else float(prob.max()) if prob is not None else None)

    boxes = [{"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"],
              "label": f"#{r['id']} {r['consensus'] or 'region'}",
              "confidence": (r.get("yolo") or {}).get("confidence"),
              "manual": r["source"] != "YOLO"} for r in regions]

    return make_result(
        model_label,
        model_version=", ".join(f"{k}: {v}" for k, v in versions.items()) or None,
        status=status, defect_category=category, score=score, confidence=confidence,
        bounding_boxes=boxes, segmentation_mask=mask,
        inference_time_ms=round(sum(s.get("time_ms", 0) for s in status_by_model.values()), 2),
        is_real_prediction=True, notes=notes,
        regions=regions, models=status_by_model, mask_coverage=coverage,
    )
