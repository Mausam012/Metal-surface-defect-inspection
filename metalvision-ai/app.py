"""METALVISION AI - Flask application (routes only; logic lives in backend/).

Run:  python app.py      then open http://127.0.0.1:5000
"""
import shutil
import uuid
from pathlib import Path

import cv2
from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   send_file, send_from_directory, url_for)
from werkzeug.utils import secure_filename

import config as cfg
from backend import history, preprocessing, report_generator, validation, visualization
from backend.inference import predict
from backend.model_manager import ModelManager

# Only these generated files may be served (prevents path traversal / directory listing).
SERVABLE = {"original_preview.png", "gray.png", "normalized.png", "processed.png",
            "overlay_heatmap.png", "overlay_boxes.png", "overlay_mask.png"}

NAV = [("dashboard", "Dashboard", "dashboard"), ("upload_page", "Upload", "upload"),
       ("preprocess_entry", "Preprocessing", "sliders"), ("results_entry", "Results", "scan"),
       ("models_page", "ML Models", "cpu"), ("history_page", "History", "clock"), ("about_page", "About", "info")]


def create_app(test_config=None):
    app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
    app.config.from_object(cfg.Config)
    if test_config:
        app.config.update(test_config)
    for key in ("UPLOAD_DIR", "PROCESSED_DIR", "REPORT_DIR"):
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)
    Path(app.config["DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    DB = app.config["DB_PATH"]
    history.init_db(DB)
    manager = ModelManager(app.config["MODELS_DIR"])
    app.extensions["mv_manager"] = manager          # lets tests / other code reach the registry

    # ------------------------------------------------------------------ helpers
    def insp_dir(insp_id) -> Path:
        return Path(app.config["PROCESSED_DIR"]) / str(int(insp_id))

    def sanitize_result(result: dict) -> dict:
        """JSON-safe copy of a model result (arrays are replaced by flags)."""
        out = {k: v for k, v in result.items() if k not in ("heatmap", "segmentation_mask")}
        out["has_heatmap"] = result.get("heatmap") is not None
        out["has_mask"] = result.get("segmentation_mask") is not None
        return out

    def run_analysis(insp_id: int, model_name: str):
        d = insp_dir(insp_id)
        gray = cv2.imread(str(d / "processed.png"), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError("Processed image is missing. Run preprocessing again.")
        result = predict(gray, model_name, manager)
        real = bool(result["is_real_prediction"])
        overlays = visualization.render_overlays(gray, result, d, simulated=not real)
        history.update_inspection(
            DB, insp_id, model_used=result["model_name"], model_version=result["model_version"],
            status=result["status"], defect_category=result["defect_category"], score=result["score"],
            confidence=result["confidence"], inference_time_ms=result["inference_time_ms"],
            is_real_prediction=int(real), mode="real" if real else "demonstration",
            overlays=overlays, result_json=sanitize_result(result), analyzed_at=history._now())
        return result

    def api_error(message, code=400):
        return jsonify(error=message), code

    def get_or_404(insp_id):
        rec = history.get_inspection(DB, insp_id)
        if rec is None:
            abort(404)
        return rec

    # ---------------------------------------------------------- template helpers
    @app.template_filter("mv_id")
    def mv_id(i):
        return f"MV-{int(i):06d}"

    @app.template_filter("num")
    def num(v, digits=3):
        return "Not available" if v is None else f"{v:.{digits}f}"

    @app.context_processor
    def inject_globals():
        return {"APP_NAME": cfg.APP_NAME, "APP_TAGLINE": cfg.APP_TAGLINE, "APP_VERSION": cfg.APP_VERSION,
                "NAV": NAV, "MODEL_NAMES": manager.names(), "models_integrated": manager.integrated_count(),
                "vendor_local": (Path(app.static_folder) / "vendor" / "bootstrap.min.css").exists(),
                "STATUSES": history.STATUSES, "REVIEW_STATUSES": history.REVIEW_STATUSES}

    # -------------------------------------------------------------------- pages
    @app.get("/")
    def dashboard():
        show_demo = request.args.get("demo") == "1"
        return render_template("index.html", stats=history.stats(DB, show_demo), show_demo=show_demo,
                               recent=history.list_inspections(DB, include_demo=show_demo, limit=6),
                               models=manager.describe_all(), project_title=cfg.PROJECT_TITLE)

    @app.get("/upload")
    def upload_page():
        return render_template("upload.html", max_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024))

    @app.get("/preprocess")
    def preprocess_entry():
        latest = history.latest_id(DB)
        if latest is None:
            flash("Upload an image first.", "info")
            return redirect(url_for("upload_page"))
        return redirect(url_for("preprocess_page", insp_id=latest))

    @app.get("/results")
    def results_entry():
        latest = history.latest_id(DB)
        if latest is None:
            flash("No inspections yet. Upload an image to create one.", "info")
            return redirect(url_for("upload_page"))
        return redirect(url_for("results_page", insp_id=latest))

    @app.get("/preprocess/<int:insp_id>")
    def preprocess_page(insp_id):
        rec = get_or_404(insp_id)
        if rec["is_demo_data"]:
            abort(404)
        return render_template("preprocess.html", rec=rec, meta=rec.get("preprocess_meta") or {},
                               defaults=preprocessing.DEFAULT_SETTINGS)

    @app.get("/results/<int:insp_id>")
    def results_page(insp_id):
        rec = get_or_404(insp_id)
        return render_template("results.html", rec=rec, result=rec.get("result_json") or {},
                               overlays=rec.get("overlays") or {}, watermark=visualization.WATERMARK)

    @app.get("/models")
    def models_page():
        return render_template("models.html", models=manager.describe_all())

    @app.get("/history")
    def history_page():
        return render_template("history.html")

    @app.get("/about")
    def about_page():
        return render_template("about.html", project_title=cfg.PROJECT_TITLE)

    # ---------------------------------------------------------------------- API
    @app.post("/api/upload")
    def api_upload():
        f = request.files.get("image")
        if f is None or not f.filename:
            return api_error("No file selected.")
        data = f.read()
        try:
            info = validation.validate_image_bytes(data, f.filename, app.config["ALLOWED_EXTENSIONS"], app.config["MAX_PIXELS"])
        except validation.ImageValidationError as exc:
            return api_error(str(exc))

        model = request.form.get("analyze_with", "").strip()
        if model:
            try:
                manager.get(model)
            except ValueError as exc:
                return api_error(str(exc))

        stored = f"{uuid.uuid4().hex}.{info['extension']}"          # server-generated name
        path = Path(app.config["UPLOAD_DIR"]) / stored
        path.write_bytes(data)                                      # original is never modified afterwards
        insp_id = history.create_inspection(
            DB, original_filename=(secure_filename(f.filename)[:120] or "image"), stored_filename=stored,
            file_size=len(data), image_format=info["format"], width=info["width"], height=info["height"])
        try:
            meta = preprocessing.preprocess_image(path, insp_dir(insp_id))
            history.update_inspection(DB, insp_id, bit_depth=meta["bit_depth"], preprocess_settings=meta["settings"], preprocess_meta=meta)
            if model:
                run_analysis(insp_id, model)
        except Exception:
            history.delete_inspection(DB, insp_id)
            shutil.rmtree(insp_dir(insp_id), ignore_errors=True)
            path.unlink(missing_ok=True)
            return api_error("The image could not be processed. Check that it is a valid radiograph image.", 422)
        target = url_for("results_page", insp_id=insp_id) if model else url_for("preprocess_page", insp_id=insp_id)
        return jsonify(id=insp_id, redirect=target)

    @app.post("/api/preprocess/<int:insp_id>")
    def api_preprocess(insp_id):
        rec = get_or_404(insp_id)
        if not rec.get("stored_filename"):
            return api_error("This record has no image.", 404)
        try:
            meta = preprocessing.preprocess_image(Path(app.config["UPLOAD_DIR"]) / rec["stored_filename"],
                                                  insp_dir(insp_id), request.get_json(silent=True) or {})
        except Exception:
            return api_error("Preprocessing failed for this image.", 422)
        history.update_inspection(DB, insp_id, bit_depth=meta["bit_depth"], preprocess_settings=meta["settings"],
                                  preprocess_meta=meta, status="Uploaded", mode=None, overlays=None, result_json=None,
                                  model_used=None, is_real_prediction=0, score=None, confidence=None,
                                  defect_category=None, inference_time_ms=None, analyzed_at=None)
        return jsonify(meta=meta)

    @app.post("/api/analyze/<int:insp_id>")
    def api_analyze(insp_id):
        rec = get_or_404(insp_id)
        if rec["is_demo_data"] or not rec.get("stored_filename"):
            return api_error("Demo records have no image to analyze.", 400)
        model = (request.get_json(silent=True) or {}).get("model", "ResNet")
        try:
            manager.get(model)
            run_analysis(insp_id, model)
        except ValueError as exc:
            return api_error(str(exc))
        except Exception:
            return api_error("Analysis failed. Run preprocessing again and retry.", 500)
        return jsonify(redirect=url_for("results_page", insp_id=insp_id))

    @app.get("/api/models")
    def api_models():
        return jsonify(models=manager.describe_all())

    @app.get("/api/stats")
    def api_stats():
        return jsonify(history.stats(DB, request.args.get("demo") == "1"))

    @app.get("/api/inspections")
    def api_list():
        rows = history.list_inspections(DB, request.args.get("q", "").strip(), request.args.get("status", "").strip())
        slim = ("id", "original_filename", "uploaded_at", "model_used", "status", "defect_category", "score",
                "review_status", "is_demo_data", "is_real_prediction", "mode")
        return jsonify(inspections=[{k: r[k] for k in slim} for r in rows])

    @app.get("/api/inspections/<int:insp_id>")
    def api_detail(insp_id):
        rec = get_or_404(insp_id)
        rec.pop("stored_filename", None)                     # never expose server file names / paths
        return jsonify(rec)

    @app.post("/api/inspections/<int:insp_id>/review")
    def api_review(insp_id):
        get_or_404(insp_id)
        body = request.get_json(silent=True) or {}
        if body.get("review_status") not in history.REVIEW_STATUSES:
            return api_error("Invalid review status.")
        history.update_inspection(DB, insp_id, review_status=body["review_status"], review_note=(body.get("note") or "")[:500])
        return jsonify(ok=True)

    @app.delete("/api/inspections/<int:insp_id>")
    def api_delete(insp_id):
        rec = get_or_404(insp_id)
        history.delete_inspection(DB, insp_id)
        shutil.rmtree(insp_dir(insp_id), ignore_errors=True)
        if rec.get("stored_filename"):
            (Path(app.config["UPLOAD_DIR"]) / rec["stored_filename"]).unlink(missing_ok=True)
        return jsonify(ok=True)

    @app.post("/api/history/seed-demo")
    def api_seed():
        return jsonify(added=history.seed_demo(DB))

    @app.delete("/api/history/demo")
    def api_clear_demo():
        return jsonify(removed=history.delete_demo(DB))

    # -------------------------------------------------------------------- files
    @app.get("/files/<int:insp_id>/<name>")
    def serve_file(insp_id, name):
        if name not in SERVABLE:
            abort(404)
        resp = send_from_directory(insp_dir(insp_id), name, max_age=0)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/report/<int:insp_id>.pdf")
    def report(insp_id):
        rec = get_or_404(insp_id)
        out = Path(app.config["REPORT_DIR"]) / f"MV-{insp_id:06d}.pdf"
        report_generator.build_report(rec, insp_dir(insp_id), out, cfg.APP_NAME, cfg.APP_VERSION)
        return send_file(out, mimetype="application/pdf", as_attachment=True, download_name=out.name)

    # ------------------------------------------------------------------- errors
    @app.errorhandler(413)
    def too_large(_e):
        msg = f"File is too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."
        return (api_error(msg, 413)) if request.path.startswith("/api") else (msg, 413)

    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api"):
            return api_error("Not found.", 404)
        return render_template("error.html", code=404, message="That page or record does not exist."), 404

    @app.errorhandler(500)
    def server_error(_e):
        if request.path.startswith("/api"):
            return api_error("Internal server error.", 500)
        return render_template("error.html", code=500, message="Something went wrong on the server."), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
