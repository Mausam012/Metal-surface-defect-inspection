"""Common interface shared by CNN, YOLO, U-Net and ResNet.

Every architecture subclasses BaseModel and only has to implement:
    load(checkpoint_path)  -> prepare self._model (raise ModelNotAvailable if impossible)
    _predict(gray8)        -> dict with the keys of make_result() that the model really produces

BaseModel takes care of: finding the checkpoint, caching, error handling,
timing, and making sure that `is_real_prediction` is True ONLY when a loaded
model actually ran. Nothing in here ever invents a score.
"""
import json
import time
from pathlib import Path

# Uncalibrated decision bands for classifiers (tune on the validation set!).
NORMAL_BELOW = 0.35        # defect probability below this  -> Normal
SUSPICIOUS_FROM = 0.65     # defect probability at/above    -> Suspicious
                           # in between                      -> Requires Manual Review

STATUSES = ["Normal", "Suspicious", "Requires Manual Review", "Model Not Available"]


class ModelNotAvailable(Exception):
    """Checkpoint missing, library missing or checkpoint incompatible."""


def make_result(model_name: str, **overrides) -> dict:
    """The one result schema used by every model (all values null/empty by default)."""
    result = {
        "model_name": model_name,
        "model_version": None,
        "status": "Model Not Available",
        "defect_category": None,
        "score": None,                # anomaly / defect score, only if genuinely computed
        "confidence": None,
        "bounding_boxes": [],         # [{"x","y","w","h","label","confidence"}] in processed-image pixels
        "segmentation_mask": None,    # uint8 array (H, W), 1 = suspicious pixel
        "heatmap": None,              # float32 array (H, W) in [0, 1]
        "inference_time_ms": None,
        "is_real_prediction": False,
        "error": None,
        "notes": [],
    }
    result.update(overrides)
    return result


def score_to_status(p_defect: float) -> str:
    if p_defect >= SUSPICIOUS_FROM:
        return "Suspicious"
    if p_defect >= NORMAL_BELOW:
        return "Requires Manual Review"
    return "Normal"


class BaseModel:
    # --- descriptive metadata shown on the Models page (override in subclasses) ---
    name = "Base"
    dir_name = "base"
    architecture = ""
    description = ""
    purpose = ""
    expected_output = ""
    output_type = ""
    checkpoint_hint = ""
    extras: list = []
    weight_exts = (".pt", ".pth")

    def __init__(self, models_dir):
        self.model_dir = Path(models_dir) / self.dir_name
        self._model = None
        self.meta: dict = {}
        self._sig = None
        self._load_error = None

    # ---------- checkpoint handling ----------
    def find_checkpoint(self):
        if not self.model_dir.exists():
            return None
        files = [p for p in self.model_dir.iterdir() if p.suffix.lower() in self.weight_exts]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def load(self, checkpoint_path: Path):  # pragma: no cover - implemented by subclasses
        raise ModelNotAvailable("This architecture has no loader yet.")

    def _reset(self):
        self._model, self.meta, self._load_error, self._sig = None, {}, None, None

    def ensure_loaded(self):
        ck = self.find_checkpoint()
        if ck is None:
            self._reset()
            raise ModelNotAvailable(f"No checkpoint found. Place a trained file in models/{self.dir_name}/.")
        sig = (str(ck), ck.stat().st_mtime)
        if sig != self._sig:                      # new or changed checkpoint -> (re)load once
            self._reset()
            self._sig = sig
            try:
                self.load(ck)
            except ModelNotAvailable as exc:
                self._load_error = str(exc)
            except Exception as exc:              # corrupt / incompatible file
                self._load_error = f"Checkpoint could not be loaded ({type(exc).__name__}). It may be corrupt or incompatible."
        if self._load_error:
            raise ModelNotAvailable(self._load_error)

    # ---------- status & metrics ----------
    def status(self) -> dict:
        ck = self.find_checkpoint()
        try:
            self.ensure_loaded()
            return {"state": "Integrated", "message": "Checkpoint loaded.", "checkpoint": ck.name if ck else None}
        except ModelNotAvailable as exc:
            state = "Not Integrated" if ck is None else "Load Error"
            return {"state": state, "message": str(exc), "checkpoint": ck.name if ck else None}

    def metrics(self):
        """Real evaluation metrics, only if the training script wrote metrics.json."""
        path = self.model_dir / "metrics.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
        return None

    def version(self):
        ck = self.find_checkpoint()
        return self.meta.get("version") or (ck.name if ck else None)

    # ---------- prediction ----------
    def _predict(self, gray8):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def predict(self, gray8) -> dict:
        try:
            self.ensure_loaded()
        except ModelNotAvailable as exc:
            return make_result(self.name, error=str(exc))
        try:
            start = time.perf_counter()
            out = self._predict(gray8)
            elapsed = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            return make_result(self.name, error=f"Inference failed ({type(exc).__name__}).")
        result = make_result(self.name, **out)
        result.update(inference_time_ms=round(elapsed, 2), is_real_prediction=True,
                      model_version=self.version())
        return result
