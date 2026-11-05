"""ResNet training pipeline  (owner: ResNet team member).

Run from the project root, e.g.:

    # 1) Folder dataset:  dataset/{train,validation,test}/{normal,defective}/
    python -m backend.resnet_training train --data-dir dataset --arch resnet18 --epochs 20

    # 2) CSV manifest (recommended - supports group-aware splitting):
    python -m backend.resnet_training make-split --csv all_images.csv --out manifest.csv
    python -m backend.resnet_training train --manifest manifest.csv --arch resnet18

    # 3) FINAL evaluation on the untouched test split (run ONCE, after model selection):
    python -m backend.resnet_training evaluate --checkpoint models/resnet/resnet18_best.pt --data-dir dataset

MANIFEST COLUMNS:  path,label,group[,split]
    path  : image file path        label : 'normal', 'defective' (or defect names for multi-class)
    group : specimen / component / weld-series ID  (ALL images of one specimen share it)
    split : train | validation | test   (added by make-split)

CLASS IMBALANCE: real inspection data has far more normal than defective images.
    A model that always answers "normal" can reach 95 % accuracy and be useless.
    We therefore (1) report precision / recall / F1 per class instead of accuracy,
    (2) weight the loss inversely to class frequency (default), optionally
    (3) oversample with WeightedRandomSampler (--sampler), and
    (4) select the best checkpoint by validation MACRO-F1.

DATA LEAKAGE RULES (enforced / checked below):
    * split ORIGINAL images by specimen/group BEFORE cropping or augmenting;
    * augmentation is applied to the training split only;
    * validation split = model selection; test split = touched once at the end;
    * a group that appears in two splits aborts training.

The script writes only REAL results: models/resnet/metrics.json is created from
actual evaluation runs. No number in this project is made up.
"""
import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import normalize_float

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models" / "resnet"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------- data
def read_gray8(path) -> np.ndarray:
    """Read any bit depth and stretch to uint8 with the SAME rule the web app uses (1-99 percentile)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return (normalize_float(img, "percentile", 1.0, 99.0) * 255).round().astype(np.uint8)


def samples_from_folders(data_dir: Path):
    """dataset/<split>/<class>/*.png -> {split: [(path, class_name, group)]}.

    NOTE: folders carry no specimen IDs. If images of one specimen exist in several
    splits you have leakage. Prefer the manifest workflow when IDs are available.
    """
    out = {}
    for split in ("train", "validation", "test"):
        items = []
        for cls_dir in sorted((data_dir / split).glob("*")):
            if cls_dir.is_dir():
                for p in sorted(cls_dir.iterdir()):
                    if p.suffix.lower() in IMG_EXTS:
                        items.append((str(p), cls_dir.name.lower(), p.stem.split("__")[0]))
        out[split] = items
    return out


def samples_from_manifest(manifest: Path):
    out = {"train": [], "validation": [], "test": []}
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "").strip().lower()
            if split not in out:
                raise ValueError("Manifest needs a 'split' column (train/validation/test). Run make-split first.")
            out[split].append((row["path"], row["label"].strip().lower(), row.get("group") or row["path"]))
    return out


def check_leakage(splits: dict):
    """Abort if one group (specimen) appears in more than one split."""
    seen = {}
    for split, items in splits.items():
        for _p, _l, g in items:
            if g in seen and seen[g] != split:
                raise SystemExit(f"DATA LEAKAGE: group '{g}' appears in both '{seen[g]}' and '{split}'. "
                                 "Re-split by specimen (make-split).")
            seen[g] = split
    print("Leakage check passed: no group appears in more than one split.")


def make_split(csv_in, csv_out, val_frac=0.15, test_frac=0.15, seed=42):
    """Group-aware split: every group goes entirely into ONE of train/validation/test."""
    from sklearn.model_selection import GroupShuffleSplit
    with open(csv_in, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = np.array([r.get("group") or r["path"] for r in rows])
    idx = np.arange(len(rows))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval, test = next(outer.split(idx, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
    tr, va = next(inner.split(trainval, groups=groups[trainval]))
    for i in trainval[tr]: rows[i]["split"] = "train"
    for i in trainval[va]: rows[i]["split"] = "validation"
    for i in test: rows[i]["split"] = "test"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {csv_out}. Check class balance in every split and the dataset's own rules before training.")


def make_datasets(splits, class_names, img_size, in_channels):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms as T

    mean = IMAGENET_MEAN if in_channels == 3 else [0.45]
    std = IMAGENET_STD if in_channels == 3 else [0.225]
    to_ch = T.Grayscale(num_output_channels=in_channels)
    # Augmentation: TRAIN ONLY. Mild, physically plausible. Flips/rotations are only
    # valid if defect appearance does not depend on orientation (check for welds!).
    train_tf = T.Compose([T.Resize((img_size, img_size)), to_ch, T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                          T.RandomRotation(10), T.ColorJitter(brightness=0.1, contrast=0.1),
                          T.ToTensor(), T.Normalize(mean, std)])
    eval_tf = T.Compose([T.Resize((img_size, img_size)), to_ch, T.ToTensor(), T.Normalize(mean, std)])

    class XrayDataset(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            path, label, _g = self.items[i]
            img = Image.fromarray(read_gray8(path))
            return self.tf(img), class_names.index(label)

    return (XrayDataset(splits["train"], train_tf), XrayDataset(splits["validation"], eval_tf),
            XrayDataset(splits["test"], eval_tf), mean, std)


# ------------------------------------------------------------------------ evaluation
def evaluate_model(model, loader, device, class_names, criterion=None):
    import torch
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    model.eval()
    ys, ps, total_loss = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * x.size(0)
            ys += y.cpu().tolist(); ps += logits.argmax(1).cpu().tolist()
    labels = list(range(len(class_names)))
    rep = classification_report(ys, ps, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    return {
        "loss": total_loss / max(1, len(ys)),
        "macro_f1": float(f1_score(ys, ps, labels=labels, average="macro", zero_division=0)),
        "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])}
                      for c in class_names},
        "confusion_matrix": [[int(v) for v in row] for row in confusion_matrix(ys, ps, labels=labels)],
        "n_images": len(ys),
    }


def measure_inference_ms(model, dataset, device, n=30):
    """Mean single-image forward time (ms) on the given device."""
    import torch
    model.eval()
    n = min(n, len(dataset))
    if n == 0:
        return None
    x = dataset[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        model(x)                                    # warm-up
        t0 = time.perf_counter()
        for i in range(n):
            model(dataset[i][0].unsqueeze(0).to(device))
    return round((time.perf_counter() - t0) / n * 1000, 2)


def save_confusion_matrix(cm, class_names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="validation")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["val_macro_f1"], color="tab:green"); ax[1].set_title("Validation macro-F1"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------- train
def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from .resnet_inference import build_resnet

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    splits = samples_from_manifest(Path(args.manifest)) if args.manifest else samples_from_folders(Path(args.data_dir))
    check_leakage(splits)
    class_names = sorted({lbl for items in splits.values() for _p, lbl, _g in items})
    if "normal" not in class_names or len(class_names) < 2:
        raise SystemExit("Need a 'normal' class plus at least one defect class. Found: " + str(class_names))
    if not splits["train"] or not splits["validation"]:
        raise SystemExit("Train and validation splits must not be empty.")
    print("Classes:", class_names, {s: len(v) for s, v in splits.items()})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds, mean, std = make_datasets(splits, class_names, args.img_size, args.in_channels)

    # ---- class imbalance handling
    counts = np.bincount([class_names.index(l) for _p, l, _g in splits["train"]], minlength=len(class_names))
    print("Training class counts:", dict(zip(class_names, counts.tolist())))
    weights = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)            # inverse-frequency class weights
    sampler = None
    if args.sampler:
        per_sample = [1.0 / max(1, counts[class_names.index(l)]) for _p, l, _g in splits["train"]]
        sampler = WeightedRandomSampler(per_sample, num_samples=len(per_sample), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    # ---- model: pretrained backbone, NEW final layer
    model = build_resnet(args.arch, len(class_names), pretrained=not args.no_pretrained, in_channels=args.in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def set_backbone_trainable(flag):
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = flag

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / f"{args.arch}_best.pt"
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        set_backbone_trainable(epoch > args.freeze_epochs)       # optional: train only fc first
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()
        val = evaluate_model(model, val_loader, device, class_names, criterion)
        history["train_loss"].append(running / len(train_ds)); history["val_loss"].append(val["loss"])
        history["val_macro_f1"].append(val["macro_f1"])
        print(f"epoch {epoch:02d}/{args.epochs}  train_loss={history['train_loss'][-1]:.4f}  "
              f"val_loss={val['loss']:.4f}  val_macro_f1={val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:                            # model selection uses VALIDATION only
            best_f1 = val["macro_f1"]
            torch.save({
                "arch": args.arch, "state_dict": model.state_dict(), "class_names": class_names,
                "in_channels": args.in_channels, "img_size": args.img_size, "mean": mean, "std": std,
                "epoch": epoch, "val_macro_f1": float(best_f1),
                "dataset_info": f"{len(train_ds)} train / {len(val_ds)} val images; source: {args.dataset_note}",
                "created": datetime.now().isoformat(timespec="seconds"), "torch_version": torch.__version__,
            }, ckpt_path)

    save_curves(history, OUT_DIR / "training_curves.png")
    best = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    val = evaluate_model(model, val_loader, device, class_names)
    save_confusion_matrix(val["confusion_matrix"], class_names, OUT_DIR / "confusion_matrix_validation.png", "Validation")
    metrics = {"evaluated_on": "validation", "arch": args.arch, "class_names": class_names, "dataset_note": args.dataset_note,
               "created": datetime.now().isoformat(timespec="seconds"), **val}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {ckpt_path}  (validation macro-F1 {best_f1:.4f})")
    print("Test split untouched. When you are done tuning run the 'evaluate' command ONCE.")


def evaluate_cli(args):
    import torch
    from torch.utils.data import DataLoader
    from .resnet_inference import build_resnet
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    class_names = list(ck["class_names"])
    splits = samples_from_manifest(Path(args.manifest)) if args.manifest else samples_from_folders(Path(args.data_dir))
    check_leakage(splits)
    if not splits["test"]:
        raise SystemExit("Test split is empty.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tr, _va, test_ds, _m, _s = make_datasets(splits, class_names, ck["img_size"], ck["in_channels"])
    model = build_resnet(ck["arch"], len(class_names), False, ck["in_channels"]).to(device)
    model.load_state_dict(ck["state_dict"])
    res = evaluate_model(model, DataLoader(test_ds, batch_size=32), device, class_names)
    res["inference_ms_per_image"] = measure_inference_ms(model, test_ds, device)
    res["device"] = str(device)
    save_confusion_matrix(res["confusion_matrix"], class_names, OUT_DIR / "confusion_matrix_test.png", "Test")
    metrics = {"evaluated_on": "test", "arch": ck["arch"], "class_names": class_names,
               "dataset_note": ck.get("dataset_info"), "created": datetime.now().isoformat(timespec="seconds"), **res}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser(description="METALVISION AI - ResNet training pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data-dir", default=str(ROOT / "dataset"))
    t.add_argument("--manifest")
    t.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--img-size", type=int, default=224)
    t.add_argument("--in-channels", type=int, default=3, choices=[1, 3])
    t.add_argument("--freeze-epochs", type=int, default=0, help="epochs to train only the new final layer")
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--sampler", action="store_true", help="oversample minority classes")
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--dataset-note", default="unspecified - document source, geometry and license")
    t.set_defaults(func=train)

    e = sub.add_parser("evaluate", help="final evaluation on the untouched test split")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--data-dir", default=str(ROOT / "dataset"))
    e.add_argument("--manifest")
    e.set_defaults(func=evaluate_cli)

    s = sub.add_parser("make-split", help="group-aware train/validation/test split of a CSV")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--val-frac", type=float, default=0.15)
    s.add_argument("--test-frac", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=lambda a: make_split(a.csv, a.out, a.val_frac, a.test_frac, a.seed))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
