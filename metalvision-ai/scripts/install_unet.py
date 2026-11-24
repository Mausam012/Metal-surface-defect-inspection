"""Install the trained U-Net weights into models/unet/ from whatever you received.

Usage (from the metalvision-ai folder, with the virtual environment active):

    python scripts/install_unet.py "C:\\Users\\you\\Downloads\\<what you got>"

Accepted inputs:
  1. the file itself:            best_unet_opencv_improved.pth
  2. a .zip that contains it:    e.g. a WhatsApp / Google Drive download
  3. an extracted folder:        Windows sometimes unzips a .pth into a folder
                                 (it contains data.pkl, data/, byteorder, version)

The script writes models/unet/best_unet_opencv_improved.pth and then checks
that it loads into the team's U-Net architecture.
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
TARGET = APP_DIR / "models" / "unet" / "best_unet_opencv_improved.pth"


def is_torch_archive(zf: zipfile.ZipFile) -> bool:
    """A PyTorch checkpoint is a zip whose entries look like <name>/data.pkl."""
    return any(n.replace("\\", "/").count("/") == 1 and n.endswith("data.pkl") for n in zf.namelist())


def repack_folder(folder: Path, out: Path):
    """Rebuild a .pth from an extracted folder (entries must use forward slashes)."""
    root = next((p.parent for p in folder.rglob("data.pkl")), None)
    if root is None:
        raise SystemExit(f"No data.pkl found inside {folder}. This is not an extracted PyTorch file.")
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: (p.name != "data.pkl", str(p)))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, f"{root.name}/{p.relative_to(root).as_posix()}")
    print(f"Repacked extracted folder ({len(files)} files).")


def from_zip(path: Path, out: Path):
    with zipfile.ZipFile(path) as zf:
        # Windows' zip tool writes backslash paths, which torch rejects. Python hides them in
        # .filename (it converts them to '/'), so check the raw names.
        backslashes = any("\\" in zi.orig_filename for zi in zf.infolist())
        if is_torch_archive(zf) and not backslashes:   # the checkpoint itself, just named .zip
            shutil.copyfile(path, out)
            print("Input is the checkpoint itself; copied.")
            return
        inner = [n for n in zf.namelist() if n.lower().endswith((".pth", ".pt")) and not n.endswith("/")]
        if inner:                                      # a zip that contains the .pth
            with zf.open(inner[0]) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"Extracted {inner[0]} from the zip.")
            return
        # a zip of an already-extracted folder
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            repack_folder(Path(tmp), out)


def verify(path: Path):
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is not installed. Activate .venv and install requirements first.")
    sys.path.insert(0, str(APP_DIR))
    from backend.unet_inference import build_team_unet

    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        model = build_team_unet()
        model.load_state_dict(state, strict=True)       # raises if anything is missing or different
    except Exception as exc:
        raise SystemExit(f"This is not the team's U-Net checkpoint ({type(exc).__name__}). "
                         "Nothing was installed. Ask for best_unet_opencv_improved.pth again.")
    params = sum(t.numel() for t in state.values())
    print(f"Verified: {len(state)} tensors, {params:,} parameters (expected 482,033).")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1].strip('"')).expanduser()
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = TARGET.with_suffix(".tmp")
    try:
        if src.is_dir():
            repack_folder(src, tmp_out)
        elif zipfile.is_zipfile(src):
            from_zip(src, tmp_out)
        else:
            raise SystemExit(f"{src.name} is neither a PyTorch file, a zip, nor a folder.")
        verify(tmp_out)
        tmp_out.replace(TARGET)
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        raise
    print(f"Installed: {TARGET}")
    print("Restart the app; U-Net should show as Integrated on the ML Models page.")


if __name__ == "__main__":
    main()
