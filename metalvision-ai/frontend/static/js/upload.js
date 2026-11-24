
/* Upload page: file validation, preview, crop selection and upload. */
document.addEventListener("DOMContentLoaded", () => {
  const $ = (id) => document.getElementById(id);

  const dz = $("dropzone");
  const input = $("fileInput");
  const err = $("errorBox");
  const modelSelect = $("model-select");
  const analyzeBtn = $("analyzeBtn");
  const prepOnlyBtn = $("prepOnlyBtn");
  const previewBox = $("previewBox");
  const cropPanel = $("cropPanel");
  const cropInstructions = $("cropInstructions");
  const canvas = $("cropCanvas");
  const ctx = canvas.getContext("2d");

  const OK_EXT = ["png", "jpg", "jpeg", "tif", "tiff"];
  const MAX_CANVAS_SIDE = 1600;

  let file = null;
  let image = null;
  let imageUrl = null;
  let crop = null;
  let dragging = false;
  let dragStart = null;

  const isResNet = () => modelSelect.value === "ResNet";

  const showError = (message) => {
    err.textContent = message;
    err.style.display = message ? "flex" : "none";
  };

  const size = (bytes) =>
    bytes > 1048576
      ? (bytes / 1048576).toFixed(2) + " MB"
      : (bytes / 1024).toFixed(1) + " KB";

  function clearCrop() {
    crop = null;
    dragging = false;
    dragStart = null;

    if (canvas.width && canvas.height && image) {
      drawCanvas();
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    $("cropStatus").textContent = "No region selected";
    setButtons();
  }

  function setButtons() {
    const hasFile = Boolean(file);
    const hasCrop = Boolean(crop);

    analyzeBtn.disabled =
      !hasFile || (isResNet() && !hasCrop);

    prepOnlyBtn.disabled = !hasFile;
  }

  function updateCropInterface() {
    const active = isResNet();

    cropInstructions.style.display = active ? "block" : "none";
    cropPanel.style.display = active && image ? "block" : "none";

    if (active && file && !image) {
      $("cropStatus").textContent =
        "ResNet crop selection requires a PNG or JPEG image.";
    }

    setButtons();
  }

  function drawCanvas(currentRect = null) {
    if (!image) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    const rect = currentRect || crop;
    if (!rect) return;

    ctx.save();
    ctx.strokeStyle = "#00d4ff";
    ctx.lineWidth = Math.max(2, canvas.width / 500);
    ctx.fillStyle = "rgba(0, 180, 220, 0.18)";
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    ctx.restore();
  }

  function canvasPosition(event) {
    const rect = canvas.getBoundingClientRect();

    return {
      x: Math.max(
        0,
        Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)
      ),
      y: Math.max(
        0,
        Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height)
      )
    };
  }

  function finishCrop(position) {
    if (!dragStart || !image) return;

    const left = Math.min(dragStart.x, position.x);
    const top = Math.min(dragStart.y, position.y);
    const right = Math.max(dragStart.x, position.x);
    const bottom = Math.max(dragStart.y, position.y);

    const width = right - left;
    const height = bottom - top;

    if (width < 8 || height < 8) {
      clearCrop();
      $("cropStatus").textContent =
        "Selection too small. Draw a larger rectangle.";
      return;
    }

    const scaleX = image.naturalWidth / canvas.width;
    const scaleY = image.naturalHeight / canvas.height;

    crop = {
      x: Math.round(left * scaleX),
      y: Math.round(top * scaleY),
      width: Math.round(width * scaleX),
      height: Math.round(height * scaleY)
    };

    drawCanvas();

    $("cropStatus").textContent =
      `Selected region: ${crop.width} × ${crop.height} px`;

    setButtons();
  }

  function prepareCropCanvas(img) {
    image = img;

    const scale = Math.min(
      1,
      MAX_CANVAS_SIDE / Math.max(img.naturalWidth, img.naturalHeight)
    );

    canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));

    drawCanvas();
    updateCropInterface();
  }

  function setFile(selectedFile) {
    showError("");

    const ext = (selectedFile.name.split(".").pop() || "").toLowerCase();

    if (!OK_EXT.includes(ext)) {
      showError(`Unsupported file type ".${ext}". Use PNG, JPG, JPEG or TIFF.`);
      return;
    }

    if (selectedFile.size > window.MAX_MB * 1048576) {
      showError(`File is too large (${size(selectedFile.size)}). Maximum is ${window.MAX_MB} MB.`);
      return;
    }

    if (selectedFile.size === 0) {
      showError("The file is empty.");
      return;
    }

    file = selectedFile;
    image = null;
    clearCrop();

    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
      imageUrl = null;
    }

    $("mName").textContent = selectedFile.name;
    $("mSize").textContent = size(selectedFile.size);
    $("mFormat").textContent =
      ext === "jpg" || ext === "jpeg"
        ? "JPEG"
        : ext.startsWith("tif")
          ? "TIFF"
          : "PNG";

    $("mDim").textContent = "—";

    if (ext.startsWith("tif")) {
      previewBox.innerHTML =
        "<span>TIFF cannot be previewed in this browser. It can still be preprocessed.</span>";

      updateCropInterface();
      setButtons();
      return;
    }

    const img = new Image();

    imageUrl = URL.createObjectURL(selectedFile);

    img.onload = () => {
      $("mDim").textContent =
        `${img.naturalWidth} × ${img.naturalHeight} px`;

      previewBox.replaceChildren(img);
      prepareCropCanvas(img);
    };

    img.onerror = () => {
      showError("This image could not be read. It may be corrupted.");
      file = null;
      image = null;
      updateCropInterface();
      setButtons();
    };

    img.src = imageUrl;
    setButtons();
  }

  async function send(analyze) {
    if (!file) return;

    const model = modelSelect.value;

    if (analyze && model === "ResNet" && !crop) {
      showError("Please select a suspected defect region before running ResNet.");
      return;
    }

    const fd = new FormData();
    fd.append("image", file);

    if (analyze) {
      fd.append("analyze_with", model);

      if (model === "ResNet") {
        fd.append("resnet_crop", JSON.stringify(crop));
      }
    }

    analyzeBtn.disabled = true;
    prepOnlyBtn.disabled = true;
    analyzeBtn.textContent = "Working...";

    try {
      const out = await MV.api("/api/upload", {
        method: "POST",
        body: fd
      });

      window.location = out.redirect;
    } catch (error) {
      showError(error.message);
      analyzeBtn.textContent = "Analyze image";
      setButtons();
    }
  }

  $("browseBtn").addEventListener("click", (event) => {
    event.stopPropagation();
    input.click();
  });

  dz.addEventListener("click", () => input.click());

  dz.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });

  input.addEventListener("change", () => {
    if (input.files[0]) setFile(input.files[0]);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dz.addEventListener(eventName, (event) => {
      event.preventDefault();
      dz.classList.add("over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dz.addEventListener(eventName, (event) => {
      event.preventDefault();
      dz.classList.remove("over");
    });
  });

  dz.addEventListener("drop", (event) => {
    if (event.dataTransfer.files[0]) {
      setFile(event.dataTransfer.files[0]);
    }
  });

  modelSelect.addEventListener("change", updateCropInterface);

  canvas.addEventListener("pointerdown", (event) => {
    if (!image || !isResNet()) return;

    event.preventDefault();
    dragging = true;
    dragStart = canvasPosition(event);

    canvas.setPointerCapture(event.pointerId);
    drawCanvas({
      x: dragStart.x,
      y: dragStart.y,
      width: 0,
      height: 0
    });
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!dragging || !dragStart) return;

    const current = canvasPosition(event);

    drawCanvas({
      x: Math.min(dragStart.x, current.x),
      y: Math.min(dragStart.y, current.y),
      width: Math.abs(current.x - dragStart.x),
      height: Math.abs(current.y - dragStart.y)
    });
  });

  canvas.addEventListener("pointerup", (event) => {
    if (!dragging) return;

    dragging = false;
    finishCrop(canvasPosition(event));
  });

  canvas.addEventListener("pointercancel", () => {
    if (dragging) clearCrop();
  });

  $("clearCropBtn").addEventListener("click", clearCrop);

  analyzeBtn.addEventListener("click", () => send(true));
  prepOnlyBtn.addEventListener("click", () => send(false));

  updateCropInterface();
});
