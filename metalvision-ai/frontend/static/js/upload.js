/* Upload page: drag & drop, client-side checks, preview, POST to /api/upload. */
document.addEventListener("DOMContentLoaded", () => {
  const $ = (id) => document.getElementById(id);
  const dz = $("dropzone"), input = $("fileInput"), err = $("errorBox");
  const OK_EXT = ["png", "jpg", "jpeg", "tif", "tiff"];
  let file = null;

  const showError = (m) => { err.textContent = m; err.style.display = m ? "flex" : "none"; };
  const size = (b) => (b > 1048576 ? (b / 1048576).toFixed(2) + " MB" : (b / 1024).toFixed(1) + " KB");

  function setFile(f) {
    showError("");
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    if (!OK_EXT.includes(ext)) { showError(`Unsupported file type ".${ext}". Use PNG, JPG, JPEG or TIFF.`); return; }
    if (f.size > window.MAX_MB * 1048576) { showError(`File is too large (${size(f.size)}). Maximum is ${window.MAX_MB} MB.`); return; }
    if (f.size === 0) { showError("The file is empty."); return; }
    file = f;
    $("mName").textContent = f.name; $("mSize").textContent = size(f.size);
    $("mFormat").textContent = ext === "jpg" ? "JPEG" : ext.replace("tif", "TIFF").toUpperCase();
    $("mDim").textContent = "—";
    const box = $("previewBox");
    if (ext.startsWith("tif")) {                          // browsers cannot draw TIFF
      box.innerHTML = "<span>TIFF cannot be previewed in the browser.<br>It will be shown after upload.</span>";
    } else {
      const img = new Image();
      img.onload = () => { $("mDim").textContent = `${img.naturalWidth} x ${img.naturalHeight} px`; };
      img.onerror = () => { showError("This image could not be read. It may be corrupted."); file = null; setButtons(); };
      img.src = URL.createObjectURL(f);
      box.replaceChildren(img);
    }
    setButtons();
  }
  const setButtons = () => { $("analyzeBtn").disabled = $("prepOnlyBtn").disabled = !file; };

  async function send(analyze) {
    if (!file) return;
    const fd = new FormData(); fd.append("image", file);
    if (analyze) fd.append("analyze_with", $("model-select").value);
    $("analyzeBtn").disabled = $("prepOnlyBtn").disabled = true;
    $("analyzeBtn").textContent = "Working...";
    try { const out = await MV.api("/api/upload", { method: "POST", body: fd }); window.location = out.redirect; }
    catch (e) { showError(e.message); $("analyzeBtn").textContent = "Analyze image"; setButtons(); }
  }

  $("browseBtn").addEventListener("click", (e) => { e.stopPropagation(); input.click(); });
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => input.files[0] && setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));
  $("analyzeBtn").addEventListener("click", () => send(true));
  $("prepOnlyBtn").addEventListener("click", () => send(false));
});
