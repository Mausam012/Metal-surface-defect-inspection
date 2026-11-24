/* Shared helpers: toasts, fetch wrapper, zoom/pan image viewer, model analyze buttons. */
const MV = { viewers: {} };

MV.toast = function (message, kind = "danger") {
  const el = document.createElement("div");
  el.className = `toast align-items-center text-bg-${kind} border-0`;
  el.setAttribute("role", "alert");
  el.innerHTML = `<div class="d-flex"><div class="toast-body"></div><button class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
  el.querySelector(".toast-body").textContent = message;          // textContent: never inject HTML
  document.getElementById("toasts").appendChild(el);
  new bootstrap.Toast(el, { delay: 6000 }).show();
};

MV.api = async function (url, options = {}) {
  let res;
  try { res = await fetch(url, options); }
  catch (e) { throw new Error("Cannot reach the server. Is Flask still running?"); }
  let data = {};
  try { data = await res.json(); } catch (e) { /* non-JSON error page */ }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
  return data;
};

/* ---- image viewer: wheel zoom, drag pan, buttons, double-click reset ---- */
MV.initViewer = function (root) {
  const stage = root.querySelector(".stage"), img = root.querySelector("img");
  let scale = 1, tx = 0, ty = 0, drag = null;
  const apply = () => { img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`; };
  const zoom = (f) => { scale = Math.min(12, Math.max(1, scale * f)); if (scale === 1) { tx = ty = 0; } apply(); };
  const reset = () => { scale = 1; tx = ty = 0; apply(); };
  stage.addEventListener("wheel", (e) => { e.preventDefault(); zoom(e.deltaY < 0 ? 1.15 : 1 / 1.15); }, { passive: false });
  stage.addEventListener("pointerdown", (e) => { if (scale > 1) { drag = { x: e.clientX - tx, y: e.clientY - ty }; stage.classList.add("dragging"); stage.setPointerCapture(e.pointerId); } });
  stage.addEventListener("pointermove", (e) => { if (drag) { tx = e.clientX - drag.x; ty = e.clientY - drag.y; apply(); } });
  stage.addEventListener("pointerup", () => { drag = null; stage.classList.remove("dragging"); });
  stage.addEventListener("dblclick", reset);
  root.querySelectorAll("[data-zoom]").forEach((b) => b.addEventListener("click", () => {
    b.dataset.zoom === "in" ? zoom(1.4) : b.dataset.zoom === "out" ? zoom(1 / 1.4) : reset();
  }));
  const api = { setSrc(src) { img.src = src; reset(); }, reset };
  if (root.id) MV.viewers[root.id] = api;
  return api;
};

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("menuBtn")?.addEventListener("click", () => document.getElementById("sidebar").classList.toggle("open"));
  document.querySelectorAll(".mv-viewer").forEach(MV.initViewer);

  /* any button with data-analyze="<inspection id>" runs analysis with the model chosen in #model-select */
  document.querySelectorAll("[data-analyze]").forEach((btn) => btn.addEventListener("click", async () => {
    const model = document.getElementById("model-select").value;
    btn.disabled = true; const old = btn.textContent; btn.textContent = "Analyzing...";
    try {
      const out = await MV.api(`/api/analyze/${btn.dataset.analyze}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) });
      window.location = out.redirect;
    } catch (e) { MV.toast(e.message); btn.disabled = false; btn.textContent = old; }
  }));
});
