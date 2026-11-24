/* History page: live search/filter, detail links, delete with confirmation. */
document.addEventListener("DOMContentLoaded", () => {
  const rows = document.getElementById("rows"), q = document.getElementById("q"), sf = document.getElementById("statusFilter");
  const modal = new bootstrap.Modal(document.getElementById("delModal"));
  let pending = null, timer = null;

  const td = (text) => { const c = document.createElement("td"); c.textContent = text ?? "—"; return c; };

  async function load() {
    const url = `/api/inspections?q=${encodeURIComponent(q.value)}&status=${encodeURIComponent(sf.value)}`;
    try {
      const { inspections } = await MV.api(url);
      document.getElementById("count").textContent = `${inspections.length} record(s)`;
      rows.replaceChildren();
      if (!inspections.length) {
        rows.innerHTML = '<tr><td colspan="9" class="empty">No inspections match. Upload an image or add demo records.</td></tr>'; return;
      }
      for (const r of inspections) {
        const tr = document.createElement("tr");
        const id = document.createElement("td"), a = document.createElement("a");
        a.href = `/results/${r.id}`; a.textContent = `MV-${String(r.id).padStart(6, "0")}`; id.appendChild(a); tr.appendChild(id);
        const file = td(r.original_filename);
        if (r.is_demo_data) { const b = document.createElement("span"); b.className = "badge-demo"; b.textContent = "DEMO"; file.appendChild(b); }
        tr.appendChild(file);
        tr.appendChild(td(r.uploaded_at.replace("T", " ")));
        tr.appendChild(td(r.model_used));
        const st = document.createElement("td"), bd = document.createElement("span");
        bd.className = `badge-s s-${r.status.replaceAll(" ", "-")}`; bd.textContent = r.status; st.appendChild(bd); tr.appendChild(st);
        tr.appendChild(td(r.defect_category));
        tr.appendChild(td(r.score == null ? "Not available" : r.score.toFixed(3)));
        tr.appendChild(td(r.review_status));
        const act = document.createElement("td"); act.className = "text-nowrap";
        const open = document.createElement("a"); open.className = "btn-ghost btn-sm me-1"; open.href = `/results/${r.id}`; open.textContent = "Open";
        const del = document.createElement("button"); del.className = "btn-ghost btn-sm btn-danger-ghost"; del.textContent = "Delete";
        del.addEventListener("click", () => { pending = r.id; document.getElementById("delName").textContent = `MV-${String(r.id).padStart(6, "0")} (${r.original_filename})`; modal.show(); });
        act.append(open, del); tr.appendChild(act);
        rows.appendChild(tr);
      }
    } catch (e) { MV.toast(e.message); }
  }

  q.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(load, 250); });
  sf.addEventListener("change", load);
  document.getElementById("delConfirm").addEventListener("click", async () => {
    try { await MV.api(`/api/inspections/${pending}`, { method: "DELETE" }); modal.hide(); MV.toast("Inspection deleted.", "success"); load(); }
    catch (e) { MV.toast(e.message); }
  });
  document.getElementById("seedBtn").addEventListener("click", async () => { await MV.api("/api/history/seed-demo", { method: "POST" }); load(); });
  document.getElementById("clearDemoBtn").addEventListener("click", async () => { await MV.api("/api/history/demo", { method: "DELETE" }); load(); });
  load();
});
