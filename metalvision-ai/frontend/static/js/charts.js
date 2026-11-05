/* Dashboard charts (Chart.js). Data comes from /api/stats and contains only stored records. */
(async () => {
  const sc = document.getElementById("statusChart");
  if (!sc || typeof Chart === "undefined") return;
  document.getElementById("seedDemo")?.addEventListener("click", async () => {
    await MV.api("/api/history/seed-demo", { method: "POST" });
    window.location = "/?demo=1";
  });
  const s = await MV.api(`/api/stats?demo=${sc.dataset.demo}`);
  const labels = ["Normal", "Suspicious", "Requires Manual Review", "Model Not Available", "Uploaded"];
  const colors = ["#1e7f55", "#b02a2a", "#d99a22", "#7c8aa0", "#c5cfdd"];
  const values = labels.map((l) => s.by_status[l] || 0);
  if (values.every((v) => v === 0)) {
    sc.parentElement.innerHTML = '<div class="empty">No data yet</div>';
  } else {
    new Chart(sc, { type: "doughnut", data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }] },
      options: { maintainAspectRatio: false, cutout: "62%", plugins: { legend: { position: "right" } } } });
  }
  const dc = document.getElementById("dayChart");
  if (s.by_day.length === 0) { dc.parentElement.innerHTML = '<div class="empty">No data yet</div>'; return; }
  new Chart(dc, { type: "bar", data: { labels: s.by_day.map((d) => d.date), datasets: [{ label: "Images", data: s.by_day.map((d) => d.count), backgroundColor: "#1fb5c9", borderRadius: 4 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } } });
})();
