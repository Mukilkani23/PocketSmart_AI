// PocketSmart frontend: vanilla JS, no build step. Every number shown comes from the API.
"use strict";

const $ = (s) => document.querySelector(s);
const inr = (x) => "₹" + Math.round(x).toLocaleString("en-IN");
const pct = (x, d = 1) => (x == null ? "–" : `${x > 0 ? "+" : ""}${x.toFixed(d)}%`);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const MODEL_NAMES = { naive: "Naive (last week)", seasonal_naive: "Seasonal naive (4 wks ago)", linear: "Linear regression", category_mean: "Category mean" };

const state = { summary: null, forecast: [], anomalies: null, charts: {}, lastCategorised: null };

async function api(path, opts = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), opts.timeout || 15000);
  try {
    const r = await fetch(path, { ...opts, signal: ctrl.signal, headers: { "Content-Type": "application/json" } });
    if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 120)}`);
    return await r.json();
  } finally { clearTimeout(t); }
}

// ---------- charts (optional: tables always render; charts only if Chart.js loaded) ----------
const hasChart = () => typeof window.Chart !== "undefined";
function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
const PALETTE = ["#0f766e", "#2563eb", "#d97706", "#db2777", "#7c3aed", "#059669", "#dc2626", "#0891b2", "#65a30d", "#6b7280"];
function drawChart(key, canvas, config) {
  if (!hasChart()) return;
  if (state.charts[key]) state.charts[key].destroy();
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--line");
  Chart.defaults.font.family = "system-ui, sans-serif";
  state.charts[key] = new Chart(canvas, config);
}

// ---------- try a merchant string ----------
const EXAMPLES = [
  "UPI/98765/SWIGGY/YESB0001", "POS 4521*ZOMATO ONLINE BANGA", "NEFT-DR-HDFC-AMAZON PAY INDIA",
  "ATM WDL 0912 SALEM TN", "upi-ani technologies-olacabs@ybl", "IMPS/P2A/8812/KAVITHA R/rent",
];
function initTry() {
  const chips = $("#chips");
  EXAMPLES.forEach((e) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = e;
    b.onclick = () => { $("#try-input").value = e; categorise(e); };
    chips.appendChild(b);
  });
  $("#try-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const v = $("#try-input").value.trim();
    if (v) categorise(v);
  });
}
async function categorise(s) {
  const out = $("#try-out"), btn = $("#try-btn");
  btn.disabled = true; out.innerHTML = '<span class="spinner"></span>Categorising…';
  try {
    const r = await api("/categorize", { method: "POST", body: JSON.stringify({ merchants: [s] }) });
    const x = r.results[0];
    state.lastCategorised = x;
    const badge = x.uncertain
      ? `<span class="badge warn" title="confidence below ${r.abstain_threshold}">UNCERTAIN: best guess</span>`
      : `<span class="badge ok">confident</span>`;
    out.innerHTML = `
      <div class="cat">${esc(x.category)} ${badge}</div>
      <div class="muted">confidence ${(x.confidence * 100).toFixed(1)}% · abstain below ${(r.abstain_threshold * 100).toFixed(0)}% · ${esc(r.model)}</div>
      <div class="bars">${x.top_3.map((t) => `
        <div class="bar-row"><span>${esc(t.label)}</span><span class="bar"><i style="width:${(t.prob * 100).toFixed(1)}%"></i></span><span class="r">${(t.prob * 100).toFixed(0)}%</span></div>`).join("")}
      </div>
      <div class="norm">model saw: ${esc(x.normalised)}</div>`;
  } catch (e) {
    out.innerHTML = `<span class="error">Couldn't categorise: ${esc(e.message)}</span>`;
  } finally { btn.disabled = false; }
}

// ---------- overview + categories ----------
async function loadSummary(month) {
  const s = await api("/summary" + (month ? `?month=${month}` : ""));
  state.summary = s;
  const sel = $("#month");
  if (!sel.options.length) {
    s.available_months.slice().reverse().forEach((m) => sel.add(new Option(m, m)));
    sel.onchange = () => loadSummary(sel.value).catch(showErr);
  }
  sel.value = s.month;
  $("#st-total").textContent = inr(s.total_spend);
  $("#st-top").textContent = s.top_category ? `${s.top_category.category}` : "–";
  const mom = $("#st-mom");
  mom.textContent = pct(s.mom_delta_pct);
  mom.className = "value " + (s.mom_delta_pct > 0 ? "up" : "down");

  const cats = s.categories;
  drawChart("mom", $("#ch-mom"), {
    type: "bar",
    data: {
      labels: cats.map((c) => c.category),
      datasets: [
        { label: s.prev_month || "prev", data: cats.map((c) => c.prev_total), backgroundColor: css("--line") },
        { label: s.month, data: cats.map((c) => c.total), backgroundColor: css("--accent") },
      ],
    },
    options: {
      indexAxis: "y", maintainAspectRatio: false, responsive: true,
      plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${inr(c.raw)}` } } },
      scales: { x: { ticks: { callback: (v) => "₹" + (v / 1000).toFixed(0) + "k" } } },
    },
  });

  $("#cat-table tbody").innerHTML = cats.map((c) => `
    <tr><td>${esc(c.category)}</td><td class="r">${inr(c.total)}</td><td class="r">${c.share_pct.toFixed(1)}%</td>
    <td class="r ${c.mom_delta_pct > 0 ? "up" : "down"}">${pct(c.mom_delta_pct, 0)}</td></tr>`).join("");
  drawChart("share", $("#ch-share"), {
    type: "doughnut",
    data: { labels: cats.map((c) => c.category), datasets: [{ data: cats.map((c) => c.total), backgroundColor: PALETTE, borderWidth: 0 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: (c) => `${c.label}: ${inr(c.raw)} (${cats[c.dataIndex].share_pct.toFixed(1)}%)` } } } },
  });
}

async function loadAnomalies() {
  const a = await api("/anomalies?days=30");
  state.anomalies = a;
  const ul = $("#anoms");
  ul.innerHTML = a.items.length ? a.items.slice(0, 6).map((i) => `
    <li><b>${inr(i.amount)}</b> · ${esc(i.category)} · ${esc(i.date)}<br>${esc(i.reason)}<div class="m">${esc(i.merchant)}</div></li>`).join("")
    : '<li class="muted">Nothing unusual in the last 30 days.</li>';
  if (a.items.length > 6) ul.insertAdjacentHTML("beforeend", `<li class="muted">+${a.items.length - 6} more (see /anomalies)</li>`);
}

// ---------- forecast ----------
async function loadForecast() {
  const f = await api("/forecast");
  state.forecast = f.items;
  const sel = $("#fc-cat");
  f.items.forEach((i) => sel.add(new Option(i.category, i.category)));
  sel.onchange = () => showForecast(sel.value);
  $("#fc-table tbody").innerHTML = f.items.map((i) => `
    <tr><td>${esc(i.category)}</td><td class="r">${inr(i.prediction)}</td><td class="r">${inr(i.lower_80)}–${inr(i.upper_80)}</td>
    <td>${esc(i.model_used)}</td><td class="r">${inr(i.holdout_mae)}</td></tr>`).join("");
  showForecast(f.items[0].category);
}
function showForecast(cat) {
  const i = state.forecast.find((x) => x.category === cat);
  $("#fc-card").innerHTML = `
    <div class="big">${inr(i.prediction)} <small class="muted">week of ${esc(i.week_start)}</small></div>
    <div class="stat"><span class="label">80% range</span><span class="value">${inr(i.lower_80)}–${inr(i.upper_80)}</span></div>
    <div class="stat"><span class="label">Model used</span><span class="value">${esc(MODEL_NAMES[i.model_used] || i.model_used)}</span></div>
    <div class="stat"><span class="label">Holdout MAE</span><span class="value">${inr(i.holdout_mae)}</span></div>
    <div class="stat"><span class="label">Interval hit rate</span><span class="value">${(i.interval_coverage_holdout * 100).toFixed(0)}% <small class="muted">(${esc(i.weeks_beat_reference)})</small></span></div>`;
  const labels = [...i.history_weeks, i.week_start];
  const hist = [...i.history_actual, null];
  const pt = labels.map((_, k) => (k === labels.length - 1 ? i.prediction : null));
  const lo = labels.map((_, k) => (k === labels.length - 1 ? i.lower_80 : null));
  const hi = labels.map((_, k) => (k === labels.length - 1 ? i.upper_80 : null));
  drawChart("fc", $("#ch-fc"), {
    type: "line",
    data: { labels, datasets: [
      { label: "actual weekly spend", data: hist, borderColor: css("--ink"), borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
      { label: "80% upper", data: hi, borderColor: "transparent", backgroundColor: css("--accent-soft"), pointRadius: 0, fill: "+1" },
      { label: "80% lower", data: lo, borderColor: "transparent", pointRadius: 0 },
      { label: `forecast (${i.model_used})`, data: pt, borderColor: css("--accent"), backgroundColor: css("--accent"), pointRadius: 6, showLine: false },
    ] },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { filter: (l) => !l.text.startsWith("80% lower") } },
        tooltip: { callbacks: { label: (c) => (c.raw == null ? "" : `${c.dataset.label}: ${inr(c.raw)}`) } } },
      scales: { x: { ticks: { maxTicksLimit: 6 } }, y: { ticks: { callback: (v) => "₹" + (v / 1000).toFixed(0) + "k" } } },
    },
  });
}

// ---------- advice ----------
async function getAdvice() {
  const btn = $("#advice-btn"), out = $("#advice-out");
  btn.disabled = true;
  out.innerHTML = '<span class="spinner"></span>Asking Gemini to explain the numbers…';
  try {
    const body = {
      summary: state.summary,
      forecast: [...state.forecast].sort((a, b) => b.prediction - a.prediction).slice(0, 3),
      anomalies: state.anomalies,
      categorization: state.lastCategorised ? [state.lastCategorised] : null,
    };
    const r = await api("/advice", { method: "POST", body: JSON.stringify(body), timeout: 20000 });
    const src = { llm: "Gemini", cache: "Gemini (cached)", fallback: "offline template (LLM unavailable)" }[r.source];
    out.innerHTML = `<div>${esc(r.text)}</div><div class="meta">Narrated by ${esc(src)} · ${esc(r.model)}${r.reason ? " · " + esc(r.reason) : ""}</div>`;
  } catch (e) {
    out.innerHTML = `<span class="error">Couldn't reach the server (${esc(e.message)}). The numbers above are still valid.</span>`;
  } finally { btn.disabled = false; }
}

function showErr(e) { console.error(e); $("#asof").textContent = "API error"; }

async function boot() {
  initTry();
  $("#advice-btn").onclick = getAdvice;
  if (!hasChart()) document.body.classList.add("no-chart");
  try {
    const h = await api("/health");
    $("#asof").textContent = `data to ${h.data_range[1]}${h.metrics_verified ? " · verified" : ""}`;
  } catch (e) { showErr(e); }
  await Promise.allSettled([loadSummary(), loadAnomalies(), loadForecast()]).then((rs) =>
    rs.filter((r) => r.status === "rejected").forEach((r) => showErr(r.reason)));
}
// Chart.js is deferred from a CDN; if it fails on mobile data the tables still render.
window.addEventListener("DOMContentLoaded", boot);
