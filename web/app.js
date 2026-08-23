const DATA = "data/";
const fmtPct = (x) => (x == null ? "—" : `${x.toFixed(1)}%`);
const fmtDate = (d) => d || "—";

async function getJSON(name) {
  const res = await fetch(DATA + name, { cache: "force-cache" });
  if (!res.ok) throw new Error(`No se pudo cargar ${name}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Navegacion entre paneles
// ---------------------------------------------------------------------------
document.querySelectorAll("#mainNav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#mainNav button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll("section.panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`panel-${btn.dataset.panel}`).classList.add("active");
  });
});

// ---------------------------------------------------------------------------
// Stat strip (portada)
// ---------------------------------------------------------------------------
(async () => {
  const catalog = await getJSON("catalog.json");
  const totalMatches = catalog.reduce((a, c) => a + Math.round(c.n_matches_rows / 2), 0);
  const totalTeams = catalog.reduce((a, c) => a + c.n_teams, 0);
  const strip = document.getElementById("statStrip");
  strip.innerHTML = `
    <div class="stat"><b>${catalog.length}</b><span>competiciones analizadas</span></div>
    <div class="stat"><b>${totalMatches.toLocaleString("es-MX")}</b><span>partidos en la base</span></div>
    <div class="stat"><b>${totalTeams}</b><span>equipos (con repetición entre ligas)</span></div>
    <div class="stat"><b>2</b><span>ligas con predicción en vivo</span></div>
  `;
})();

// ===========================================================================
// PANEL: Predictor en vivo
// ===========================================================================
const LIVE_LEAGUES = [
  { slug: "ligamx", label: "Liga MX — Apertura 2026" },
  { slug: "laliga", label: "La Liga — 2026/27" },
];
let liveModel = null;
let currentLeague = LIVE_LEAGUES[0].slug;

const liveToggle = document.getElementById("liveLeagueToggle");
LIVE_LEAGUES.forEach((l) => {
  const b = document.createElement("button");
  b.textContent = l.label;
  b.dataset.slug = l.slug;
  if (l.slug === currentLeague) b.classList.add("active");
  b.addEventListener("click", () => loadLiveLeague(l.slug));
  liveToggle.appendChild(b);
});

function fillTeamSelects(model) {
  const teams = Object.keys(model.teams).sort((a, b) => a.localeCompare(b, "es"));
  for (const sel of [document.getElementById("homeSelect"), document.getElementById("awaySelect")]) {
    sel.innerHTML = teams.map((t) => `<option value="${t}">${t}</option>`).join("");
  }
  document.getElementById("awaySelect").selectedIndex = Math.min(1, teams.length - 1);
}

async function loadLiveLeague(slug) {
  currentLeague = slug;
  document.querySelectorAll("#liveLeagueToggle button").forEach((b) => b.classList.toggle("active", b.dataset.slug === slug));
  document.getElementById("predictResult").innerHTML = "";
  liveModel = await getJSON(`live_${slug}.json`);
  fillTeamSelects(liveModel);
  document.getElementById("liveCaveat").innerHTML = `
    <b>${liveModel.competition} — ${liveModel.season}.</b>
    Modelo entrenado con ${liveModel.n_train.toLocaleString("es-MX")} partidos históricos, usando solo
    ${liveModel.feature_cols.length} variables (ranking dinámico + forma reciente de goles) — el mismo
    subconjunto disponible para la temporada actual. No incluye tiros, xG ni posesión: para algunos
    equipos la "forma reciente" todavía arrastra su último partido histórico conocido, que puede ser
    de temporadas anteriores. Tómalo como una estimación, no como una certeza.
  `;
  runPredict();
}

function teamStateCard(team, model) {
  const s = model.teams[team];
  if (!s) return `<div class="team-state"><h4>${team}</h4><p class="muted">Sin datos.</p></div>`;
  const rows = model.feature_cols.map((c) => `<dt>${c}</dt><dd>${s[c] == null ? "—" : s[c]}</dd>`).join("");
  const freshness = s.current_season
    ? `<dt>Estado</dt><dd>temporada en curso</dd>`
    : `<dt>Estado</dt><dd>histórico (${s.as_of})</dd>`;
  return `<div class="team-state"><h4>${team}</h4><dl>${rows}${freshness}</dl></div>`;
}

function runPredict() {
  if (!liveModel) return;
  const home = document.getElementById("homeSelect").value;
  const away = document.getElementById("awaySelect").value;
  if (!home || !away || home === away) {
    document.getElementById("predictResult").innerHTML = `<p class="muted">Elige dos equipos distintos.</p>`;
    return;
  }
  const row = buildFeatureRow(liveModel, home, away);
  const out = predictHierarchical(liveModel, row);

  const labelMap = { H: `Gana ${home} (local)`, D: "Empate", A: `Gana ${away} (visita)` };
  const bars = [
    ["H", `Gana ${home}`, out.pH],
    ["D", "Empate", out.pD],
    ["A", `Gana ${away}`, out.pA],
  ].map(([cls, label, p]) => `
    <div class="pbar-row">
      <div class="pbar-label">${label}</div>
      <div class="pbar-track"><div class="pbar-fill ${cls}" style="width:${(p * 100).toFixed(1)}%"></div></div>
      <div class="pbar-pct">${fmtPct(p * 100)}</div>
    </div>`).join("");

  document.getElementById("predictResult").innerHTML = `
    <div class="pred-headline">Predicción: <b>${labelMap[out.pred]}</b> <span class="muted" style="font-size:0.85rem;">(confianza ${fmtPct(out.confidence * 100)})</span></div>
    ${bars}
    <div class="team-state-grid">
      ${teamStateCard(home, liveModel)}
      ${teamStateCard(away, liveModel)}
    </div>
  `;
}

document.getElementById("predictBtn").addEventListener("click", runPredict);
document.getElementById("homeSelect").addEventListener("change", runPredict);
document.getElementById("awaySelect").addEventListener("change", runPredict);

loadLiveLeague(currentLeague);

// ===========================================================================
// PANEL: Explorar equipos
// ===========================================================================
let catalogCache = null;
let currentComp = null;
let exploreData = { rankings: [], pca: null, styles: null, backtest: null };

async function initExplore() {
  catalogCache = await getJSON("catalog.json");
  const compSel = document.getElementById("compSelect");
  compSel.innerHTML = catalogCache.map((c) => `<option value="${c.slug}">${c.name}</option>`).join("");
  compSel.addEventListener("change", () => loadCompetition(compSel.value));
  document.getElementById("teamSelect").addEventListener("change", renderTeamViews);
  loadCompetition(catalogCache[0].slug);
}

async function loadCompetition(slug) {
  currentComp = slug;
  const [rankings, pca, styles] = await Promise.all([
    getJSON(`rankings_${slug}.json`),
    getJSON(`pca_${slug}.json`),
    getJSON(`styles_${slug}.json`),
  ]);
  let backtest = null;
  try { backtest = await getJSON(`backtest_${slug}.json`); } catch (e) { backtest = null; }
  exploreData = { rankings, pca, styles, backtest };

  const teams = [...new Set(pca.points.map((p) => p.team))].sort((a, b) => a.localeCompare(b, "es"));
  const teamSel = document.getElementById("teamSelect");
  teamSel.innerHTML = teams.map((t) => `<option value="${t}">${t}</option>`).join("");

  renderTeamViews();
}

function renderTeamViews() {
  const team = document.getElementById("teamSelect").value;
  renderRankChart(team);
  renderPcaChart(team);
  renderStylesTable(team);
  renderBacktest(team);
}

function renderRankChart(team) {
  const rows = exploreData.rankings.filter((r) => r.team === team).sort((a, b) => a.date.localeCompare(b.date));
  const el = document.getElementById("rankChart");
  if (!rows.length) { el.innerHTML = `<p class="muted">Esta competición no tiene ranking dinámico calculado.</p>`; return; }
  Plotly.newPlot(el, [{
    x: rows.map((r) => r.date), y: rows.map((r) => r.rank),
    mode: "lines+markers", line: { color: "#1c6b4a", width: 2 }, marker: { size: 6 },
    text: rows.map((r) => `vs ${r.opponent} (${r.result}) ${r.gf}-${r.ga}`),
    hovertemplate: "%{x}<br>Posición %{y}<br>%{text}<extra></extra>",
  }], {
    margin: { t: 10, r: 10, b: 40, l: 50 }, yaxis: { autorange: "reversed", title: "Posición (1 = líder)" },
    xaxis: { title: "" }, paper_bgcolor: "transparent", plot_bgcolor: "transparent",
    font: { color: getComputedStyle(document.body).getPropertyValue("--ink") },
  }, { displayModeBar: false, responsive: true });
}

function renderPcaChart(team) {
  const pts = exploreData.pca.points;
  const el = document.getElementById("pcaChart");
  const others = pts.filter((p) => p.team !== team);
  const mine = pts.filter((p) => p.team === team);
  const trace = (arr, name, color, opacity, size) => ({
    x: arr.map((p) => p.pc1), y: arr.map((p) => p.pc2), mode: "markers", name,
    marker: { color, opacity, size },
    text: arr.map((p) => `${p.team} vs ${p.opponent} (${p.result})`),
    hovertemplate: "%{text}<br>PC1=%{x:.2f} PC2=%{y:.2f}<extra></extra>",
  });
  Plotly.newPlot(el, [
    trace(others, "Otros equipos", "#8b9aa8", 0.45, 6),
    trace(mine, team, "#e0553e", 0.9, 9),
  ], {
    margin: { t: 10, r: 10, b: 40, l: 50 }, xaxis: { title: "PC1" }, yaxis: { title: "PC2" },
    paper_bgcolor: "transparent", plot_bgcolor: "transparent", legend: { orientation: "h", y: -0.2 },
    font: { color: getComputedStyle(document.body).getPropertyValue("--ink") },
  }, { displayModeBar: false, responsive: true });
}

function renderStylesTable(team) {
  const rows = exploreData.styles.points.filter((p) => p.team === team).sort((a, b) => b.date.localeCompare(a.date));
  const tbody = document.querySelector("#stylesTable tbody");
  tbody.innerHTML = rows.map((r) => `
    <tr><td>${fmtDate(r.date)}</td><td>${r.opponent}</td>
    <td><span class="pill ${r.result}">${r.result}</span></td><td>${r.cluster}</td></tr>
  `).join("") || `<tr><td colspan="4" class="muted">Sin datos.</td></tr>`;
}

function renderBacktest(team) {
  const card = document.getElementById("backtestCard");
  const bt = exploreData.backtest;
  if (!bt) {
    card.querySelector("#backtestHint").textContent = "Muestra insuficiente en esta competición para un backtest confiable (se necesitan al menos 2 temporadas con volumen razonable).";
    document.getElementById("backtestMetrics").innerHTML = "";
    document.querySelector("#backtestTable tbody").innerHTML = "";
    return;
  }
  card.querySelector("#backtestHint").innerHTML =
    `Entrena con ${bt.train_seasons.length} temporada(s) anteriores y prueba contra <b>${bt.test_season}</b> (${bt.n_test} partidos) — nunca ve la temporada de prueba durante el entrenamiento.`;

  const beat = bt.accuracy_pct >= bt.naive_accuracy_pct;
  document.getElementById("backtestMetrics").innerHTML = `
    <div class="m"><b>${fmtPct(bt.accuracy_pct)}</b><span>accuracy del modelo (${bt.aciertos}/${bt.n_test})</span></div>
    <div class="m"><b>${fmtPct(bt.naive_accuracy_pct)}</b><span>baseline ingenuo: siempre "${bt.naive_class}" (${bt.naive_aciertos}/${bt.n_test})</span></div>
    <div class="m"><b style="color:${beat ? 'var(--win)' : 'var(--loss)'}">${beat ? "Le gana" : "No le gana"}</b><span>al baseline ingenuo</span></div>
  `;

  const rows = team ? bt.matches.filter((m) => m.home === team || m.away === team) : bt.matches;
  document.querySelector("#backtestTable tbody").innerHTML = rows.map((m) => `
    <tr>
      <td>${m.date}</td><td>${m.home}</td>
      <td class="mono">${m.hs}–${m.as_}</td>
      <td>${m.away}</td>
      <td><span class="pill ${m.pred}">${m.pred}</span> ${m.acierto ? "✓" : ""}</td>
      <td class="mono">${fmtPct(m.conf_pct)}</td>
    </tr>
  `).join("") || `<tr><td colspan="6" class="muted">Sin partidos de este equipo en el set de prueba.</td></tr>`;
}

initExplore();
