// Inferencia 100% en el navegador del modelo jerarquico (StandardScaler + PCA +
// 2 Random Forest binarios) exportado por export_web_data.py. Sin backend,
// sin WebAssembly, sin librerias externas -- solo el mismo algebra lineal y
// recorrido de arboles que hace scikit-learn, reimplementado en JS.

function standardize(x, scaler) {
  return x.map((v, i) => (v - scaler.mean[i]) / scaler.scale[i]);
}

function pcaTransform(xStd, pca) {
  const centered = xStd.map((v, i) => v - pca.mean[i]);
  return pca.components.map((row) => row.reduce((acc, w, i) => acc + w * centered[i], 0));
}

function treePredictProba(tree, xPca) {
  let node = 0;
  while (tree.left[node] !== -1) {
    const feat = tree.feature[node];
    node = xPca[feat] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
  }
  return tree.leaf_p1[node];
}

function forestPredictProba(trees, xPca) {
  const sum = trees.reduce((acc, t) => acc + treePredictProba(t, xPca), 0);
  return sum / trees.length;
}

// modelJson: payload de live_<slug>.json
// featureRow: dict con las mismas claves que modelJson.model_cols (faltantes -> 0)
function predictHierarchical(modelJson, featureRow) {
  const x = modelJson.model_cols.map((c) => (c in featureRow && featureRow[c] !== null && !Number.isNaN(featureRow[c])) ? featureRow[c] : 0.0);
  const xStd = standardize(x, modelJson.scaler);
  const xPca = pcaTransform(xStd, modelJson.pca);

  const pDraw = forestPredictProba(modelJson.draw_trees, xPca);
  const pHomeGivenNotDraw = forestPredictProba(modelJson.home_trees, xPca);

  const pH = (1 - pDraw) * pHomeGivenNotDraw;
  const pA = (1 - pDraw) * (1 - pHomeGivenNotDraw);
  const pD = pDraw;

  const probs = { H: pH, D: pD, A: pA };
  const pred = Object.keys(probs).reduce((a, b) => (probs[a] > probs[b] ? a : b));
  return { pH, pD, pA, pred, confidence: probs[pred] };
}

// Arma el featureRow home_*/away_*/gap a partir del estado guardado de cada equipo
function buildFeatureRow(modelJson, homeTeam, awayTeam) {
  const teams = modelJson.teams;
  const row = {};
  for (const c of modelJson.feature_cols) {
    row[`home_${c}`] = teams[homeTeam] ? teams[homeTeam][c] : null;
    row[`away_${c}`] = teams[awayTeam] ? teams[awayTeam][c] : null;
  }
  if ("home_rank_dynamic" in row && "away_rank_dynamic" in row && row.home_rank_dynamic != null && row.away_rank_dynamic != null) {
    row.rank_gap = Math.abs(row.home_rank_dynamic - row.away_rank_dynamic);
  }
  if ("home_win_pct_dynamic" in row && "away_win_pct_dynamic" in row && row.home_win_pct_dynamic != null && row.away_win_pct_dynamic != null) {
    row.win_pct_gap = Math.abs(row.home_win_pct_dynamic - row.away_win_pct_dynamic);
  }
  if ("home_form_goals_for" in row && "away_form_goals_for" in row && row.home_form_goals_for != null && row.away_form_goals_for != null) {
    row.form_goals_gap = Math.abs(row.home_form_goals_for - row.away_form_goals_for);
  }
  return row;
}
