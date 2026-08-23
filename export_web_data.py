"""Exporta toda la data y los modelos necesarios para el dashboard estatico
(web/) que se puede publicar en Netlify sin backend: JSON precalculado para
las graficas (ranking dinamico, PCA, k-means, backtest de predicciones) y
arboles de Random Forest exportados a JSON para correr el modelo jerarquico
de Liga MX y La Liga EN EL NAVEGADOR (sin servidor Python).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")

from src.match_prediction import (
    fit_hierarchical_model, predict_hierarchical, prediction_feature_cols, temporal_split,
)
from src.pca_analysis import run_kmeans, run_pca
from src.team_database import TEAM_ID_COLS, load_team_database, team_feature_cols, to_team_perspective
from src.team_form import add_rolling_form, build_prediction_dataset, prematch_feature_cols

OUT = Path(__file__).resolve().parent / "web" / "data"
OUT.mkdir(parents=True, exist_ok=True)

EXPLORE_COMPS = [
    "Liga MX", "La Liga", "FA Women's Super League", "Ligue 1",
    "Premier League", "Serie A", "NWSL", "1. Bundesliga",
]


def slug(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def jnum(x):
    """JSON-safe float: NaN/inf -> None, numpy types -> python."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(xf):
        return None
    return round(xf, 4)


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    print(f"  {path.relative_to(OUT.parent.parent)}  ({path.stat().st_size/1024:.0f} KB)")


# ---------------------------------------------------------------------------
# 1) Seccion "explorar equipos": ranking dinamico + PCA + k-means + backtest
# ---------------------------------------------------------------------------
print("Cargando base de equipos...")
wide = load_team_database()
tp_all = to_team_perspective(wide)

catalog = []
for comp in EXPLORE_COMPS:
    scope = tp_all[tp_all["competition_name"] == comp].copy()
    if scope.empty:
        continue
    sl = slug(comp)
    seasons = sorted(scope["season_name"].dropna().unique().tolist())
    teams = sorted(scope["team"].dropna().unique().tolist())
    print(f"\n{comp} ({len(scope)} filas, {len(seasons)} temporadas, {len(teams)} equipos)")

    feature_cols = team_feature_cols(scope)

    # --- ranking dinamico ---
    rank_series = []
    if "rank_dynamic" in scope.columns and scope["rank_dynamic"].notna().any():
        rk = scope.dropna(subset=["rank_dynamic"]).sort_values(["team", "match_date"])
        for _, r in rk.iterrows():
            rank_series.append({
                "team": r["team"], "season": r["season_name"], "date": str(r["match_date"])[:10],
                "opponent": r["opponent"], "result": r["result"],
                "gf": jnum(r.get("goals_for")), "ga": jnum(r.get("goals_against")),
                "rank": jnum(r["rank_dynamic"]),
            })

    # --- PCA ---
    n_comp = max(2, min(5, len(feature_cols), len(scope) - 1))
    pca_res = run_pca(scope, feature_cols=feature_cols, n_components=n_comp, id_cols=TEAM_ID_COLS)
    pca_points = []
    sc = pca_res["scores"]
    for _, r in sc.iterrows():
        pca_points.append({
            "team": r["team"], "season": r["season_name"], "date": str(r.get("match_date", ""))[:10],
            "opponent": r.get("opponent"), "result": r.get("result"),
            "pc1": jnum(r["PC1"]), "pc2": jnum(r["PC2"]),
        })
    explained = [
        {"component": row["component"], "pct": jnum(row["explained_var_pct"]), "cum_pct": jnum(row["cumulative_pct"])}
        for _, row in pca_res["explained"].iterrows()
    ]
    loadings = {
        pc: {feat: jnum(pca_res["loadings"].loc[feat, pc]) for feat in pca_res["loadings"].index}
        for pc in pca_res["loadings"].columns
    }

    # --- k-means (k=4, fijo para la version estatica) ---
    clustered, profile = run_kmeans(scope, feature_cols=feature_cols, k=4)
    style_points = [
        {"team": r["team"], "season": r["season_name"], "date": str(r.get("match_date", ""))[:10],
         "opponent": r.get("opponent"), "result": r.get("result"), "cluster": r["cluster_label"]}
        for _, r in clustered.iterrows()
    ]
    style_profile = {
        label: {feat: jnum(profile.loc[label, feat]) for feat in profile.columns}
        for label in profile.index
    }

    # --- backtest de prediccion (temporada mas reciente vs anteriores) ---
    backtest = None
    tp_form = add_rolling_form(scope, window=5)
    pred_data = build_prediction_dataset(tp_form)
    feat_cols2 = prediction_feature_cols(pred_data)
    train, test, test_season = temporal_split(pred_data)
    if len(train) >= 20 and len(test) >= 5:
        model = fit_hierarchical_model(train, feat_cols2, n_components=8)
        pred = predict_hierarchical(model, test)
        proba_cols = [c for c in pred.columns if c.startswith("proba_")]
        aciertos = int(pred["acierto"].sum())
        naive_class = train["result"].value_counts().idxmax()
        naive_aciertos = int((pred["result"] == naive_class).sum())
        backtest = {
            "train_seasons": sorted(pred_data[pred_data["season_name"] != test_season]["season_name"].unique().tolist()),
            "test_season": test_season,
            "n_train": len(train), "n_test": len(test),
            "aciertos": aciertos, "accuracy_pct": jnum(aciertos / len(pred) * 100),
            "naive_class": naive_class, "naive_aciertos": naive_aciertos,
            "naive_accuracy_pct": jnum(naive_aciertos / len(pred) * 100),
            "matches": [
                {
                    "date": str(r["match_date"])[:10], "home": r["home_team"], "away": r["away_team"],
                    "hs": int(r["home_score"]), "as_": int(r["away_score"]), "result": r["result"],
                    "pred": r["prediccion"], "conf_pct": jnum(r["confianza"] * 100),
                    "pA": jnum(r["proba_A"] * 100), "pD": jnum(r["proba_D"] * 100), "pH": jnum(r["proba_H"] * 100),
                    "acierto": bool(r["acierto"]),
                }
                for _, r in pred.sort_values("confianza", ascending=False).iterrows()
            ],
        }
        print(f"  backtest {test_season}: {aciertos}/{len(pred)} aciertos")
    else:
        print("  (sin backtest: muestra insuficiente)")

    dump(OUT / f"rankings_{sl}.json", rank_series)
    dump(OUT / f"pca_{sl}.json", {"explained": explained, "loadings": loadings, "points": pca_points})
    dump(OUT / f"styles_{sl}.json", {"profile": style_profile, "points": style_points})
    if backtest:
        dump(OUT / f"backtest_{sl}.json", backtest)

    catalog.append({
        "name": comp, "slug": sl, "n_matches_rows": len(scope), "n_teams": len(teams),
        "seasons": seasons, "has_rankings": len(rank_series) > 0, "has_backtest": backtest is not None,
    })

dump(OUT / "catalog.json", catalog)


# ---------------------------------------------------------------------------
# 2) Modelos "en vivo" (Liga MX Apertura 2026, La Liga 2026/27) -> JSON de
#    arboles para correr inferencia 100% en el navegador (sin backend).
# ---------------------------------------------------------------------------
REDUCED_COLS = ["win_pct_dynamic", "rank_dynamic", "points_before", "goal_diff_before",
                "games_before", "form_goals_for", "form_goals_against"]
WEB_N_ESTIMATORS = 60   # menos arboles que el modelo de analisis (300) -> JSON manejable en el navegador
WEB_N_COMPONENTS = 6


def export_tree(tree, class_of_interest_idx: int) -> dict:
    t = tree.tree_
    return {
        "feature": t.feature.tolist(),
        "threshold": [jnum(x) for x in t.threshold.tolist()],
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        # proba de la clase de interes en cada nodo (solo se usa en las hojas)
        "leaf_p1": [jnum(v[0][class_of_interest_idx] / max(v[0].sum(), 1e-9)) for v in t.value],
    }


def export_forest(clf: RandomForestClassifier) -> list:
    idx1 = list(clf.classes_).index(1)
    return [export_tree(est, idx1) for est in clf.estimators_]


def export_live_model(comp_name: str, season_name: str, out_slug: str) -> None:
    scope = tp_all[tp_all["competition_name"] == comp_name].copy()
    form = add_rolling_form(scope, window=5)
    feat_cols = [c for c in prematch_feature_cols(form) if c in REDUCED_COLS]

    latest = (
        form.sort_values("match_date").groupby("team").tail(1)
        .set_index("team")[feat_cols + ["match_date", "season_name"]]
    )
    latest = latest.loc[:, ~latest.columns.duplicated()]

    train_scope = form[form["season_name"] != season_name]
    train_data = build_prediction_dataset(train_scope)
    home_cols = [f"home_{c}" for c in feat_cols]
    away_cols = [f"away_{c}" for c in feat_cols]
    extra = [c for c in ["rank_gap", "win_pct_gap", "form_goals_gap"] if c in train_data.columns]
    cols = home_cols + away_cols + extra
    cols = [c for c in cols if c in train_data.columns and train_data[c].notna().any() and train_data[c].std(skipna=True) > 0]

    model = fit_hierarchical_model(train_data, cols, n_components=WEB_N_COMPONENTS, random_state=42)
    # re-entrenar los clasificadores con menos arboles solo para exportar (mismo scaler/pca)
    X_std = model["scaler"].transform(train_data[cols].fillna(0.0))
    X_pca = model["pca"].transform(X_std)
    is_draw = (train_data["result"] == "D").astype(int).values
    draw_clf = RandomForestClassifier(n_estimators=WEB_N_ESTIMATORS, random_state=42, class_weight="balanced")
    draw_clf.fit(X_pca, is_draw)
    non_draw = train_data["result"] != "D"
    home_clf = RandomForestClassifier(n_estimators=WEB_N_ESTIMATORS, random_state=42, class_weight="balanced")
    home_clf.fit(X_pca[non_draw.values], (train_data.loc[non_draw, "result"] == "H").astype(int).values)

    teams_state = {}
    for team, row in latest.iterrows():
        teams_state[team] = {
            **{c: jnum(row[c]) for c in feat_cols},
            "as_of": str(row["match_date"])[:10],
            "current_season": bool(row["season_name"] == season_name),
        }

    payload = {
        "competition": comp_name, "season": season_name, "feature_cols": feat_cols,
        "model_cols": cols, "n_train": len(train_data),
        "scaler": {"mean": [jnum(x) for x in model["scaler"].mean_], "scale": [jnum(x) for x in model["scaler"].scale_]},
        "pca": {
            "mean": [jnum(x) for x in model["pca"].mean_],
            "components": [[jnum(x) for x in row] for row in model["pca"].components_],
        },
        "draw_trees": export_forest(draw_clf),
        "home_trees": export_forest(home_clf),
        "teams": teams_state,
    }
    dump(OUT / f"live_{out_slug}.json", payload)
    print(f"  {out_slug}: {len(teams_state)} equipos, {len(cols)} variables de entrada, {len(draw_clf.estimators_)}+{len(home_clf.estimators_)} arboles")


print("\nExportando modelos en vivo...")
export_live_model("Liga MX", "Apertura 2026", "ligamx")
export_live_model("La Liga", "2026/2027", "laliga")

print("\nListo.")
