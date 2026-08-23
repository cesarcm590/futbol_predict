"""Prediccion en vivo: La Liga, partidos del 24 de agosto de 2026
(Osasuna vs Levante UD, Malaga vs RC Deportivo La Coruna).

IMPORTANTE: nuestro historico de La Liga via StatsBomb llega hasta 2020/2021
(datos ricos: tiros, xG, posesion, pases). La temporada 2026-27 actual solo
la tenemos via resultados publicos (11 partidos jugados al 22-ago-2026), sin
esas estadisticas de evento. Para que el modelo sea honesto con lo que
realmente sabe DE AHORA, se entrena y predice usando SOLO el subconjunto de
variables que existen para la temporada actual (ranking dinamico + forma
reciente basada en goles) -- igual que Liga MX -- aunque el historico tenga
mas columnas disponibles.
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.standings import compute_dynamic_standings
from src.team_database import load_team_database, to_team_perspective, _append_to_db, DB_PATH
from src.team_form import add_rolling_form, prematch_feature_cols
from src.match_prediction import fit_hierarchical_model

COMPETITION_ID = 900002  # id sintetico, distinto del de Liga MX (900001)
SEASON_ID = 900998
SEASON_NAME = "2026/2027"

# --- 11 partidos confirmados de la temporada 2026-27 (nombres mapeados al historico StatsBomb) ---
CONFIRMED = [
    ("2026-08-15", "Deportivo Alavés", "Getafe", 3, 0),
    ("2026-08-16", "Espanyol", "Levante UD", 3, 0),
    ("2026-08-16", "Racing Santander", "Villarreal", 2, 2),
    ("2026-08-17", "Atlético Madrid", "Málaga", 2, 0),
    ("2026-08-17", "RC Deportivo La Coruña", "Elche", 1, 1),
    ("2026-08-17", "Real Betis", "Real Sociedad", 1, 0),
    ("2026-08-17", "Valencia", "Celta Vigo", 0, 0),
    ("2026-08-19", "Athletic Club", "Sevilla", 1, 3),
    ("2026-08-19", "Rayo Vallecano", "Deportivo Alavés", 1, 1),
    ("2026-08-21", "Espanyol", "Real Madrid", 1, 2),
    ("2026-08-21", "Sevilla", "Rayo Vallecano", 2, 1),
]

FIXTURES = [
    ("Osasuna", "Levante UD"),
    ("Málaga", "RC Deportivo La Coruña"),
]

rows = []
mid = COMPETITION_ID * 100000
for d, home, away, hs, as_ in CONFIRMED:
    rows.append({
        "match_id": mid, "competition_id": COMPETITION_ID, "season_id": SEASON_ID,
        "competition_name": "La Liga", "season_name": SEASON_NAME,
        "match_date": d, "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_,
    })
    mid += 1

confirmed = pd.DataFrame(rows)
standings = compute_dynamic_standings(
    confirmed[["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]]
)
confirmed_wide = confirmed.merge(standings, on="match_id", how="left")
print(f"Partidos confirmados La Liga 2026/2027 (al 22-ago-2026): {len(confirmed_wide)}")

_append_to_db(confirmed_wide)
print(f"Guardado en {DB_PATH}\n")

wide_all = load_team_database()
tp_all = to_team_perspective(wide_all)
ll = tp_all[tp_all["competition_name"] == "La Liga"].copy()
ll_form = add_rolling_form(ll, window=5)

# --- cuantos partidos de 2026-27 lleva jugados cada equipo de manana ---
print("Partidos jugados en 2026/2027 por cada equipo del fixture de mañana:")
for t in ["Osasuna", "Levante UD", "Málaga", "RC Deportivo La Coruña"]:
    n = len(ll_form[(ll_form["team"] == t) & (ll_form["season_name"] == SEASON_NAME)])
    last_hist = ll_form[(ll_form["team"] == t) & (ll_form["season_name"] != SEASON_NAME)]["match_date"].max()
    print(f"  {t:<26} {n} partido(s) esta temporada · último partido histórico previo: {last_hist}")

# --- SOLO el subconjunto de variables reducido (coherente con lo disponible hoy) ---
REDUCED_COLS = [c for c in prematch_feature_cols(ll_form) if c in (
    "win_pct_dynamic", "rank_dynamic", "points_before", "goal_diff_before", "games_before",
    "form_goals_for", "form_goals_against",
)]
print(f"\nVariables usadas (subconjunto reducido, sin tiros/xG/posesión): {REDUCED_COLS}")

latest = (
    ll_form.sort_values("match_date")
    .groupby("team")
    .tail(1)
    .set_index("team")[REDUCED_COLS + ["match_date"]]
)
latest = latest.loc[:, ~latest.columns.duplicated()]

from src.team_form import build_prediction_dataset
train_scope = ll_form[ll_form["season_name"] != SEASON_NAME]
train_data = build_prediction_dataset(train_scope)

home_cols = [f"home_{c}" for c in REDUCED_COLS]
away_cols = [f"away_{c}" for c in REDUCED_COLS]
extra = [c for c in ["rank_gap", "win_pct_gap", "form_goals_gap"] if c in train_data.columns]
cols = home_cols + away_cols + extra
cols = [c for c in cols if c in train_data.columns and train_data[c].notna().any() and train_data[c].std(skipna=True) > 0]

model = fit_hierarchical_model(train_data, cols, n_components=6)
print(f"Modelo entrenado con {len(train_data)} partidos históricos de La Liga (2004/05–2020/21), variables reducidas.\n")


def predict_future(home_team, away_team):
    row = {}
    for c in REDUCED_COLS:
        row[f"home_{c}"] = latest.loc[home_team, c] if home_team in latest.index else np.nan
        row[f"away_{c}"] = latest.loc[away_team, c] if away_team in latest.index else np.nan
    if "home_rank_dynamic" in row and "away_rank_dynamic" in row:
        row["rank_gap"] = abs(row["home_rank_dynamic"] - row["away_rank_dynamic"])
    if "home_win_pct_dynamic" in row and "away_win_pct_dynamic" in row:
        row["win_pct_gap"] = abs(row["home_win_pct_dynamic"] - row["away_win_pct_dynamic"])
    if "home_form_goals_for" in row and "away_form_goals_for" in row:
        row["form_goals_gap"] = abs(row["home_form_goals_for"] - row["away_form_goals_for"])
    return pd.DataFrame([row])


print("=== Predicción — 24 de agosto de 2026 ===")
for home, away in FIXTURES:
    X = predict_future(home, away).reindex(columns=cols).fillna(0.0)
    X_std = model["scaler"].transform(X)
    X_pca = model["pca"].transform(X_std)
    p_draw = model["draw_clf"].predict_proba(X_pca)[:, list(model["draw_clf"].classes_).index(1)][0]
    p_home_given_not_draw = model["home_clf"].predict_proba(X_pca)[:, list(model["home_clf"].classes_).index(1)][0]
    p_H = (1 - p_draw) * p_home_given_not_draw
    p_A = (1 - p_draw) * (1 - p_home_given_not_draw)
    p_D = p_draw
    probs = {"A": p_A, "D": p_D, "H": p_H}
    pred = max(probs, key=probs.get)
    print(f"  {home:<26} vs {away:<26} -> {pred}  (L {p_H*100:4.1f}% / E {p_D*100:4.1f}% / V {p_A*100:4.1f}%)  conf={probs[pred]*100:.1f}%")
