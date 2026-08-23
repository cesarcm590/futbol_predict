"""Prediccion en vivo para el Torneo Apertura 2026 de Liga MX.

Usa los resultados reales de las jornadas 1-5 (confirmados via Wikipedia,
21-23 agosto 2026) como estado actual de cada equipo, entrena el modelo
jerarquico con las 30 temporadas historicas (Apertura 2010 - Clausura 2025,
sin incluir Apertura 2026 para mantener la separacion temporal), y predice
los partidos pendientes de la jornada 5 + toda la jornada 6.
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.standings import compute_dynamic_standings
from src.team_database import load_team_database, to_team_perspective, _append_to_db, DB_PATH
from src.team_form import add_rolling_form, prematch_feature_cols
from src.match_prediction import prediction_feature_cols, fit_hierarchical_model

COMPETITION_ID = 900001
SEASON_ID = 900999
SEASON_NAME = "Apertura 2026"

# --- 41 partidos confirmados, jornadas 1-5 (nombres mapeados al dataset historico) ---
J1 = [
    ("Club Necaxa", "Atlante", 2, 1),
    ("Club Tijuana", "UANL Tigres", 3, 1),
    ("Atlético San Luis", "Cruz Azul", 2, 3),
    ("Club León", "Atlas Guadalajara", 2, 3),
    ("FC Juárez", "Puebla FC", 0, 1),
    ("Pumas UNAM", "CF Pachuca", 0, 3),
    ("CF Monterrey", "Santos Laguna", 3, 2),
    ("Deportivo Guadalajara", "Deportivo Toluca", 0, 2),
    ("Gallos Blancos", "CF América", 0, 1),
]
J2 = [
    ("Cruz Azul", "Puebla FC", 2, 1),
    ("Deportivo Toluca", "Pumas UNAM", 1, 2),
    ("Atlante", "CF América", 1, 1),
    ("Club Tijuana", "Club León", 1, 0),
    ("Deportivo Guadalajara", "FC Juárez", 1, 0),
    ("Santos Laguna", "Atlas Guadalajara", 0, 1),
    ("UANL Tigres", "Atlético San Luis", 2, 2),
    ("Club Necaxa", "CF Monterrey", 2, 1),
    ("CF Pachuca", "Gallos Blancos", 1, 2),
]
J3 = [
    ("Puebla FC", "Deportivo Guadalajara", 1, 1),
    ("Atlético San Luis", "Club Tijuana", 0, 0),
    ("FC Juárez", "Pumas UNAM", 1, 5),
    ("Gallos Blancos", "UANL Tigres", 3, 2),
    ("Atlas Guadalajara", "CF Monterrey", 0, 2),
    ("Club León", "CF Pachuca", 1, 0),
    ("Cruz Azul", "Atlante", 2, 3),
    ("CF América", "Santos Laguna", 3, 0),
    ("Deportivo Toluca", "Club Necaxa", 3, 1),
]
J4 = [
    ("Atlante", "Deportivo Toluca", 0, 0),
    ("CF Monterrey", "FC Juárez", 6, 1),
    ("Atlas Guadalajara", "UANL Tigres", 2, 1),
    ("Pumas UNAM", "Gallos Blancos", 0, 0),
    ("CF América", "Atlético San Luis", 3, 0),
    ("Santos Laguna", "Deportivo Guadalajara", 0, 1),
    ("Club Tijuana", "Cruz Azul", 2, 1),
    ("Club Necaxa", "Club León", 1, 2),
    ("CF Pachuca", "Puebla FC", 2, 3),
]
J5_confirmados = [
    ("Club León", "CF Monterrey", 2, 0),
    ("UANL Tigres", "Atlante", 2, 0),
    ("FC Juárez", "CF América", 1, 2),
    ("Gallos Blancos", "Deportivo Toluca", 1, 2),
    ("Puebla FC", "Santos Laguna", 3, 2),
]

J5_pendientes = [
    ("Deportivo Guadalajara", "Club Tijuana"),
    ("Cruz Azul", "Atlas Guadalajara"),
    ("Atlético San Luis", "CF Pachuca"),
    ("Pumas UNAM", "Club Necaxa"),
]
J6 = [
    ("Club Necaxa", "Cruz Azul"),
    ("Atlante", "Club León"),
    ("Club Tijuana", "Pumas UNAM"),
    ("Atlas Guadalajara", "Gallos Blancos"),
    ("CF Pachuca", "Deportivo Guadalajara"),
    ("CF América", "Puebla FC"),
    ("Santos Laguna", "UANL Tigres"),
    ("Deportivo Toluca", "FC Juárez"),
    ("CF Monterrey", "Atlético San Luis"),
]

JORNADA_DATES = {
    1: date(2026, 7, 17), 2: date(2026, 7, 24), 3: date(2026, 8, 1),
    4: date(2026, 8, 16), 5: date(2026, 8, 22),
}

rows = []
mid = COMPETITION_ID * 100000 + 900000
for jornada, matches in [(1, J1), (2, J2), (3, J3), (4, J4), (5, J5_confirmados)]:
    for home, away, hs, as_ in matches:
        rows.append({
            "match_id": mid, "competition_id": COMPETITION_ID, "season_id": SEASON_ID,
            "competition_name": "Liga MX", "season_name": SEASON_NAME,
            "match_date": JORNADA_DATES[jornada].isoformat(),
            "home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
        })
        mid += 1

confirmed = pd.DataFrame(rows)
standings = compute_dynamic_standings(
    confirmed[["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]]
)
confirmed_wide = confirmed.merge(standings, on="match_id", how="left")

print(f"Partidos confirmados Apertura 2026 (J1-J5): {len(confirmed_wide)}")

# Persistir a la base para que el dashboard tambien lo vea
_append_to_db(confirmed_wide)
print(f"Guardado en {DB_PATH}")

# --- Reconstruir team_perspective con TODO el historico + Apertura 2026 ---
wide_all = load_team_database()
tp_all = to_team_perspective(wide_all)
mx = tp_all[tp_all["competition_name"] == "Liga MX"].copy()
mx_form = add_rolling_form(mx, window=5)

# --- Estado MAS RECIENTE de cada equipo (su ultimo partido jugado, sea J4 o J5) ---
feat_cols = prematch_feature_cols(mx_form)
latest = (
    mx_form.sort_values("match_date")
    .groupby("team")
    .tail(1)
    .set_index("team")[feat_cols + ["match_date"]]
)
latest = latest.loc[:, ~latest.columns.duplicated()]

print("\nEstado actual (tras su ultimo partido jugado) de los equipos en juego:")
print(latest[["match_date", "rank_dynamic", "win_pct_dynamic"]].sort_values("rank_dynamic").to_string())

# --- Entrenar el modelo jerarquico SOLO con historico anterior a Apertura 2026 ---
train_scope = mx_form[mx_form["season_name"] != SEASON_NAME]
from src.team_form import build_prediction_dataset
train_data = build_prediction_dataset(train_scope)
cols = prediction_feature_cols(train_data)
model = fit_hierarchical_model(train_data, cols, n_components=6)
print(f"\nModelo entrenado con {len(train_data)} partidos historicos (30 torneos, Apertura 2010 - Clausura 2025).")


def predict_future(home_team, away_team):
    row = {}
    for c in feat_cols:
        row[f"home_{c}"] = latest.loc[home_team, c] if home_team in latest.index else np.nan
        row[f"away_{c}"] = latest.loc[away_team, c] if away_team in latest.index else np.nan
    if "home_rank_dynamic" in row and "away_rank_dynamic" in row:
        row["rank_gap"] = abs(row["home_rank_dynamic"] - row["away_rank_dynamic"])
    if "home_win_pct_dynamic" in row and "away_win_pct_dynamic" in row:
        row["win_pct_gap"] = abs(row["home_win_pct_dynamic"] - row["away_win_pct_dynamic"])
    if "home_form_goals_for" in row and "away_form_goals_for" in row:
        row["form_goals_gap"] = abs(row["home_form_goals_for"] - row["away_form_goals_for"])
    return pd.DataFrame([row])


def predict_and_show(label, fixtures, is_tuple2):
    print(f"\n=== {label} ===")
    results = []
    for fx in fixtures:
        home, away = fx[0], fx[1]
        X = predict_future(home, away)[cols].fillna(0.0)
        X_std = model["scaler"].transform(X)
        X_pca = model["pca"].transform(X_std)
        p_draw = model["draw_clf"].predict_proba(X_pca)[:, list(model["draw_clf"].classes_).index(1)][0]
        p_home_given_not_draw = model["home_clf"].predict_proba(X_pca)[:, list(model["home_clf"].classes_).index(1)][0]
        p_H = (1 - p_draw) * p_home_given_not_draw
        p_A = (1 - p_draw) * (1 - p_home_given_not_draw)
        p_D = p_draw
        probs = {"A": p_A, "D": p_D, "H": p_H}
        pred = max(probs, key=probs.get)
        results.append({
            "home": home, "away": away, "p_H": p_H, "p_D": p_D, "p_A": p_A,
            "pred": pred, "conf": probs[pred],
        })
        print(f"  {home:<24} vs {away:<24}  ->  {pred}  (L {p_H*100:4.1f}% / E {p_D*100:4.1f}% / V {p_A*100:4.1f}%)")
    return pd.DataFrame(results)


res_j5 = predict_and_show("Jornada 5 - partidos pendientes", J5_pendientes, True)
res_j6 = predict_and_show("Jornada 6 - calendario completo", J6, True)

res_j5.to_csv("predicciones_j5_pendiente.csv", index=False)
res_j6.to_csv("predicciones_j6.csv", index=False)
print("\nGuardado: predicciones_j5_pendiente.csv, predicciones_j6.csv")
