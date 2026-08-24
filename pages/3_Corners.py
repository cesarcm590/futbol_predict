import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.corner_model import CORNER_LINES, evaluate_corners, fit_corners_model, predict_corners
from src.match_prediction import prediction_feature_cols, temporal_split
from src.team_database import load_team_database, to_team_perspective
from src.team_form import add_rolling_form, build_prediction_dataset

st.set_page_config(page_title="Corners — Fútbol", layout="wide")

st.title("Corners totales por partido")
st.caption(
    "Random Forest de regresión (sobre componentes principales) que estima corners totales "
    "(local + visita) usando solo información previa al partido: forma reciente de corners a "
    "favor/en contra de cada equipo, ranking dinámico y qué tan parejos están. Cada árbol del "
    "bosque da su propia estimación — la fracción de árboles por encima de una línea es la "
    "probabilidad de 'Over' para esa línea, sin entrenar un modelo aparte por cada una."
)
st.warning(
    "Solo funciona en competiciones con conteo de corners por partido (StatsBomb o "
    "football-data.co.uk) — no en Liga MX vía openfootball, que solo trae resultados. Esto es "
    "un ejercicio analítico, no una recomendación de apuesta. Con la muestra actual, en la "
    "mayoría de las líneas el modelo queda prácticamente empatado con el baseline ingenuo — "
    "no lo tomes como una señal fuerte."
)


@st.cache_data(show_spinner="Cargando base de equipos...")
def _load():
    wide = load_team_database()
    tp = to_team_perspective(wide)
    return tp


if st.button("Refrescar base de datos"):
    _load.clear()

tp = _load()
if tp.empty or "corners" not in tp.columns:
    st.info("Todavía no hay datos de corners sincronizados. Corre `resync_corners.py` para agregarlos a la base.")
    st.stop()

has_corners = tp.groupby("competition_name")["corners"].apply(lambda s: s.notna().sum())
season_counts = tp.groupby("competition_name")["season_name"].nunique()
eligible = sorted(set(has_corners[has_corners >= 40].index) & set(season_counts[season_counts >= 2].index))

if not eligible:
    st.info("Ninguna competición tiene todavía suficientes partidos con corners y 2+ temporadas. Espera a que termine la sincronización.")
    st.stop()

with st.sidebar:
    st.header("Filtros")
    comp_choice = st.selectbox("Competición", eligible)
    window = st.slider("Partidos de forma reciente (rolling)", 3, 10, 5)
    n_comp = st.slider("Componentes PCA", 2, 15, 8)

scope = tp[tp["competition_name"] == comp_choice].copy()

if "data_source" in scope.columns:
    src_counts = (scope["data_source"].value_counts() / 2).round(0).astype(int)  # /2: cada partido son 2 filas (una por equipo)
    src_txt = " · ".join(f"**{n:,}** partidos de *{s}*" for s, n in src_counts.items())
    st.caption(f"Fuente de datos en este filtro: {src_txt}.")

tp_form = add_rolling_form(scope, window=window)
pred_data = build_prediction_dataset(tp_form)
pred_data = pred_data.dropna(subset=["total_corners"]).copy()

if len(pred_data) < 30:
    st.warning(f"Muestra chica con corners disponibles ({len(pred_data)} partidos) — resultados poco confiables.")

SAFE_FEATURE_BASES = (
    "win_pct_dynamic", "rank_dynamic", "points_before", "goal_diff_before", "games_before",
    "form_corners", "form_opp_corners_against",
)
feature_cols = [
    c for c in prediction_feature_cols(pred_data)
    if c.replace("home_", "").replace("away_", "") in SAFE_FEATURE_BASES
    or c in ("rank_gap", "win_pct_gap", "form_goals_gap")
]
feature_cols = [c for c in feature_cols if c in pred_data.columns]

train, test, test_season = temporal_split(pred_data)
st.caption(
    f"**{comp_choice}** · entrenando con {len(train)} partidos "
    f"({', '.join(sorted(pred_data[pred_data['season_name'] != test_season]['season_name'].unique()))}) "
    f"→ probando en **{test_season}** ({len(test)} partidos)."
)

if len(train) < 30 or len(test) < 10:
    st.warning(f"Muestra chica (train={len(train)}, test={len(test)}) — toma esto como exploratorio.")

model = fit_corners_model(train, feature_cols, n_components=n_comp)
pred = predict_corners(model, test)
metrics = evaluate_corners(pred, train_mean_corners=train["total_corners"].mean())

col1, col2, col3 = st.columns(3)
col1.metric("Error promedio (MAE)", f"{metrics['mae']:.2f} corners")
col2.metric("Promedio real (test)", f"{test['total_corners'].mean():.2f}")
col3.metric("Promedio real (train)", f"{train['total_corners'].mean():.2f}")

st.markdown("**Líneas Over/Under — accuracy del modelo vs. baseline ingenuo**")
st.caption(
    "El baseline ingenuo apuesta siempre al mismo lado (Over o Under) según el promedio histórico de "
    "corners, sin ver el partido — si el modelo no le gana, no está aportando nada sobre lo obvio."
)
per_line = metrics["per_line"].copy()
per_line["model_accuracy"] = (per_line["model_accuracy"] * 100).round(1)
per_line["naive_accuracy"] = (per_line["naive_accuracy"] * 100).round(1)
per_line["pct_over_real"] = (per_line["pct_over_real"] * 100).round(1)
per_line.columns = ["Accuracy modelo (%)", "Pick ingenuo", "Accuracy ingenuo (%)", "% partidos Over real"]
st.dataframe(per_line, width='stretch')

fig = px.bar(
    per_line.reset_index(), x="line", y=["Accuracy modelo (%)", "Accuracy ingenuo (%)"],
    barmode="group", title="Accuracy por línea: modelo vs. baseline ingenuo",
)
fig.update_layout(yaxis_title="Accuracy (%)", xaxis_title="Línea de corners")
st.plotly_chart(fig, width='stretch')

st.markdown("**Predicción por partido (temporada de prueba)**")
line_choice = st.selectbox("Línea a mostrar", CORNER_LINES, index=1)
show_cols = ["match_date", "home_team", "away_team", "actual_home_corners", "actual_away_corners", "total_corners",
             "corners_esperados", f"p_over_{line_choice}", f"pick_{line_choice}", f"acierto_{line_choice}"]
show_cols += [c for c in ["data_source"] if c in pred.columns]
show = pred[show_cols].copy()
show["corners_esperados"] = show["corners_esperados"].round(2)
show[f"p_over_{line_choice}"] = (show[f"p_over_{line_choice}"] * 100).round(1)
show = show.sort_values(f"p_over_{line_choice}", ascending=False)
st.dataframe(show, width='stretch', height=450)

aciertos = pred[f"acierto_{line_choice}"].sum()
st.caption(f"Línea {line_choice}: {aciertos} de {len(pred)} partidos acertados ({aciertos/len(pred)*100:.1f}%).")
