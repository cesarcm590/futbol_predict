import numpy as np
import plotly.express as px
import streamlit as st

from src.match_prediction import prediction_feature_cols, temporal_split
from src.team_database import load_team_database, to_team_perspective
from src.team_form import add_rolling_form, build_prediction_dataset
from src.totals_model import evaluate_totals, fit_totals_model, predict_totals

st.set_page_config(page_title="NBA — Puntos totales", layout="wide")

st.title("NBA — Puntos totales por partido")
st.caption(
    "Mismo motor que el de corners de fútbol (Random Forest de regresión sobre componentes "
    "principales), aplicado a puntos totales (local + visita) de un partido de NBA. Las líneas "
    "se calculan por cuantiles de la propia base de entrenamiento — el ritmo de anotación de la "
    "NBA cambió mucho entre 2010 y hoy, así que una línea fija no tendría sentido en todas las "
    "épocas."
)
st.warning("Ejercicio analítico, no una recomendación de apuesta.")


@st.cache_data(show_spinner="Cargando base de equipos...")
def _load():
    wide = load_team_database()
    tp = to_team_perspective(wide)
    return tp


if st.button("Refrescar base de datos"):
    _load.clear()

tp = _load()
nba = tp[tp["competition_name"] == "NBA"].copy()
if nba.empty:
    st.info("Todavía no hay datos de NBA sincronizados. Corre `sync_nba()` (src/team_database.py) para cargarlos.")
    st.stop()

if "data_source" in nba.columns:
    src_counts = (nba["data_source"].value_counts() / 2).round(0).astype(int)
    src_txt = " · ".join(f"**{n:,}** partidos de *{s}*" for s, n in src_counts.items())
    st.caption(f"Fuente de datos: {src_txt}.")

with st.sidebar:
    st.header("Filtros")
    window = st.slider("Partidos de forma reciente (rolling)", 3, 15, 10)
    n_comp = st.slider("Componentes PCA", 2, 15, 8)

form = add_rolling_form(nba, window=window)
pred_data = build_prediction_dataset(form)
pred_data["total_pts"] = pred_data["home_score"] + pred_data["away_score"]

feature_cols = prediction_feature_cols(pred_data)
train, test, test_season = temporal_split(pred_data)
st.caption(
    f"Entrenando con {len(train)} partidos "
    f"({sorted(pred_data[pred_data['season_name'] != test_season]['season_name'].unique())[0]}"
    f"–{sorted(pred_data[pred_data['season_name'] != test_season]['season_name'].unique())[-1]}) "
    f"→ probando en **{test_season}** ({len(test)} partidos, la temporada más reciente disponible)."
)

quantiles = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
lines = sorted({round(train["total_pts"].quantile(q)) - 0.5 for q in quantiles})

model = fit_totals_model(train, feature_cols, target_col="total_pts", n_components=n_comp)
pred = predict_totals(model, test, lines=lines)
metrics = evaluate_totals(pred, target_col="total_pts", train_mean=train["total_pts"].mean(), lines=lines)

col1, col2, col3 = st.columns(3)
col1.metric("Error promedio (MAE)", f"{metrics['mae']:.1f} pts")
col2.metric("Promedio real (test)", f"{test['total_pts'].mean():.1f}")
col3.metric("Promedio real (train)", f"{train['total_pts'].mean():.1f}")
st.caption(
    "Si el promedio de train y test difiere bastante, es la NBA anotando más (o menos) ahora que "
    "en el histórico — el baseline ingenuo usa el promedio de train, así que un modelo útil debe "
    "ganarle precisamente en esa diferencia de época."
)

st.markdown("**Líneas Over/Under — accuracy del modelo vs. baseline ingenuo**")
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
fig.update_layout(yaxis_title="Accuracy (%)", xaxis_title="Línea de puntos totales")
st.plotly_chart(fig, width='stretch')

st.markdown("**Predicción por partido (temporada de prueba)**")
line_choice = st.selectbox("Línea a mostrar", lines, index=len(lines) // 2)
show_cols = ["match_date", "home_team", "away_team", "total_pts", "esperado",
             f"p_over_{line_choice}", f"pick_{line_choice}", f"acierto_{line_choice}"]
show_cols += [c for c in ["data_source"] if c in pred.columns]
show = pred[show_cols].rename(columns={"esperado": "pts_esperados", "total_pts": "pts_reales"}).copy()
show["pts_esperados"] = show["pts_esperados"].round(1)
show[f"p_over_{line_choice}"] = (show[f"p_over_{line_choice}"] * 100).round(1)
show = show.sort_values(f"p_over_{line_choice}", ascending=False)
st.dataframe(show, width='stretch', height=450)

aciertos = pred[f"acierto_{line_choice}"].sum()
st.caption(f"Línea {line_choice}: {aciertos} de {len(pred)} partidos acertados ({aciertos/len(pred)*100:.1f}%).")
