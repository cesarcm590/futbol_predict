import numpy as np
import plotly.express as px
import streamlit as st

from src.match_prediction import prediction_feature_cols, temporal_split
from src.pca_analysis import run_kmeans
from src.team_database import TEAM_ID_COLS, load_team_database, team_feature_cols, to_team_perspective
from src.team_form import add_rolling_form, build_prediction_dataset
from src.totals_model import evaluate_totals, fit_totals_model, predict_totals

st.set_page_config(page_title="NBA — Puntos totales", layout="wide")

st.title("NBA — Rango de puntos totales por partido")
st.caption(
    "No busca acertarle al ganador, sino al RANGO de puntos totales (local + visita) que va a "
    "tener el partido, y cómo ese rango cambia según el estilo de juego de cada equipo (k-means "
    "sobre las mismas métricas de PCA). Random Forest de regresión sobre componentes principales: "
    "cada árbol da su propia estimación, y los percentiles de esas ~300 estimaciones arman el "
    "rango — no un intervalo estadístico formal, pero mucho más útil que un solo número cuando lo "
    "que importa es dónde probablemente va a caer el total."
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
    st.info("Todavía no hay datos de NBA sincronizados.")
    st.stop()

if "data_source" in nba.columns:
    src_counts = (nba["data_source"].value_counts() / 2).round(0).astype(int)
    src_txt = " · ".join(f"**{n:,}** partidos de *{s}*" for s, n in src_counts.items())
    st.caption(f"Fuente de datos: {src_txt}.")

with st.sidebar:
    st.header("Filtros")
    window = st.slider("Partidos de forma reciente (rolling)", 3, 15, 10)
    n_comp = st.slider("Componentes PCA", 2, 15, 8)
    k = st.slider("Estilos de equipo (k)", 2, 8, 4)

form = add_rolling_form(nba, window=window)
pred_data = build_prediction_dataset(form)
pred_data["total_pts"] = pred_data["home_score"] + pred_data["away_score"]

feature_cols = prediction_feature_cols(pred_data)
train, test, test_season = temporal_split(pred_data)
st.caption(
    f"Entrenando con {len(train)} partidos "
    f"({sorted(pred_data[pred_data['season_name'] != test_season]['season_name'].unique())[0]}"
    f"–{sorted(pred_data[pred_data['season_name'] != test_season]['season_name'].unique())[-1]}) "
    f"→ probando en **{test_season}** ({len(test)} partidos, la temporada más reciente y completa disponible)."
)

lines = sorted({round(train["total_pts"].quantile(q)) - 0.5 for q in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]})

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

tab_estilos, tab_lineas, tab_partidos = st.tabs(["Rango por estilo de equipo", "Líneas Over/Under", "Predicción por partido"])

with tab_estilos:
    st.subheader("¿Qué tan alto anota cada estilo de equipo?")
    st.caption(
        "k-means sobre las mismas métricas del PCA agrupa cada partido-equipo por parecido "
        "estadístico (ritmo, tiro de 3, rebote, etc.) — el rango de abajo es el P25–P75 de puntos "
        "totales REALES en los partidos de cada estilo (no la predicción del modelo, la data cruda)."
    )
    feature_cols_style = team_feature_cols(nba)
    clustered, profile = run_kmeans(nba, feature_cols=feature_cols_style, k=k)
    # total del partido = puntos propios + puntos del rival, ambos ya estan en la vista por equipo
    clustered["total_pts"] = clustered["goals_for"] + clustered["goals_against"]

    style_stats = clustered.groupby("cluster_label")["total_pts"].agg(
        partidos="count", promedio="mean", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75),
    ).round(1).sort_values("promedio", ascending=False)
    st.dataframe(style_stats, width='stretch')

    fig = px.bar(
        style_stats.reset_index(), x="cluster_label", y="promedio",
        error_y=style_stats["p75"] - style_stats["promedio"], error_y_minus=style_stats["promedio"] - style_stats["p25"],
        title="Puntos totales promedio por estilo (barras = rango P25–P75)",
    )
    fig.update_layout(xaxis_title="Estilo", yaxis_title="Puntos totales del partido")
    st.plotly_chart(fig, width='stretch')

    st.markdown("**Perfil de cada estilo (z-score sobre el promedio de la liga)**")
    st.dataframe(profile.round(2), width='stretch')

with tab_lineas:
    st.markdown("**Líneas Over/Under — accuracy del modelo vs. baseline ingenuo**")
    per_line = metrics["per_line"].copy()
    per_line["model_accuracy"] = (per_line["model_accuracy"] * 100).round(1)
    per_line["naive_accuracy"] = (per_line["naive_accuracy"] * 100).round(1)
    per_line["pct_over_real"] = (per_line["pct_over_real"] * 100).round(1)
    per_line.columns = ["Accuracy modelo (%)", "Pick ingenuo", "Accuracy ingenuo (%)", "% partidos Over real"]
    st.dataframe(per_line, width='stretch')

    fig2 = px.bar(
        per_line.reset_index(), x="line", y=["Accuracy modelo (%)", "Accuracy ingenuo (%)"],
        barmode="group", title="Accuracy por línea: modelo vs. baseline ingenuo",
    )
    fig2.update_layout(yaxis_title="Accuracy (%)", xaxis_title="Línea de puntos totales")
    st.plotly_chart(fig2, width='stretch')

with tab_partidos:
    st.markdown("**Predicción por partido (temporada de prueba) — rango P10–P90**")
    show = pred[["match_date", "home_team", "away_team", "total_pts", "p10", "p25", "esperado", "p75", "p90"]].copy()
    for c in ["p10", "p25", "esperado", "p75", "p90"]:
        show[c] = show[c].round(1)
    show = show.rename(columns={"total_pts": "pts_reales", "esperado": "esperado_p50"})
    show = show.sort_values("match_date", ascending=False)
    st.dataframe(show, width='stretch', height=500)
    dentro_rango = ((pred["total_pts"] >= pred["p10"]) & (pred["total_pts"] <= pred["p90"])).mean()
    st.caption(f"El total real cayó dentro del rango P10–P90 predicho en el {dentro_rango*100:.1f}% de los partidos.")
