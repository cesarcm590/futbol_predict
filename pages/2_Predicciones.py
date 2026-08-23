import plotly.express as px
import streamlit as st

from src.match_prediction import (
    evaluate, fit_hierarchical_model, fit_result_model, predict_hierarchical,
    predict_with_confidence, prediction_feature_cols, temporal_split,
)
from src.team_database import load_team_database, to_team_perspective
from src.team_form import add_rolling_form, build_prediction_dataset

st.set_page_config(page_title="Predicción de resultados — Fútbol", layout="wide")

st.title("Predicción de resultados")
st.caption(
    "Clasificador (Random Forest sobre componentes principales) que predice el "
    "resultado (Local / Empate / Visita) usando SOLO información conocida antes del "
    "partido: ranking dinámico en la tabla + forma reciente + qué tan parejos están "
    "los dos equipos. Se entrena con temporadas pasadas y se prueba en la temporada "
    "más reciente disponible — igual que el modelo 'PC1-PC8 previos + estilos' de la "
    "metodología de basquetball, aplicado aquí para clasificar resultados."
)
st.warning(
    "Los resultados de fútbol son mucho más ruidosos que el total de puntos de un "
    "partido de basquetball — no esperes accuracy tipo 80%+. Esto ya es útil si le "
    "gana consistentemente a 'siempre predecir el más probable'."
)


@st.cache_data(show_spinner="Cargando base de equipos...")
def _load():
    wide = load_team_database()
    tp = to_team_perspective(wide)
    return tp


if st.button("Refrescar base de datos"):
    _load.clear()

tp = _load()
if tp.empty:
    st.info("Todavía no hay datos. Ve a la página 'Equipos' o espera a que termine la sincronización.")
    st.stop()

season_counts = tp.groupby("competition_name")["season_name"].nunique()
eligible = sorted(season_counts[season_counts >= 2].index.tolist())

if not eligible:
    st.info(
        "Ninguna competición tiene todavía 2+ temporadas sincronizadas (se necesita al "
        "menos una para entrenar y otra para probar). Espera a que avance la "
        "sincronización en segundo plano y refresca."
    )
    st.stop()

with st.sidebar:
    st.header("Filtros")
    comp_choice = st.selectbox("Competición", eligible)
    window = st.slider("Partidos de forma reciente (rolling)", 3, 10, 5)
    n_comp = st.slider("Componentes PCA para el modelo", 2, 15, 8)
    use_hierarchical = st.radio(
        "Modelo",
        ["Jerárquico (empate primero)", "Conjunto (3 clases a la vez)"],
        help="El jerárquico separa 'empate sí/no' de 'quién gana' en dos pasos — detecta muchos más empates reales.",
    ) == "Jerárquico (empate primero)"

scope = tp[tp["competition_name"] == comp_choice]
tp_form = add_rolling_form(scope, window=window)
pred_data = build_prediction_dataset(tp_form)
feature_cols = prediction_feature_cols(pred_data)

train, test, test_season = temporal_split(pred_data)
st.caption(
    f"**{comp_choice}** · entrenando con {len(train)} partidos "
    f"({', '.join(sorted(pred_data[pred_data['season_name'] != test_season]['season_name'].unique()))}) "
    f"→ probando en **{test_season}** ({len(test)} partidos, la temporada más reciente disponible)."
)

if len(train) < 20 or len(test) < 5:
    st.warning(
        f"Muestra chica (train={len(train)}, test={len(test)}) — los resultados van a "
        "ser ruidosos. Esto va a mejorar conforme se sincronicen más temporadas de esta liga."
    )

if use_hierarchical:
    model = fit_hierarchical_model(train, feature_cols, n_components=n_comp)
    pred = predict_hierarchical(model, test)
else:
    model = fit_result_model(train, feature_cols, n_components=n_comp)
    pred = predict_with_confidence(model, test)

proba_cols = [c for c in pred.columns if c.startswith("proba_")]
metrics, cm, per_class = evaluate(pred, proba_cols=proba_cols)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
col2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']*100:.1f}%")
if "log_loss" in metrics:
    col3.metric("Log-loss (menor = mejor calibrado)", f"{metrics['log_loss']:.3f}")

st.markdown("**Precisión y recall por resultado**")
st.caption(
    "`recall` de Empate = de todos los empates reales, cuántos detectó el modelo. "
    "Es la métrica clave para saber si el modelo de verdad se anima a predecir empates "
    "o los ignora por completo."
)
st.dataframe(per_class.round(3), width='stretch')

tab_preds, tab_cm = st.tabs(["Predicción por partido", "Matriz de confusión"])

with tab_preds:
    st.caption(
        "`confianza` = probabilidad que el modelo le dio a su propia predicción para "
        "ESE partido — así lees qué tan seguro estaba caso por caso, no solo en promedio."
    )
    show = pred[["match_date", "home_team", "away_team", "home_score", "away_score",
                 "result", "prediccion", "confianza"] + proba_cols].copy()
    show["confianza"] = (show["confianza"] * 100).round(1)
    for c in proba_cols:
        show[c] = (show[c] * 100).round(1)
    show = show.sort_values("confianza", ascending=False)
    st.dataframe(show, width='stretch', height=500)

    aciertos = pred["acierto"].sum()
    st.caption(f"{aciertos} de {len(pred)} partidos acertados ({aciertos/len(pred)*100:.1f}%).")

with tab_cm:
    st.caption("Filas = resultado real, columnas = lo que predijo el modelo. La diagonal son los aciertos.")
    st.dataframe(cm, width='stretch')
