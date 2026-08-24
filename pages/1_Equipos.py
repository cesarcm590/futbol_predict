import pandas as pd
import plotly.express as px
import streamlit as st

from src.pca_analysis import run_pca, run_kmeans
from src.team_database import TEAM_ID_COLS, load_team_database, team_feature_cols, to_team_perspective

st.set_page_config(page_title="Rendimiento de equipos — Fútbol", layout="wide")

st.title("Rendimiento de equipos — StatsBomb Open Data")
st.caption(
    "Base estandarizada de partidos (una fila por partido, prefijos home_/away_) con "
    "ranking dinámico previo a cada partido — misma lógica que la metodología de "
    "basquetball, a nivel equipo. Se va llenando en segundo plano conforme se "
    "sincronizan más ligas; presiona 'Refrescar' para ver los datos más recientes."
)


@st.cache_data(show_spinner="Cargando base de equipos...")
def _load():
    wide = load_team_database()
    tp = to_team_perspective(wide)
    return wide, tp


if st.button("Refrescar base de datos"):
    _load.clear()

wide, tp = _load()

if tp.empty:
    st.info("Todavía no hay datos en la base de equipos. Se están sincronizando en segundo plano — vuelve en unos minutos y presiona Refrescar.")
    st.stop()

st.caption(f"Base actual: **{wide['match_id'].nunique()}** partidos · **{tp['competition_name'].nunique()}** competencias · **{tp['team'].nunique()}** equipos.")

with st.sidebar:
    st.header("Filtros")
    comps = sorted(tp["competition_name"].dropna().unique().tolist())
    comp_choice = st.selectbox("Competición", comps)

    seasons = sorted(tp[tp["competition_name"] == comp_choice]["season_name"].dropna().unique().tolist())
    season_choice = st.multiselect("Temporada(s)", seasons, default=seasons)

scope = tp[(tp["competition_name"] == comp_choice) & (tp["season_name"].isin(season_choice))]

with st.sidebar:
    teams = sorted(scope["team"].dropna().unique().tolist())
    team_choice = st.selectbox("Equipo", teams) if teams else None

if not team_choice:
    st.warning("No hay equipos para este filtro.")
    st.stop()

st.caption(f"{len(scope)} observaciones equipo-partido en el filtro actual ({scope['team'].nunique()} equipos).")
if "data_source" in scope.columns and not scope.empty:
    src_counts = (scope["data_source"].value_counts() / 2).round(0).astype(int)
    src_txt = " · ".join(f"**{n:,}** partidos de *{s}*" for s, n in src_counts.items())
    st.caption(f"Fuente de datos: {src_txt}.")
if len(scope) < 20:
    st.warning(
        "Muy pocos partidos en este filtro — StatsBomb Open Data no siempre libera la "
        "temporada completa. El PCA / k-means necesitan más observaciones para ser "
        "confiables; trata con 'Todas las temporadas' de esta competición."
    )

team_df = scope[scope["team"] == team_choice].sort_values("match_date")
feature_cols = team_feature_cols(scope)

tab_evo, tab_pca, tab_styles = st.tabs(["Evolución en la temporada", "PCA", "Estilos de equipo"])

with tab_evo:
    st.subheader(f"{team_choice} — evolución partido a partido")
    st.caption(
        "`rank_dynamic` = posición en la tabla justo ANTES de jugar ese partido "
        "(1 = líder). Se calcula solo con resultados previos, nunca con el propio partido."
    )
    if "rank_dynamic" in team_df.columns and team_df["rank_dynamic"].notna().any():
        fig = px.line(
            team_df, x="match_date", y="rank_dynamic", markers=True,
            hover_data=["opponent", "result", "goals_for", "goals_against", "season_name"],
            title=f"Posición en la tabla antes de cada partido — {team_choice}",
        )
        fig.update_yaxes(autorange="reversed", title="Posición en la tabla (1 = líder)")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Esta competición no tiene ranking dinámico calculado (formato de grupos/eliminación, no de liga).")

    st.markdown("**Resultados**")
    show_cols = [c for c in [
        "match_date", "season_name", "opponent", "is_home", "goals_for", "goals_against",
        "result", "xg_total", "possession_pct", "rank_dynamic",
    ] if c in team_df.columns]
    st.dataframe(team_df[show_cols], width='stretch', height=400)

with tab_pca:
    st.subheader("PCA — estructura de rendimiento en la competición")
    st.caption(
        f"Cada punto es UN partido de UN equipo (toda la competición/temporada filtrada, "
        f"no solo {team_choice}) — sirve para ver qué tan parecidos o distintos fueron los rendimientos."
    )
    n_max = max(2, min(10, len(feature_cols), len(scope) - 1))
    n_comp = st.slider("Número de componentes", 2, n_max, min(5, n_max), key="team_pca_n")
    pca_res = run_pca(scope, feature_cols=feature_cols, n_components=n_comp, id_cols=TEAM_ID_COLS)

    col1, col2 = st.columns([2, 1])
    with col1:
        scores = pca_res["scores"].copy()
        scores["Equipo"] = scores["team"].where(scores["team"] == team_choice, "Otros equipos")
        fig = px.scatter(
            scores, x="PC1", y="PC2", color="Equipo", hover_name="team",
            hover_data=["opponent", "match_date", "result"], height=550,
            title="Partidos en el plano PC1–PC2",
            color_discrete_map={team_choice: "#f24e6a", "Otros equipos": "#4e9af2"},
        )
        fig.update_traces(marker=dict(size=8, opacity=0.75))
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("**Varianza explicada**")
        st.dataframe(pca_res["explained"].round(2), width='stretch', hide_index=True)

    st.markdown("**Loadings — qué variables definen cada componente**")
    pc_choice = st.selectbox("Componente", pca_res["loadings"].columns.tolist(), key="team_pc_choice")
    load_sorted = pca_res["loadings"][pc_choice].sort_values()
    fig_load = px.bar(
        load_sorted, orientation="h", height=450,
        title=f"Loadings de {pc_choice}",
        color=load_sorted.values, color_continuous_scale="RdBu_r",
    )
    fig_load.update_layout(showlegend=False, yaxis_title="", xaxis_title="Loading", coloraxis_showscale=False)
    st.plotly_chart(fig_load, width='stretch')

with tab_styles:
    st.subheader("Estilos de equipo (k-means)")
    st.caption(
        "Agrupa cada partido-equipo de la competición/temporada filtrada por parecido "
        "estadístico (incluyendo el ranking dinámico) — misma lógica que los estilos de "
        "juego (Ofensivo_eficiente, Perimetral, ...) de la metodología de basquetball."
    )
    k = st.slider("Número de clusters (k)", 2, 8, 4, key="team_k")
    clustered, profile = run_kmeans(scope, feature_cols=feature_cols, k=k)

    st.markdown(f"**Estilo(s) de {team_choice} por partido**")
    st.dataframe(
        clustered[clustered["team"] == team_choice][
            [c for c in ["match_date", "opponent", "result", "cluster_label"] if c in clustered.columns]
        ],
        width='stretch',
    )

    st.markdown("**Perfil promedio (z-score) por estilo**")
    st.caption("Arriba de 0 = por encima del promedio de la competición filtrada en esa métrica; abajo de 0 = por debajo.")
    st.dataframe(profile.round(2), width='stretch')
