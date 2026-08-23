import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import get_competitions, get_matches, get_events
from src.features import build_player_features
from src.heatmaps import player_heatmap, EVENT_GROUPS
from src.pca_analysis import run_pca, run_kmeans

st.set_page_config(page_title="Rendimiento de jugadores — Fútbol", layout="wide")

ROLE_ORDER = ["Portero", "Defensa", "Mediocampo", "Delantero", "Otro"]
ROLE_COLORS = {
    "Portero": "#f2c14e",
    "Defensa": "#4e9af2",
    "Mediocampo": "#4ef2a3",
    "Delantero": "#f24e6a",
    "Otro": "#9a9a9a",
}


@st.cache_data(show_spinner=False)
def load_competitions_df():
    return get_competitions()


@st.cache_data(show_spinner=False)
def load_events_for_season(competition_id: int, season_id: int):
    matches = get_matches(competition_id, season_id)
    match_ids = matches["match_id"].tolist()

    progress = st.progress(0.0, text=f"Descargando eventos: 0/{len(match_ids)} partidos")
    frames = []
    for i, mid in enumerate(match_ids, start=1):
        try:
            frames.append(get_events(int(mid)))
        except Exception:
            pass
        progress.progress(i / len(match_ids), text=f"Descargando eventos: {i}/{len(match_ids)} partidos")
    progress.empty()

    events = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return events, matches


@st.cache_data(show_spinner="Construyendo tabla de metricas por jugador...")
def load_features(competition_id: int, season_id: int, min_minutes: float):
    events, _ = load_events_for_season(competition_id, season_id)
    return build_player_features(events, min_minutes=min_minutes)


st.title("Rendimiento de jugadores — StatsBomb Open Data")
st.caption(
    "Heatmaps + PCA + estilos de juego (k-means) por jugador-temporada, "
    "misma metodologia que el analisis de estilos de juego en basquetball, "
    "portada a futbol con datos abiertos de StatsBomb."
)

comps = load_competitions_df()
comps = comps.copy()
comps["label"] = (
    comps["competition_name"] + " " + comps["season_name"]
    + " ("+ comps["country_name"] + ", " + comps["competition_gender"] + ")"
)
comps = comps.sort_values(["competition_name", "season_name"])

options = comps["label"].tolist()
default_idx = next((i for i, l in enumerate(options) if "Indian Super league" in l), 0)

with st.sidebar:
    st.header("Filtros")
    comp_label = st.selectbox("Competición / temporada", options, index=default_idx)
    row = comps[comps["label"] == comp_label].iloc[0]
    competition_id, season_id = int(row["competition_id"]), int(row["season_id"])

    min_minutes = st.slider("Minutos mínimos jugados en la temporada", 0, 1800, 270, step=30)

    load_clicked = st.button("Cargar / actualizar temporada", type="primary")

state_key = (competition_id, season_id)
if load_clicked:
    st.session_state["loaded_season"] = state_key

loaded_season = st.session_state.get("loaded_season")

if loaded_season != state_key:
    st.info(
        "Selecciona competición/temporada y minutos mínimos en el panel izquierdo, "
        "luego presiona **Cargar / actualizar temporada**.\n\n"
        "La primera carga de una temporada descarga los eventos de todos sus partidos "
        "desde StatsBomb Open Data (puede tardar); las siguientes veces queda en cache local."
    )
    st.stop()

feat = load_features(competition_id, season_id, min_minutes)

if feat.empty:
    st.warning("No hay suficientes datos para esta combinación de filtros. Prueba bajando el mínimo de minutos.")
    st.stop()

with st.sidebar:
    teams = ["Todos"] + sorted(feat["team"].dropna().unique().tolist())
    team_filter = st.selectbox("Equipo", teams)

feat_view = feat if team_filter == "Todos" else feat[feat["team"] == team_filter]

tab_heat, tab_table, tab_pca, tab_styles = st.tabs(
    ["Heatmap", "Tabla de métricas", "PCA", "Estilos de juego"]
)

with tab_heat:
    st.subheader("Heatmap de jugador (ubicación de eventos)")
    st.caption(
        "Zonas más claras/cálidas = donde el jugador tocó el balón con más frecuencia "
        "en la temporada. Los puntos blancos son cada evento individual."
    )
    player = st.selectbox("Jugador", feat_view.sort_values("minutes_played", ascending=False)["player"].tolist())
    event_group = st.selectbox("Tipo de evento", list(EVENT_GROUPS.keys()))

    events, _ = load_events_for_season(competition_id, season_id)
    fig = player_heatmap(events, player, event_group)
    st.pyplot(fig)
    st.caption(
        "Heatmap construido con la ubicación (x,y) de los eventos del jugador. "
        "No es tracking posicional continuo (StatsBomb Open Data no lo incluye)."
    )

with tab_table:
    st.subheader("Métricas por 90 minutos")
    st.caption(
        "Todo normalizado 'por 90 minutos' (`_p90`) para poder comparar jugadores con "
        "distintos minutos jugados en igualdad de condiciones."
    )
    display_cols = ["player", "team", "position", "minutes_played", "matches_played", "pass_pct"] + [
        c for c in feat_view.columns if c.endswith("_p90")
    ]
    st.dataframe(
        feat_view[display_cols].sort_values("minutes_played", ascending=False),
        width='stretch',
        height=600,
    )

with tab_pca:
    st.subheader("Análisis de componentes principales (PCA)")
    n_comp = st.slider("Número de componentes", 2, min(10, len(feat_view) - 1 if len(feat_view) > 2 else 2), 5)
    pca_res = run_pca(feat_view, n_components=n_comp)

    st.caption(
        "Cada punto es un jugador de la temporada seleccionada, resumido en 2 ejes que "
        "capturan la mayor variabilidad posible de todas sus métricas por-90 a la vez. "
        "Jugadores cercanos entre sí tuvieron temporadas estadísticamente parecidas."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        scores = pca_res["scores"]
        fig = px.scatter(
            scores, x="PC1", y="PC2", color="role", hover_name="player",
            hover_data=["team", "position", "minutes_played"], height=550,
            title="Jugadores en el plano PC1–PC2",
            category_orders={"role": ROLE_ORDER}, color_discrete_map=ROLE_COLORS,
        )
        fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")))
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("**Varianza explicada**")
        st.dataframe(pca_res["explained"].round(2), width='stretch', hide_index=True)
        st.caption(
            "`cumulative_pct` = qué % de toda la información original ya capturan "
            "los primeros N componentes. Con 2-3 componentes normalmente se resume "
            "gran parte del rendimiento."
        )

    st.markdown("**Loadings — qué variables definen cada componente**")
    st.caption(
        "Cada barra es una métrica. Mientras más larga (positiva o negativa), más peso "
        "tiene esa métrica en este componente. Barras del mismo lado (misma dirección) "
        "significan que esas métricas suben y bajan juntas en los jugadores."
    )
    pc_choice = st.selectbox("Componente", pca_res["loadings"].columns.tolist())
    load_sorted = pca_res["loadings"][pc_choice].sort_values()
    fig_load = px.bar(
        load_sorted, orientation="h",
        title=f"Loadings de {pc_choice} (qué tanto pesa cada métrica)",
        height=500, color=load_sorted.values, color_continuous_scale="RdBu_r",
    )
    fig_load.update_layout(showlegend=False, yaxis_title="", xaxis_title="Loading", coloraxis_showscale=False)
    st.plotly_chart(fig_load, width='stretch')

with tab_styles:
    st.subheader("Estilos de juego / roles de jugador (k-means)")
    st.caption(
        "k-means agrupa a los jugadores por parecido estadístico en sus métricas por-90 "
        "(estandarizadas), sin usar su posición oficial. La etiqueta de cada grupo se "
        "arma automáticamente con las 2 métricas donde ese grupo más se aleja del promedio."
    )
    k = st.slider("Número de clusters (k)", 2, 8, 4)
    clustered, profile = run_kmeans(feat_view, k=k)

    col1, col2 = st.columns([2, 1])
    with col1:
        pca_for_plot = run_pca(feat_view, n_components=2)["scores"][["player", "PC1", "PC2"]]
        plot_df = clustered.merge(pca_for_plot, on="player", how="left")
        fig_c = px.scatter(
            plot_df, x="PC1", y="PC2", color="cluster_label", hover_name="player",
            hover_data=["team", "position", "minutes_played"], height=550,
            title="Clusters proyectados en el plano PC1–PC2",
        )
        fig_c.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")))
        st.plotly_chart(fig_c, width='stretch')
    with col2:
        st.markdown("**Jugadores por estilo**")
        st.dataframe(
            clustered["cluster_label"].value_counts().rename_axis("Estilo").reset_index(name="n"),
            width='stretch', hide_index=True,
        )

    st.markdown("**Perfil promedio (z-score) por estilo**")
    st.caption(
        "Valores arriba de 0 = ese estilo está por encima del promedio de la liga en esa "
        "métrica; abajo de 0 = por debajo. Así se lee qué hace distinto a cada estilo."
    )
    st.dataframe(profile.round(2), width='stretch')

    st.markdown("**Jugadores y su estilo asignado**")
    st.dataframe(
        clustered[["player", "team", "position", "minutes_played", "cluster_label"]]
        .sort_values("minutes_played", ascending=False),
        width='stretch', height=400,
    )
