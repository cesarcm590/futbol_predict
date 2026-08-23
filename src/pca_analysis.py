"""PCA + k-means sobre la tabla de metricas por jugador-temporada.

Misma logica que la metodologia de basquetball (Estancia, Carrillo Martinez
2026): estandarizar variables deportivas, reducir dimension con PCA para ver
que agrupa la variabilidad, y usar k-means sobre esa estructura para
proponer "estilos de juego" / roles de jugador interpretables a partir de
los promedios por cluster (las etiquetas no las impone el algoritmo).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_ID_COLS = ["player", "team", "position", "role", "minutes_played", "matches_played"]


def default_feature_cols(feat_df: pd.DataFrame) -> list[str]:
    cols = [c for c in feat_df.columns if c.endswith("_p90")]
    if "pass_pct" in feat_df.columns:
        cols.append("pass_pct")
    return cols


def run_pca(feat_df: pd.DataFrame, feature_cols: list[str] | None = None, n_components: int = 8,
            id_cols: list[str] | None = None):
    feature_cols = feature_cols or default_feature_cols(feat_df)
    data = feat_df[feature_cols].fillna(0.0)

    n_components = min(n_components, len(feature_cols), len(data))
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    pc_names = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=pc_names, index=feat_df.index)
    id_cols = [c for c in (id_cols or DEFAULT_ID_COLS) if c in feat_df.columns]
    scores_df = pd.concat([feat_df[id_cols].reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

    loadings_df = pd.DataFrame(pca.components_.T, index=feature_cols, columns=pc_names)

    explained = pd.DataFrame({
        "component": pc_names,
        "explained_var_pct": pca.explained_variance_ratio_ * 100,
        "cumulative_pct": np.cumsum(pca.explained_variance_ratio_) * 100,
    })

    return {
        "scores": scores_df,
        "loadings": loadings_df,
        "explained": explained,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "standardized": X,
    }


def run_kmeans(feat_df: pd.DataFrame, feature_cols: list[str] | None = None, k: int = 4, random_state: int = 42):
    feature_cols = feature_cols or default_feature_cols(feat_df)
    data = feat_df[feature_cols].fillna(0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    k = min(k, len(data)) if len(data) > 0 else 0
    if k < 1:
        return feat_df.assign(cluster=np.nan, cluster_label="Sin datos")

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    result = feat_df.copy()
    result["cluster"] = labels

    # Etiqueta automatica por cluster: las 2 variables con mayor z-score promedio
    z = pd.DataFrame(X, columns=feature_cols, index=feat_df.index)
    z["cluster"] = labels
    cluster_profile = z.groupby("cluster")[feature_cols].mean()

    labels_map = {}
    for cl, row in cluster_profile.iterrows():
        top2 = row.sort_values(ascending=False).head(2).index.tolist()
        pretty = " + ".join(c.replace("_p90", "").replace("_", " ") for c in top2)
        labels_map[cl] = pretty.title()

    result["cluster_label"] = result["cluster"].map(labels_map)
    profile_named = cluster_profile.rename(index=labels_map)
    return result, profile_named
