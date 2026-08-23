"""Prediccion de resultado (H/D/A) a partir de componentes principales de
variables pre-partido, con separacion temporal (entrena en temporadas
pasadas, prueba en la ultima temporada disponible). Misma idea que el
modelo "PC1-PC8 previos + estilos" de la metodologia de basquetball, pero
usado para clasificar el resultado en vez de regresion de puntos.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


CLOSENESS_COLS = ["rank_gap", "win_pct_gap", "form_goals_gap"]


def prediction_feature_cols(df: pd.DataFrame) -> list[str]:
    """Columnas home_/away_ + cercania, descartando las que no tienen señal
    real para esta competición (todo NaN o varianza cero) -- pasa cuando una
    fuente de datos no trae ciertas estadísticas, como Liga MX sin eventos."""
    exclude = {"home_team", "away_team", "home_score", "away_score"}
    prefixed = [c for c in df.columns if (c.startswith("home_") or c.startswith("away_")) and c not in exclude]
    closeness = [c for c in CLOSENESS_COLS if c in df.columns]
    cols = prefixed + closeness
    return [c for c in cols if df[c].notna().any() and df[c].std(skipna=True) > 0]


def temporal_split(df: pd.DataFrame, test_season: str | None = None):
    """train = todas las temporadas salvo `test_season`; test = `test_season`
    (por defecto, la mas reciente disponible para esta competicion)."""
    season_order = df.groupby("season_name")["match_date"].max().sort_values()
    if test_season is None:
        test_season = season_order.index[-1]
    train = df[df["season_name"] != test_season].copy()
    test = df[df["season_name"] == test_season].copy()
    return train, test, test_season


def fit_result_model(train: pd.DataFrame, feature_cols: list[str], n_components: int = 8, random_state: int = 42):
    X_train_raw = train[feature_cols].fillna(0.0)
    scaler = StandardScaler().fit(X_train_raw)
    X_train_std = scaler.transform(X_train_raw)

    n_components = max(1, min(n_components, len(feature_cols), len(X_train_std)))
    pca = PCA(n_components=n_components, random_state=random_state).fit(X_train_std)
    X_train_pca = pca.transform(X_train_std)

    clf = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    clf.fit(X_train_pca, train["result"])

    return {"scaler": scaler, "pca": pca, "clf": clf, "feature_cols": feature_cols}


def fit_hierarchical_model(train: pd.DataFrame, feature_cols: list[str], n_components: int = 8, random_state: int = 42):
    """Dos clasificadores binarios en cascada en vez de uno de 3 clases:

    1) empate vs no-empate
    2) (solo en los no-empate) local vs visita

    El empate es la clase mas dificil de separar en un modelo conjunto de 3
    clases porque casi nunca es la opcion con mayor probabilidad frente a
    'gana el local' o 'gana la visita'. Aislarlo en su propio clasificador
    binario le da una oportunidad real de aparecer como prediccion.
    """
    X_train_raw = train[feature_cols].fillna(0.0)
    scaler = StandardScaler().fit(X_train_raw)
    X_train_std = scaler.transform(X_train_raw)

    n_components = max(1, min(n_components, len(feature_cols), len(X_train_std)))
    pca = PCA(n_components=n_components, random_state=random_state).fit(X_train_std)
    X_train_pca = pca.transform(X_train_std)

    is_draw = (train["result"] == "D").astype(int).values
    draw_clf = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    draw_clf.fit(X_train_pca, is_draw)

    non_draw = train["result"] != "D"
    home_clf = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    home_clf.fit(X_train_pca[non_draw.values], (train.loc[non_draw, "result"] == "H").astype(int).values)

    return {"scaler": scaler, "pca": pca, "draw_clf": draw_clf, "home_clf": home_clf, "feature_cols": feature_cols}


def predict_hierarchical(model: dict, df: pd.DataFrame) -> pd.DataFrame:
    X_raw = df[model["feature_cols"]].fillna(0.0)
    X_std = model["scaler"].transform(X_raw)
    X_pca = model["pca"].transform(X_std)

    draw_classes = list(model["draw_clf"].classes_)
    p_draw = model["draw_clf"].predict_proba(X_pca)[:, draw_classes.index(1)]

    home_classes = list(model["home_clf"].classes_)
    p_home_given_not_draw = model["home_clf"].predict_proba(X_pca)[:, home_classes.index(1)]

    p_H = (1 - p_draw) * p_home_given_not_draw
    p_A = (1 - p_draw) * (1 - p_home_given_not_draw)
    p_D = p_draw

    keep = [c for c in ["match_id", "match_date", "home_team", "away_team", "home_score", "away_score", "result"] if c in df.columns]
    out = df[keep].reset_index(drop=True).copy()
    out["proba_A"] = p_A
    out["proba_D"] = p_D
    out["proba_H"] = p_H

    proba_mat = out[["proba_A", "proba_D", "proba_H"]].values
    labels = ["A", "D", "H"]
    out["prediccion"] = [labels[i] for i in proba_mat.argmax(axis=1)]
    out["confianza"] = proba_mat.max(axis=1)
    out["acierto"] = out["prediccion"] == out["result"]
    return out


def predict_with_confidence(model: dict, df: pd.DataFrame) -> pd.DataFrame:
    X_raw = df[model["feature_cols"]].fillna(0.0)
    X_std = model["scaler"].transform(X_raw)
    X_pca = model["pca"].transform(X_std)
    proba = model["clf"].predict_proba(X_pca)
    classes = model["clf"].classes_

    keep = [c for c in ["match_id", "match_date", "home_team", "away_team", "home_score", "away_score", "result"] if c in df.columns]
    out = df[keep].reset_index(drop=True).copy()
    for i, cls in enumerate(classes):
        out[f"proba_{cls}"] = proba[:, i]
    out["prediccion"] = classes[np.argmax(proba, axis=1)]
    out["confianza"] = proba.max(axis=1)
    out["acierto"] = out["prediccion"] == out["result"]
    return out


def evaluate(pred_df: pd.DataFrame, proba_cols: list[str] | None = None) -> tuple[dict, pd.DataFrame]:
    y_true = pred_df["result"]
    y_pred = pred_df["prediccion"]
    labels = sorted(y_true.unique().tolist())

    metrics = {
        "n_partidos": len(pred_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    if proba_cols:
        classes = [c.replace("proba_", "") for c in proba_cols]
        try:
            metrics["log_loss"] = log_loss(y_true, pred_df[proba_cols].values, labels=classes)
        except ValueError:
            pass

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"real_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({
        "clase": labels, "precision": precision, "recall": recall, "f1": f1, "n_real": support,
    }).set_index("clase")

    return metrics, cm_df, per_class
