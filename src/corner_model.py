"""Prediccion de corners totales por partido (local + visita), con lineas de
tipo casa de apuestas (7.5, 8.5, ... 12.5) para clasificar Over/Under.

Misma logica de PCA + separacion temporal que match_prediction.py, pero con
un Random Forest de REGRESION en vez de clasificacion: cada arbol del bosque
da su propia estimacion de corners totales, y la fraccion de arboles que
predice por encima de una linea es la probabilidad de "Over" para esa linea
-- el bosque completo funciona como una distribucion empirica, sin tener que
entrenar un clasificador binario separado por cada linea.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

CORNER_LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]


def fit_corners_model(train: pd.DataFrame, feature_cols: list[str], n_components: int = 8,
                       n_estimators: int = 300, random_state: int = 42):
    X_train_raw = train[feature_cols].fillna(0.0)
    scaler = StandardScaler().fit(X_train_raw)
    X_train_std = scaler.transform(X_train_raw)

    n_components = max(1, min(n_components, len(feature_cols), len(X_train_std)))
    pca = PCA(n_components=n_components, random_state=random_state).fit(X_train_std)
    X_train_pca = pca.transform(X_train_std)

    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, min_samples_leaf=3)
    reg.fit(X_train_pca, train["total_corners"])

    return {"scaler": scaler, "pca": pca, "reg": reg, "feature_cols": feature_cols}


def predict_corners(model: dict, df: pd.DataFrame, lines: list[float] | None = None) -> pd.DataFrame:
    lines = lines or CORNER_LINES
    X_raw = df[model["feature_cols"]].fillna(0.0)
    X_std = model["scaler"].transform(X_raw)
    X_pca = model["pca"].transform(X_std)

    tree_preds = np.stack([est.predict(X_pca) for est in model["reg"].estimators_], axis=1)  # (n_rows, n_trees)
    mean_pred = tree_preds.mean(axis=1)

    keep = [c for c in ["match_id", "match_date", "home_team", "away_team", "data_source",
                         "actual_home_corners", "actual_away_corners", "total_corners"] if c in df.columns]
    out = df[keep].reset_index(drop=True).copy()
    out["corners_esperados"] = mean_pred

    for line in lines:
        p_over = (tree_preds > line).mean(axis=1)
        out[f"p_over_{line}"] = p_over
        out[f"pick_{line}"] = np.where(p_over >= 0.5, "Over", "Under")
        if "total_corners" in out.columns:
            actual_over = out["total_corners"] > line
            out[f"acierto_{line}"] = (out[f"pick_{line}"] == np.where(actual_over, "Over", "Under"))
    return out


def evaluate_corners(pred_df: pd.DataFrame, train_mean_corners: float, lines: list[float] | None = None) -> dict:
    lines = lines or CORNER_LINES
    mae = float((pred_df["corners_esperados"] - pred_df["total_corners"]).abs().mean())

    per_line = []
    naive_pick_global = "Over" if train_mean_corners > lines[0] else "Under"  # se recalcula por linea abajo
    for line in lines:
        naive_pick = "Over" if train_mean_corners > line else "Under"
        actual_over = pred_df["total_corners"] > line
        model_acc = float(pred_df[f"acierto_{line}"].mean())
        naive_acc = float((np.where(actual_over, "Over", "Under") == naive_pick).mean())
        per_line.append({
            "line": line, "model_accuracy": model_acc, "naive_pick": naive_pick,
            "naive_accuracy": naive_acc, "pct_over_real": float(actual_over.mean()),
        })

    return {"mae": mae, "n_test": len(pred_df), "per_line": pd.DataFrame(per_line).set_index("line")}
