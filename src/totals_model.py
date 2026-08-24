"""Prediccion de un TOTAL por partido (corners, puntos, lo que sea) contra
lineas tipo casa de apuestas, con Random Forest de regresion sobre PCA.

Cada arbol del bosque da su propia estimacion del total; la fraccion de
arboles que predice por encima de una linea es la probabilidad de "Over"
para esa linea -- el bosque completo funciona como una distribucion
empirica, sin entrenar un clasificador binario separado por cada linea.

Generico sobre el dominio: se uso primero para corners de futbol
(src/corner_model.py originalmente) y ahora tambien para puntos totales de
NBA -- basta con pasarle el nombre de la columna objetivo y las lineas.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def fit_totals_model(train: pd.DataFrame, feature_cols: list[str], target_col: str,
                      n_components: int = 8, n_estimators: int = 300, random_state: int = 42):
    X_train_raw = train[feature_cols].fillna(0.0)
    scaler = StandardScaler().fit(X_train_raw)
    X_train_std = scaler.transform(X_train_raw)

    n_components = max(1, min(n_components, len(feature_cols), len(X_train_std)))
    pca = PCA(n_components=n_components, random_state=random_state).fit(X_train_std)
    X_train_pca = pca.transform(X_train_std)

    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, min_samples_leaf=3)
    reg.fit(X_train_pca, train[target_col])

    return {"scaler": scaler, "pca": pca, "reg": reg, "feature_cols": feature_cols, "target_col": target_col}


def predict_totals(model: dict, df: pd.DataFrame, lines: list[float], extra_cols: list[str] | None = None) -> pd.DataFrame:
    target_col = model["target_col"]
    X_raw = df[model["feature_cols"]].fillna(0.0)
    X_std = model["scaler"].transform(X_raw)
    X_pca = model["pca"].transform(X_std)

    tree_preds = np.stack([est.predict(X_pca) for est in model["reg"].estimators_], axis=1)  # (n_rows, n_trees)
    mean_pred = tree_preds.mean(axis=1)

    keep = [c for c in (["match_id", "match_date", "home_team", "away_team", "data_source", target_col] + (extra_cols or [])) if c in df.columns]
    out = df[keep].reset_index(drop=True).copy()
    out["esperado"] = mean_pred
    # rango esperado a partir de la distribucion empirica del bosque (percentiles
    # de las predicciones de cada arbol) -- no un intervalo estadistico formal,
    # pero mucho mas informativo que un solo numero cuando lo que importa es el
    # rango probable, no acertarle al promedio exacto.
    out["p10"] = np.percentile(tree_preds, 10, axis=1)
    out["p25"] = np.percentile(tree_preds, 25, axis=1)
    out["p75"] = np.percentile(tree_preds, 75, axis=1)
    out["p90"] = np.percentile(tree_preds, 90, axis=1)

    for line in lines:
        p_over = (tree_preds > line).mean(axis=1)
        out[f"p_over_{line}"] = p_over
        out[f"pick_{line}"] = np.where(p_over >= 0.5, "Over", "Under")
        if target_col in out.columns:
            actual_over = out[target_col] > line
            out[f"acierto_{line}"] = (out[f"pick_{line}"] == np.where(actual_over, "Over", "Under"))
    return out


def evaluate_totals(pred_df: pd.DataFrame, target_col: str, train_mean: float, lines: list[float]) -> dict:
    mae = float((pred_df["esperado"] - pred_df[target_col]).abs().mean())

    per_line = []
    for line in lines:
        naive_pick = "Over" if train_mean > line else "Under"
        actual_over = pred_df[target_col] > line
        model_acc = float(pred_df[f"acierto_{line}"].mean())
        naive_acc = float((np.where(actual_over, "Over", "Under") == naive_pick).mean())
        per_line.append({
            "line": line, "model_accuracy": model_acc, "naive_pick": naive_pick,
            "naive_accuracy": naive_acc, "pct_over_real": float(actual_over.mean()),
        })

    return {"mae": mae, "n_test": len(pred_df), "per_line": pd.DataFrame(per_line).set_index("line")}
