"""Forma reciente de cada equipo y ensamblado del dataset de prediccion.

Todo aqui se calcula usando SOLO informacion anterior al partido (shift(1)
antes de cualquier rolling), para que el dataset de prediccion nunca vea el
resultado del partido que esta tratando de predecir -- mismo cuidado que la
metodologia de basquetball tuvo al separar variables "previas al partido" de
variables que solo se conocen despues.
"""
import numpy as np
import pandas as pd

ROLLING_STAT_COLS = [
    "goals_for", "goals_against", "xg_total", "shots", "possession_pct",
    "passes_completed", "pass_pct", "progressive_passes", "progressive_carries",
    "tackles_won", "interceptions", "pressures", "corners",
]

DYNAMIC_COLS = ["win_pct_dynamic", "rank_dynamic", "points_before", "goal_diff_before", "games_before"]


def add_rolling_form(team_perspective: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Agrega columnas form_* = promedio movil de las ULTIMAS `window` partidos
    ANTERIORES (sin incluir el partido actual) de cada equipo."""
    df = team_perspective.sort_values(["team", "match_date"]).copy()

    for col in ROLLING_STAT_COLS:
        if col not in df.columns:
            continue
        df[f"form_{col}"] = df.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    if "opp_xg_total" in df.columns:
        df["form_opp_xg_against"] = df.groupby("team")["opp_xg_total"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    if "opp_corners" in df.columns:
        df["form_opp_corners_against"] = df.groupby("team")["opp_corners"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    return df


def prematch_feature_cols(team_perspective_with_form: pd.DataFrame) -> list[str]:
    dynamic = [c for c in DYNAMIC_COLS if c in team_perspective_with_form.columns]
    form = [c for c in team_perspective_with_form.columns if c.startswith("form_")]
    return dynamic + form


def build_prediction_dataset(team_perspective_with_form: pd.DataFrame) -> pd.DataFrame:
    """De la vista equipo-partido (con forma) arma 1 fila por partido con
    prefijos home_/away_ SOLO de variables conocidas antes del partido, mas
    el resultado real (result: H/D/A, total_goals) para entrenar/evaluar.
    """
    feat_cols = prematch_feature_cols(team_perspective_with_form)
    shared = ["match_id", "competition_name", "season_name", "match_date"]
    shared += [c for c in ["data_source"] if c in team_perspective_with_form.columns]

    home = team_perspective_with_form[team_perspective_with_form["is_home"] == 1]
    away = team_perspective_with_form[team_perspective_with_form["is_home"] == 0]

    home_feat = home[["match_id"] + feat_cols].rename(columns={c: f"home_{c}" for c in feat_cols})
    away_feat = away[["match_id"] + feat_cols].rename(columns={c: f"away_{c}" for c in feat_cols})

    meta_cols = ["team", "opponent", "goals_for", "goals_against"]
    home_corners = None
    if "corners" in home.columns and "corners" in away.columns:
        # OJO: "actual_" y no "home_"/"away_" a proposito -- prediction_feature_cols()
        # y filtros similares recogen cualquier columna con esos prefijos como feature de
        # entrada; estos son los corners REALES del propio partido (la respuesta que se
        # quiere predecir), asi que un prefijo home_/away_ aqui seria fuga de datos.
        home_corners = home[["match_id", "corners"]].rename(columns={"corners": "actual_home_corners"})
        away_corners = away[["match_id", "corners"]].rename(columns={"corners": "actual_away_corners"})

    meta = home[shared + meta_cols].rename(columns={
        "team": "home_team", "opponent": "away_team", "goals_for": "home_score", "goals_against": "away_score",
    })

    out = meta.merge(home_feat, on="match_id").merge(away_feat, on="match_id")
    out["total_goals"] = out["home_score"] + out["away_score"]
    if home_corners is not None:
        out = out.merge(home_corners, on="match_id").merge(away_corners, on="match_id")
        out["total_corners"] = out["actual_home_corners"] + out["actual_away_corners"]
    out["result"] = np.select(
        [out["home_score"] > out["away_score"], out["home_score"] == out["away_score"]],
        ["H", "D"], default="A",
    )

    # "Que tan parejos" estan los dos equipos antes del partido -- la senial mas
    # directa de empate probable (dos equipos muy disparejos casi nunca empatan).
    if "home_rank_dynamic" in out.columns and "away_rank_dynamic" in out.columns:
        out["rank_gap"] = (out["home_rank_dynamic"] - out["away_rank_dynamic"]).abs()
    if "home_win_pct_dynamic" in out.columns and "away_win_pct_dynamic" in out.columns:
        out["win_pct_gap"] = (out["home_win_pct_dynamic"] - out["away_win_pct_dynamic"]).abs()
    if "home_form_goals_for" in out.columns and "away_form_goals_for" in out.columns:
        out["form_goals_gap"] = (out["home_form_goals_for"] - out["away_form_goals_for"]).abs()

    return out.sort_values("match_date").reset_index(drop=True)


CLOSENESS_COLS = ["rank_gap", "win_pct_gap", "form_goals_gap"]
