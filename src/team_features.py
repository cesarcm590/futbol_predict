"""Metricas de equipo por partido, agregadas desde eventos.

Misma logica que features.py pero a nivel equipo-partido en vez de
jugador-temporada: una fila por partido con prefijos home_/away_, igual que
la base final de la metodologia de basquetball (Estancia, Carrillo Martinez
2026), lista para PCA + k-means + ranking dinamico a nivel equipo.
"""
import numpy as np
import pandas as pd

from src.geometry import progress_fraction, team_directions


def _possession_pct(events: pd.DataFrame, team: str) -> float:
    sub = events.dropna(subset=["possession_team"])
    if sub.empty:
        return np.nan
    return (sub["possession_team"] == team).mean() * 100.0


def _team_match_stats(events: pd.DataFrame, team: str, directions: dict) -> dict:
    tdf = events[events["team"] == team]

    passes = tdf[tdf["type"] == "Pass"]
    passes_completed = passes["pass_outcome"].isna().sum()
    passes_attempted = len(passes)

    prog_passes = 0
    for _, row in passes.dropna(subset=["location", "pass_end_location"]).iterrows():
        d = directions.get((row["match_id"], row["team"], row["period"]), 1)
        if progress_fraction(row["location"][0], row["pass_end_location"][0], d) >= 0.25:
            prog_passes += 1

    carries = tdf[tdf["type"] == "Carry"]
    prog_carries = 0
    for _, row in carries.dropna(subset=["location", "carry_end_location"]).iterrows():
        d = directions.get((row["match_id"], row["team"], row["period"]), 1)
        if progress_fraction(row["location"][0], row["carry_end_location"][0], d) >= 0.25:
            prog_carries += 1

    shots = tdf[tdf["type"] == "Shot"]
    goals = (shots["shot_outcome"] == "Goal").sum()
    xg_total = shots["shot_statsbomb_xg"].fillna(0).sum()

    duels = tdf[tdf["type"] == "Duel"]
    tackles = duels[duels["duel_type"] == "Tackle"]
    tackles_won = tackles["duel_outcome"].isin(["Won", "Success In Play", "Success Out"]).sum()

    corners = int((passes["pass_type"] == "Corner").sum()) if "pass_type" in passes.columns else np.nan

    return {
        "passes_attempted": passes_attempted,
        "passes_completed": passes_completed,
        "pass_pct": (passes_completed / passes_attempted * 100) if passes_attempted else np.nan,
        "possession_pct": _possession_pct(events, team),
        "progressive_passes": prog_passes,
        "progressive_carries": prog_carries,
        "shots": len(shots),
        "goals_from_events": int(goals),
        "xg_total": xg_total,
        "tackles_won": int(tackles_won),
        "interceptions": int((tdf["type"] == "Interception").sum()),
        "pressures": int((tdf["type"] == "Pressure").sum()),
        "fouls_committed": int((tdf["type"] == "Foul Committed").sum()),
        "touches": int(tdf["location"].notna().sum()),
        "corners": corners,
    }


def build_team_match_row(events: pd.DataFrame, match_meta: dict) -> dict:
    """Una fila por partido: identificadores + prefijos home_/away_ con las metricas de cada equipo."""
    home, away = match_meta["home_team"], match_meta["away_team"]
    directions = team_directions(events)

    row = {
        "match_id": match_meta["match_id"],
        "competition_id": match_meta["competition_id"],
        "season_id": match_meta["season_id"],
        "competition_name": match_meta.get("competition_name"),
        "season_name": match_meta.get("season"),
        "match_date": match_meta.get("match_date"),
        "home_team": home,
        "away_team": away,
        "home_score": match_meta["home_score"],
        "away_score": match_meta["away_score"],
    }

    for k, v in _team_match_stats(events, home, directions).items():
        row[f"home_{k}"] = v
    for k, v in _team_match_stats(events, away, directions).items():
        row[f"away_{k}"] = v

    row["total_goals"] = row["home_score"] + row["away_score"]
    row["goal_margin_home"] = row["home_score"] - row["away_score"]
    row["home_win"] = int(row["home_score"] > row["away_score"])
    if pd.notna(row.get("home_corners")) and pd.notna(row.get("away_corners")):
        row["total_corners"] = row["home_corners"] + row["away_corners"]
    else:
        row["total_corners"] = np.nan
    return row
