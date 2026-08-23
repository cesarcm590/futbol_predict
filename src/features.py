"""Tabla de metricas por jugador-temporada, normalizadas por 90 minutos.

Construida a partir del dataframe de eventos (uno o varios partidos) de
StatsBomb Open Data + los minutos jugados calculados en `minutes.py`.
"""
import numpy as np
import pandas as pd

from src.geometry import progress_fraction, team_directions
from src.minutes import minutes_for_matches

PASS_COMPLETE_OK = {None}  # pass_outcome NaN == pase completo


def role_bucket(position: str | None) -> str:
    """Agrupa las ~18 posiciones detalladas de StatsBomb en 4 roles amplios.

    Sirve para colorear graficas (PCA, clusters) de forma legible: 18 colores
    de posicion exacta son ilegibles en una leyenda, 4 roles no.
    """
    if not position or pd.isna(position):
        return "Otro"
    p = str(position)
    if "Goalkeeper" in p:
        return "Portero"
    if "Wing Back" in p or "Back" in p:
        return "Defensa"
    if "Midfield" in p:
        return "Mediocampo"
    if "Wing" in p or "Forward" in p or "Striker" in p:
        return "Delantero"
    return "Otro"


def build_player_features(events: pd.DataFrame, min_minutes: float = 90.0) -> pd.DataFrame:
    """Una fila por jugador con metricas por-90 agregadas sobre `events`.

    `events` puede contener uno o varios partidos (misma competicion/temporada).
    """
    if events.empty:
        return pd.DataFrame()

    events = events.copy()
    events["match_id"] = events["match_id"].astype(int)

    events_by_match = {mid: g for mid, g in events.groupby("match_id")}
    minutes_df = minutes_for_matches(events_by_match)
    minutes_df = minutes_df[minutes_df["minutes_played"] >= min_minutes]
    if minutes_df.empty:
        return pd.DataFrame()

    directions = team_directions(events)

    def direction_for(row):
        return directions.get((row["match_id"], row["team"], row["period"]), 1)

    # posicion mas frecuente por jugador (para colorear PCA / clusters)
    position = (
        events.dropna(subset=["position"])
        .groupby("player")["position"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .rename("position")
    )

    stats = []
    for player, pdf in events.groupby("player"):
        if player not in set(minutes_df["player"]):
            continue

        passes = pdf[pdf["type"] == "Pass"]
        passes_completed = passes["pass_outcome"].isna().sum()
        passes_attempted = len(passes)

        prog_passes = 0
        if not passes.empty:
            for _, row in passes.dropna(subset=["location", "pass_end_location"]).iterrows():
                d = direction_for(row)
                frac = progress_fraction(row["location"][0], row["pass_end_location"][0], d)
                if frac >= 0.25:
                    prog_passes += 1

        carries = pdf[pdf["type"] == "Carry"]
        prog_carries = 0
        if not carries.empty:
            for _, row in carries.dropna(subset=["location", "carry_end_location"]).iterrows():
                d = direction_for(row)
                frac = progress_fraction(row["location"][0], row["carry_end_location"][0], d)
                if frac >= 0.25:
                    prog_carries += 1

        shots = pdf[pdf["type"] == "Shot"]
        goals = (shots["shot_outcome"] == "Goal").sum()
        xg_total = shots["shot_statsbomb_xg"].fillna(0).sum()

        assists = (pdf["pass_goal_assist"] == True).sum()  # noqa: E712
        key_passes = (pdf["pass_shot_assist"] == True).sum()  # noqa: E712

        dribbles = pdf[pdf["type"] == "Dribble"]
        dribbles_won = (dribbles["dribble_outcome"] == "Complete").sum()

        duels = pdf[pdf["type"] == "Duel"]
        tackles = duels[duels["duel_type"] == "Tackle"]
        tackles_won = tackles["duel_outcome"].isin(["Won", "Success In Play", "Success Out"]).sum()
        aerials = duels[duels["duel_type"] == "Aerial Lost"]
        aerials_won = (aerials["duel_outcome"] == "Won").sum()

        interceptions = (pdf["type"] == "Interception").sum()
        pressures = (pdf["type"] == "Pressure").sum()
        clearances = (pdf["type"] == "Clearance").sum()
        touches = pdf["location"].notna().sum()

        stats.append(
            {
                "player": player,
                "passes_attempted": passes_attempted,
                "passes_completed": passes_completed,
                "pass_pct": (passes_completed / passes_attempted * 100) if passes_attempted else np.nan,
                "progressive_passes": prog_passes,
                "progressive_carries": prog_carries,
                "shots": len(shots),
                "goals": goals,
                "xg_total": xg_total,
                "assists": assists,
                "key_passes": key_passes,
                "dribbles_won": dribbles_won,
                "tackles_won": tackles_won,
                "aerials_won": aerials_won,
                "interceptions": interceptions,
                "pressures": pressures,
                "clearances": clearances,
                "touches": touches,
            }
        )

    feat = pd.DataFrame(stats).merge(minutes_df, on="player", how="inner")
    feat = feat.merge(position, on="player", how="left")
    feat["role"] = feat["position"].apply(role_bucket)

    per90_cols = [c for c in feat.columns if c not in (
        "player", "team", "position", "role", "minutes_played", "matches_played", "pass_pct"
    )]
    for c in per90_cols:
        feat[f"{c}_p90"] = feat[c] / feat["minutes_played"] * 90.0

    return feat.sort_values("minutes_played", ascending=False).reset_index(drop=True)
