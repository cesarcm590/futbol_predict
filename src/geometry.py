"""Geometria de cancha compartida: direccion de ataque y progresion de balon.

Usado tanto por features.py (jugador-partido) como por team_features.py
(equipo-partido) para no duplicar la logica de "hacia donde ataca cada
equipo" y "que tan progresiva fue una accion".
"""
import numpy as np
import pandas as pd


def team_directions(events: pd.DataFrame) -> dict:
    """Direccion de ataque (+1 hacia x=120, -1 hacia x=0) por (match_id, team, period).

    Heuristica: en cada periodo, el equipo cuyo promedio de x de eventos es
    menor se asume que ataca hacia x=120 (pasa mas tiempo construyendo cerca
    de su propia porteria en x=0), y viceversa.
    """
    ev = events.dropna(subset=["location"]).copy()
    ev["x"] = ev["location"].apply(lambda loc: loc[0] if isinstance(loc, list) and len(loc) == 2 else np.nan)
    ev = ev.dropna(subset=["x"])
    grp = ev.groupby(["match_id", "period", "team"])["x"].mean().reset_index()

    directions = {}
    for (mid, period), sub in grp.groupby(["match_id", "period"]):
        if len(sub) < 2:
            for _, r in sub.iterrows():
                directions[(mid, r["team"], period)] = 1
            continue
        sub = sub.sort_values("x")
        low_team = sub.iloc[0]["team"]
        high_team = sub.iloc[-1]["team"]
        directions[(mid, low_team, period)] = 1
        directions[(mid, high_team, period)] = -1
    return directions


def progress_fraction(start_x: float, end_x: float, direction: int) -> float:
    goal_x = 120.0 if direction == 1 else 0.0
    dist_start = abs(goal_x - start_x)
    if dist_start == 0:
        return 0.0
    dist_end = abs(goal_x - end_x)
    return (dist_start - dist_end) / dist_start
