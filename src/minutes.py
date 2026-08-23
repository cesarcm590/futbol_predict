"""Minutos jugados por jugador y partido.

StatsBomb Open Data no trae minutos jugados directamente: se derivan de los
eventos 'Starting XI' (titulares, arrancan en el minuto 0) y 'Substitution'
(cambia el jugador en cancha en el minuto del evento). El final del partido
se aproxima con el minuto+segundo del ultimo evento registrado.
"""
import pandas as pd


def _match_end_minute(events: pd.DataFrame) -> float:
    last = events.sort_values(["minute", "second"]).iloc[-1]
    return float(last["minute"]) + float(last.get("second", 0) or 0) / 60.0


def player_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """Minutos jugados por cada jugador dentro de UN partido (un match_id)."""
    if events.empty:
        return pd.DataFrame(columns=["player", "team", "minutes_played"])

    match_end = _match_end_minute(events)
    spans: dict[str, dict] = {}
    player_team: dict[str, str] = {}

    for _, row in events[events["type"] == "Starting XI"].iterrows():
        team = row["team"]
        lineup = (row.get("tactics") or {}).get("lineup", [])
        for entry in lineup:
            name = entry["player"]["name"]
            spans[name] = {"on": 0.0, "off": match_end}
            player_team[name] = team

    subs = events[events["type"] == "Substitution"].sort_values(["minute", "second"])
    for _, row in subs.iterrows():
        team = row["team"]
        off_player = row.get("player")
        on_player = row.get("substitution_replacement")
        t = float(row["minute"]) + float(row.get("second", 0) or 0) / 60.0

        if off_player in spans:
            spans[off_player]["off"] = t
        elif pd.notna(off_player):
            spans[off_player] = {"on": 0.0, "off": t}
            player_team[off_player] = team

        if pd.notna(on_player):
            spans[on_player] = {"on": t, "off": match_end}
            player_team[on_player] = team

    rows = [
        {"player": name, "team": player_team.get(name), "minutes_played": max(span["off"] - span["on"], 0.0)}
        for name, span in spans.items()
    ]
    return pd.DataFrame(rows)


def minutes_for_matches(events_by_match: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Agrega minutos jugados de un jugador a traves de varios partidos."""
    frames = []
    for match_id, events in events_by_match.items():
        pm = player_minutes(events)
        pm["match_id"] = match_id
        frames.append(pm)
    if not frames:
        return pd.DataFrame(columns=["player", "team", "minutes_played"])

    all_minutes = pd.concat(frames, ignore_index=True)
    agg = (
        all_minutes.groupby("player", as_index=False)
        .agg(minutes_played=("minutes_played", "sum"), matches_played=("match_id", "nunique"))
    )
    team_mode = all_minutes.groupby("player")["team"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
    agg = agg.merge(team_mode.rename("team"), on="player", how="left")
    return agg
