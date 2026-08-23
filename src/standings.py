"""Tabla de posiciones dinamica (estado ANTES de cada partido).

Mismo concepto que win_pct_dynamic / rank_dynamic / diff_* en la metodologia
de basquetball: para cada partido, el estado acumulado de cada equipo usando
solo informacion previa a ese partido (nunca el resultado del propio partido).

Solo tiene sentido para competencias de formato liga (todos contra todos en
una sola tabla) -- ver `team_database.is_league_format`.
"""
import numpy as np
import pandas as pd


def compute_dynamic_standings(matches: pd.DataFrame) -> pd.DataFrame:
    """matches necesita: match_id, match_date, home_team, away_team, home_score, away_score.

    Devuelve una fila por match_id con el estado dinamico previo al partido
    para home y away, mas las diferencias home-away.
    """
    m = matches.sort_values("match_date").reset_index(drop=True)
    teams = pd.unique(pd.concat([m["home_team"], m["away_team"]]))
    state = {t: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0, "gf": 0, "ga": 0} for t in teams}

    def snapshot(s):
        games = s["games"]
        win_pct = s["wins"] / games if games > 0 else np.nan
        return games, win_pct, s["points"], s["gf"] - s["ga"]

    rows = []
    for _, r in m.iterrows():
        home, away = r["home_team"], r["away_team"]
        hs, as_ = state[home], state[away]

        h_games, h_winpct, h_pts, h_gd = snapshot(hs)
        a_games, a_winpct, a_pts, a_gd = snapshot(as_)

        table = sorted(state.items(), key=lambda kv: (-kv[1]["points"], -(kv[1]["gf"] - kv[1]["ga"])))
        rank_map = {team: i + 1 for i, (team, _) in enumerate(table)}

        rows.append({
            "match_id": r["match_id"],
            "home_games_before": h_games,
            "home_win_pct_dynamic": h_winpct,
            "home_points_before": h_pts,
            "home_goal_diff_before": h_gd,
            "home_rank_dynamic": rank_map[home],
            "away_games_before": a_games,
            "away_win_pct_dynamic": a_winpct,
            "away_points_before": a_pts,
            "away_goal_diff_before": a_gd,
            "away_rank_dynamic": rank_map[away],
        })

        hs["games"] += 1
        as_["games"] += 1
        hs["gf"] += r["home_score"]
        hs["ga"] += r["away_score"]
        as_["gf"] += r["away_score"]
        as_["ga"] += r["home_score"]
        if r["home_score"] > r["away_score"]:
            hs["wins"] += 1
            hs["points"] += 3
            as_["losses"] += 1
        elif r["home_score"] < r["away_score"]:
            as_["wins"] += 1
            as_["points"] += 3
            hs["losses"] += 1
        else:
            hs["draws"] += 1
            as_["draws"] += 1
            hs["points"] += 1
            as_["points"] += 1

    out = pd.DataFrame(rows)
    out["diff_win_pct_dynamic"] = out["home_win_pct_dynamic"] - out["away_win_pct_dynamic"]
    out["diff_rank_dynamic"] = out["away_rank_dynamic"] - out["home_rank_dynamic"]
    return out
