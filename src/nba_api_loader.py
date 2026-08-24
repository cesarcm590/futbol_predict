"""Carga de temporadas de NBA via nba_api (stats.nba.com oficial) -- a
diferencia de basketball-reference.com (detras de un challenge de
Cloudflare que bloquea requests programaticos), esta API responde JSON
directo y sin bloqueos. Se usa para completar/actualizar temporadas que
data_cache/nba_raw/ (scrapeado a mano, via Drive) no cubre por completo, y
para sincronizar la temporada en curso una vez que empiece.

Mismo esquema que nba_loader.py (home_/away_ prefijos) para que
team_database.py trate ambas fuentes de forma identica.
"""
import time

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from src.nba_loader import ABBR_CANON, COMPETITION_ID, COMPETITION_NAME, TEAM_FULL_NAMES

_COL_MAP = {
    "FGM": "fg", "FGA": "fga", "FG_PCT": "fg_pct", "FG3M": "tp", "FG3A": "tpa",
    "FG3_PCT": "tp_pct", "FTM": "ft", "FTA": "fta", "FT_PCT": "ft_pct",
    "OREB": "orb", "DREB": "drb", "REB": "trb", "AST": "ast", "STL": "stl",
    "BLK": "blk", "TOV": "tov", "PF": "pf", "PTS": "pts_box",
}


def _canon(abbr: str) -> str:
    return ABBR_CANON.get(abbr, abbr)


def load_nba_season(season: str, season_types: tuple[str, ...] = ("Regular Season", "Playoffs")) -> pd.DataFrame:
    """season: '2025-26'. Devuelve 1 fila por partido en el esquema estandar
    (home_/away_), con la fuente marcada como 'nba_api (stats.nba.com)'."""
    frames = []
    for i, season_type in enumerate(season_types):
        if i > 0:
            time.sleep(0.6)  # no saturar la API
        lg = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type, timeout=30)
        df = lg.get_data_frames()[0]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)
    long_df["is_home"] = long_df["MATCHUP"].str.contains(" vs. ")
    long_df["team_abbr"] = long_df["TEAM_ABBREVIATION"].map(_canon)
    long_df = long_df.rename(columns=_COL_MAP)
    stat_cols = list(_COL_MAP.values())

    rows = []
    for game_id, g in long_df.groupby("GAME_ID"):
        if len(g) != 2:
            continue
        home = g[g["is_home"]]
        away = g[~g["is_home"]]
        if len(home) != 1 or len(away) != 1:
            continue
        home, away = home.iloc[0], away.iloc[0]
        row = {
            "match_date": home["GAME_DATE"],
            "season_name": season,
            "home_team": TEAM_FULL_NAMES.get(home["team_abbr"], home["team_abbr"]),
            "away_team": TEAM_FULL_NAMES.get(away["team_abbr"], away["team_abbr"]),
            "home_score": int(home["pts_box"]),
            "away_score": int(away["pts_box"]),
        }
        for c in stat_cols:
            row[f"home_{c}"] = home.get(c)
            row[f"away_{c}"] = away.get(c)
        # ts_pct/efg_pct calculados a mano -- nba_api no los trae directo, pero
        # sin ellos esas columnas quedan NaN en las filas de esta temporada y
        # se rellenan con 0.0 en la forma reciente (rolling), envenenando la
        # prediccion progresivamente conforme mas partidos de esta temporada
        # entran a la ventana de cada equipo. Formulas estandar de basketball-reference.
        for side, r in (("home", home), ("away", away)):
            fga = r.get("fga") or 0
            fta = r.get("fta") or 0
            fgm = r.get("fg") or 0
            tpm = r.get("tp") or 0
            row[f"{side}_ts_pct"] = (r.get("pts_box") / (2 * (fga + 0.44 * fta))) if (fga + fta) else np.nan
            row[f"{side}_efg_pct"] = ((fgm + 0.5 * tpm) / fga) if fga else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
    out["match_date"] = pd.to_datetime(out["match_date"])
    out["match_id"] = COMPETITION_ID * 1_000_000 + 900_000 + out.index  # offset para no chocar con nba_loader
    out["competition_id"] = COMPETITION_ID
    out["competition_name"] = COMPETITION_NAME
    start_year = int(season[:4])
    out["season_id"] = COMPETITION_ID * 100 + (start_year - 2000)  # deterministico, no cambia entre corridas
    out["data_source"] = "nba_api (stats.nba.com)"
    return out
