"""Carga de los box scores de NBA (team totals por partido, ya agregados,
scrapeados de basketball-reference.com) provistos por el usuario -- misma
metodologia del pipeline de la tesis (PCA de 8 componentes + k-means de
estilos + Random Forest), portada al formato estandarizado de este proyecto
para reusar directamente team_database.py / team_form.py / match_prediction.py
/ pca_analysis.py sin cambios.

Cada CSV (data_cache/nba_raw/nba_team_totals_<AÑO>.csv) trae 2 filas por
partido ("Team Totals" de basketball-reference), una por equipo, con
columnas basic_(...) y adv_(...) de box score basico y avanzado.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "nba_raw"

COMPETITION_NAME = "NBA"
COMPETITION_ID = 900020

SEASON_FILES = list(range(2010, 2027))  # nba_team_totals_2010.csv .. 2026.csv

# Franquicias reubicadas/renombradas -> abreviatura canonica (para que la
# forma reciente y el ranking dinamico no se corten al haber un rebrand).
ABBR_CANON = {
    "NJN": "BRK", "NOH": "NOP", "NOK": "NOP", "CHA": "CHO", "SEA": "OKC",
}

TEAM_FULL_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
    "CHI": "Chicago Bulls", "CHO": "Charlotte Hornets", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
    "BKN": "Brooklyn Nets",
}

# columna_original (extraida del multi-index aplanado) -> nombre limpio
_STAT_RENAME = {
    "fg": "fg", "fga": "fga", "fg%": "fg_pct", "3p": "tp", "3pa": "tpa", "3p%": "tp_pct",
    "ft": "ft", "fta": "fta", "ft%": "ft_pct", "orb": "orb", "drb": "drb", "trb": "trb",
    "ast": "ast", "stl": "stl", "blk": "blk", "tov": "tov", "pf": "pf", "pts": "pts_box",
    "ts%": "ts_pct", "efg%": "efg_pct", "3par": "tpar", "ftr": "ftr", "orb%": "orb_pct",
    "drb%": "drb_pct", "trb%": "trb_pct", "ast%": "ast_pct", "stl%": "stl_pct",
    "blk%": "blk_pct", "tov%": "tov_pct", "usg%": "usg_pct", "ortg": "ortg", "drtg": "drtg",
}


def _clean_col(col: str) -> str | None:
    m = re.search(r",\s*'([^']+)'\)$", col)
    if not m:
        return None
    return _STAT_RENAME.get(m.group(1).lower())


def _canon(abbr: str) -> str:
    return ABBR_CANON.get(abbr, abbr)


def _load_season(year: int) -> pd.DataFrame:
    path = RAW_DIR / f"nba_team_totals_{year}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)

    starters_col = "basic_('unnamed: 0_level_0', 'starters')"
    if starters_col in df.columns:
        df = df[df[starters_col] == "Team Totals"].copy()

    rename = {}
    for c in df.columns:
        if c.startswith("basic_(") or c.startswith("adv_("):
            clean = _clean_col(c)
            if clean and clean not in rename.values():
                rename[c] = clean
    df = df.rename(columns=rename)

    stat_cols = [c for c in set(rename.values()) if c in df.columns]
    df["team_abbr"] = df["team_abbr"].map(_canon)
    df["match_date"] = pd.to_datetime(
        df["game_date_raw"].str.extract(r",\s*(.+)$")[0], format="%B %d, %Y", errors="coerce"
    )
    df["is_home"] = np.isclose(df["pts_box"], df["home_pts"])

    keep = ["boxscore_url", "match_date", "team_abbr", "is_home", "home_pts", "away_pts"] + stat_cols
    return df[keep].dropna(subset=["match_date"])


def load_nba_matches() -> pd.DataFrame:
    """Una fila por partido (home_/away_) en el formato estandarizado del
    proyecto: match_id, competition_id/name, season_id/name, match_date,
    home_team/away_team, home_score/away_score, mas stats home_*/away_*."""
    all_games = []
    season_names = []
    for year in SEASON_FILES:
        season = _load_season(year)
        if season.empty:
            continue
        season_name = f"{year-1}-{str(year)[2:]}"
        season["season_name"] = season_name
        season_names.append(season_name)
        all_games.append(season)

    if not all_games:
        return pd.DataFrame()

    long_df = pd.concat(all_games, ignore_index=True)
    stat_cols = [c for c in long_df.columns if c not in (
        "boxscore_url", "match_date", "team_abbr", "is_home", "home_pts", "away_pts", "season_name",
    )]

    rows = []
    for boxscore_url, g in long_df.groupby("boxscore_url"):
        if len(g) != 2:
            continue
        home = g[g["is_home"]]
        away = g[~g["is_home"]]
        if len(home) != 1 or len(away) != 1:
            continue
        home, away = home.iloc[0], away.iloc[0]
        row = {
            "boxscore_url": boxscore_url,
            "match_date": home["match_date"],
            "season_name": home["season_name"],
            "home_team": TEAM_FULL_NAMES.get(home["team_abbr"], home["team_abbr"]),
            "away_team": TEAM_FULL_NAMES.get(away["team_abbr"], away["team_abbr"]),
            "home_score": int(home["home_pts"]),
            "away_score": int(home["away_pts"]),
        }
        for c in stat_cols:
            row[f"home_{c}"] = home.get(c)
            row[f"away_{c}"] = away.get(c)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
    out["match_id"] = COMPETITION_ID * 1_000_000 + out.index
    out["competition_id"] = COMPETITION_ID
    out["competition_name"] = COMPETITION_NAME
    season_ids = {s: COMPETITION_ID * 100 + i for i, s in enumerate(sorted(set(season_names)))}
    out["season_id"] = out["season_name"].map(season_ids)
    out["data_source"] = "basketball-reference.com (via Drive)"
    return out.drop(columns=["boxscore_url"])
