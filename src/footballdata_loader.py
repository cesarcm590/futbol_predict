"""Carga de football-data.co.uk: resultados + estadisticas de partido (tiros,
corners, faltas, tarjetas) para las "Main Leagues" europeas, gratis y sin
login, desde 2000/01. A diferencia de StatsBomb (eventos completos) o
openfootball (solo resultados), esta fuente da estadisticas agregadas por
partido -- justo lo que falta para correr el modelo de corners en mas
temporadas de las que StatsBomb tiene liberadas.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "footballdata"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEASON_CODES = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2000, 2025)]  # 0001 .. 2425

# division code (en football-data.co.uk) -> nombre de competicion en nuestra base
DIVISIONS = {
    "E0": "Premier League",
}

# nombres cortos de football-data.co.uk -> nombres largos (convencion StatsBomb),
# para que un mismo equipo se reconozca como el mismo a traves de fuentes.
TEAM_NAME_MAP = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers", "Blackpool": "Blackpool", "Bolton": "Bolton Wanderers",
    "Bournemouth": "AFC Bournemouth", "Bradford": "Bradford City", "Brentford": "Brentford",
    "Brighton": "Brighton & Hove Albion", "Burnley": "Burnley", "Cardiff": "Cardiff City",
    "Charlton": "Charlton Athletic", "Chelsea": "Chelsea", "Coventry": "Coventry City",
    "Crystal Palace": "Crystal Palace", "Derby": "Derby County", "Everton": "Everton",
    "Fulham": "Fulham", "Huddersfield": "Huddersfield Town", "Hull": "Hull City",
    "Ipswich": "Ipswich Town", "Leeds": "Leeds United", "Leicester": "Leicester City",
    "Liverpool": "Liverpool", "Luton": "Luton Town", "Man City": "Manchester City",
    "Man United": "Manchester United", "Middlesbrough": "Middlesbrough",
    "Newcastle": "Newcastle United", "Norwich": "Norwich City", "Nott'm Forest": "Nottingham Forest",
    "Portsmouth": "Portsmouth", "QPR": "Queens Park Rangers", "Reading": "Reading",
    "Sheffield United": "Sheffield United", "Southampton": "Southampton", "Stoke": "Stoke City",
    "Sunderland": "Sunderland", "Swansea": "Swansea City", "Tottenham": "Tottenham Hotspur",
    "Watford": "Watford", "West Brom": "West Bromwich Albion", "West Ham": "West Ham United",
    "Wigan": "Wigan Athletic", "Wolves": "Wolverhampton Wanderers",
}

COL_MAP = {
    "HC": "home_corners", "AC": "away_corners",
    "HS": "home_shots", "AS": "away_shots",
    "HF": "home_fouls_committed", "AF": "away_fouls_committed",
}


def _season_name(code: str) -> str:
    y1 = int(code[:2])
    y1 = 2000 + y1
    return f"{y1}/{y1+1}"


def _fetch_raw(division: str, code: str) -> str | None:
    path = CACHE_DIR / f"{division}_{code}.csv"
    if path.exists():
        return path.read_text(errors="ignore")
    url = f"https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200 or not resp.content:
        return None
    text = resp.content.decode("utf-8-sig", errors="ignore")
    path.write_text(text)
    return text


def load_footballdata(division: str = "E0") -> pd.DataFrame:
    """Todas las temporadas disponibles de una division: una fila por partido
    con home_/away_ corners, tiros y faltas, mas match_date/teams/score."""
    competition_name = DIVISIONS[division]
    rows = []
    for code in SEASON_CODES:
        raw = _fetch_raw(division, code)
        if not raw:
            continue
        try:
            df = pd.read_csv(pd.io.common.StringIO(raw), on_bad_lines="skip")
        except Exception:
            continue
        if "HomeTeam" not in df.columns or "Date" not in df.columns:
            continue
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        match_date = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
        clean = pd.DataFrame({
            "match_date": match_date,
            "home_team": df["HomeTeam"].map(TEAM_NAME_MAP).fillna(df["HomeTeam"]),
            "away_team": df["AwayTeam"].map(TEAM_NAME_MAP).fillna(df["AwayTeam"]),
            "home_score": df["FTHG"].astype(int),
            "away_score": df["FTAG"].astype(int),
            "season_name": _season_name(code),
        })
        for src, dst in COL_MAP.items():
            clean[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
        clean = clean.dropna(subset=["match_date"])
        rows.append(clean)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("match_date").reset_index(drop=True)
    out["competition_name"] = competition_name
    out["match_id"] = 900010_000000 + out.index
    out["competition_id"] = 900010
    season_ids = {s: 900200 + i for i, s in enumerate(sorted(out["season_name"].unique()))}
    out["season_id"] = out["season_name"].map(season_ids)
    out["data_source"] = "football-data.co.uk"
    return out
