"""Liga MX (Mexico) via openfootball (dominio publico, github.com/openfootball/world).

A diferencia de StatsBomb, esta fuente SOLO trae fecha/equipos/marcador -- sin
eventos (sin xG, pases, posesion, tiros). Por eso Liga MX solo alimenta el
ranking dinamico y la prediccion de resultados (que solo necesitan goles),
pero no los heatmaps ni el PCA de jugador, que requieren datos de evento.

Liga MX juega 2 torneos cortos por año (Apertura y Clausura), cada uno con
fase regular (todos contra todos) + liguilla de eliminacion directa. Cada
torneo corto se trata como su propia "temporada" con su propia tabla
dinamica; los partidos de liguilla se excluyen (formato de bracket, no de
tabla, y con marcadores de penales/ida-vuelta que no encajan en el esquema).
"""
import re
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "ligamx"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/openfootball/world/master/north-america/mexico/{season}_mx1.txt"

SEASONS = [
    "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]

STAGE_RE = re.compile(r"^▪\s*(.+)$")
DATE_RE = re.compile(r"^\s{2}(\w{3})\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
MATCH_RE = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?(?P<home>.+?)\s{2,}v\s+(?P<away>.+?)\s{2,}(?P<hs>\d+)-(?P<as_>\d+)(?:\s+.*)?$"
)

# competition_id/season_id sinteticos (StatsBomb usa enteros; usamos un rango
# alto reservado para no chocar con ids reales de StatsBomb).
COMPETITION_ID = 900001
_SEASON_ID_BASE = 900100


def _fetch_raw(season: str) -> str:
    path = CACHE_DIR / f"{season}_mx1.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    url = BASE_URL.format(season=season)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    return resp.text


def _parse_regular_season_matches(season: str) -> pd.DataFrame:
    text = _fetch_raw(season)

    rows = []
    current_stage = ""
    current_tournament = None  # "Apertura" o "Clausura"
    current_date = None

    for line in text.splitlines():
        sm = STAGE_RE.match(line.strip())
        if sm:
            current_stage = sm.group(1).strip()
            current_tournament = current_stage.split(",")[0].strip()
            continue

        dm = DATE_RE.match(line)
        if dm:
            _, mon, day, year = dm.groups()
            if year:
                current_date = pd.to_datetime(f"{mon} {day} {year}", format="%b %d %Y", errors="coerce")
            elif current_date is not None:
                current_date = pd.to_datetime(f"{mon} {day} {current_date.year}", format="%b %d %Y", errors="coerce")
            continue

        if "Playoffs" in current_stage or current_date is None:
            continue

        m = MATCH_RE.match(line)
        if m:
            rows.append({
                "season_raw": season,
                "tournament": current_tournament,
                "match_date": current_date,
                "home_team": m.group("home").strip(),
                "away_team": m.group("away").strip(),
                "home_score": int(m.group("hs")),
                "away_score": int(m.group("as_")),
            })

    return pd.DataFrame(rows)


def load_ligamx_matches() -> pd.DataFrame:
    """Todas las temporadas disponibles, ya con match_id/season_name listos
    para entrar al mismo esquema que team_database (una fila por partido)."""
    frames = [_parse_regular_season_matches(s) for s in SEASONS]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["match_date"]).reset_index(drop=True)

    df["season_name"] = df["tournament"] + " " + df["match_date"].dt.year.astype(str)
    season_ids = {name: _SEASON_ID_BASE + i for i, name in enumerate(sorted(df["season_name"].unique()))}
    df["season_id"] = df["season_name"].map(season_ids)
    df["competition_id"] = COMPETITION_ID
    df["competition_name"] = "Liga MX"

    df = df.sort_values("match_date").reset_index(drop=True)
    df["match_id"] = COMPETITION_ID * 100000 + df.index
    return df[[
        "match_id", "competition_id", "season_id", "competition_name", "season_name",
        "match_date", "home_team", "away_team", "home_score", "away_score",
    ]]
