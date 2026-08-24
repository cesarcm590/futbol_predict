"""Carga de football-data.co.uk: resultados + estadisticas de partido (tiros,
corners, faltas, tarjetas) para las "Main Leagues" europeas, gratis y sin
login, desde 2000/01. A diferencia de StatsBomb (eventos completos) o
openfootball (solo resultados), esta fuente da estadisticas agregadas por
partido -- justo lo que falta para correr el modelo de corners en mas
temporadas de las que StatsBomb tiene liberadas.

Nota: Liga MX NO esta disponible aqui -- football-data.co.uk solo trae
resultados + momios para las ligas "extra" (fuera de Europa), sin
estadisticas de partido. Confirmado revisando su CSV de Mexico.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "footballdata"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEASON_CODES = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2000, 2025)]  # 0001 .. 2425

# division code (en football-data.co.uk) -> (nombre de competicion en nuestra base, competition_id sintetico)
DIVISIONS = {
    "E0": ("Premier League", 900010),
    "SP1": ("La Liga", 900011),
    "D1": ("1. Bundesliga", 900012),
    "I1": ("Serie A", 900013),
    "F1": ("Ligue 1", 900014),
}

# nombres cortos de football-data.co.uk -> nombres largos (convencion StatsBomb),
# por division, para que un mismo equipo se reconozca como el mismo a traves de fuentes.
TEAM_NAME_MAPS: dict[str, dict[str, str]] = {
    "E0": {
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
    },
    "SP1": {
        "Alaves": "Deportivo Alavés", "Albacete": "Albacete", "Almeria": "Almería",
        "Ath Bilbao": "Athletic Club", "Ath Madrid": "Atlético Madrid", "Barcelona": "Barcelona",
        "Betis": "Real Betis", "Cadiz": "Cádiz", "Celta": "Celta Vigo", "Cordoba": "Córdoba CF",
        "Eibar": "Eibar", "Elche": "Elche", "Espanol": "Espanyol", "Getafe": "Getafe",
        "Gimnastic": "Gimnàstic Tarragona", "Girona": "Girona", "Granada": "Granada",
        "Hercules": "Hércules", "Huesca": "Huesca", "La Coruna": "RC Deportivo La Coruña",
        "Las Palmas": "Las Palmas", "Leganes": "Leganés", "Levante": "Levante UD",
        "Malaga": "Málaga", "Mallorca": "Mallorca", "Murcia": "Real Murcia CF",
        "Numancia": "CD Numancia de Soria", "Osasuna": "Osasuna", "Oviedo": "Real Oviedo",
        "Real Madrid": "Real Madrid", "Recreativo": "Recreativo Huelva",
        "Santander": "Racing Santander", "Sevilla": "Sevilla", "Sociedad": "Real Sociedad",
        "Sp Gijon": "Sporting Gijón", "Tenerife": "Tenerife", "Valencia": "Valencia",
        "Valladolid": "Real Valladolid", "Vallecano": "Rayo Vallecano", "Villarreal": "Villarreal",
        "Xerez": "Xerez", "Zaragoza": "Real Zaragoza",
    },
    "D1": {
        "Aachen": "Alemannia Aachen", "Augsburg": "Augsburg", "Bayern Munich": "Bayern Munich",
        "Bielefeld": "Arminia Bielefeld", "Bochum": "Bochum", "Braunschweig": "Eintracht Braunschweig",
        "Cottbus": "Energie Cottbus", "Darmstadt": "Darmstadt 98", "Dortmund": "Borussia Dortmund",
        "Duisburg": "MSV Duisburg", "Ein Frankfurt": "Eintracht Frankfurt", "FC Koln": "FC Köln",
        "Fortuna Dusseldorf": "Fortuna Düsseldorf", "Freiburg": "Freiburg",
        "Greuther Furth": "Greuther Fürth", "Hamburg": "Hamburger SV", "Hannover": "Hannover 96",
        "Hansa Rostock": "Hansa Rostock", "Heidenheim": "FC Heidenheim", "Hertha": "Hertha Berlin",
        "Hoffenheim": "Hoffenheim", "Holstein Kiel": "Holstein Kiel", "Ingolstadt": "Ingolstadt",
        "Kaiserslautern": "Kaiserslautern", "Karlsruhe": "Karlsruher SC", "Leverkusen": "Bayer Leverkusen",
        "M'gladbach": "Borussia Mönchengladbach", "Mainz": "FSV Mainz 05", "Munich 1860": "1860 Munich",
        "Nurnberg": "Nürnberg", "Paderborn": "Paderborn", "RB Leipzig": "RB Leipzig",
        "Schalke 04": "Schalke 04", "St Pauli": "St. Pauli", "Stuttgart": "VfB Stuttgart",
        "Union Berlin": "Union Berlin", "Unterhaching": "Unterhaching", "Werder Bremen": "Werder Bremen",
        "Wolfsburg": "Wolfsburg",
    },
    "I1": {
        "Ancona": "Ancona", "Ascoli": "Ascoli", "Atalanta": "Atalanta", "Bari": "Bari",
        "Benevento": "Benevento", "Bologna": "Bologna", "Brescia": "Brescia", "Cagliari": "Cagliari",
        "Carpi": "Carpi", "Catania": "Catania", "Cesena": "Cesena", "Chievo": "Chievo",
        "Como": "Como", "Cremonese": "Cremonese", "Crotone": "Crotone", "Empoli": "Empoli",
        "Fiorentina": "Fiorentina", "Frosinone": "Frosinone", "Genoa": "Genoa", "Inter": "Inter Milan",
        "Juventus": "Juventus", "Lazio": "Lazio", "Lecce": "Lecce", "Livorno": "Livorno",
        "Messina": "Messina", "Milan": "AC Milan", "Modena": "Modena", "Monza": "Monza",
        "Napoli": "Napoli", "Novara": "Novara", "Palermo": "Palermo", "Parma": "Parma",
        "Perugia": "Perugia", "Pescara": "Pescara", "Piacenza": "Piacenza", "Reggina": "Reggina",
        "Roma": "AS Roma", "Salernitana": "Salernitana", "Sampdoria": "Sampdoria",
        "Sassuolo": "Sassuolo", "Siena": "Siena", "Spal": "SPAL", "Spezia": "Spezia",
        "Torino": "Torino", "Treviso": "Treviso", "Udinese": "Udinese", "Venezia": "Venezia",
        "Verona": "Hellas Verona", "Vicenza": "Vicenza",
    },
    "F1": {
        "Ajaccio": "AC Ajaccio", "Ajaccio GFCO": "Gazélec Ajaccio", "Amiens": "Amiens",
        "Angers": "Angers", "Arles": "Arles-Avignon", "Auxerre": "Auxerre", "Bastia": "Bastia",
        "Bordeaux": "Bordeaux", "Boulogne": "Boulogne", "Brest": "Stade Brestois",
        "Caen": "Stade Malherbe Caen", "Clermont": "Clermont Foot", "Dijon": "Dijon",
        "Evian Thonon Gaillard": "Evian Thonon Gaillard", "Grenoble": "Grenoble",
        "Guingamp": "Guingamp", "Istres": "Istres", "Le Havre": "Le Havre", "Le Mans": "Le Mans",
        "Lens": "Lens", "Lille": "Lille", "Lorient": "Lorient", "Lyon": "Lyon",
        "Marseille": "Olympique de Marseille", "Metz": "Metz", "Monaco": "AS Monaco",
        "Montpellier": "Montpellier", "Nancy": "Nancy", "Nantes": "Nantes", "Nice": "OGC Nice",
        "Nimes": "Nimes", "Paris SG": "Paris Saint-Germain", "Reims": "Stade de Reims",
        "Rennes": "Rennes", "Sedan": "Sedan", "Sochaux": "Sochaux", "St Etienne": "Saint-Étienne",
        "Strasbourg": "Strasbourg", "Toulouse": "Toulouse", "Troyes": "Troyes",
        "Valenciennes": "Valenciennes",
    },
}

COL_MAP = {
    "HC": "home_corners", "AC": "away_corners",
    "HS": "home_shots", "AS": "away_shots",
    "HF": "home_fouls_committed", "AF": "away_fouls_committed",
}


def _season_name(code: str) -> str:
    y1 = 2000 + int(code[:2])
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
    competition_name, competition_id = DIVISIONS[division]
    name_map = TEAM_NAME_MAPS.get(division, {})
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
            "home_team": df["HomeTeam"].map(name_map).fillna(df["HomeTeam"]),
            "away_team": df["AwayTeam"].map(name_map).fillna(df["AwayTeam"]),
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
    out["match_id"] = competition_id * 1_000_000 + out.index
    out["competition_id"] = competition_id
    season_ids = {s: competition_id * 100 + i for i, s in enumerate(sorted(out["season_name"].unique()))}
    out["season_id"] = out["season_name"].map(season_ids)
    out["data_source"] = "football-data.co.uk"
    return out
