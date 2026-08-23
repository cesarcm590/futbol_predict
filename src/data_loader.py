"""Carga de datos de StatsBomb Open Data con cache local en disco.

Los datos abiertos de StatsBomb son historicos y estaticos, asi que una vez
descargado un partido/competicion no vuelve a cambiar: cachear indefinidamente
en `data_cache/` es seguro y evita golpear GitHub en cada rerun de Streamlit.
"""
from pathlib import Path
import pickle

import pandas as pd
from statsbombpy import sb

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
MATCHES_DIR = CACHE_DIR / "matches"
EVENTS_DIR = CACHE_DIR / "events"
for d in (CACHE_DIR, MATCHES_DIR, EVENTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _cached(path: Path, fetch_fn):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    df = fetch_fn()
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


def get_competitions() -> pd.DataFrame:
    return _cached(CACHE_DIR / "competitions.pkl", lambda: sb.competitions())


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    path = MATCHES_DIR / f"{competition_id}_{season_id}.pkl"
    return _cached(path, lambda: sb.matches(competition_id=competition_id, season_id=season_id))


def get_events(match_id: int) -> pd.DataFrame:
    path = EVENTS_DIR / f"{match_id}.pkl"
    return _cached(path, lambda: sb.events(match_id=match_id))


def get_events_for_matches(match_ids: list[int]) -> pd.DataFrame:
    frames = []
    for mid in match_ids:
        try:
            frames.append(get_events(int(mid)))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
