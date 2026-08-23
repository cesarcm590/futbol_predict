"""Sincroniza TODAS las competencias de formato liga de StatsBomb Open Data
hacia la base estandarizada local (data_cache/team_matches.parquet).

Uso: ./.venv/bin/python sync_all_leagues.py
"""
import sys
import time

from src.data_loader import get_competitions
from src.team_database import is_league_format, sync_competition_season

comps = get_competitions()
league = comps[comps["competition_name"].apply(is_league_format)]
league = league.sort_values(["competition_name", "season_name"])

total = len(league)
for i, (_, r) in enumerate(league.iterrows(), start=1):
    comp_id, season_id = int(r["competition_id"]), int(r["season_id"])
    label = f"{r['competition_name']} {r['season_name']}"
    t0 = time.time()
    try:
        rows = sync_competition_season(comp_id, season_id)
        n = len(rows)
        status = "ok"
    except Exception as e:
        n = 0
        status = f"ERROR: {e}"
    print(f"[{i}/{total}] {label} -> {n} partidos ({status}) en {time.time()-t0:.1f}s", flush=True)

print("Listo.", flush=True)
