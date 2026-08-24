"""Reprocesa todas las competencias-temporada de StatsBomb ya sincronizadas
para agregar el conteo de corners (nuevo campo en team_features.py). Usa el
cache de eventos en disco -- no vuelve a descargar nada."""
import sys
import time

import pandas as pd

sys.path.insert(0, ".")

from src.team_database import load_team_database, sync_competition_season

wide = load_team_database()
pairs = (
    wide[~wide["competition_id"].isin([900001, 900002])]
    [["competition_id", "season_id", "competition_name"]]
    .drop_duplicates()
    .sort_values(["competition_name", "season_id"])
)

t0 = time.time()
for i, (_, row) in enumerate(pairs.iterrows(), 1):
    cid, sid, name = int(row["competition_id"]), int(row["season_id"]), row["competition_name"]
    out = sync_competition_season(cid, sid)
    n = len(out)
    has_corners = "home_corners" in out.columns and out["home_corners"].notna().any() if n else False
    print(f"[{i}/{len(pairs)}] {name} (cid={cid}, sid={sid}): {n} partidos, corners={'ok' if has_corners else 'NO'}  ({time.time()-t0:.0f}s)")

print("Listo.")
