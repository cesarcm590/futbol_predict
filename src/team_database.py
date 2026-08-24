"""Base de datos estandarizada de partidos-equipo, acumulada entre TODAS las
ligas/temporadas que se vayan sincronizando. Una fila por partido, con
prefijos home_/away_, igual que la base final de la metodologia de basquetball.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import get_events, get_matches
from src.standings import compute_dynamic_standings
from src.team_features import build_team_match_row

DB_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "team_matches.parquet"

# Torneos de eliminacion / grupos: no tienen una tabla de liga continua, asi
# que se excluyen del ranking dinamico (aunque si entran a la base para PCA/k-means).
_KNOCKOUT_MARKERS = [
    "world cup", "euro", "champions league", "copa america",
    "cup of nations", "copa del rey", "europa league", "u20",
]


def is_league_format(competition_name: str) -> bool:
    name = str(competition_name).lower()
    return not any(marker in name for marker in _KNOCKOUT_MARKERS)


def sync_competition_season(competition_id: int, season_id: int) -> pd.DataFrame:
    """Descarga (o reusa cache local) los partidos de una competicion-temporada,
    construye las filas equipo-partido y las agrega a la base estandarizada."""
    matches = get_matches(competition_id, season_id)
    if matches.empty:
        return pd.DataFrame()

    rows = []
    for _, mrow in matches.iterrows():
        try:
            events = get_events(int(mrow["match_id"]))
        except Exception:
            continue
        if events.empty:
            continue
        rows.append(build_team_match_row(events, mrow.to_dict()))

    team_rows = pd.DataFrame(rows)
    if team_rows.empty:
        return team_rows

    comp_name = matches["competition_name"].iloc[0] if "competition_name" in matches.columns else ""
    if is_league_format(comp_name):
        standings = compute_dynamic_standings(matches[["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]])
        team_rows = team_rows.merge(standings, on="match_id", how="left")

    team_rows["data_source"] = "StatsBomb"
    _append_to_db(team_rows)
    return team_rows


FOOTBALLDATA_DIVISIONS = ["E0", "SP1", "D1", "I1", "F1"]


def sync_footballdata(division: str = "E0") -> pd.DataFrame:
    """football-data.co.uk: resultados + tiros/corners/faltas por partido,
    gratis desde 2000/01 -- se usa para sumar temporadas donde StatsBomb no
    tiene datos liberados (mismo equipo, mas historia), sin duplicar
    temporadas que ya existen via StatsBomb."""
    from src.footballdata_loader import load_footballdata

    rows = load_footballdata(division)
    if rows.empty:
        return rows

    comp_name = rows["competition_name"].iloc[0]
    existing = load_team_database()
    already = set()
    if not existing.empty:
        already = set(existing[existing["competition_name"] == comp_name]["season_name"].unique())

    rows = rows[~rows["season_name"].isin(already)].copy()
    if rows.empty:
        return rows

    all_new = []
    for season_name, season_matches in rows.groupby("season_name"):
        standings = compute_dynamic_standings(
            season_matches[["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]]
        )
        merged = season_matches.merge(standings, on="match_id", how="left")
        all_new.append(merged)

    team_rows = pd.concat(all_new, ignore_index=True)
    team_rows["match_date"] = pd.to_datetime(team_rows["match_date"]).dt.strftime("%Y-%m-%d")
    _append_to_db(team_rows)
    return team_rows


def _append_to_db(new_rows: pd.DataFrame) -> None:
    if new_rows.empty:
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        existing = pd.read_parquet(DB_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset="match_id", keep="last")
    else:
        combined = new_rows
    combined.to_parquet(DB_PATH, index=False)


def sync_ligamx() -> pd.DataFrame:
    """Liga MX via openfootball: solo resultados (sin datos de evento), asi
    que aqui solo se calcula el ranking dinamico -- no hay stats de equipo
    tipo StatsBomb (esas columnas quedan NaN para estas filas)."""
    from src.ligamx_loader import load_ligamx_matches

    matches = load_ligamx_matches()
    if matches.empty:
        return matches

    rows = matches.rename(columns={"match_date": "match_date"}).copy()
    all_rows = []
    for season_name, season_matches in rows.groupby("season_name"):
        standings = compute_dynamic_standings(
            season_matches[["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]]
        )
        merged = season_matches.merge(standings, on="match_id", how="left")
        all_rows.append(merged)

    team_rows = pd.concat(all_rows, ignore_index=True)
    team_rows["match_date"] = pd.to_datetime(team_rows["match_date"]).dt.strftime("%Y-%m-%d")
    team_rows["data_source"] = "openfootball"
    _append_to_db(team_rows)
    return team_rows


def sync_nba() -> pd.DataFrame:
    """NBA via basketball-reference.com (box scores 'Team Totals' provistos
    por el usuario, ver src/nba_loader.py) -- ya trae estadisticas ricas de
    equipo por partido (FG%, ORtg, DRtg, etc.), a diferencia de las fuentes
    de solo-resultados como openfootball."""
    from src.nba_loader import load_nba_matches

    matches = load_nba_matches()
    if matches.empty:
        return matches

    all_rows = []
    for season_name, season_matches in matches.groupby("season_name"):
        standings = compute_dynamic_standings(
            season_matches[["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]]
        )
        merged = season_matches.merge(standings, on="match_id", how="left")
        all_rows.append(merged)

    team_rows = pd.concat(all_rows, ignore_index=True)
    team_rows["match_date"] = pd.to_datetime(team_rows["match_date"]).dt.strftime("%Y-%m-%d")
    _append_to_db(team_rows)
    return team_rows


def sync_nba_season(season: str) -> pd.DataFrame:
    """Actualiza/completa UNA temporada de NBA via nba_api (stats.nba.com) --
    reemplaza cualquier fila previa de esa misma temporada (p.ej. las
    parciales que vienen del CSV de basketball-reference via Drive, que
    llegan hasta una fecha de corte) para no duplicar partidos."""
    from src.nba_api_loader import load_nba_season

    matches = load_nba_season(season)
    if matches.empty:
        return matches

    standings = compute_dynamic_standings(
        matches[["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]]
    )
    team_rows = matches.merge(standings, on="match_id", how="left")
    team_rows["match_date"] = pd.to_datetime(team_rows["match_date"]).dt.strftime("%Y-%m-%d")

    existing = load_team_database()
    if not existing.empty:
        keep = ~((existing["competition_name"] == "NBA") & (existing["season_name"] == season))
        existing = existing[keep]
    combined = pd.concat([existing, team_rows], ignore_index=True, sort=False) if not existing.empty else team_rows
    combined = combined.drop_duplicates(subset="match_id", keep="last")
    combined.to_parquet(DB_PATH, index=False)
    return team_rows


def load_team_database() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DB_PATH)
    if "data_source" not in df.columns:
        df["data_source"] = pd.NA
    unknown = df["data_source"].isna()
    df.loc[unknown & (df["competition_id"] == 900001), "data_source"] = "openfootball"
    df.loc[unknown & df["competition_id"].isin([900002]), "data_source"] = "manual (Wikipedia/ESPN)"
    df.loc[unknown & ~df["competition_id"].isin([900001, 900002]), "data_source"] = "StatsBomb"
    return df


_SHARED_COLS = ["match_id", "competition_id", "season_id", "competition_name", "season_name", "match_date", "total_goals", "data_source"]


def to_team_perspective(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Reacomoda la base ancha (1 fila = 1 partido, home_/away_) a 1 fila = 1
    equipo-partido: cada partido aparece 2 veces, una desde la perspectiva de
    cada equipo (team/opponent, is_home, goals_for/against, stats propias sin
    prefijo, stats del rival con prefijo opp_). Esto permite filtrar por un
    solo equipo y ver su trayectoria partido a partido en la temporada.
    """
    if team_matches.empty:
        return pd.DataFrame()

    def build_side(is_home: bool) -> pd.DataFrame:
        own, opp = ("home_", "away_") if is_home else ("away_", "home_")
        goals_for = team_matches["home_score"] if is_home else team_matches["away_score"]
        goals_against = team_matches["away_score"] if is_home else team_matches["home_score"]

        # Se arma todo en un dict y se concatena UNA sola vez -- asignar columna
        # por columna sobre un DataFrame ancho (60+ columnas) lo fragmenta y
        # cada insercion se vuelve mas lenta que la anterior.
        cols = {
            "team": team_matches["home_team"] if is_home else team_matches["away_team"],
            "opponent": team_matches["away_team"] if is_home else team_matches["home_team"],
            "is_home": int(is_home),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "result": np.select([goals_for > goals_against, goals_for == goals_against], ["W", "D"], default="L"),
        }
        for c in [c for c in team_matches.columns if c.startswith(own)]:
            cols[c[len(own):]] = team_matches[c]
        for c in [c for c in team_matches.columns if c.startswith(opp)]:
            cols[f"opp_{c[len(opp):]}"] = team_matches[c]

        shared = team_matches[[c for c in _SHARED_COLS if c in team_matches.columns]]
        df = pd.concat([shared, pd.DataFrame(cols, index=team_matches.index)], axis=1)

        if "win_pct_dynamic" in df.columns and "opp_win_pct_dynamic" in df.columns:
            df["diff_win_pct_dynamic"] = df["win_pct_dynamic"] - df["opp_win_pct_dynamic"]
        if "rank_dynamic" in df.columns and "opp_rank_dynamic" in df.columns:
            df["diff_rank_dynamic"] = df["opp_rank_dynamic"] - df["rank_dynamic"]
        return df

    out = pd.concat([build_side(True), build_side(False)], ignore_index=True, sort=False)
    return out.sort_values(["team", "match_date"]).reset_index(drop=True)


TEAM_ID_COLS = ["team", "opponent", "match_date", "result", "is_home", "competition_name", "season_name"]

TEAM_FEATURE_COLS = [
    "passes_attempted", "passes_completed", "pass_pct", "possession_pct",
    "progressive_passes", "progressive_carries", "shots", "xg_total", "corners",
    "tackles_won", "interceptions", "pressures", "fouls_committed", "touches",
    # NBA (basketball-reference "Team Totals" + nba_api para la temporada en curso;
    # ortg/drtg/tpar/ftr/usg_pct se quedan fuera porque nba_api no las trae y
    # calcularlas bien requiere estimar posesiones -- mejor no tenerlas que
    # tenerlas a medias, rellenas de NaN->0 para la temporada mas reciente.)
    "fg_pct", "tp_pct", "ft_pct", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf",
    "ts_pct", "efg_pct",
    "win_pct_dynamic", "rank_dynamic", "goal_diff_before",
]


def team_feature_cols(team_perspective_df: pd.DataFrame) -> list[str]:
    """Metricas de equipo disponibles, descartando las que no tienen señal
    real en el subconjunto filtrado (todo NaN o varianza cero) -- pasa con
    Liga MX, que solo trae goles/ranking dinamico, no stats de evento."""
    candidates = [c for c in TEAM_FEATURE_COLS if c in team_perspective_df.columns]
    return [c for c in candidates if team_perspective_df[c].notna().any() and team_perspective_df[c].std(skipna=True) > 0]
