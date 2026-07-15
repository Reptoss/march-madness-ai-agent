"""Load and validate the raw men's March Madness competition data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REQUIRED_COLUMNS = {
    "MTeams.csv": {"TeamID", "TeamName", "FirstD1Season", "LastD1Season"},
    "MRegularSeasonDetailedResults.csv": {
        "Season",
        "DayNum",
        "WTeamID",
        "WScore",
        "LTeamID",
        "LScore",
        "WLoc",
        "NumOT",
    },
    "MNCAATourneyDetailedResults.csv": {
        "Season",
        "DayNum",
        "WTeamID",
        "WScore",
        "LTeamID",
        "LScore",
    },
    "MNCAATourneySeeds.csv": {"Season", "Seed", "TeamID"},
    "MSeasons.csv": {"Season", "DayZero", "RegionW", "RegionX", "RegionY", "RegionZ"},
    "MMasseyOrdinals.csv": {
        "Season",
        "RankingDayNum",
        "SystemName",
        "TeamID",
        "OrdinalRank",
    },
}


@dataclass(frozen=True)
class MensData:
    """The core tables used to build a men's tournament predictor."""

    teams: pd.DataFrame
    regular_season_results: pd.DataFrame
    tournament_results: pd.DataFrame
    tournament_seeds: pd.DataFrame
    seasons: pd.DataFrame
    rankings: pd.DataFrame | None = None


def _read_table(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {filename}. Extract the Kaggle competition ZIP into {data_dir}."
        )

    table = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS[filename] - set(table.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{filename} is missing required columns: {missing}")

    return table


def load_mens_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    include_rankings: bool = False,
) -> MensData:
    """Load the core men's tables from a local raw-data directory.

    Massey rankings are optional because that table is much larger than the
    other inputs and is not needed for a basic setup check.
    """

    raw_dir = Path(data_dir).expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. "
            "Create data/raw and extract the Kaggle competition ZIP there."
        )

    rankings = (
        _read_table(raw_dir, "MMasseyOrdinals.csv") if include_rankings else None
    )

    data = MensData(
        teams=_read_table(raw_dir, "MTeams.csv"),
        regular_season_results=_read_table(
            raw_dir, "MRegularSeasonDetailedResults.csv"
        ),
        tournament_results=_read_table(
            raw_dir, "MNCAATourneyDetailedResults.csv"
        ),
        tournament_seeds=_read_table(raw_dir, "MNCAATourneySeeds.csv"),
        seasons=_read_table(raw_dir, "MSeasons.csv"),
        rankings=rankings,
    )
    _validate_team_ids(data)
    return data


def _validate_team_ids(data: MensData) -> None:
    """Ensure every game and seed references a team in MTeams.csv."""

    known_ids = set(data.teams["TeamID"])
    referenced_ids = set(data.regular_season_results["WTeamID"])
    referenced_ids.update(data.regular_season_results["LTeamID"])
    referenced_ids.update(data.tournament_results["WTeamID"])
    referenced_ids.update(data.tournament_results["LTeamID"])
    referenced_ids.update(data.tournament_seeds["TeamID"])

    unknown_ids = referenced_ids - known_ids
    if unknown_ids:
        preview = ", ".join(str(team_id) for team_id in sorted(unknown_ids)[:10])
        raise ValueError(f"Data references unknown TeamID values: {preview}")


def _season_span(table: pd.DataFrame) -> str:
    return f"{int(table['Season'].min())}-{int(table['Season'].max())}"


def print_summary(data: MensData) -> None:
    """Print a compact confirmation that the raw data loaded successfully."""

    print("Men's March Madness data loaded successfully")
    print(f"  Teams: {len(data.teams):,}")
    print(
        "  Regular-season games: "
        f"{len(data.regular_season_results):,} "
        f"({_season_span(data.regular_season_results)})"
    )
    print(
        "  Tournament games: "
        f"{len(data.tournament_results):,} "
        f"({_season_span(data.tournament_results)})"
    )
    print(
        f"  Tournament seeds: {len(data.tournament_seeds):,} "
        f"({_season_span(data.tournament_seeds)})"
    )
    if data.rankings is not None:
        print(
            f"  Ranking records: {len(data.rankings):,} "
            f"({_season_span(data.rankings)})"
        )
