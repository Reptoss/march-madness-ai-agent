"""Create leakage-safe team and matchup features for model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data_loader import MensData, PROJECT_ROOT


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

BOX_SCORE_STATS = {
    "FGM": "FieldGoalsMade",
    "FGA": "FieldGoalsAttempted",
    "FGM3": "ThreePointersMade",
    "FGA3": "ThreePointersAttempted",
    "FTM": "FreeThrowsMade",
    "FTA": "FreeThrowsAttempted",
    "OR": "OffensiveRebounds",
    "DR": "DefensiveRebounds",
    "Ast": "Assists",
    "TO": "Turnovers",
    "Stl": "Steals",
    "Blk": "Blocks",
    "PF": "Fouls",
}

RATE_FEATURES = {
    "FieldGoalPct": ("FieldGoalsMade", "FieldGoalsAttempted"),
    "ThreePointPct": ("ThreePointersMade", "ThreePointersAttempted"),
    "FreeThrowPct": ("FreeThrowsMade", "FreeThrowsAttempted"),
}


@dataclass(frozen=True)
class FeatureData:
    """Processed tables used for training and future predictions."""

    team_seasons: pd.DataFrame
    tournament_matchups: pd.DataFrame


def _required_regular_season_columns() -> set[str]:
    columns = {
        "Season",
        "DayNum",
        "WTeamID",
        "WScore",
        "LTeamID",
        "LScore",
        "WLoc",
        "NumOT",
    }
    for short_name in BOX_SCORE_STATS:
        columns.update({f"W{short_name}", f"L{short_name}"})
    return columns


def _check_columns(table: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = required - set(table.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{table_name} is missing required columns: {missing_text}")


def _team_game_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Turn each winner/loser game row into two team-perspective rows."""

    _check_columns(
        results,
        _required_regular_season_columns(),
        "MRegularSeasonDetailedResults.csv",
    )

    common_columns = ["Season", "DayNum", "NumOT"]
    winners = results[common_columns].copy()
    winners["TeamID"] = results["WTeamID"]
    winners["OpponentID"] = results["LTeamID"]
    winners["Won"] = 1
    winners["Points"] = results["WScore"]
    winners["PointsAllowed"] = results["LScore"]
    winners["Location"] = results["WLoc"]

    losers = results[common_columns].copy()
    losers["TeamID"] = results["LTeamID"]
    losers["OpponentID"] = results["WTeamID"]
    losers["Won"] = 0
    losers["Points"] = results["LScore"]
    losers["PointsAllowed"] = results["WScore"]
    losers["Location"] = results["WLoc"].map({"H": "A", "A": "H", "N": "N"})

    for source_name, feature_name in BOX_SCORE_STATS.items():
        winners[feature_name] = results[f"W{source_name}"]
        losers[feature_name] = results[f"L{source_name}"]

    games = pd.concat([winners, losers], ignore_index=True)
    games["PointMargin"] = games["Points"] - games["PointsAllowed"]
    games["TotalRebounds"] = (
        games["OffensiveRebounds"] + games["DefensiveRebounds"]
    )
    return games


def build_team_season_features(
    regular_season_results: pd.DataFrame,
    tournament_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate detailed regular-season games into one row per team-season."""

    _check_columns(
        tournament_seeds,
        {"Season", "Seed", "TeamID"},
        "MNCAATourneySeeds.csv",
    )
    games = _team_game_rows(regular_season_results)
    group_keys = ["Season", "TeamID"]

    summed_columns = [
        "FieldGoalsMade",
        "FieldGoalsAttempted",
        "ThreePointersMade",
        "ThreePointersAttempted",
        "FreeThrowsMade",
        "FreeThrowsAttempted",
    ]
    average_columns = [
        "Points",
        "PointsAllowed",
        "PointMargin",
        "OffensiveRebounds",
        "DefensiveRebounds",
        "TotalRebounds",
        "Assists",
        "Turnovers",
        "Steals",
        "Blocks",
        "Fouls",
    ]

    grouped = games.groupby(group_keys, sort=True)
    features = grouped.agg(
        Games=("Won", "size"),
        Wins=("Won", "sum"),
        WinPct=("Won", "mean"),
        **{f"Avg{name}": (name, "mean") for name in average_columns},
        **{f"Total{name}": (name, "sum") for name in summed_columns},
    ).reset_index()

    for rate_name, (made_name, attempted_name) in RATE_FEATURES.items():
        features[rate_name] = (
            features[f"Total{made_name}"] / features[f"Total{attempted_name}"]
        )

    opponent_win_pct = features[["Season", "TeamID", "WinPct"]].rename(
        columns={"TeamID": "OpponentID", "WinPct": "OpponentWinPct"}
    )
    games_with_strength = games.merge(
        opponent_win_pct,
        on=["Season", "OpponentID"],
        how="left",
        validate="many_to_one",
    )
    schedule_strength = (
        games_with_strength.groupby(group_keys, sort=True)["OpponentWinPct"]
        .mean()
        .rename("StrengthOfSchedule")
        .reset_index()
    )
    features = features.merge(
        schedule_strength, on=group_keys, how="left", validate="one_to_one"
    )

    seeds = tournament_seeds[["Season", "TeamID", "Seed"]].copy()
    seeds["SeedNumber"] = pd.to_numeric(
        seeds["Seed"].str.extract(r"(\d+)", expand=False), errors="raise"
    )
    if seeds.duplicated(group_keys).any():
        raise ValueError("Tournament seeds contain duplicate team-season rows")

    features = features.merge(
        seeds, on=group_keys, how="left", validate="one_to_one"
    )
    return features.sort_values(group_keys).reset_index(drop=True)


def build_tournament_matchups(
    team_seasons: pd.DataFrame,
    tournament_results: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """Create labeled historical matchups using pre-tournament features."""

    _check_columns(
        tournament_results,
        {"Season", "DayNum", "WTeamID", "LTeamID"},
        "MNCAATourneyDetailedResults.csv",
    )
    _check_columns(teams, {"TeamID", "TeamName"}, "MTeams.csv")

    games = tournament_results[
        ["Season", "DayNum", "WTeamID", "LTeamID"]
    ].copy()
    games["TeamAID"] = games[["WTeamID", "LTeamID"]].min(axis=1)
    games["TeamBID"] = games[["WTeamID", "LTeamID"]].max(axis=1)
    games["TeamAWin"] = (games["WTeamID"] == games["TeamAID"]).astype(int)

    excluded = {"Season", "TeamID", "Seed"}
    feature_columns = [
        column
        for column in team_seasons.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(team_seasons[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric team-season features were available")

    side_a = team_seasons[["Season", "TeamID", *feature_columns]].rename(
        columns={
            "TeamID": "TeamAID",
            **{column: f"A_{column}" for column in feature_columns},
        }
    )
    side_b = team_seasons[["Season", "TeamID", *feature_columns]].rename(
        columns={
            "TeamID": "TeamBID",
            **{column: f"B_{column}" for column in feature_columns},
        }
    )
    matchups = games.merge(
        side_a, on=["Season", "TeamAID"], how="left", validate="many_to_one"
    ).merge(
        side_b, on=["Season", "TeamBID"], how="left", validate="many_to_one"
    )

    required_side_columns = [
        *(f"A_{column}" for column in feature_columns),
        *(f"B_{column}" for column in feature_columns),
    ]
    missing_rows = matchups[required_side_columns].isna().any(axis=1)
    if missing_rows.any():
        examples = matchups.loc[
            missing_rows, ["Season", "TeamAID", "TeamBID"]
        ].head(5)
        raise ValueError(
            "Tournament games are missing regular-season features:\n"
            f"{examples.to_string(index=False)}"
        )

    for column in feature_columns:
        matchups[f"Diff_{column}"] = matchups[f"A_{column}"] - matchups[f"B_{column}"]

    names = teams[["TeamID", "TeamName"]].drop_duplicates("TeamID")
    matchups = matchups.merge(
        names.rename(columns={"TeamID": "TeamAID", "TeamName": "TeamAName"}),
        on="TeamAID",
        how="left",
        validate="many_to_one",
    ).merge(
        names.rename(columns={"TeamID": "TeamBID", "TeamName": "TeamBName"}),
        on="TeamBID",
        how="left",
        validate="many_to_one",
    )

    identity_columns = [
        "Season",
        "DayNum",
        "TeamAID",
        "TeamAName",
        "TeamBID",
        "TeamBName",
        "TeamAWin",
    ]
    difference_columns = [f"Diff_{column}" for column in feature_columns]
    return matchups[identity_columns + difference_columns].sort_values(
        ["Season", "DayNum", "TeamAID", "TeamBID"]
    ).reset_index(drop=True)


def build_feature_data(data: MensData) -> FeatureData:
    """Run the complete regular-season-to-training-matchup pipeline."""

    team_seasons = build_team_season_features(
        data.regular_season_results, data.tournament_seeds
    )
    tournament_matchups = build_tournament_matchups(
        team_seasons, data.tournament_results, data.teams
    )
    return FeatureData(
        team_seasons=team_seasons,
        tournament_matchups=tournament_matchups,
    )


def write_feature_data(
    feature_data: FeatureData,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write reproducible processed tables and return their paths."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    team_path = destination / "mens_team_season_features.csv"
    matchup_path = destination / "mens_tournament_training.csv"
    feature_data.team_seasons.to_csv(team_path, index=False)
    feature_data.tournament_matchups.to_csv(matchup_path, index=False)
    return team_path, matchup_path
