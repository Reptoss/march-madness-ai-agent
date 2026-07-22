"""Use a saved model to predict a matchup between two tournament teams."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.modeling import DEFAULT_MODEL_PATH


@dataclass(frozen=True)
class MatchupPrediction:
    """A win probability expressed in the user's requested team order."""

    season: int
    team_a_id: int
    team_a_name: str
    team_b_id: int
    team_b_name: str
    team_a_win_probability: float

    @property
    def predicted_winner(self) -> str:
        if self.team_a_win_probability >= 0.5:
            return self.team_a_name
        return self.team_b_name


def load_model_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load a locally generated model and verify its required metadata."""

    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run python trainModel.py first."
        )
    bundle = joblib.load(model_path)
    required = {"model", "feature_columns", "trained_through_season"}
    missing = required - set(bundle)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Model bundle is missing metadata: {missing_text}")
    return bundle


def resolve_team_id(teams: pd.DataFrame, team: str | int) -> int:
    """Resolve an exact team name or numeric TeamID into a TeamID."""

    if isinstance(team, int) or str(team).strip().isdigit():
        team_id = int(team)
        if team_id not in set(teams["TeamID"]):
            raise ValueError(f"Unknown TeamID: {team_id}")
        return team_id

    query = str(team).strip().casefold()
    name_lookup = {
        str(name).casefold(): int(team_id)
        for team_id, name in teams[["TeamID", "TeamName"]].itertuples(index=False)
    }
    if query in name_lookup:
        return name_lookup[query]

    display_names = [str(name) for name in teams["TeamName"]]
    normalized_to_display = {name.casefold(): name for name in display_names}
    suggestions = get_close_matches(query, normalized_to_display, n=3, cutoff=0.6)
    suggestion_text = ", ".join(normalized_to_display[name] for name in suggestions)
    if suggestion_text:
        raise ValueError(f"Unknown team '{team}'. Did you mean: {suggestion_text}?")
    raise ValueError(f"Unknown team: {team}")


def _team_name(teams: pd.DataFrame, team_id: int) -> str:
    return str(teams.loc[teams["TeamID"].eq(team_id), "TeamName"].iloc[0])


def _canonical_feature_row(
    team_seasons: pd.DataFrame,
    feature_columns: list[str],
    season: int,
    lower_team_id: int,
    higher_team_id: int,
) -> pd.DataFrame:
    season_rows = team_seasons.loc[team_seasons["Season"].eq(season)].set_index(
        "TeamID"
    )
    missing_teams = {
        team_id
        for team_id in (lower_team_id, higher_team_id)
        if team_id not in season_rows.index
    }
    if missing_teams:
        missing_text = ", ".join(str(team_id) for team_id in sorted(missing_teams))
        raise ValueError(f"No team-season features for TeamID values: {missing_text}")

    row: dict[str, float] = {}
    missing_features: list[str] = []
    for feature_column in feature_columns:
        source_column = feature_column.removeprefix("Diff_")
        if source_column not in season_rows.columns:
            missing_features.append(source_column)
            continue
        lower_value = season_rows.at[lower_team_id, source_column]
        higher_value = season_rows.at[higher_team_id, source_column]
        if pd.isna(lower_value) or pd.isna(higher_value):
            missing_features.append(source_column)
            continue
        row[feature_column] = float(lower_value - higher_value)

    if missing_features:
        missing_text = ", ".join(sorted(set(missing_features)))
        raise ValueError(
            "The matchup lacks required pre-tournament features: " + missing_text
        )
    return pd.DataFrame([row], columns=feature_columns)


def predict_matchup(
    bundle: dict[str, Any],
    team_seasons: pd.DataFrame,
    teams: pd.DataFrame,
    season: int,
    team_a: str | int,
    team_b: str | int,
) -> MatchupPrediction:
    """Predict Team A's win probability against Team B."""

    team_a_id = resolve_team_id(teams, team_a)
    team_b_id = resolve_team_id(teams, team_b)
    if team_a_id == team_b_id:
        raise ValueError("Choose two different teams")

    lower_team_id, higher_team_id = sorted((team_a_id, team_b_id))
    features = _canonical_feature_row(
        team_seasons,
        list(bundle["feature_columns"]),
        int(season),
        lower_team_id,
        higher_team_id,
    )
    lower_team_probability = float(bundle["model"].predict_proba(features)[0, 1])
    team_a_probability = (
        lower_team_probability
        if team_a_id == lower_team_id
        else 1.0 - lower_team_probability
    )
    return MatchupPrediction(
        season=int(season),
        team_a_id=team_a_id,
        team_a_name=_team_name(teams, team_a_id),
        team_b_id=team_b_id,
        team_b_name=_team_name(teams, team_b_id),
        team_a_win_probability=team_a_probability,
    )
