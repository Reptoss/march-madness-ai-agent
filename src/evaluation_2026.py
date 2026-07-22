"""Evaluate the trained baseline against the completed 2026 tournament."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.data_loader import PROJECT_ROOT
from src.prediction import predict_matchup


DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT / "data" / "external" / "mens_2026_tournament_results.csv"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "generated"
REQUIRED_COLUMNS = {
    "Season",
    "Round",
    "Region",
    "WTeamID",
    "WTeamName",
    "WScore",
    "LTeamID",
    "LTeamName",
    "LScore",
}


@dataclass(frozen=True)
class Evaluation2026:
    predictions: pd.DataFrame
    round_metrics: pd.DataFrame
    overall_metrics: dict[str, float | int]


def load_2026_results(path: str | Path = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
    """Load and validate the fixed 67-game 2026 evaluation set."""

    results_path = Path(path).expanduser().resolve()
    if not results_path.is_file():
        raise FileNotFoundError(f"2026 results file not found: {results_path}")
    results = pd.read_csv(results_path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(results.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"2026 results are missing columns: {missing_text}")
    if len(results) != 67:
        raise ValueError(f"Expected 67 tournament games, found {len(results)}")
    if set(results["Season"]) != {2026}:
        raise ValueError("The 2026 evaluation file contains another season")
    if (results["WTeamID"] == results["LTeamID"]).any():
        raise ValueError("A 2026 game lists the same team on both sides")
    if (results["WScore"] <= results["LScore"]).any():
        raise ValueError("A 2026 winner has a non-winning score")
    if results.duplicated(["Round", "WTeamID", "LTeamID"]).any():
        raise ValueError("The 2026 results contain duplicate games")
    return results


def evaluate_2026(
    bundle: dict,
    team_seasons: pd.DataFrame,
    teams: pd.DataFrame,
    results: pd.DataFrame,
) -> Evaluation2026:
    """Predict all actual matchups without training on 2026 results."""

    if int(bundle["trained_through_season"]) >= 2026:
        raise ValueError("The model was trained on 2026 or later and cannot be a holdout")

    rows: list[dict] = []
    for game in results.itertuples(index=False):
        team_a_id, team_b_id = sorted((int(game.WTeamID), int(game.LTeamID)))
        prediction = predict_matchup(
            bundle,
            team_seasons,
            teams,
            2026,
            team_a_id,
            team_b_id,
        )
        actual_team_a_win = int(team_a_id == int(game.WTeamID))
        predicted_team_id = (
            team_a_id if prediction.team_a_win_probability >= 0.5 else team_b_id
        )
        rows.append(
            {
                "Season": 2026,
                "Round": game.Round,
                "Region": game.Region,
                "TeamAID": team_a_id,
                "TeamAName": prediction.team_a_name,
                "TeamBID": team_b_id,
                "TeamBName": prediction.team_b_name,
                "ActualWinnerID": int(game.WTeamID),
                "ActualWinnerName": game.WTeamName,
                "TeamAWin": actual_team_a_win,
                "PredictedProbability": prediction.team_a_win_probability,
                "PredictedWinnerID": predicted_team_id,
                "Correct": int(predicted_team_id == int(game.WTeamID)),
            }
        )

    predictions = pd.DataFrame(rows)
    actual = predictions["TeamAWin"]
    probability = predictions["PredictedProbability"]
    overall_metrics: dict[str, float | int] = {
        "Season": 2026,
        "Games": len(predictions),
        "Correct": int(predictions["Correct"].sum()),
        "Accuracy": float(accuracy_score(actual, probability >= 0.5)),
        "LogLoss": float(log_loss(actual, probability, labels=[0, 1])),
        "BrierScore": float(brier_score_loss(actual, probability)),
        "ModelTrainedThrough": int(bundle["trained_through_season"]),
    }
    round_metrics = (
        predictions.groupby("Round", sort=False)
        .agg(Games=("Correct", "size"), Correct=("Correct", "sum"), Accuracy=("Correct", "mean"))
        .reset_index()
    )
    return Evaluation2026(predictions, round_metrics, overall_metrics)


def write_2026_evaluation(
    evaluation: Evaluation2026,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> tuple[Path, Path, Path]:
    destination = Path(report_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / "2026_predictions.csv"
    rounds_path = destination / "2026_round_metrics.csv"
    summary_path = destination / "2026_summary.json"
    evaluation.predictions.to_csv(predictions_path, index=False)
    evaluation.round_metrics.to_csv(rounds_path, index=False)
    summary_path.write_text(
        json.dumps(evaluation.overall_metrics, indent=2) + "\n", encoding="utf-8"
    )
    return predictions_path, rounds_path, summary_path
