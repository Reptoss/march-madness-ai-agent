"""Train and evaluate a baseline March Madness win-probability model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import PROJECT_ROOT


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mens_logistic_regression.joblib"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "generated"
TARGET_COLUMN = "TeamAWin"
FEATURE_PREFIX = "Diff_"


@dataclass(frozen=True)
class BacktestResult:
    """Predictions and metrics from chronological season holdouts."""

    predictions: pd.DataFrame
    season_metrics: pd.DataFrame
    overall_metrics: dict[str, float | int]


def get_feature_columns(matchups: pd.DataFrame) -> list[str]:
    """Return the matchup-difference columns used by the model."""

    feature_columns = sorted(
        column for column in matchups.columns if column.startswith(FEATURE_PREFIX)
    )
    if not feature_columns:
        raise ValueError("No Diff_ feature columns were found in the training data")
    return feature_columns


def _validate_training_data(
    matchups: pd.DataFrame, feature_columns: list[str]
) -> None:
    required = {"Season", TARGET_COLUMN, *feature_columns}
    missing = required - set(matchups.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Training data is missing columns: {missing_text}")
    if matchups.empty:
        raise ValueError("Training data is empty")
    if matchups[feature_columns].isna().any().any():
        raise ValueError("Training features contain missing values")
    labels = set(matchups[TARGET_COLUMN].unique())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError("TeamAWin must contain both binary labels 0 and 1")


def create_baseline_model() -> Pipeline:
    """Create a scaled logistic regression probability model."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2_000, random_state=42),
            ),
        ]
    )


def _score_predictions(actual: pd.Series, probability: pd.Series) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "LogLoss": float(log_loss(actual, probability, labels=[0, 1])),
        "BrierScore": float(brier_score_loss(actual, probability)),
    }


def backtest_by_season(
    matchups: pd.DataFrame,
    *,
    test_seasons: list[int] | None = None,
) -> BacktestResult:
    """Evaluate on whole seasons after training only on earlier seasons."""

    feature_columns = get_feature_columns(matchups)
    _validate_training_data(matchups, feature_columns)

    available_seasons = sorted(int(season) for season in matchups["Season"].unique())
    if test_seasons is None:
        test_seasons = available_seasons[-5:]
    unavailable = set(test_seasons) - set(available_seasons)
    if unavailable:
        missing_text = ", ".join(str(season) for season in sorted(unavailable))
        raise ValueError(f"Requested test seasons are unavailable: {missing_text}")

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int]] = []
    identity_columns = [
        column
        for column in [
            "Season",
            "DayNum",
            "TeamAID",
            "TeamAName",
            "TeamBID",
            "TeamBName",
            TARGET_COLUMN,
        ]
        if column in matchups.columns
    ]

    for season in sorted(test_seasons):
        train = matchups.loc[matchups["Season"] < season]
        test = matchups.loc[matchups["Season"] == season]
        if train.empty:
            raise ValueError(f"No earlier seasons are available to train for {season}")
        if train[TARGET_COLUMN].nunique() < 2:
            raise ValueError(f"Training rows before {season} do not contain both labels")

        model = create_baseline_model()
        model.fit(train[feature_columns], train[TARGET_COLUMN])
        probabilities = pd.Series(
            model.predict_proba(test[feature_columns])[:, 1],
            index=test.index,
            name="PredictedProbability",
        )

        scored = _score_predictions(test[TARGET_COLUMN], probabilities)
        metric_rows.append(
            {
                "TestSeason": season,
                "TrainingRows": len(train),
                "TestRows": len(test),
                "TrainThroughSeason": int(train["Season"].max()),
                **scored,
            }
        )
        predictions = test[identity_columns].copy()
        predictions["PredictedProbability"] = probabilities
        predictions["PredictedWinner"] = (probabilities >= 0.5).astype(int)
        predictions["TrainThroughSeason"] = int(train["Season"].max())
        prediction_frames.append(predictions)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    overall_scores = _score_predictions(
        all_predictions[TARGET_COLUMN], all_predictions["PredictedProbability"]
    )
    overall_metrics: dict[str, float | int] = {
        "TestSeasons": len(test_seasons),
        "TestRows": len(all_predictions),
        **overall_scores,
    }
    return BacktestResult(
        predictions=all_predictions,
        season_metrics=pd.DataFrame(metric_rows),
        overall_metrics=overall_metrics,
    )


def train_final_model(matchups: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """Train the baseline on every completed historical tournament."""

    feature_columns = get_feature_columns(matchups)
    _validate_training_data(matchups, feature_columns)
    model = create_baseline_model()
    model.fit(matchups[feature_columns], matchups[TARGET_COLUMN])
    return model, feature_columns


def save_model_bundle(
    model: Pipeline,
    feature_columns: list[str],
    matchups: pd.DataFrame,
    path: str | Path = DEFAULT_MODEL_PATH,
) -> Path:
    """Save the fitted model together with the metadata needed for prediction."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "model": model,
        "feature_columns": feature_columns,
        "trained_through_season": int(matchups["Season"].max()),
        "training_rows": len(matchups),
        "target_column": TARGET_COLUMN,
    }
    joblib.dump(bundle, destination)
    return destination


def write_backtest_report(
    result: BacktestResult,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> tuple[Path, Path, Path]:
    """Write season metrics, predictions, and overall metrics."""

    destination = Path(report_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    season_path = destination / "baseline_season_metrics.csv"
    prediction_path = destination / "baseline_predictions.csv"
    summary_path = destination / "baseline_summary.json"

    result.season_metrics.to_csv(season_path, index=False)
    result.predictions.to_csv(prediction_path, index=False)
    summary_path.write_text(
        json.dumps(result.overall_metrics, indent=2) + "\n", encoding="utf-8"
    )
    return season_path, prediction_path, summary_path
