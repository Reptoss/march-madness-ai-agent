"""Train and backtest the baseline men's tournament prediction model."""

from pathlib import Path

import pandas as pd

from src.data_loader import PROJECT_ROOT, load_mens_data
from src.features import build_feature_data, write_feature_data
from src.modeling import (
    backtest_by_season,
    save_model_bundle,
    train_final_model,
    write_backtest_report,
)


TRAINING_PATH = PROJECT_ROOT / "data" / "processed" / "mens_tournament_training.csv"


def load_or_build_training_data(path: Path = TRAINING_PATH) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)

    raw_data = load_mens_data()
    feature_data = build_feature_data(raw_data)
    write_feature_data(feature_data)
    return feature_data.tournament_matchups


if __name__ == "__main__":
    training_data = load_or_build_training_data()
    backtest = backtest_by_season(training_data)
    model, feature_columns = train_final_model(training_data)
    model_path = save_model_bundle(model, feature_columns, training_data)
    season_path, prediction_path, summary_path = write_backtest_report(backtest)

    print("Baseline model completed successfully")
    print("  Evaluation: chronological holdout of the latest five tournaments")
    print(f"  Test games: {backtest.overall_metrics['TestRows']:,}")
    print(f"  Accuracy: {backtest.overall_metrics['Accuracy']:.3f}")
    print(f"  Log loss: {backtest.overall_metrics['LogLoss']:.3f}")
    print(f"  Brier score: {backtest.overall_metrics['BrierScore']:.3f}")
    print(f"  Model: {model_path}")
    print(f"  Season metrics: {season_path}")
    print(f"  Predictions: {prediction_path}")
    print(f"  Summary: {summary_path}")
