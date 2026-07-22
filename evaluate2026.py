"""Score the trained model against all actual 2026 tournament games."""

from src.data_loader import load_mens_data
from src.evaluation_2026 import evaluate_2026, load_2026_results, write_2026_evaluation
from src.features import build_team_season_features
from src.prediction import load_model_bundle


if __name__ == "__main__":
    raw_data = load_mens_data()
    team_seasons = build_team_season_features(
        raw_data.regular_season_results, raw_data.tournament_seeds
    )
    bundle = load_model_bundle()
    results = load_2026_results()
    evaluation = evaluate_2026(bundle, team_seasons, raw_data.teams, results)
    predictions_path, rounds_path, summary_path = write_2026_evaluation(evaluation)

    metrics = evaluation.overall_metrics
    print("2026 holdout evaluation completed successfully")
    print(f"  Model trained through: {metrics['ModelTrainedThrough']}")
    print(f"  Games: {metrics['Games']}")
    print(f"  Correct winners: {metrics['Correct']}")
    print(f"  Accuracy: {metrics['Accuracy']:.3f}")
    print(f"  Log loss: {metrics['LogLoss']:.3f}")
    print(f"  Brier score: {metrics['BrierScore']:.3f}")
    print("\nAccuracy by round:")
    print(evaluation.round_metrics.to_string(index=False, formatters={"Accuracy": "{:.1%}".format}))
    print(f"\n  Predictions: {predictions_path}")
    print(f"  Round metrics: {rounds_path}")
    print(f"  Summary: {summary_path}")
