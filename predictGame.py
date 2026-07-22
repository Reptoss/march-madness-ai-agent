"""Predict one matchup with the trained baseline model."""

import argparse

from src.data_loader import load_mens_data
from src.features import build_team_season_features
from src.prediction import load_model_bundle, predict_matchup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a men's March Madness matchup."
    )
    parser.add_argument("team_a", help="First team name or Kaggle TeamID")
    parser.add_argument("team_b", help="Second team name or Kaggle TeamID")
    parser.add_argument("--season", type=int, default=2026, help="Season to predict")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        raw_data = load_mens_data()
        team_seasons = build_team_season_features(
            raw_data.regular_season_results, raw_data.tournament_seeds
        )
        model_bundle = load_model_bundle()
        prediction = predict_matchup(
            model_bundle,
            team_seasons,
            raw_data.teams,
            args.season,
            args.team_a,
            args.team_b,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Prediction failed: {error}") from error

    print(f"{prediction.season} matchup prediction")
    print(
        f"  {prediction.team_a_name} win probability: "
        f"{prediction.team_a_win_probability:.1%}"
    )
    print(
        f"  {prediction.team_b_name} win probability: "
        f"{1.0 - prediction.team_a_win_probability:.1%}"
    )
    print(f"  Predicted winner: {prediction.predicted_winner}")
