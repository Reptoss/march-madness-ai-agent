import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.modeling import (
    backtest_by_season,
    get_feature_columns,
    save_model_bundle,
    train_final_model,
)


def _training_rows() -> pd.DataFrame:
    rows = []
    for season in range(2018, 2024):
        for game_number in range(6):
            strength = float(game_number - 2)
            rows.append(
                {
                    "Season": season,
                    "DayNum": 136 + game_number,
                    "TeamAID": 1000 + game_number,
                    "TeamBID": 1100 + game_number,
                    "TeamAWin": int(strength > 0),
                    "Diff_WinPct": strength + (season - 2020) * 0.01,
                    "Diff_SeedNumber": -strength,
                }
            )
    return pd.DataFrame(rows)


class ModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matchups = _training_rows()

    def test_feature_columns_only_include_differences(self) -> None:
        self.assertEqual(
            get_feature_columns(self.matchups),
            ["Diff_SeedNumber", "Diff_WinPct"],
        )

    def test_backtest_never_trains_on_test_or_future_seasons(self) -> None:
        result = backtest_by_season(self.matchups, test_seasons=[2022, 2023])

        self.assertEqual(list(result.season_metrics["TestSeason"]), [2022, 2023])
        self.assertTrue(
            (
                result.season_metrics["TrainThroughSeason"]
                < result.season_metrics["TestSeason"]
            ).all()
        )
        self.assertEqual(result.overall_metrics["TestRows"], 12)

    def test_saved_bundle_contains_model_metadata(self) -> None:
        model, feature_columns = train_final_model(self.matchups)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.joblib"
            save_model_bundle(model, feature_columns, self.matchups, path)
            bundle = joblib.load(path)

        self.assertEqual(bundle["feature_columns"], feature_columns)
        self.assertEqual(bundle["trained_through_season"], 2023)
        self.assertEqual(bundle["training_rows"], len(self.matchups))


if __name__ == "__main__":
    unittest.main()
