import unittest

import pandas as pd

from src.evaluation_2026 import evaluate_2026
from tests.test_prediction import FakeModel


class Evaluation2026Tests(unittest.TestCase):
    def test_evaluation_uses_a_model_trained_before_2026(self) -> None:
        bundle = {
            "model": FakeModel(),
            "feature_columns": ["Diff_WinPct", "Diff_SeedNumber"],
            "trained_through_season": 2025,
        }
        teams = pd.DataFrame(
            [
                {"TeamID": 1, "TeamName": "Alpha"},
                {"TeamID": 2, "TeamName": "Beta"},
            ]
        )
        team_seasons = pd.DataFrame(
            [
                {"Season": 2026, "TeamID": 1, "WinPct": 0.8, "SeedNumber": 1},
                {"Season": 2026, "TeamID": 2, "WinPct": 0.6, "SeedNumber": 8},
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "Season": 2026,
                    "Round": "Championship",
                    "Region": "",
                    "WTeamID": 1,
                    "WTeamName": "Alpha",
                    "WScore": 80,
                    "LTeamID": 2,
                    "LTeamName": "Beta",
                    "LScore": 70,
                }
            ]
        )

        evaluation = evaluate_2026(bundle, team_seasons, teams, results)

        self.assertEqual(evaluation.overall_metrics["Games"], 1)
        self.assertEqual(evaluation.overall_metrics["Correct"], 1)
        self.assertEqual(evaluation.overall_metrics["ModelTrainedThrough"], 2025)

    def test_model_trained_on_2026_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be a holdout"):
            evaluate_2026(
                {"trained_through_season": 2026},
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )


if __name__ == "__main__":
    unittest.main()
