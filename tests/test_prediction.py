import unittest

import numpy as np
import pandas as pd

from src.prediction import predict_matchup, resolve_team_id


class FakeModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        lower_team_probability = 0.75 if features.iloc[0]["Diff_WinPct"] > 0 else 0.25
        return np.array([[1.0 - lower_team_probability, lower_team_probability]])


class PredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.teams = pd.DataFrame(
            [
                {"TeamID": 1, "TeamName": "Alpha"},
                {"TeamID": 2, "TeamName": "Beta"},
            ]
        )
        self.team_seasons = pd.DataFrame(
            [
                {"Season": 2026, "TeamID": 1, "WinPct": 0.8, "SeedNumber": 1},
                {"Season": 2026, "TeamID": 2, "WinPct": 0.6, "SeedNumber": 8},
            ]
        )
        self.bundle = {
            "model": FakeModel(),
            "feature_columns": ["Diff_WinPct", "Diff_SeedNumber"],
            "trained_through_season": 2025,
        }

    def test_team_names_are_case_insensitive(self) -> None:
        self.assertEqual(resolve_team_id(self.teams, "alpha"), 1)

    def test_requested_team_order_is_respected(self) -> None:
        forward = predict_matchup(
            self.bundle, self.team_seasons, self.teams, 2026, "Alpha", "Beta"
        )
        reverse = predict_matchup(
            self.bundle, self.team_seasons, self.teams, 2026, "Beta", "Alpha"
        )

        self.assertAlmostEqual(forward.team_a_win_probability, 0.75)
        self.assertAlmostEqual(reverse.team_a_win_probability, 0.25)
        self.assertEqual(forward.predicted_winner, "Alpha")
        self.assertEqual(reverse.predicted_winner, "Alpha")

    def test_same_team_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "two different teams"):
            predict_matchup(
                self.bundle, self.team_seasons, self.teams, 2026, "Alpha", "Alpha"
            )


if __name__ == "__main__":
    unittest.main()
