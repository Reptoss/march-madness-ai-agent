import unittest

import pandas as pd

from src.features import build_team_season_features, build_tournament_matchups


def _game(
    winner: int,
    loser: int,
    winner_score: int,
    loser_score: int,
) -> dict[str, int | str]:
    row: dict[str, int | str] = {
        "Season": 2025,
        "DayNum": 10,
        "WTeamID": winner,
        "WScore": winner_score,
        "LTeamID": loser,
        "LScore": loser_score,
        "WLoc": "N",
        "NumOT": 0,
    }
    winner_stats = [25, 50, 8, 20, 12, 16, 10, 20, 15, 9, 5, 3, 14]
    loser_stats = [20, 50, 5, 20, 10, 14, 8, 18, 10, 12, 4, 2, 16]
    short_names = [
        "FGM",
        "FGA",
        "FGM3",
        "FGA3",
        "FTM",
        "FTA",
        "OR",
        "DR",
        "Ast",
        "TO",
        "Stl",
        "Blk",
        "PF",
    ]
    row.update({f"W{name}": value for name, value in zip(short_names, winner_stats)})
    row.update({f"L{name}": value for name, value in zip(short_names, loser_stats)})
    return row


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.regular_results = pd.DataFrame(
            [
                _game(1, 2, 70, 60),
                _game(2, 1, 80, 65),
            ]
        )
        self.seeds = pd.DataFrame(
            [
                {"Season": 2025, "Seed": "W01", "TeamID": 1},
                {"Season": 2025, "Seed": "W02", "TeamID": 2},
            ]
        )

    def test_team_features_use_both_winner_and_loser_rows(self) -> None:
        features = build_team_season_features(self.regular_results, self.seeds)
        team_one = features.loc[features["TeamID"] == 1].iloc[0]

        self.assertEqual(team_one["Games"], 2)
        self.assertEqual(team_one["Wins"], 1)
        self.assertAlmostEqual(team_one["WinPct"], 0.5)
        self.assertAlmostEqual(team_one["AvgPointMargin"], -2.5)
        self.assertEqual(team_one["SeedNumber"], 1)

    def test_matchup_differences_are_team_a_minus_team_b(self) -> None:
        features = build_team_season_features(self.regular_results, self.seeds)
        tournament_results = pd.DataFrame(
            [
                {
                    "Season": 2025,
                    "DayNum": 136,
                    "WTeamID": 1,
                    "LTeamID": 2,
                }
            ]
        )
        teams = pd.DataFrame(
            [
                {"TeamID": 1, "TeamName": "Alpha"},
                {"TeamID": 2, "TeamName": "Beta"},
            ]
        )

        matchups = build_tournament_matchups(features, tournament_results, teams)
        matchup = matchups.iloc[0]

        self.assertEqual(matchup["TeamAID"], 1)
        self.assertEqual(matchup["TeamBID"], 2)
        self.assertEqual(matchup["TeamAWin"], 1)
        self.assertEqual(matchup["Diff_SeedNumber"], -1)
        self.assertEqual(matchup["TeamAName"], "Alpha")


if __name__ == "__main__":
    unittest.main()
