"""Build model-ready features from the local raw competition data."""

from src.data_loader import load_mens_data
from src.features import build_feature_data, write_feature_data


if __name__ == "__main__":
    data = load_mens_data()
    features = build_feature_data(data)
    team_path, matchup_path = write_feature_data(features)

    print("Feature pipeline completed successfully")
    print(f"  Team-season rows: {len(features.team_seasons):,}")
    print(f"  Tournament training rows: {len(features.tournament_matchups):,}")
    print(f"  Team features: {team_path}")
    print(f"  Training matchups: {matchup_path}")
