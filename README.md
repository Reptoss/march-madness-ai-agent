# March Madness AI Agent

This project uses historical NCAA basketball data to build a men's March
Madness game predictor.

## Setup

1. Clone this repository and create a Python virtual environment.
2. Install the dependency:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Download the official March Machine Learning Mania data from Kaggle.
4. Extract the CSV files into `data/raw/`. Raw data is intentionally excluded
   from Git, so every contributor must download their own copy.
5. Verify the setup:

   ```powershell
   python loadData.py
   ```

The command validates the required files and columns, then prints the number
of teams and games available. The large Massey rankings table can be loaded
from Python when needed:

```python
from src.data_loader import load_mens_data

data = load_mens_data(include_rankings=True)
```

## Build features

Create team-season summaries and labeled historical tournament matchups:

```powershell
python buildFeatures.py
```

The generated files are written to `data/processed/`:

- `mens_team_season_features.csv` contains one row per team and season. Its
  features include win percentage, scoring margin, shooting rates, rebounds,
  assists, turnovers, defense, schedule strength, and tournament seed.
- `mens_tournament_training.csv` contains one row per historical tournament
  game. `TeamAWin` is the prediction target, and every `Diff_` column is Team
  A's pre-tournament feature minus Team B's corresponding feature.

Processed files are ignored by Git because every contributor can reproduce
them from the raw Kaggle download.

Run the automated checks with:

```powershell
python -m unittest discover -s tests -v
```

## Train the baseline model

Train logistic regression and evaluate it by holding out the latest five
completed tournaments one season at a time:

```powershell
python trainModel.py
```

For every held-out season, the model trains only on earlier seasons. This
prevents future tournament results from leaking into the evaluation. The
command reports accuracy, log loss, and Brier score, then writes reproducible
model artifacts to `models/` and detailed results to `reports/generated/`.
Those generated directories are excluded from Git.

## Predict a 2026 matchup

After running `python trainModel.py`, give the prediction command two exact
Kaggle team names:

```powershell
python predictGame.py "Duke" "Arizona"
```

The command uses 2026 regular-season statistics and seeds with the model
trained through the 2025 tournament. It prints both win probabilities and the
predicted winner. Team names are not case-sensitive; Kaggle TeamID values also
work.

## Evaluate the completed 2026 tournament

Score the model trained through 2025 against all 67 actual 2026 tournament
games:

```powershell
python evaluate2026.py
```

The fixed evaluation data is in `data/external/` with its sources documented.
Detailed generated predictions and metrics are written to the ignored
`reports/generated/` directory.
