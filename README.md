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
