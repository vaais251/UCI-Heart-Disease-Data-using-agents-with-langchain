"""
CardioTriage AI — Phase A, Step A1: Load & inspect the dataset.

Goal of this step: just LOOK at the raw data. We do not clean, change, or
model anything yet. We want to understand what we are working with:
  - how many rows and columns,
  - what each column is and its data type,
  - a sample of real rows,
  - how much data is missing, and where.

Run with:  uv run python ml/a1_load_inspect.py
"""

from pathlib import Path

import pandas as pd

# Path to the dataset. Path(__file__).parent is the folder this script lives in
# (ml/), so ".." steps up to the project root, then into data/.
DATA_PATH = Path(__file__).parent.parent / "data" / "heart_disease_uci.csv"


def main() -> None:
    # Load the CSV into a DataFrame (an in-memory table).
    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("1) SHAPE  (rows, columns)")
    print("=" * 70)
    print(df.shape)

    print("\n" + "=" * 70)
    print("2) COLUMNS and their data types")
    print("=" * 70)
    # dtypes tells us how pandas interpreted each column:
    #   int64/float64 = numeric, object = text/category, bool = True/False
    print(df.dtypes)

    print("\n" + "=" * 70)
    print("3) FIRST 5 ROWS")
    print("=" * 70)
    print(df.head())

    print("\n" + "=" * 70)
    print("4) MISSING VALUES per column (count, then % of rows)")
    print("=" * 70)
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(1)
    missing = pd.DataFrame({"missing_count": missing_count, "missing_%": missing_pct})
    # Show only columns that actually have missing values, worst first.
    print(missing[missing["missing_count"] > 0].sort_values("missing_count", ascending=False))

    print("\n" + "=" * 70)
    print("5) TARGET COLUMN 'num' — distribution of raw values (0-4)")
    print("=" * 70)
    # The proposal: 0 = no disease, 1-4 = disease present. We only LOOK here.
    print(df["num"].value_counts().sort_index())

    print("\n" + "=" * 70)
    print("6) 'dataset' column — which hospital each row came from")
    print("=" * 70)
    print(df["dataset"].value_counts())


if __name__ == "__main__":
    main()
