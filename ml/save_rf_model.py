"""
CardioTriage AI — train & save a Random Forest for the UI model comparison.

Uses the SAME leakage-safe preprocessing pipeline as the Gradient Boosting
model, with the regularized hyperparameters from A6b (so it generalizes
well rather than overfitting). Saved to models/rf_model.joblib.

Run with:  uv run python ml/save_rf_model.py
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    load_raw,
    prepare_frame,
)

MODELS_DIR = Path(__file__).parent.parent / "models"


def main() -> None:
    full = prepare_frame(load_raw())
    X, y = full[FEATURE_COLUMNS], full["target"]

    rf = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=5,
                                       min_samples_leaf=10, random_state=42)),
    ])
    rf.fit(X, y)

    out = MODELS_DIR / "rf_model.joblib"
    joblib.dump(rf, out)
    print(f"Saved Random Forest -> {out}")


if __name__ == "__main__":
    main()
