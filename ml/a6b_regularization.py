"""
CardioTriage AI — Phase A, Step A6b: Regularization.

We measure each model's train vs 5-fold cross-validation accuracy BEFORE and
AFTER regularization, so the train-CV gap (variance) is visible shrinking.
We also show L1 driving Logistic Regression coefficients to exactly zero
(automatic feature selection).

All on the TRAINING set via cross-validation; test set untouched.

Run with:  uv run python ml/a6b_regularization.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, load_and_split

X_train, _, y_train, _ = load_and_split()


def cv_gap(variant: str, classifier) -> dict:
    """5-fold CV returning mean train accuracy, mean CV accuracy, and gap."""
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", classifier)])
    res = cross_validate(
        pipe, X_train, y_train, cv=5, scoring="accuracy",
        return_train_score=True, n_jobs=-1,
    )
    train = res["train_score"].mean()
    cv = res["test_score"].mean()
    return {"variant": variant, "train": train, "cv": cv, "gap": train - cv}


def main() -> None:
    # --- Logistic Regression: L2 strength + L1 sparsity -------------------
    lr_rows = [
        cv_gap("LR L2  C=1.0 (default)",
               LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        cv_gap("LR L2  C=0.01 (strong reg)",
               LogisticRegression(max_iter=1000, C=0.01, random_state=42)),
        # New sklearn 1.8 API: l1_ratio=1.0 = pure L1 (lasso); needs 'saga'.
        cv_gap("LR L1  C=1.0 (lasso)",
               LogisticRegression(max_iter=5000, C=1.0, l1_ratio=1.0,
                                  solver="saga", random_state=42)),
    ]

    # --- Random Forest: default vs depth/leaf limits ----------------------
    rf_rows = [
        cv_gap("RF default (no limits)",
               RandomForestClassifier(n_estimators=300, random_state=42)),
        cv_gap("RF regularized (depth=5, leaf=10)",
               RandomForestClassifier(n_estimators=300, max_depth=5,
                                      min_samples_leaf=10, random_state=42)),
    ]

    # --- Gradient Boosting: default vs smaller steps/shallower ------------
    gb_rows = [
        cv_gap("GB default",
               GradientBoostingClassifier(random_state=42)),
        cv_gap("GB regularized (lr=0.05, depth=2, sub=0.8)",
               GradientBoostingClassifier(learning_rate=0.05, max_depth=2,
                                          n_estimators=200, subsample=0.8,
                                          random_state=42)),
    ]

    table = pd.DataFrame(lr_rows + rf_rows + gb_rows).set_index("variant")
    print("=" * 78)
    print("REGULARIZATION: train vs CV accuracy (5-fold) and the gap")
    print("=" * 78)
    print(table.round(3).to_string())

    # --- L1 sparsity: how many coefficients got zeroed out ----------------
    l1 = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=5000, C=0.2, l1_ratio=1.0,
                                   solver="saga", random_state=42)),
    ])
    l1.fit(X_train, y_train)
    coefs = l1.named_steps["clf"].coef_.ravel()
    n_total = coefs.size
    n_zero = int(np.sum(np.isclose(coefs, 0.0)))
    print("\n" + "=" * 78)
    print("L1 (lasso) FEATURE SELECTION at C=0.2")
    print("=" * 78)
    print(f"Total coefficients : {n_total}")
    print(f"Driven to zero     : {n_zero}  (these features were dropped)")
    print(f"Kept (non-zero)    : {n_total - n_zero}")


if __name__ == "__main__":
    main()
