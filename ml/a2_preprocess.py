"""
CardioTriage AI — Phase A, Step A2: Run & verify preprocessing.

This script DEMONSTRATES the preprocessing module: it splits the data,
fits the preprocessor on the TRAINING set only, transforms both halves, and
prints checks that prove the pipeline did its job (no missing values left,
correct shapes, preserved class balance).

Run with:  uv run python ml/a2_preprocess.py
"""

import numpy as np

from preprocessing import (
    build_preprocessor,
    load_and_split,
    load_raw,
    prepare_frame,
)


def main() -> None:
    # --- Show the cleaning effect on hidden missingness --------------------
    raw = load_raw()
    prepared = prepare_frame(raw)
    print("=" * 70)
    print("CLEANING: hidden missing cholesterol (chol == 0)")
    print("=" * 70)
    print(f"chol == 0 in raw data : {(raw['chol'] == 0).sum()} patients")
    print(f"chol NaN after cleaning: {prepared['chol'].isna().sum()} patients")
    print(f"ca_missing flag = 1    : {prepared['ca_missing'].sum()} patients")
    print(f"thal_missing flag = 1  : {prepared['thal_missing'].sum()} patients")

    # --- Stratified split ---------------------------------------------------
    X_train, X_test, y_train, y_test = load_and_split()
    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT (stratified, 80/20)")
    print("=" * 70)
    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(f"Disease rate  -- train: {y_train.mean():.3f}   test: {y_test.mean():.3f}")
    print("(rates should be almost identical -> stratification worked)")

    # --- Fit on TRAIN only, then transform both ----------------------------
    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)  # learns medians/modes/scales here
    X_test_t = pre.transform(X_test)         # reuses what it learned on train

    feature_names = pre.get_feature_names_out()
    print("\n" + "=" * 70)
    print("AFTER PREPROCESSING")
    print("=" * 70)
    print(f"Raw features in     : {X_train.shape[1]}")
    print(f"Model features out  : {X_train_t.shape[1]}  (one-hot expands categories)")
    print(f"X_train_t shape     : {X_train_t.shape}")
    print(f"X_test_t shape      : {X_test_t.shape}")
    print(f"Any missing left?   : train={np.isnan(X_train_t).any()}  test={np.isnan(X_test_t).any()}")

    print("\nGenerated feature columns:")
    for name in feature_names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
