"""
CardioTriage AI — Phase A, Step A2: Reusable preprocessing.

This module is the SINGLE source of truth for how raw patient data becomes
model-ready features. Every later modeling step (A4, A5, ...) imports from
here, so all models share identical, leakage-safe preprocessing.

Two public pieces:
  - load_and_split(): clean the raw data + make a stratified train/test split.
  - build_preprocessor(): build an (unfitted) ColumnTransformer that imputes,
    encodes, and scales. It is fit ONLY on training data inside a Pipeline,
    which is what prevents test data from leaking into training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent.parent / "data" / "heart_disease_uci.csv"

# --- Feature groups ---------------------------------------------------------
# Continuous numbers: get median-imputed, then standardized (scaled).
# 'ca' (0-3 vessel count) is numeric/ordinal, so it lives here too.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

# Text categories: get mode-imputed, then one-hot encoded into 0/1 columns.
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# Binary "was this value originally missing?" flags. Passed through as-is.
FLAG_FEATURES = ["ca_missing", "thal_missing"]

# Full ordered feature list the risk model expects as input.
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES

# Core numeric features used for K-means phenotype clustering (Step A7).
CLUSTER_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

# Deliberately EXCLUDED from features:
#   id      -> just a row number, not predictive
#   dataset -> which hospital; a confound, not a clinical measurement
#   num     -> the raw 0-4 target; we derive a binary 'target' from it


def load_raw() -> pd.DataFrame:
    """Read the raw CSV exactly as stored on disk."""
    return pd.read_csv(DATA_PATH)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning that is safe to do BEFORE the train/test split.

    These operations look at one cell at a time (no column-wide statistics),
    so they cannot leak information between rows.
    """
    df = df.copy()

    # Binary target: 0 = no disease, 1-4 = disease present -> 1.
    # Guarded so this also works at inference time, when a new patient
    # record has no 'num' column.
    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)

    # Hidden missingness: a serum cholesterol of 0 mg/dl is biologically
    # impossible, so it really means "not recorded". Turn it into a real NaN
    # so the imputer fills it instead of treating 0 as a true measurement.
    df["chol"] = df["chol"].replace(0, np.nan)

    # Missing-indicator flags for the two heavily-missing clinical tests.
    # Computed from raw missingness, which is observable for any new patient,
    # so this is not leakage.
    df["ca_missing"] = df["ca"].isna().astype(int)
    df["thal_missing"] = df["thal"].isna().astype(int)

    return df


def load_and_split(test_size: float = 0.2, random_state: int = 42):
    """Clean the data and return a stratified train/test split.

    Returns: X_train, X_test, y_train, y_test
    'stratify=y' keeps the disease/no-disease ratio identical in both halves.
    """
    df = prepare_frame(load_raw())
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES
    X = df[feature_cols]
    y = df["target"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) preprocessing transformer.

    Fit it on training data only (a Pipeline does this for us), then it can
    transform any new data the same way.
    """
    # Numeric branch: fill blanks with the column median, then standardize
    # (mean 0, std 1) so no large-numbered feature dominates the model.
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    # Categorical branch: fill blanks with the most frequent category, then
    # one-hot encode (turn each category into its own 0/1 column).
    # handle_unknown='ignore' keeps us safe if a category appears only in test.
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("flag", "passthrough", FLAG_FEATURES),
        ]
    )
