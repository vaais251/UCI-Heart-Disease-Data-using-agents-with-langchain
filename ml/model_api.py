"""
CardioTriage AI — ML <-> AI bridge.

The single clean interface the AI phase imports. It hides all ML details
(pipelines, scaling, thresholds) behind three functions that take a plain
patient dict and return plain results:

    predict_risk(patient)      -> probability + high-risk flag
    predict_phenotype(patient) -> cluster id, name, triage priority
    assess(patient)            -> both, combined

Example:
    from model_api import assess
    assess({"age": 67, "sex": "Male", "cp": "asymptomatic", ...})
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import (
    CATEGORICAL_FEATURES,
    CLUSTER_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    prepare_frame,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
# Raw clinical columns a caller may supply (flags are derived for them).
RAW_INPUT_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@lru_cache(maxsize=1)
def _load():
    """Load and cache the saved artifacts (loaded once per process)."""
    risk_model = joblib.load(MODELS_DIR / "risk_model.joblib")
    phenotype_model = joblib.load(MODELS_DIR / "phenotype_model.joblib")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    return risk_model, phenotype_model, metadata


def _to_frame(patient: dict) -> pd.DataFrame:
    """Turn one patient dict into a model-ready 1-row DataFrame.

    Missing raw columns become NaN (the pipelines impute them), and the
    ca/thal missing-flags are derived by prepare_frame.
    """
    df = pd.DataFrame([patient])
    # Ensure every raw input column exists so derivation/selection won't fail.
    df = df.reindex(columns=df.columns.union(RAW_INPUT_COLUMNS))
    return prepare_frame(df)


def predict_risk(patient: dict) -> dict:
    """Return P(disease) and a high-risk flag using the tuned threshold."""
    risk_model, _, metadata = _load()
    frame = _to_frame(patient)
    probability = float(risk_model.predict_proba(frame[FEATURE_COLUMNS])[0, 1])
    threshold = metadata["threshold"]
    return {
        "probability": round(probability, 4),
        "is_high_risk": probability >= threshold,
        "threshold": threshold,
    }


def predict_phenotype(patient: dict) -> dict:
    """Assign the patient to a risk phenotype with a triage priority."""
    _, phenotype_model, metadata = _load()
    frame = _to_frame(patient)
    cluster = int(phenotype_model.predict(frame[CLUSTER_FEATURES])[0])
    info = metadata["phenotype_map"][str(cluster)]
    return {"cluster": cluster, **info}


def assess(patient: dict) -> dict:
    """Combined risk + phenotype assessment — the agent's entry point."""
    return {"risk": predict_risk(patient), "phenotype": predict_phenotype(patient)}


if __name__ == "__main__":
    # Round-trip self-test: load the saved artifacts and assess two patients.
    high_risk = {
        "age": 67, "sex": "Male", "cp": "asymptomatic", "trestbps": 160,
        "chol": 286, "fbs": False, "restecg": "lv hypertrophy", "thalch": 108,
        "exang": True, "oldpeak": 1.5, "slope": "flat", "ca": 3,
        "thal": "reversable defect",
    }
    low_risk = {
        "age": 37, "sex": "Female", "cp": "atypical angina", "trestbps": 120,
        "chol": 200, "fbs": False, "restecg": "normal", "thalch": 180,
        "exang": False, "oldpeak": 0.0, "slope": "upsloping", "ca": 0,
        "thal": "normal",
    }
    for label, patient in [("HIGH-risk profile", high_risk),
                           ("LOW-risk profile", low_risk)]:
        print(f"\n{label}:")
        print(json.dumps(assess(patient), indent=2))
