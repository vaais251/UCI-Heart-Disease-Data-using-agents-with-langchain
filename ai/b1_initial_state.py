"""
CardioTriage AI — AI Phase, Step B1: ML model -> agent initial state.

This is the bridge from the Machine Learning phase to the AI phase. We:
  1. Reconnect the trained model saved in models/ (via ml/model_api.py).
  2. Run it on a few contrasting patients.
  3. Package each result into an explicit `initial_state` dict — the starting
     snapshot the A* search agent (built in B3-B4) will reason from.

Run with:  uv run python ai/b1_initial_state.py
"""

import sys
from pathlib import Path

# The trained-model interface lives in ../ml. Make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from model_api import assess  # noqa: E402  (from ../ml/model_api.py)


def build_initial_state(patient: dict) -> dict:
    """Turn a raw patient record into the search agent's INITIAL STATE.

    The initial state captures everything the agent needs to start planning:
      - risk_probability : the ML model's P(disease), 0..1
      - is_high_risk     : risk crosses the tuned clinical threshold (0.35)
      - phenotype/priority: the K-means risk group + its triage priority
      - confirmed/actions : nothing resolved yet -> the search hasn't begun
    """
    result = assess(patient)
    risk = result["risk"]
    pheno = result["phenotype"]
    return {
        "risk_probability": risk["probability"],
        "is_high_risk": risk["is_high_risk"],
        "phenotype": pheno["name"],
        "priority": pheno["priority"],
        # These two fields start "empty" and get filled as the agent acts.
        "confirmed": [],        # findings confirmed so far (none at the start)
        "actions_taken": [],    # clinical actions performed so far (none yet)
    }


# Three contrasting patients to verify the bridge end-to-end.
SAMPLE_PATIENTS = {
    "HIGH-risk (elderly, asymptomatic, 3 vessels)": {
        "age": 67, "sex": "Male", "cp": "asymptomatic", "trestbps": 160,
        "chol": 286, "fbs": False, "restecg": "lv hypertrophy", "thalch": 108,
        "exang": True, "oldpeak": 1.5, "slope": "flat", "ca": 3,
        "thal": "reversable defect",
    },
    "BORDERLINE (middle-aged, mixed signals)": {
        "age": 54, "sex": "Male", "cp": "non-anginal", "trestbps": 130,
        "chol": 240, "fbs": False, "restecg": "normal", "thalch": 150,
        "exang": False, "oldpeak": 1.0, "slope": "flat", "ca": 0,
        "thal": "normal",
    },
    "LOW-risk (young, atypical, clean)": {
        "age": 37, "sex": "Female", "cp": "atypical angina", "trestbps": 120,
        "chol": 200, "fbs": False, "restecg": "normal", "thalch": 180,
        "exang": False, "oldpeak": 0.0, "slope": "upsloping", "ca": 0,
        "thal": "normal",
    },
}


def main() -> None:
    for label, patient in SAMPLE_PATIENTS.items():
        state = build_initial_state(patient)
        print("=" * 64)
        print(label)
        print("=" * 64)
        print(f"  risk_probability : {state['risk_probability']:.3f}")
        print(f"  is_high_risk     : {state['is_high_risk']}")
        print(f"  phenotype        : {state['phenotype']} (priority {state['priority']})")
        print(f"  confirmed        : {state['confirmed']}")
        print(f"  actions_taken    : {state['actions_taken']}")
        print("  --> this dict is the INITIAL STATE for the A* search agent\n")


if __name__ == "__main__":
    main()
