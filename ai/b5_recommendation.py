"""
CardioTriage AI - AI Phase, Step B5: Path -> clinical recommendation.

7d: translate the A* action sequence into a plain-English recommendation.
7e: demonstrate that high-risk vs low-risk predictions yield DIFFERENT paths
    and DIFFERENT final decisions.

Returns a STRUCTURED recommendation (used later by the LangChain agent in B9)
and renders it as readable text.

Run with:  uv run python ai/b5_recommendation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from b1_initial_state import SAMPLE_PATIENTS  # noqa: E402
from b4_astar import astar  # noqa: E402
from state_space import build_initial_state  # noqa: E402

# Each abstract action -> a clinical instruction.
ACTION_NARRATION = {
    "order ECG": "Obtain a 12-lead ECG to assess cardiac electrical activity.",
    "order troponin": "Draw blood for cardiac troponin to check for heart-muscle injury.",
    "order stress test": "Arrange an exercise stress test.",
    "refer cardiologist": "Refer to cardiology for specialist evaluation.",
    "admit & treat": "Admit to a monitored cardiac bed and begin cardiology-guided treatment.",
    "discharge": "Discharge with reassurance and outpatient follow-up advice.",
}

DECISION_NARRATION = {
    "admitted_treated": "ADMIT & TREAT - monitored cardiac bed, cardiology-guided care.",
    "discharged": "DISCHARGE - safe for outpatient follow-up.",
}


def plan_and_recommend(patient: dict) -> dict:
    """Run A* and return a structured clinical recommendation."""
    s0 = build_initial_state(patient)
    path, total_cost, _ = astar(s0)

    actions = [action for action, _cost, _state in path]
    steps = [ACTION_NARRATION[a] for a in actions]
    final_disposition = path[-1][2].disposition

    if s0.risk_level == "high":
        rationale = (f"High ML risk ({s0.risk_probability:.0%}) and phenotype "
                     f"'{s0.phenotype}' (priority {s0.priority}) -> urgent "
                     f"confirm-then-treat pathway.")
    else:
        rationale = (f"Low ML risk ({s0.risk_probability:.0%}) -> minimal safe "
                     f"work-up; no specialist referral or admission needed.")

    return {
        "risk_level": s0.risk_level,
        "risk_probability": round(s0.risk_probability, 3),
        "phenotype": s0.phenotype,
        "priority": s0.priority,
        "actions": actions,
        "steps": steps,
        "decision": DECISION_NARRATION[final_disposition],
        "total_cost": total_cost,
        "rationale": rationale,
    }


def render(label: str, rec: dict) -> None:
    print("=" * 72)
    print(label)
    print("=" * 72)
    print(f"Risk: {rec['risk_level'].upper()} ({rec['risk_probability']:.0%})   "
          f"Phenotype: {rec['phenotype']} (priority {rec['priority']})")
    print("\nRecommended clinical pathway:")
    for i, step in enumerate(rec["steps"], 1):
        print(f"  {i}. {step}")
    print(f"\nFinal decision : {rec['decision']}")
    print(f"Rationale      : {rec['rationale']}")
    print(f"(plan cost = {rec['total_cost']})\n")


def main() -> None:
    labels = list(SAMPLE_PATIENTS.keys())
    recs = {lab: plan_and_recommend(SAMPLE_PATIENTS[lab]) for lab in labels}

    # --- 7d: readable recommendations ------------------------------------
    for lab in labels:
        render(lab, recs[lab])

    # --- 7e: contrast high-risk vs low-risk ------------------------------
    high = recs[labels[0]]   # HIGH-risk sample
    low = recs[labels[2]]    # LOW-risk sample
    print("#" * 72)
    print("7e - SAME AGENT, DIFFERENT PREDICTIONS -> DIFFERENT PATHS & DECISIONS")
    print("#" * 72)
    print(f"{'':18s} | {'HIGH-risk patient':32s} | {'LOW-risk patient':32s}")
    print("-" * 90)
    print(f"{'ML risk':18s} | {high['risk_probability']:<32.0%} | {low['risk_probability']:<32.0%}")
    print(f"{'Path length':18s} | {str(len(high['actions']))+' actions':32s} | {str(len(low['actions']))+' actions':32s}")
    print(f"{'Path':18s} | {' -> '.join(high['actions']):32s} | {' -> '.join(low['actions']):32s}")
    print(f"{'Final decision':18s} | {high['decision'][:32]:32s} | {low['decision'][:32]:32s}")
    print(f"{'Plan cost':18s} | {high['total_cost']:<32} | {low['total_cost']:<32}")


if __name__ == "__main__":
    main()
