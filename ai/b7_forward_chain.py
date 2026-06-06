"""
CardioTriage AI — AI Phase, Step B7: Forward-chaining inference engine.

Fire the knowledge-base rules over a patient's facts until no new facts
emerge (the fixpoint), printing the full inference trace. Demonstrated on two
contrasting patients: one high-risk (with synthetic safety flags) and one
low-risk.

Run with:  uv run python ai/b7_forward_chain.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from model_api import assess  # noqa: E402
from knowledge_base import RULES, build_facts  # noqa: E402


def forward_chain(facts: dict, rules=RULES):
    """Repeatedly fire applicable rules until no new facts appear.

    Returns (facts, trace). trace is a list of
    (pass_number, rule_name, [new_fact_keys], is_override).
    """
    trace = []
    pass_num = 0
    changed = True
    while changed:
        changed = False
        pass_num += 1
        for rule in rules:
            if rule.condition(facts):
                # Only "fires" if it asserts something not already known.
                new = {k: v for k, v in rule.conclusions.items() if facts.get(k) != v}
                if new:
                    facts.update(new)
                    trace.append((pass_num, rule.name, list(new.keys()), rule.override))
                    changed = True
    return facts, trace, pass_num


def run_case(label: str, patient: dict) -> None:
    ml = assess(patient)
    facts = build_facts(patient, ml)
    initial_keys = set(facts)               # to separate derived facts later

    print("=" * 72)
    print(label)
    print("=" * 72)
    print("INITIAL FACTS (from ML + profile + synthetic flags):")
    print(f"  risk_probability = {facts['risk_probability']:.3f}   "
          f"is_high_risk = {facts['is_high_risk']}")
    print(f"  phenotype = {facts['phenotype']}")
    print(f"  kidney_function = {facts['kidney_function']}   "
          f"on_anticoagulant = {facts['on_anticoagulant']}")

    facts, trace, passes = forward_chain(facts)

    print("\nINFERENCE TRACE (rules firing until no new facts):")
    if not trace:
        print("  (no rules fired)")
    current_pass = 0
    for p, name, new_facts, override in trace:
        if p != current_pass:
            current_pass = p
            print(f"  -- pass {p} --")
        tag = "  [SAFETY OVERRIDE]" if override else ""
        print(f"    {name} fired -> {', '.join(new_facts)}{tag}")
    print(f"  (fixpoint reached after pass {passes}: no new facts)")

    derived = sorted(k for k in facts if k not in initial_keys)
    print("\nDERIVED CONCLUSIONS:")
    for k in derived:
        print(f"  - {k}")
    print()


# High-risk patient WITH synthetic safety flags set to trigger overrides.
HIGH_RISK = {
    "age": 67, "sex": "Male", "cp": "asymptomatic", "trestbps": 160,
    "chol": 286, "fbs": False, "restecg": "lv hypertrophy", "thalch": 108,
    "exang": True, "oldpeak": 1.5, "slope": "flat", "ca": 3,
    "thal": "reversable defect",
    "kidney_function": "impaired", "on_anticoagulant": True,   # synthetic
}

# Low-risk patient, clean profile, default (safe) synthetic flags.
LOW_RISK = {
    "age": 37, "sex": "Female", "cp": "atypical angina", "trestbps": 120,
    "chol": 200, "fbs": False, "restecg": "normal", "thalch": 180,
    "exang": False, "oldpeak": 0.0, "slope": "upsloping", "ca": 0,
    "thal": "normal",
}


def main() -> None:
    run_case("HIGH-RISK case (impaired kidneys + on anticoagulant)", HIGH_RISK)
    run_case("LOW-RISK case (clean profile)", LOW_RISK)


if __name__ == "__main__":
    main()
