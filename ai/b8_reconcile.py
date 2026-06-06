"""
CardioTriage AI - AI Phase, Step B8: Reconcile search agent + knowledge base.

Compares the A* recommendation (B5) with the forward-chained KB conclusions
(B7) and classifies each KB conclusion as CONFIRM / COMPLEMENT / OVERRIDE,
then analyses any disagreement (safety vs speed). The knowledge base wins on
safety. No LLM yet - this is the deterministic integration layer.

Run with:  uv run python ai/b8_reconcile.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from model_api import assess  # noqa: E402
from b5_recommendation import plan_and_recommend  # noqa: E402
from b7_forward_chain import HIGH_RISK, LOW_RISK, forward_chain  # noqa: E402
from knowledge_base import build_facts  # noqa: E402

# Each derived fact -> (relationship, human-readable meaning).
FACT_INFO = {
    # CONFIRM - supports an escalate-or-discharge disposition
    "elevated_cardiac_risk": ("confirm", "risk is clinically elevated"),
    "silent_ischemia_suspected": ("confirm", "silent ischemia suspected"),
    "urgent_cardiology_review": ("confirm", "urgent cardiology review"),
    "inducible_ischemia": ("confirm", "inducible ischemia"),
    "recommend_cardiology": ("confirm", "cardiology referral advised"),
    "significant_vessel_disease": ("confirm", "significant vessel disease"),
    "recommend_admission": ("confirm", "admission advised"),
    "reversible_perfusion_defect": ("confirm", "reversible perfusion defect"),
    "high_priority_elderly": ("confirm", "high-priority elderly"),
    "low_risk_profile": ("confirm", "low-risk profile"),
    "safe_for_discharge": ("confirm", "safe for discharge"),
    # COMPLEMENT - extra management beyond the acute disposition
    "hyperlipidemia": ("complement", "high cholesterol noted"),
    "recommend_statin": ("complement", "start statin + lipid follow-up"),
    "dysglycemia": ("complement", "elevated fasting glucose"),
    "recommend_diabetes_workup": ("complement", "arrange diabetes work-up"),
    "hypertensive_urgency": ("complement", "hypertensive urgency"),
    "control_bp_first": ("complement", "control BP before stress testing"),
    # OVERRIDE - safety vetoes that constrain the plan
    "renal_impairment": ("override", "impaired kidney function"),
    "avoid_contrast_angiography": ("override", "do NOT use contrast angiography"),
    "bleeding_risk": ("override", "high bleeding risk"),
    "avoid_thrombolysis": ("override", "do NOT give clot-busting (thrombolytic) drugs"),
}


def derived_facts(patient: dict) -> set:
    """Run forward chaining and return only the NEW (derived) facts."""
    facts = build_facts(patient, assess(patient))
    initial = set(facts)
    facts, _trace, _passes = forward_chain(facts)
    return {k for k in facts if k not in initial}


def reconcile(label: str, patient: dict) -> None:
    rec = plan_and_recommend(patient)        # A* plan
    derived = derived_facts(patient)         # KB conclusions

    buckets = {"confirm": [], "complement": [], "override": []}
    for fact in sorted(derived):
        category, meaning = FACT_INFO.get(fact, ("complement", fact))
        buckets[category].append(meaning)

    agent_escalates = rec["risk_level"] == "high"
    kb_escalates = bool({"recommend_admission", "recommend_cardiology",
                         "elevated_cardiac_risk"} & derived)
    kb_safe = "safe_for_discharge" in derived

    print("=" * 74)
    print(label)
    print("=" * 74)
    print(f"SEARCH AGENT (A*) : {rec['decision']}")
    print(f"                    path = {' -> '.join(rec['actions'])}")
    print(f"KNOWLEDGE BASE    : {len(derived)} conclusions derived\n")

    print("RECONCILIATION")
    print(f"  CONFIRM    (KB agrees with the plan) : {buckets['confirm'] or '-'}")
    print(f"  COMPLEMENT (KB adds extra care)      : {buckets['complement'] or '-'}")
    print(f"  OVERRIDE   (KB safety vetoes)        : {buckets['override'] or '-'}")

    # --- disposition agreement -------------------------------------------
    if agent_escalates and kb_escalates:
        print("\n  Disposition: AGREE - both escalate (admit/treat).")
    elif (not agent_escalates) and kb_safe and not kb_escalates:
        print("\n  Disposition: AGREE - both favour discharge.")
    else:
        print("\n  Disposition: ** CONFLICT ** - agent and KB disagree on escalation!")

    # --- safety-override analysis ----------------------------------------
    if buckets["override"]:
        print("\n  DISAGREEMENT ANALYSIS (safety vs speed):")
        print("    The A* agent optimises for the fastest safe disposition and would,")
        print("    by default, escalate to contrast angiography / thrombolysis. The KB")
        print("    vetoes these for THIS patient:")
        for o in buckets["override"]:
            print(f"      - {o}")
        print("    RESOLUTION: keep the agent's disposition, but the knowledge base")
        print("    OVERRIDES the method - safety takes precedence over speed.")

    # --- final integrated recommendation ---------------------------------
    print("\n  FINAL INTEGRATED RECOMMENDATION:")
    print(f"    1. Disposition: {rec['decision']}")
    if buckets["override"]:
        print("    2. Safety modifications (KB overrides agent):")
        if "do NOT use contrast angiography" in buckets["override"]:
            print("         - use a NON-contrast imaging strategy (protect kidneys)")
        if any("thrombolytic" in o for o in buckets["override"]):
            print("         - avoid thrombolysis; consider mechanical revascularisation")
    if buckets["complement"]:
        print("    3. Additional management:")
        for c in buckets["complement"]:
            print(f"         - {c}")
    print()


def main() -> None:
    reconcile("HIGH-RISK case (impaired kidneys + on anticoagulant)", HIGH_RISK)
    reconcile("LOW-RISK case (clean profile)", LOW_RISK)


if __name__ == "__main__":
    main()
