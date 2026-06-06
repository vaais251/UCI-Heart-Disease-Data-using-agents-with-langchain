"""
CardioTriage AI — AI Phase, Step B9a: Integrated triage flow + tool layer.

Wires the A* search agent (Step 7) and the forward-chaining knowledge base
(Step 9) into ONE pipeline, and exposes each capability as a string-in /
string-out TOOL — the interface a LangChain agent will call in B9b.

The LLM (B9b) only ORCHESTRATES these deterministic tools; it never makes a
medical decision itself.

Run with:  uv run python ai/b9_integration.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from model_api import assess  # noqa: E402
from b5_recommendation import plan_and_recommend  # noqa: E402
from b7_forward_chain import HIGH_RISK, LOW_RISK  # noqa: E402
from b8_reconcile import FACT_INFO, derived_facts  # noqa: E402

# Override-meaning -> concrete safety modification to the plan.
SAFETY_MODS = {
    "do NOT use contrast angiography": "use a NON-contrast imaging strategy (protect kidneys)",
    "do NOT give clot-busting (thrombolytic) drugs": "avoid thrombolysis; consider mechanical revascularisation",
}


def _categorize(derived: set) -> dict:
    buckets = {"confirm": [], "complement": [], "override": []}
    for fact in sorted(derived):
        category, meaning = FACT_INFO.get(fact, ("complement", fact))
        buckets[category].append(meaning)
    return buckets


# ==========================================================================
# TOOLS  — string in (patient JSON), string out. Used directly by LangChain.
# ==========================================================================
def assess_risk_tool(patient_json: str) -> str:
    """Run the trained ML model: return risk probability + phenotype."""
    a = assess(json.loads(patient_json))
    return (f"risk_probability={a['risk']['probability']:.3f}, "
            f"is_high_risk={a['risk']['is_high_risk']}, "
            f"phenotype={a['phenotype']['name']} (priority {a['phenotype']['priority']})")


def plan_pathway_tool(patient_json: str) -> str:
    """Run A* search: return the recommended clinical pathway."""
    rec = plan_and_recommend(json.loads(patient_json))
    return (f"Decision: {rec['decision']} | Path: {' -> '.join(rec['actions'])} | "
            f"Steps: " + "; ".join(rec["steps"]))


def knowledge_base_tool(patient_json: str) -> str:
    """Run forward-chaining KB: return confirm/complement/override findings."""
    buckets = _categorize(derived_facts(json.loads(patient_json)))
    return (f"CONFIRM={buckets['confirm']} | COMPLEMENT={buckets['complement']} | "
            f"OVERRIDE(safety vetoes)={buckets['override']}")


# ==========================================================================
# INTEGRATED FLOW  — combine everything into one structured result.
# ==========================================================================
def run_triage(patient: dict) -> dict:
    rec = plan_and_recommend(patient)
    derived = derived_facts(patient)
    buckets = _categorize(derived)

    safety_mods = [SAFETY_MODS[o] for o in buckets["override"] if o in SAFETY_MODS]

    # Genuine agreement check: does the agent's disposition match the KB stance?
    agent_escalates = rec["risk_level"] == "high"
    kb_escalates = bool({"recommend_admission", "recommend_cardiology",
                         "elevated_cardiac_risk"} & derived)
    kb_safe = "safe_for_discharge" in derived
    if (agent_escalates and kb_escalates) or (not agent_escalates and kb_safe and not kb_escalates):
        agreement = "AGREE"
    else:
        agreement = "CONFLICT"

    return {
        "decision": rec["decision"],
        "pathway": rec["actions"],
        "rationale": rec["rationale"],
        "kb_confirm": buckets["confirm"],
        "kb_complement": buckets["complement"],
        "kb_override": buckets["override"],
        "safety_modifications": safety_mods,
        "additional_care": buckets["complement"],
        "agreement": agreement,
    }


def briefing(patient: dict) -> str:
    """Deterministic plain-English briefing (the LLM will enhance this in B9b)."""
    t = run_triage(patient)
    lines = [f"Disposition: {t['decision']}",
             f"Pathway: {' -> '.join(t['pathway'])}",
             f"Knowledge base {t['agreement']}s with this disposition."]
    if t["safety_modifications"]:
        lines.append("Safety overrides applied: " + "; ".join(t["safety_modifications"]))
    if t["additional_care"]:
        lines.append("Additional management: " + "; ".join(t["additional_care"]))
    return "\n  ".join(lines)


def main() -> None:
    for label, patient in [("HIGH-RISK case", HIGH_RISK), ("LOW-RISK case", LOW_RISK)]:
        pj = json.dumps(patient)
        print("=" * 72)
        print(label)
        print("=" * 72)
        print("TOOL 1 assess_risk     :", assess_risk_tool(pj))
        print("TOOL 2 plan_pathway    :", plan_pathway_tool(pj))
        print("TOOL 3 knowledge_base  :", knowledge_base_tool(pj))
        print("\nINTEGRATED BRIEFING:")
        print("  " + briefing(patient))
        print()


if __name__ == "__main__":
    main()
