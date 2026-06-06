"""
CardioTriage AI - AI Phase, Step B6: Knowledge base (Step 9).

A set of 12 domain IF-THEN rules grounded in our dataset features, the ML
output, and a small DISCLOSED set of synthetic clinical flags. This file
DEFINES the rules + facts; the forward-chaining engine that fires them is B7.

Run with:  uv run python ai/knowledge_base.py   (prints the rule catalogue)
"""

from dataclasses import dataclass, field
from typing import Callable

# --------------------------------------------------------------------------
# Synthetic clinical flags (DISCLOSED) - NOT used by the ML risk model.
# The UCI dataset lacks these attributes, so we add a clearly-labeled few so
# the safety-override rules are demonstrable. Defaults are the "safe/normal"
# value, so a patient without the flag is unaffected.
# --------------------------------------------------------------------------
SYNTHETIC_FLAGS = {
    "kidney_function": "normal",     # "normal" | "impaired"  (synthetic eGFR)
    "on_anticoagulant": False,       # already on blood thinners (synthetic)
}


@dataclass
class Rule:
    name: str
    if_text: str          # human-readable IF
    then_text: str        # human-readable THEN
    condition: Callable[[dict], bool]
    conclusions: dict     # new facts asserted when the rule fires
    explanation: str
    override: bool = False  # True for safety rules that can veto a plan


def build_facts(patient: dict, ml_result: dict) -> dict:
    """Assemble the initial working memory: patient features + ML output +
    synthetic flags (defaulted if absent)."""
    facts = dict(patient)                       # dataset features
    facts["risk_probability"] = ml_result["risk"]["probability"]
    facts["is_high_risk"] = ml_result["risk"]["is_high_risk"]
    facts["phenotype"] = ml_result["phenotype"]["name"]
    for flag, default in SYNTHETIC_FLAGS.items():
        facts.setdefault(flag, default)
    return facts


# --------------------------------------------------------------------------
# The rule base (12 rules). Conditions read facts; conclusions assert facts.
# Several chain: R1's conclusion 'elevated_cardiac_risk' triggers R2 and R7;
# R3/R4/R5 enable the R9 bleeding-risk override.
# --------------------------------------------------------------------------
RULES = [
    Rule("R1", "ML risk probability >= 0.35", "elevated_cardiac_risk",
         lambda f: f.get("risk_probability", 0) >= 0.35,
         {"elevated_cardiac_risk": True},
         "Anchors the KB to the ML model: a risk at/above our tuned threshold "
         "is treated as clinically elevated."),

    Rule("R2", "elevated_cardiac_risk AND chest pain is asymptomatic",
         "silent_ischemia_suspected, urgent_cardiology_review",
         lambda f: f.get("elevated_cardiac_risk") and f.get("cp") == "asymptomatic",
         {"silent_ischemia_suspected": True, "urgent_cardiology_review": True},
         "Asymptomatic presentation with high risk is the dangerous 'silent' "
         "pattern our EDA flagged (78% disease) - escalate."),

    Rule("R3", "exercise angina present AND oldpeak > 1.0",
         "inducible_ischemia, recommend_cardiology",
         lambda f: f.get("exang") is True and f.get("oldpeak", 0) > 1.0,
         {"inducible_ischemia": True, "recommend_cardiology": True},
         "Exercise-induced angina with ST depression indicates ischemia "
         "provoked by exertion - needs specialist review."),

    Rule("R4", "number of diseased vessels (ca) >= 1",
         "significant_vessel_disease, recommend_admission",
         lambda f: f.get("ca", 0) >= 1,
         {"significant_vessel_disease": True, "recommend_admission": True},
         "Any vessel narrowing on fluoroscopy is anatomical disease - admit "
         "for monitoring/intervention."),

    Rule("R5", "thalassemia test = reversible defect",
         "reversible_perfusion_defect, recommend_cardiology",
         lambda f: f.get("thal") == "reversable defect",
         {"reversible_perfusion_defect": True, "recommend_cardiology": True},
         "A reversible perfusion defect signals tissue at risk that may "
         "benefit from revascularisation."),

    Rule("R6", "resting blood pressure >= 180",
         "hypertensive_urgency, control_bp_first",
         lambda f: f.get("trestbps", 0) >= 180,
         {"hypertensive_urgency": True, "control_bp_first": True},
         "Severe hypertension must be controlled before stress testing to "
         "avoid provoking an event."),

    Rule("R7", "age >= 65 AND elevated_cardiac_risk", "high_priority_elderly",
         lambda f: f.get("age", 0) >= 65 and f.get("elevated_cardiac_risk"),
         {"high_priority_elderly": True},
         "Elderly high-risk patients are prioritised; age compounds cardiac "
         "risk (chains on R1's conclusion)."),

    Rule("R8", "kidney_function = impaired (synthetic)",
         "renal_impairment, avoid_contrast_angiography",
         lambda f: f.get("kidney_function") == "impaired",
         {"renal_impairment": True, "avoid_contrast_angiography": True},
         "SAFETY: contrast dye can worsen impaired kidneys - veto contrast "
         "angiography even if the search plan implies it.", override=True),

    Rule("R9", "on_anticoagulant AND (admission or cardiology recommended)",
         "bleeding_risk, avoid_thrombolysis",
         lambda f: f.get("on_anticoagulant") is True
         and (f.get("recommend_admission") or f.get("recommend_cardiology")),
         {"bleeding_risk": True, "avoid_thrombolysis": True},
         "SAFETY: a patient already anticoagulated has high bleeding risk - "
         "veto clot-busting therapy (chains on R3/R4/R5).", override=True),

    Rule("R10", "NOT elevated risk AND ca==0 AND oldpeak<=1.0 AND no exercise angina",
         "low_risk_profile, safe_for_discharge",
         lambda f: (not f.get("elevated_cardiac_risk") and f.get("ca", 0) == 0
                    and f.get("oldpeak", 0) <= 1.0 and f.get("exang") is False),
         {"low_risk_profile": True, "safe_for_discharge": True},
         "A clean profile with low ML risk supports safe discharge - this is "
         "the KB confirming the search agent's low-risk plan."),

    Rule("R11", "cholesterol >= 240", "hyperlipidemia, recommend_statin",
         lambda f: f.get("chol", 0) >= 240,
         {"hyperlipidemia": True, "recommend_statin": True},
         "High cholesterol warrants lipid-lowering therapy and follow-up "
         "regardless of the acute decision."),

    Rule("R12", "fasting blood sugar > 120 (fbs = True)",
         "dysglycemia, recommend_diabetes_workup",
         lambda f: f.get("fbs") is True,
         {"dysglycemia": True, "recommend_diabetes_workup": True},
         "Elevated fasting glucose is a major cardiac risk modifier - refer "
         "for diabetes work-up."),
]


def main() -> None:
    print("SYNTHETIC CLINICAL FLAGS (disclosed; not used by the ML model):")
    for flag, default in SYNTHETIC_FLAGS.items():
        print(f"  - {flag} (default '{default}')")
    print(f"\nKNOWLEDGE BASE - {len(RULES)} rules\n" + "=" * 60)
    for r in RULES:
        tag = "  [SAFETY OVERRIDE]" if r.override else ""
        print(f"\n{r.name}{tag}")
        print(f"  IF   {r.if_text}")
        print(f"  THEN {r.then_text}")
        print(f"  why: {r.explanation}")


if __name__ == "__main__":
    main()
