"""
CardioTriage AI — backend bridge for the Reasoning Console.

Wraps the AI-phase modules (A* search, forward-chaining KB, Gemini orchestrator)
behind clean functions the FastAPI layer can call. The browser only ever sends
a preset id — the actual patient records (including the disclosed synthetic
safety flags) live here on the server.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# The AI modules each add ../ml themselves; we make ../ai importable here.
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "ai"))
load_dotenv(ROOT / ".env")  # for GOOGLE_API_KEY (live Gemini briefing)

from model_api import assess  # noqa: E402
from b7_forward_chain import HIGH_RISK, LOW_RISK  # noqa: E402
from b4_astar import astar  # noqa: E402
from state_space import build_initial_state  # noqa: E402
from b5_recommendation import ACTION_NARRATION, DECISION_NARRATION  # noqa: E402
from b7_forward_chain import forward_chain  # noqa: E402
from knowledge_base import RULES, build_facts  # noqa: E402
from b8_reconcile import FACT_INFO  # noqa: E402
from b9_integration import briefing as deterministic_briefing  # noqa: E402
from b9_integration import run_triage  # noqa: E402

# Rule lookup by name, so we can attach IF/THEN text to each trace line.
_RULE_BY_NAME = {r.name: r for r in RULES}

# The two showcase cases. HIGH carries the synthetic flags (impaired kidneys +
# anticoagulant) that trigger the knowledge base's SAFETY OVERRIDES.
PRESETS = {
    "high": {
        "label": "High-risk — elderly, 3-vessel (+safety flags)",
        "patient": HIGH_RISK,
    },
    "low": {
        "label": "Low-risk — young, clean profile",
        "patient": LOW_RISK,
    },
}


def list_patients() -> list[dict]:
    """Return each preset with its ML risk summary (for the dropdown + chip)."""
    out = []
    for pid, entry in PRESETS.items():
        a = assess(entry["patient"])
        out.append({
            "id": pid,
            "label": entry["label"],
            "risk_probability": a["risk"]["probability"],
            "is_high_risk": a["risk"]["is_high_risk"],
            "phenotype": a["phenotype"]["name"],
            "priority": a["phenotype"]["priority"],
        })
    return out


def get_patient(pid: str) -> dict | None:
    """Look up a preset patient record by id (None if unknown)."""
    entry = PRESETS.get(pid)
    return entry["patient"] if entry else None


def search_path(patient: dict) -> dict:
    """Run A* and serialize the path into display nodes for the UI.

    Returns the ordered chain: the ML-seeded initial state, then one node per
    action the agent takes, ending at the goal (committed disposition). Each
    node carries a 'tone' the front-end colour-codes:
        red   = initial high-risk state
        teal  = initial low-risk state
        amber = intermediate diagnostic/referral action
        green = goal (safe committed disposition)
    """
    s0 = build_initial_state(patient)
    path, total_cost, expanded = astar(s0)

    nodes = [{
        "kind": "initial",
        "label": "INITIAL",
        "title": "ML-seeded state",
        "subtitle": (f"Risk {s0.risk_probability:.0%} · {s0.phenotype} "
                     f"(priority {s0.priority})"),
        "tone": "red" if s0.risk_level == "high" else "teal",
        "confirmed": [],
    }]

    step = 0
    for action, cost, state in path:
        is_goal = state.disposition is not None
        step += 1
        nodes.append({
            "kind": "goal" if is_goal else "action",
            "label": "GOAL" if is_goal else f"STEP {step}",
            "title": action,
            "subtitle": ACTION_NARRATION.get(action, ""),
            "cost": cost,
            "tone": "green" if is_goal else "amber",
            "confirmed": sorted(state.confirmed),
            "disposition": state.disposition,
        })

    return {
        "risk_level": s0.risk_level,
        "total_cost": total_cost,
        "nodes_expanded": expanded,
        "decision": DECISION_NARRATION[path[-1][2].disposition],
        "nodes": nodes,
    }


def trace_log(patient: dict) -> dict:
    """Run forward chaining and serialize the firing trace for the UI.

    Each line = one rule firing: its name, IF/THEN text, the new facts it
    derived (with their human-readable meaning + category), and whether it is
    a SAFETY OVERRIDE.
    """
    facts = build_facts(patient, assess(patient))
    initial = set(facts)
    facts, trace, passes = forward_chain(facts)

    lines = []
    for pass_num, rule_name, new_keys, override in trace:
        rule = _RULE_BY_NAME.get(rule_name)
        derived = []
        for key in new_keys:
            category, meaning = FACT_INFO.get(key, ("complement", key))
            derived.append({"key": key, "meaning": meaning, "category": category})
        lines.append({
            "pass": pass_num,
            "rule": rule_name,
            "if_text": rule.if_text if rule else "",
            "then_text": rule.then_text if rule else "",
            "facts": derived,
            "override": override,
        })

    derived_count = len([k for k in facts if k not in initial])
    override_count = sum(1 for ln in lines if ln["override"])
    return {
        "lines": lines,
        "passes": passes,
        "derived_count": derived_count,
        "override_count": override_count,
    }


# Cache the LangChain agent so we build the Gemini conductor only once.
_AGENT = None


def _gemini_text(patient: dict) -> str:
    """Call the live Gemini orchestrator. Raises if the key/model is missing
    or the call fails (the caller falls back to the deterministic briefing)."""
    global _AGENT
    from b9b_orchestrator import build_agent, extract_text, set_patient

    set_patient(patient)
    if _AGENT is None:
        _AGENT = build_agent()
    result = _AGENT.invoke(
        {"messages": [("user",
         "Assess the current patient and produce the triage briefing.")]}
    )
    text = extract_text(result["messages"][-1].content).strip()
    if not text:
        raise ValueError("Gemini returned empty text")
    return text


def briefing(patient: dict) -> dict:
    """Produce the final triage briefing for Zone 03.

    Tries the live Gemini orchestrator; on any failure (no API key, network,
    quota) it falls back to the deterministic briefing so the demo never
    breaks. Also returns the KB safety overrides for the amber shield box.
    """
    triage = run_triage(patient)

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            text, source = _gemini_text(patient), "gemini"
        except Exception as exc:  # noqa: BLE001 - demo must stay resilient
            print(f"[briefing] Gemini unavailable, using fallback: {exc}")
            text, source = deterministic_briefing(patient), "fallback"
    else:
        text, source = deterministic_briefing(patient), "fallback"

    return {
        "text": text,
        "source": source,
        "decision": triage["decision"],
        "agreement": triage["agreement"],
        "overrides": triage["kb_override"],
        "safety_modifications": triage["safety_modifications"],
    }
