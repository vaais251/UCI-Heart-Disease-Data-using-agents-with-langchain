"""
CardioTriage AI — AI Phase, Step B3: State-space formulation.

Defines the five ingredients of the search problem:
  1. State          -> the State dataclass (a snapshot)
  2. initial state  -> build_initial_state(patient)   (seeded by the ML model)
  3. goal test      -> is_goal(state)
  4. actions + 5. transitions/costs -> successors(state)

A* search (B4) will use successors() + is_goal() to find the cheapest safe
path. This file just defines the world; it does no searching yet.

Run with:  uv run python ai/state_space.py
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from model_api import assess  # noqa: E402  (from ../ml/model_api.py)

from b1_initial_state import SAMPLE_PATIENTS  # noqa: E402  (same ai/ folder)


@dataclass(frozen=True)
class State:
    """One snapshot of the triage process. Frozen so it's hashable -> usable
    in A*'s visited-set and priority queue."""
    risk_level: str            # "high" or "low" (from the ML threshold)
    risk_probability: float    # P(disease) from the model (context, constant)
    priority: int              # phenotype triage priority 1..4 (context)
    phenotype: str             # phenotype name (context)
    confirmed: frozenset       # findings confirmed so far
    disposition: str | None    # None until a terminal action commits one


# Costs reflect clinical effort/invasiveness (used by A* as step cost g).
ACTION_COSTS = {
    "order ECG": 1,
    "order troponin": 1,
    "order stress test": 2,
    "refer cardiologist": 2,
    "admit & treat": 3,
    "discharge": 1,
}


def build_initial_state(patient: dict) -> State:
    """Seed the initial state from the ML model's assessment (B1)."""
    a = assess(patient)
    return State(
        risk_level="high" if a["risk"]["is_high_risk"] else "low",
        risk_probability=a["risk"]["probability"],
        priority=a["phenotype"]["priority"],
        phenotype=a["phenotype"]["name"],
        confirmed=frozenset(),   # nothing confirmed yet
        disposition=None,        # no decision yet
    )


def is_goal(state: State) -> bool:
    """A safe, confirmed disposition appropriate to the risk level."""
    if state.risk_level == "high":
        return state.disposition == "admitted_treated"
    return state.disposition == "discharged"


def successors(state: State):
    """Transition model: list of (action, cost, next_state) reachable in one
    step. Encodes each action's preconditions and effects."""
    if state.disposition is not None:
        return []  # terminal state: nothing more to do

    out = []
    c = state.confirmed

    def add(action, **changes):
        out.append((action, ACTION_COSTS[action], replace(state, **changes)))

    # --- Diagnostics (each can be ordered once) ---------------------------
    if "ECG" not in c:
        add("order ECG", confirmed=c | {"ECG"})
    if "troponin" not in c:
        add("order troponin", confirmed=c | {"troponin"})
    if "stress test" not in c:
        add("order stress test", confirmed=c | {"stress test"})

    # --- Referral: only with diagnostic evidence in hand ------------------
    if {"ECG", "troponin"} <= c and "cardiology" not in c:
        add("refer cardiologist", confirmed=c | {"cardiology"})

    # --- Terminal dispositions -------------------------------------------
    if state.risk_level == "high" and "cardiology" in c:
        add("admit & treat", disposition="admitted_treated")
    if state.risk_level == "low" and "ECG" in c:
        add("discharge", disposition="discharged")

    return out


def format_state(state: State) -> str:
    confirmed = ", ".join(sorted(state.confirmed)) or "(none)"
    return (f"[risk={state.risk_level} p={state.risk_probability:.2f} "
            f"prio={state.priority} | confirmed: {confirmed} | "
            f"disposition: {state.disposition or 'none'}]")


def main() -> None:
    for label, patient in SAMPLE_PATIENTS.items():
        s0 = build_initial_state(patient)
        print("=" * 70)
        print(label)
        print("=" * 70)
        print("INITIAL STATE :", format_state(s0))
        print("is goal?      :", is_goal(s0))
        print("Available actions from the initial state:")
        for action, cost, nxt in successors(s0):
            print(f"   - {action:20s} (cost {cost}) -> {format_state(nxt)}")
        print()


if __name__ == "__main__":
    main()
