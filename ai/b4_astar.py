"""
CardioTriage AI — AI Phase, Step B4: A* search over the triage state space.

A* ranks states by f = g + h:
  g = actual cost paid so far (sum of action costs)
  h = unresolved-risk x urgency  (admissible estimate of remaining cost)
It returns the cheapest safe path from the ML-seeded initial state to a goal.

Run with:  uv run python ai/b4_astar.py
"""

import heapq
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from b1_initial_state import SAMPLE_PATIENTS  # noqa: E402
from state_space import (  # noqa: E402
    State,
    build_initial_state,
    format_state,
    is_goal,
    successors,
)


def remaining_milestones(state: State):
    """How many required milestones are still unmet, and the total count.

    High-risk milestones: ECG, troponin, cardiology referral, admit&treat.
    Low-risk milestones:  ECG, discharge.
    """
    if state.risk_level == "high":
        done = (("ECG" in state.confirmed)
                + ("troponin" in state.confirmed)
                + ("cardiology" in state.confirmed)
                + (state.disposition == "admitted_treated"))
        return 4 - done, 4
    done = ("ECG" in state.confirmed) + (state.disposition == "discharged")
    return 2 - done, 2


def heuristic(state: State) -> float:
    """h = unresolved-risk x urgency  (admissible; h = 0 at any goal)."""
    remaining, total = remaining_milestones(state)
    unresolved_risk = state.risk_probability * (remaining / total)
    urgency = 5 - state.priority          # priority 1 -> 4 (most urgent)
    return unresolved_risk * urgency


def astar(start: State):
    """Standard A*. Returns (path, total_cost, nodes_expanded).

    path is a list of (action, step_cost, resulting_state) from start to goal.
    """
    counter = itertools.count()            # tie-breaker so heap never compares States
    frontier = [(heuristic(start), next(counter), start)]
    g_score = {start: 0}
    came_from = {}                         # state -> (prev_state, action, step_cost)
    expanded = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if is_goal(current):
            path = []
            node = current
            while node in came_from:
                prev, action, cost = came_from[node]
                path.append((action, cost, node))
                node = prev
            path.reverse()
            return path, g_score[current], expanded

        expanded += 1
        for action, cost, nxt in successors(current):
            tentative_g = g_score[current] + cost
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                came_from[nxt] = (current, action, cost)
                f = tentative_g + heuristic(nxt)
                heapq.heappush(frontier, (f, next(counter), nxt))

    return None, None, expanded


def main() -> None:
    for label, patient in SAMPLE_PATIENTS.items():
        s0 = build_initial_state(patient)
        path, total, expanded = astar(s0)

        print("=" * 72)
        print(label)
        print("=" * 72)
        print("INITIAL :", format_state(s0))
        print(f"h(initial) = {heuristic(s0):.2f}   (unresolved-risk x urgency)\n")

        print("SEARCH PATH (cheapest safe route to a goal):")
        g = 0
        for i, (action, cost, state) in enumerate(path, 1):
            g += cost
            print(f"  {i}. {action:20s} (cost {cost}, total g={g})")
            print(f"       -> {format_state(state)}")
        print(f"\n  Total path cost : {total}")
        print(f"  Nodes expanded  : {expanded}")
        print(f"  Goal reached    : {is_goal(path[-1][2])}\n")


if __name__ == "__main__":
    main()
