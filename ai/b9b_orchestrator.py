"""
CardioTriage AI — AI Phase, Step B9b: Gemini + LangChain orchestrator.

A Gemini chat model (via LangChain tool-calling) acts as the CONDUCTOR: it
calls our three deterministic tools (ML risk, A* pathway, knowledge base) and
writes a plain-English triage briefing. The LLM never makes a medical decision
itself — the tools do all the reasoning; the LLM only sequences and explains.

Setup: the Gemini API key is read from GOOGLE_API_KEY (loaded from .env).

Run with:  uv run python ai/b9b_orchestrator.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
load_dotenv(ROOT / ".env")

from langchain.agents import create_agent  # noqa: E402  (LangChain 1.x API)
from langchain_core.tools import tool  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402

from b7_forward_chain import HIGH_RISK, LOW_RISK  # noqa: E402
from b9_integration import (  # noqa: E402
    assess_risk_tool,
    knowledge_base_tool,
    plan_pathway_tool,
)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

# The patient currently under review. Tools read this so the LLM doesn't have
# to shuttle a JSON blob around (more reliable for a demo).
_CURRENT: dict = {}


def set_patient(patient: dict) -> None:
    _CURRENT.clear()
    _CURRENT.update(patient)


# --- LangChain tools (thin wrappers over our deterministic functions) -----
@tool
def assess_cardiac_risk() -> str:
    """Run the trained ML model. Returns the patient's cardiac risk
    probability, the high-risk flag, and the risk phenotype."""
    return assess_risk_tool(json.dumps(_CURRENT))


@tool
def plan_clinical_pathway() -> str:
    """Run the A* search agent. Returns the recommended ordered clinical
    pathway and the final disposition (admit & treat vs discharge)."""
    return plan_pathway_tool(json.dumps(_CURRENT))


@tool
def consult_knowledge_base() -> str:
    """Run the rule-based knowledge base. Returns confirmations,
    complementary care, and any SAFETY OVERRIDES (vetoes that take
    precedence over the pathway)."""
    return knowledge_base_tool(json.dumps(_CURRENT))


SYSTEM_PROMPT = (
    "You are CardioTriage AI, a clinical decision-SUPPORT assistant. You do "
    "NOT make medical decisions yourself: you orchestrate three deterministic "
    "tools and explain their combined output to a clinician.\n\n"
    "For every patient you MUST call all three tools, in this order:\n"
    "1. assess_cardiac_risk  2. plan_clinical_pathway  3. consult_knowledge_base\n\n"
    "Then write a concise briefing (<=120 words) covering: the risk level and "
    "phenotype; the recommended disposition and pathway; whether the knowledge "
    "base agrees; and CRUCIALLY any SAFETY OVERRIDES. Safety overrides take "
    "precedence — never advise something the knowledge base vetoes. Use only "
    "facts returned by the tools; do not invent clinical details."
)


def build_agent():
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    tools = [assess_cardiac_risk, plan_clinical_pathway, consult_knowledge_base]
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def extract_text(content) -> str:
    """Gemini may return a plain string or a list of content blocks; return
    just the human-readable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", str(block)))
        return "".join(parts)
    return str(content)


def print_tool_trace(messages) -> None:
    """Show which tools the LLM chose to call (the orchestration trace)."""
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            print(f"   -> LLM called tool: {call['name']}")
        if m.__class__.__name__ == "ToolMessage":
            print(f"      result: {m.content}")


def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found (check .env). Aborting.")
        return

    print(f"Using Gemini model: {MODEL_NAME}\n")
    agent = build_agent()

    for label, patient in [("HIGH-RISK case", HIGH_RISK), ("LOW-RISK case", LOW_RISK)]:
        set_patient(patient)
        print("#" * 74)
        print(f"# {label}")
        print("#" * 74)
        result = agent.invoke(
            {"messages": [("user",
             "Assess the current patient and produce the triage briefing.")]}
        )
        print("ORCHESTRATION TRACE:")
        print_tool_trace(result["messages"])
        print("\n=== GEMINI BRIEFING ===")
        print(extract_text(result["messages"][-1].content), "\n")


if __name__ == "__main__":
    main()
