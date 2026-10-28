"""Orchestrator Agent — LangGraph-based multi-agent workflow.

Flow (docs/agentic_ai_design.md):
    trigger -> profile -> rag -> content generation -> guardrail -> send/escalation

When LangGraph is not installed, a deterministic sequential fallback runs the
same nodes in order.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd

from src.agents.escalation import (
    EscalationCase,
    build_summary,
    determine_priority,
    should_escalate,
)
from src.agents.guardrails import llm_judge, run_guardrails
from src.agents.llm import LLMClient, build_email_prompt, mask_pii
from src.agents.rag import RAGRetriever, RetrievalResult
from src.config import PROJECT_ROOT, load_yaml, settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    customer_id: str
    profile: dict
    risk_reason: str
    context: str
    retrieval: RetrievalResult
    draft_content: str
    guardrail_passed: bool
    guardrail_reasons: list[str]
    escalated: bool
    escalation_summary: str
    final_email: str | None
    decision: str


def load_agents_config() -> dict:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "config.yaml")
    agents = cfg.get("agents", {})
    return agents if isinstance(agents, dict) else {}


def _native(value: Any) -> Any:
    """Convert numpy scalars to JSON-compatible native Python types."""
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def load_customer_profile(customer_id: str) -> dict:
    """Load customer features from the feature store (parquet)."""
    try:
        path = settings.data_processed / "customer_features.parquet"
        df = pd.read_parquet(path)
        mask = df["customer_id"].astype(str) == str(customer_id)
        if mask.any():
            return dict(_native(df.loc[mask].iloc[0].to_dict()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Profile could not be loaded, using an empty profile: %s", exc)
    return {}


def _profile_node(state: AgentState) -> AgentState:
    profile = state.get("profile") or load_customer_profile(state["customer_id"])
    risk_reason = state.get("risk_reason") or (
        "High churn risk: elevated recency, low purchase activity"
    )
    return {"profile": profile, "risk_reason": risk_reason}


def _rag_node(state: AgentState) -> AgentState:
    cfg = load_agents_config()
    top_k = int(cfg.get("top_k_retrieval", 5))
    profile = state.get("profile", {})
    query = (
        f"{state.get('customer_id', '')} {state.get('risk_reason', '')} "
        f"country {profile.get('country', '')} spend {profile.get('total_spend_usd', profile.get('monetary', ''))} "
        "discount campaign product offer"
    )
    result = RAGRetriever().retrieve(query, top_k)
    return {"retrieval": result, "context": "\n".join(result.chunks)}


def _generation_node(state: AgentState) -> AgentState:
    system, user = build_email_prompt(
        state.get("profile", {}),
        state.get("context", ""),
        state.get("risk_reason", ""),
    )
    resp = LLMClient().complete(system, user)
    return {"draft_content": resp.content}


def _guardrail_node(state: AgentState) -> AgentState:
    cfg = load_agents_config()
    max_discount = float(cfg.get("max_discount_pct", 30.0))
    content = state.get("draft_content", "")
    result = run_guardrails(content, max_discount)
    checks = dict(result.checks)
    reasons = list(result.reasons)
    # Model-based grounding check is part of the workflow, not just tests.
    grounded = llm_judge(content, state.get("context", ""))
    checks["grounded"] = grounded
    if not grounded:
        reasons.append("grounded")
    return {"guardrail_passed": all(checks.values()), "guardrail_reasons": reasons}


def _should_send(state: AgentState) -> str:
    cfg = load_agents_config()
    high_ltv = float(cfg.get("high_ltv_threshold", 1000.0))
    conf_thr = float(cfg.get("confidence_threshold", 0.8))

    profile = state.get("profile", {})
    ltv = float(profile.get("total_spend_usd", profile.get("monetary", 0.0)) or 0.0)
    confidence = float(profile.get("churn_probability", 0.5) or 0.5)
    complaints = int(profile.get("support_complaints", 0) or 0)

    if not state.get("guardrail_passed", False):
        return "escalate"
    if should_escalate(state["customer_id"], ltv, confidence, True, complaints, high_ltv, conf_thr):
        return "escalate"
    return "send"


def _send_node(state: AgentState) -> AgentState:
    return {
        "escalated": False,
        "decision": "SENT",
        "final_email": state.get("draft_content", ""),
    }


def _escalate_node(state: AgentState) -> AgentState:
    cfg = load_agents_config()
    high_ltv = float(cfg.get("high_ltv_threshold", 1000.0))

    profile = state.get("profile", {})
    ltv = float(profile.get("total_spend_usd", profile.get("monetary", 0.0)) or 0.0)
    confidence = float(profile.get("churn_probability", 0.5) or 0.5)
    complaints = int(profile.get("support_complaints", 0) or 0)

    reasons = state.get("guardrail_reasons", [])
    reason = (
        "; ".join(reasons) if reasons else "High LTV + low confidence (human approval required)"
    )
    priority = determine_priority(ltv, confidence, complaints, high_ltv)

    case = EscalationCase(
        customer_id=state["customer_id"],
        reason=reason,
        summary=(f"Churn risk reason: {state.get('risk_reason', '')}. Profile: {profile}"),
        recommended_action="Sales representative should call and present a personalized offer",
        priority=priority,
    )
    return {
        "escalated": True,
        "decision": "ESCALATED",
        "escalation_summary": build_summary(case),
        "final_email": None,
    }


def build_graph():
    """Build and compile the LangGraph StateGraph."""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("profile_node", _profile_node)
    graph.add_node("rag_node", _rag_node)
    graph.add_node("generate_node", _generation_node)
    graph.add_node("guardrail_node", _guardrail_node)
    graph.add_node("send_node", _send_node)
    graph.add_node("escalate_node", _escalate_node)

    try:
        from langgraph.graph import START

        graph.add_edge(START, "profile_node")
    except ImportError:
        graph.set_entry_point("profile_node")

    graph.add_edge("profile_node", "rag_node")
    graph.add_edge("rag_node", "generate_node")
    graph.add_edge("generate_node", "guardrail_node")
    graph.add_conditional_edges(
        "guardrail_node",
        _should_send,
        {"send": "send_node", "escalate": "escalate_node"},
    )
    graph.add_edge("send_node", END)
    graph.add_edge("escalate_node", END)
    return graph.compile()


def _run_sequential(customer_id: str, profile: dict) -> AgentState:
    """Deterministic fallback used when LangGraph is unavailable."""
    state: AgentState = {"customer_id": customer_id, "profile": profile}
    state.update(_profile_node(state))
    state.update(_rag_node(state))
    state.update(_generation_node(state))
    state.update(_guardrail_node(state))
    if _should_send(state) == "send":
        state.update(_send_node(state))
    else:
        state.update(_escalate_node(state))
    return state


def run_workflow(customer_id: str, profile: dict | None = None) -> AgentState:
    """Run the end-to-end agent flow for a high-risk customer."""
    try:
        graph = build_graph()
        initial: AgentState = {"customer_id": customer_id, "profile": profile or {}}
        return cast(AgentState, dict(graph.invoke(initial)))
    except ImportError:
        return _run_sequential(customer_id, profile or {})


def _serialize_state(state: AgentState) -> dict:
    out: dict = {}
    for key, value in state.items():
        if key == "retrieval" and isinstance(value, RetrievalResult):
            out[key] = {"chunks": value.chunks, "sources": value.sources, "scores": value.scores}
        elif key == "final_email":
            out[key] = mask_pii(value if isinstance(value, str) else "")
        else:
            out[key] = _native(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic AI re-engagement flow")
    parser.add_argument("customer_id", nargs="?", default="CUST1000")
    parser.add_argument(
        "--out", type=Path, default=settings.artifacts_dir / "agent_run_report.json"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_workflow(args.customer_id)
    payload = _serialize_state(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Agent report saved: %s", args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
