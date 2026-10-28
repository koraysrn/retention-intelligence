"""Agent tools (function calling) with consent, budget and idempotency guards.

A natural-language command is translated into a concrete action, e.g. "send a
coupon to every risky customer". The action layer is deliberately conservative:

- explicit ``confirm=True`` is required (human approval step),
- only customers with marketing consent are contacted (when available),
- the recipient count and spend are capped,
- idempotency keys prevent duplicate sends,
- every action is appended to an audit log with actor/model version.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.channels.notifier import ChannelNotifier
from src.config import settings

MAX_DISCOUNT_PCT = 30.0
AUDIT_LOG = settings.artifacts_dir / "action_audit.jsonl"
_IDEMPOTENCY: set[str] = set()

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "send_coupon_to_risk_segment",
            "description": (
                "Creates a discount coupon and sends it to consenting high-risk customers. "
                "Requires explicit human confirmation (confirm=true) and an idempotency key."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "discount_pct": {
                        "type": "number",
                        "description": "Discount percentage between 0 and 30",
                    },
                    "validity_days": {
                        "type": "integer",
                        "description": "Coupon validity in days (e.g. 7)",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["email", "sms", "push"],
                        "description": "Delivery channel",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true; the caller confirms the action.",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Unique key so a retry cannot send twice.",
                    },
                },
                "required": ["discount_pct", "validity_days", "channel", "confirm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_risk_segments",
            "description": "Lists the number of customers in each risk segment.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _load_scores() -> pd.DataFrame:
    path = settings.data_processed / "scores.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_consent() -> pd.DataFrame:
    """Return ``customer_id`` + marketing consent from the customer source."""
    path = settings.ecommerce_data
    if not path.exists():
        return pd.DataFrame(columns=["customer_id", "marketing_opt_in"])
    try:
        df = pd.read_csv(path, usecols=["customer_id", "marketing_opt_in"])
        df["customer_id"] = df["customer_id"].astype(str)
        df["marketing_opt_in"] = df["marketing_opt_in"].astype(bool)
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["customer_id", "marketing_opt_in"])


def _audit(record: dict) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_risk_segments(scores: pd.DataFrame | None = None) -> dict:
    df = scores if scores is not None else _load_scores()
    if df.empty:
        return {"available": False, "segments": {}}
    dist = df["risk_tier"].value_counts().to_dict()
    return {"available": True, "segments": {str(k): int(v) for k, v in dist.items()}}


def send_coupon_to_risk_segment(
    discount_pct: float,
    validity_days: int,
    channel: str = "email",
    notifier: ChannelNotifier | None = None,
    scores: pd.DataFrame | None = None,
    confirm: bool = False,
    idempotency_key: str | None = None,
    budget_limit: float | None = None,
    max_recipients: int | None = None,
    cost_per_coupon: float | None = None,
    actor: str = "system",
    role: str = "admin",
    consent: pd.DataFrame | None = None,
) -> dict:
    """Create and send coupons to consenting high-risk customers.

    The action is a no-op unless ``confirm=True``. It never sends more than
    ``max_recipients`` or spends more than ``budget_limit``.
    """
    if not (0 < discount_pct <= MAX_DISCOUNT_PCT):
        raise ValueError(f"Discount must be between 0-%{MAX_DISCOUNT_PCT:.0f}")
    if validity_days <= 0:
        raise ValueError("validity_days must be positive")
    if not confirm:
        return {
            "sent": 0,
            "requires_confirmation": True,
            "message": "Human confirmation required. Re-run with confirm=true.",
        }

    key = idempotency_key or f"auto-{uuid.uuid4().hex}"
    if key in _IDEMPOTENCY:
        return {
            "sent": 0,
            "duplicate": True,
            "idempotency_key": key,
            "message": "Duplicate idempotency key; action was already executed.",
        }

    df = scores if scores is not None else _load_scores()
    if df.empty:
        return {"sent": 0, "message": "No score data found; run batch scoring first."}

    high_risk = df[df["risk_tier"] == "high"].copy()
    high_risk["customer_id"] = high_risk["customer_id"].astype(str)

    # Consent: prefer a column on the score frame, else join the customer source.
    if "marketing_opt_in" in high_risk.columns:
        high_risk = high_risk[high_risk["marketing_opt_in"].astype(bool)]
        consent_source = "scores"
    else:
        consent_frame = consent if consent is not None else _load_consent()
        if not consent_frame.empty:
            high_risk = high_risk.merge(
                consent_frame, on="customer_id", how="left", suffixes=("", "_consent")
            )
            high_risk = high_risk[high_risk["marketing_opt_in"].astype(bool)]
            consent_source = "customer_source"
        else:
            # Prototype fallback: no consent source available; do not silently
            # claim compliance in production.
            consent_source = "unavailable_assumed_opt_in"

    budget = settings.action_budget_limit if budget_limit is None else float(budget_limit)
    limit = settings.action_max_recipients if max_recipients is None else int(max_recipients)
    coupon_cost = (
        settings.action_cost_per_coupon if cost_per_coupon is None else float(cost_per_coupon)
    )
    affordable = int(budget // coupon_cost) if coupon_cost > 0 else limit
    allowed = max(0, min(len(high_risk), limit, affordable))
    if allowed == 0:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "actor": actor,
            "role": role,
            "action": "send_coupon_to_risk_segment",
            "sent": 0,
            "budget_limit": budget,
            "reason": "budget_or_recipient_cap",
            "model_version": str(df.get("model_version", pd.Series(["unknown"])).iloc[0]),
        }
        _audit(record)
        return {
            "sent": 0,
            "budget_exceeded": affordable == 0,
            "budget_limit": budget,
            "cost_per_coupon": coupon_cost,
            "message": "Action blocked by budget/recipient limits.",
        }

    selected = high_risk.head(allowed)
    notifier = notifier or ChannelNotifier()
    codes: list[str] = []
    sent_ids: list[str] = []

    for customer_id in selected["customer_id"].astype(str):
        code = f"COUPON-{uuid.uuid4().hex[:8].upper()}"
        codes.append(code)
        sent_ids.append(customer_id)
        message = (
            f"Your exclusive {discount_pct:g}% discount coupon: {code} "
            f"(valid for {validity_days} days). Use it on the items you last viewed "
            "or left in your cart."
        )
        notifier.send(customer_id, message, channels=[channel])

    _IDEMPOTENCY.add(key)
    model_version = str(df.get("model_version", pd.Series(["unknown"])).iloc[0])
    _audit(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "actor": actor,
            "role": role,
            "action": "send_coupon_to_risk_segment",
            "customer_ids": sent_ids,
            "discount_pct": float(discount_pct),
            "validity_days": int(validity_days),
            "channel": channel,
            "sent": int(len(sent_ids)),
            "budget_limit": budget,
            "cost_per_coupon": coupon_cost,
            "consent_source": consent_source,
            "idempotency_key": key,
            "model_version": model_version,
        }
    )

    return {
        "sent": int(len(sent_ids)),
        "discount_pct": float(discount_pct),
        "validity_days": int(validity_days),
        "channel": channel,
        "coupon_codes": codes[:5],
        "total_sent": len(notifier.sent),
        "budget_limit": budget,
        "cost_per_coupon": coupon_cost,
        "consent_source": consent_source,
        "idempotency_key": key,
        "model_version": model_version,
    }


def execute_tool(name: str, arguments: dict, **kwargs: Any) -> dict:
    """Translate a tool name and its arguments into a concrete action."""
    if name == "send_coupon_to_risk_segment":
        return send_coupon_to_risk_segment(
            discount_pct=float(arguments.get("discount_pct", 15)),
            validity_days=int(arguments.get("validity_days", 7)),
            channel=str(arguments.get("channel", "email")),
            notifier=kwargs.get("notifier"),
            scores=kwargs.get("scores"),
            confirm=bool(arguments.get("confirm", kwargs.get("confirm", False))),
            idempotency_key=arguments.get("idempotency_key", kwargs.get("idempotency_key")),
            budget_limit=kwargs.get("budget_limit"),
            max_recipients=kwargs.get("max_recipients"),
            cost_per_coupon=kwargs.get("cost_per_coupon"),
            actor=str(kwargs.get("actor", "system")),
            role=str(kwargs.get("role", "admin")),
            consent=kwargs.get("consent"),
        )
    if name == "list_risk_segments":
        return list_risk_segments(scores=kwargs.get("scores"))
    raise ValueError(f"Unknown tool: {name}")


def parse_mock_tool_call(message: str) -> list[dict]:
    """Deterministic command parsing when no LLM is available (test/demo fallback)."""
    q = message.lower()
    calls: list[dict] = []

    if ("coupon" in q or "discount" in q or "offer" in q) and (
        "send" in q or "issue" in q or "apply" in q or "trigger" in q
    ):
        m = re.search(r"%?\s*(\d{1,2})\s*[-–]\s*(\d{1,2})", q)
        if m:
            discount = max(int(m.group(1)), int(m.group(2)))
        else:
            m2 = re.search(r"%?\s*(\d{1,2})", q)
            discount = int(m2.group(1)) if m2 else 15
        calls.append(
            {
                "name": "send_coupon_to_risk_segment",
                "arguments": {
                    "discount_pct": discount,
                    "validity_days": 7,
                    "channel": "email",
                    # The deterministic parser never auto-confirms an action.
                    "confirm": False,
                },
            }
        )
    elif "segment" in q or "how many" in q or "list" in q or "breakdown" in q:
        calls.append({"name": "list_risk_segments", "arguments": {}})

    return calls
