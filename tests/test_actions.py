"""Agent action (tools) tests."""

import pandas as pd
import pytest
from src.agents.actions import (
    execute_tool,
    list_risk_segments,
    parse_mock_tool_call,
    send_coupon_to_risk_segment,
)
from src.channels.notifier import ChannelNotifier


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "churn_probability": [0.9, 0.8, 0.5, 0.2],
            "risk_tier": ["high", "high", "medium", "low"],
            "marketing_opt_in": [True, True, True, False],
            "model_version": ["test-v1", "test-v1", "test-v1", "test-v1"],
        }
    )


def test_send_coupon_creates_and_sends() -> None:
    notifier = ChannelNotifier(enabled_channels=["email"])
    result = send_coupon_to_risk_segment(
        discount_pct=15,
        validity_days=7,
        channel="email",
        notifier=notifier,
        scores=_scores(),
        confirm=True,
        idempotency_key="test-send-1",
    )
    assert result["sent"] == 2  # 2 high-risk, consenting customers
    assert result["discount_pct"] == 15.0
    assert len(result["coupon_codes"]) == 2
    assert len(notifier.sent) == 2
    assert "COUPON-" in notifier.sent[0].content


def test_send_coupon_requires_confirmation() -> None:
    result = send_coupon_to_risk_segment(
        discount_pct=15, validity_days=7, scores=_scores(), confirm=False
    )
    assert result["sent"] == 0
    assert result["requires_confirmation"] is True


def test_send_coupon_is_idempotent() -> None:
    notifier = ChannelNotifier(enabled_channels=["email"])
    first = send_coupon_to_risk_segment(
        discount_pct=15,
        validity_days=7,
        notifier=notifier,
        scores=_scores(),
        confirm=True,
        idempotency_key="same-key",
    )
    second = send_coupon_to_risk_segment(
        discount_pct=15,
        validity_days=7,
        notifier=notifier,
        scores=_scores(),
        confirm=True,
        idempotency_key="same-key",
    )
    assert first["sent"] == 2
    assert second["sent"] == 0
    assert second["duplicate"] is True
    assert len(notifier.sent) == 2


def test_send_coupon_respects_consent() -> None:
    scores = _scores()
    scores.loc[scores["customer_id"] == "C2", "marketing_opt_in"] = False
    notifier = ChannelNotifier(enabled_channels=["email"])
    result = send_coupon_to_risk_segment(
        discount_pct=15,
        validity_days=7,
        notifier=notifier,
        scores=scores,
        confirm=True,
        idempotency_key="consent-test",
    )
    assert result["sent"] == 1


def test_send_coupon_respects_budget() -> None:
    notifier = ChannelNotifier(enabled_channels=["email"])
    result = send_coupon_to_risk_segment(
        discount_pct=15,
        validity_days=7,
        notifier=notifier,
        scores=_scores(),
        confirm=True,
        idempotency_key="budget-test",
        budget_limit=50,
        cost_per_coupon=50,
    )
    assert result["sent"] == 1


def test_send_coupon_invalid_discount_raises() -> None:
    with pytest.raises(ValueError):
        send_coupon_to_risk_segment(40, 7, scores=_scores())


def test_send_coupon_invalid_validity_raises() -> None:
    with pytest.raises(ValueError):
        send_coupon_to_risk_segment(15, 0, scores=_scores())


def test_list_risk_segments() -> None:
    result = list_risk_segments(scores=_scores())
    assert result["available"] is True
    assert result["segments"] == {"high": 2, "medium": 1, "low": 1}


def test_execute_tool_dispatch() -> None:
    result = execute_tool("list_risk_segments", {}, scores=_scores())
    assert result["segments"]["high"] == 2


def test_execute_tool_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        execute_tool("unknown", {})


def test_parse_mock_tool_call_coupon() -> None:
    calls = parse_mock_tool_call("send a 10-15% discount coupon to high-risk customers for 7 days")
    assert len(calls) == 1
    assert calls[0]["name"] == "send_coupon_to_risk_segment"
    assert calls[0]["arguments"]["discount_pct"] == 15
    assert calls[0]["arguments"]["validity_days"] == 7


def test_parse_mock_tool_call_segments() -> None:
    calls = parse_mock_tool_call("list the risk segments")
    assert len(calls) == 1
    assert calls[0]["name"] == "list_risk_segments"


def test_parse_mock_no_action() -> None:
    assert parse_mock_tool_call("hello") == []
