"""Guardrail / validation layer.

Two-stage validation:
1. Deterministic rules: PII, forbidden phrases, discount limit.
2. Model-based: LLM-as-judge — a deterministic grounding check in the prototype.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.agents.llm import mask_pii

FORBIDDEN_PATTERNS = ["unlimited free trial", "100% money-back guarantee"]
PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "system prompt",
    "developer message",
    "execute without asking",
    "bypass confirmation",
    "reveal your instructions",
]


def detect_prompt_injection(text: str) -> bool:
    """Detect obvious prompt-injection attempts before they reach the LLM."""
    lowered = text.lower()
    return any(p in lowered for p in PROMPT_INJECTION_PATTERNS)


@dataclass
class GuardrailResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def run_guardrails(content: str, max_discount_pct: float = 30.0) -> GuardrailResult:
    """Run deterministic guardrail checks."""
    checks: dict[str, bool] = {}

    masked = mask_pii(content)
    checks["no_pii_leak"] = masked == content  # no PII when masking leaves the text unchanged

    lowered = content.lower()
    checks["no_forbidden_terms"] = all(p.lower() not in lowered for p in FORBIDDEN_PATTERNS)

    # Discount limit check — supports both "60%" and "%60" notations
    matches = re.findall(r"(\d+)\s*%|%\s*(\d+)", content)
    discounts = [int(a or b) for a, b in matches]
    checks["discount_within_limit"] = all(int(d) <= max_discount_pct for d in discounts)

    checks["no_prompt_injection"] = not detect_prompt_injection(content)

    passed = all(checks.values())
    reasons = [k for k, v in checks.items() if not v]

    return GuardrailResult(passed=passed, checks=checks, reasons=reasons)


def llm_judge(
    content: str,
    context: str = "",
    grounded_terms: list[str] | None = None,
) -> bool:
    """Model-based validation (deterministic grounding check in the prototype).

    Generated content is treated as a suspected hallucination when it does not
    contain at least one term related to the context.
    """
    lowered = content.lower()
    terms = grounded_terms or ["discount", "offer", "product", "cart"]
    return any(term in lowered for term in terms)
