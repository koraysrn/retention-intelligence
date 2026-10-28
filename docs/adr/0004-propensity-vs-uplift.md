# ADR 0004 — Propensity risk model, not an uplift model

## Context

The business wants to avoid wasted discounts. An uplift model estimates the
*causal* effect of an offer per customer; a propensity model estimates churn
risk. They answer different questions.

## Decision

Ship a **propensity/risk** model for prioritisation, and measure campaign impact
with the A/B layer (stratified assignment, CUPED, sequential testing). Do not
present the risk score as an uplift estimate.

## Consequences

- Risk ranking is available immediately and is easy to explain with SHAP.
- High risk does not necessarily mean “persuadable”; the A/B result is the
  source of truth for campaign lift.
- A future uplift model would need treatment/control outcome data and a
  different evaluation protocol (Qini/uplift curves).
