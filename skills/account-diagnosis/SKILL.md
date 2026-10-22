---
name: account-diagnosis
description: Diagnose enterprise customer churn risk from profile facts, ML predictions and drivers, segment benchmarks, retrieved playbooks, and persisted memory. Use before proposing any retention plan, when asked why an account is risky, or whenever observed facts, model signals, prior history, and interpretation must be separated.
---

# Account diagnosis

## Establish the evidence contract

1. Require a customer profile and prediction containing churn probability, risk band, model version, and top drivers.
2. Use benchmarks, retrieved playbooks, and memory only when supplied; name missing sources instead of filling gaps.
3. Treat profile and persisted workflow data as authoritative facts, ML output as a predictive signal, memory as recorded history, and playbooks as guidance.
4. Read [references/evidence-rules.md](references/evidence-rules.md) before resolving conflicting evidence or making a material claim.

## Diagnose

1. State the risk probability, band, model version, renewal timing, and two or three decision-relevant drivers.
2. Compare account metrics with a supplied segment benchmark only when definitions and periods are comparable.
3. Distinguish symptoms from verified blockers. Never convert a model driver or correlation into a root cause.
4. Check memory for prior interventions and outcomes before recommending the next lever.
5. Identify contradictions, missing evidence, and the next question needed to improve confidence.
6. Route technical blockers to `deployment-recovery`, usage weakness to `adoption-reactivation`, prior failed tactics to `memory-aware-strategy`, value gaps to `roi-value-story`, and ownership or approval gaps to `stakeholder-alignment`.

## Contribute to the response

Return content compatible with the enclosing recommendation:

- `risk_assessment`: primary risks, evidence type, uncertainty, and renewal urgency.
- `evidence`: trace every material claim to a profile field, model signal, memory item, benchmark, or retrieved source.
- `recommended_actions`: include only the next diagnostic or recovery steps justified by evidence.
- `do_not_do`: include unsupported causal claims, invented customer statements, and premature commercial action.
- `confidence`: lower it when required inputs are absent, stale, contradictory, or weak.
- `requires_human_approval`: set it for customer-facing commitments, discounts, record changes, tickets, or production changes.
