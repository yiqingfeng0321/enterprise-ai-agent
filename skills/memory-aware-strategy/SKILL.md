---
name: memory-aware-strategy
description: Adapt an enterprise customer-retention plan using persisted prior interactions, interventions, feedback, and recorded outcomes. Use when long-term memory exists, when asked what has already been tried, or when the next plan must avoid repeating ineffective outreach, discounts, workshops, escalation, or recovery tactics.
---

# Memory-aware strategy

## Establish the memory contract

1. Use only persisted memory items supplied for the same customer.
2. Separate prior action, customer feedback, recorded outcome, unresolved item, and generated summary.
3. Treat missing outcome evidence as unknown, not as success or failure.
4. Read [references/intervention-outcome-rules.md](references/intervention-outcome-rules.md) when classifying history or deciding whether a tactic can be reused.

## Adapt the plan

1. Order relevant memories by time and identify the intervention, intended outcome, actual recorded outcome, and open dependency.
2. Mark each relevant tactic `continue`, `stop`, `change`, or `insufficient evidence`.
3. Stop a tactic explicitly rejected or recorded as ineffective unless new evidence materially changes the conditions.
4. Reuse a successful pattern only when the current workflow, cohort, blocker, and timing remain comparable.
5. Choose a materially different lever after failure and explain how it addresses the recorded reason.
6. Never let generated text overwrite profile facts, model outputs, or stored outcomes.

## Contribute to the response

- Connect every continue, stop, or change decision to a cited memory item.
- Add the adapted action with owner, rationale, expected impact, new measurement, and approval boundary.
- Put tactics that should not be repeated in `do_not_do`.
- Lower `confidence` for stale, contradictory, unscoped, or outcome-free memories.
- Require human approval before repeating or changing a customer-facing commitment, discount, ticket, production change, or customer record.
