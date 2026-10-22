---
name: stakeholder-alignment
description: Align customer champions, technical owners, executives, procurement, and internal teams around an enterprise renewal or recovery plan. Use for high-risk or near-renewal accounts, weak champion coverage, missing decision makers, ambiguous ownership, escalation, executive review, or any action requiring explicit human approval.
---

# Stakeholder alignment

## Establish the stakeholder contract

1. Use only supplied stakeholder names, roles, ownership, dates, and customer commitments.
2. Label missing roles and decision rights without inventing people.
3. Require the decision, evidence needed, accountable owner, customer counterpart if known, due date, and approval status for every alignment point.
4. Read [references/stakeholder-matrix.md](references/stakeholder-matrix.md) when mapping roles or sequencing escalation.

## Align decisions

1. Separate technical recovery, adoption, value proof, renewal, procurement, and commercial decisions.
2. Match escalation level to verified risk, revenue exposure, renewal timing, and unresolved dependency.
3. Build the smallest meeting sequence that resolves decisions; avoid meetings without an owner or decision.
4. Keep messaging consistent with verified facts, model uncertainty, memory, and approved commitments.
5. Do not treat a champion as an approver unless supplied evidence establishes that authority.
6. Stop before external outreach, discounts, tickets, record changes, production changes, or contractual commitments until explicit approval is recorded.

## Contribute to the response

Return a compact stakeholder map plus a sequenced set of actions. For each action include decision, accountable owner, customer counterpart if known, evidence required, due date or timing, completion signal, and `approval_required`.

- Put known roles and decision evidence in `evidence`.
- Put invented contacts, unowned escalations, and unapproved commitments in `do_not_do`.
- Lower `confidence` when decision rights, customer counterpart, due date, or approval status is unknown.
- Set `requires_human_approval` whenever any proposed step crosses an external side-effect or commitment boundary.
