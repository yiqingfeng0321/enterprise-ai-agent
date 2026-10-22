---
name: adoption-reactivation
description: Design measurable workflow-level actions that restore active usage, feature adoption, and time to value. Use when usage is declining, active-seat ratio or engagement is low, deployed capabilities are not becoming repeatable habits, or an account needs an adoption experiment after technical blockers are cleared.
---

# Adoption reactivation

## Establish the input contract

1. Require current usage change, active-seat ratio, onboarding status, and known customer context.
2. Use workflow, cohort, telemetry, champion, benchmark, and memory evidence when supplied.
3. Name missing workflow-level telemetry rather than inventing a feature, user cohort, or source of friction.
4. Read [references/adoption-patterns.md](references/adoption-patterns.md) when selecting an intervention or success metric.

## Design the recovery experiment

1. Stop and coordinate with `deployment-recovery` when permissions, integration, security, tenant setup, or incomplete onboarding prevents use.
2. Select one or two high-value workflows supported by evidence; do not propose a generic training campaign.
3. Define the target cohort, observed friction, desired repeatable behavior, owner, and smallest milestone.
4. Choose a leading usage signal and a dated checkpoint. Treat retention impact as a hypothesis, not a promised outcome.
5. Check `memory-aware-strategy` before repeating workshops, outreach, incentives, or enablement that previously failed.
6. Add `stakeholder-alignment` when ownership, champion coverage, or customer approval is missing.

## Contribute to the response

For every proposed action provide priority, action, owner, rationale, expected impact, measurable completion signal, checkpoint, and `approval_required`.

- Put observed adoption evidence and assumptions in `evidence`.
- Put unsupported causal claims, blanket outreach, and expansion before validation in `do_not_do`.
- Lower `confidence` when workflow-level usage or cohort evidence is missing.
- Require human approval for customer commitments, external outreach, record changes, tickets, or commercial incentives.
