---
name: deployment-recovery
description: Build an approval-aware enterprise AI rollout recovery plan for incomplete onboarding, low activation, permissions or security blockers, integration failures, unresolved support issues, or stalled production deployment. Use whenever technical enablement or governance decisions must be resolved before adoption, value, or commercial action.
---

# Deployment recovery

## Establish the input contract

1. Require onboarding status and the supplied evidence about rollout stage, usage, support burden, and renewal timing.
2. Use tenant, workflow, identity, permissions, data access, security, integration, owner, and telemetry details only when supplied.
3. Do not infer a technical root cause from low usage alone.
4. Read [references/rollout-checklist.md](references/rollout-checklist.md) when classifying blockers or sequencing recovery.

## Recover the rollout

1. Identify the last verified deployment stage and separate technical blockers, governance decisions, enablement gaps, support issues, and ownership gaps.
2. Assign an internal owner and customer counterpart where known; label missing counterparts instead of inventing names.
3. Sequence diagnose, unblock, validate, pilot, and scale. Define entry criteria, completion evidence, date, and rollback path for risky changes.
4. Choose the smallest production-representative workflow that can prove the blocker is removed.
5. Hand off to `adoption-reactivation` only after users can access and execute the target workflow.
6. Escalate unresolved ownership, customer commitments, or approval decisions through `stakeholder-alignment`.

## Contribute to the response

Return two to four prioritized actions with owner, customer counterpart if known, rationale, milestone, completion signal, expected impact, dependency, and `approval_required`.

- Put the verified rollout stage, blocker evidence, and missing diagnostics in `evidence`.
- Put scaling before pilot validation, invented root causes, and unapproved production changes in `do_not_do`.
- Lower `confidence` when the blocker category or current stage is unverified.
- Require human approval for production, permissions, customer records, external commitments, or commercial changes.
