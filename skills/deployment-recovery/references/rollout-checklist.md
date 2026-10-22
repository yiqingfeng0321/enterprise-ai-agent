# Rollout recovery checklist

## Classify the blocker

| Category | Verify | Typical owner | Approval boundary |
|---|---|---|---|
| Identity and permissions | Role assignment, tenant, access error, affected users | Customer IT + Solutions Engineering | Permission and production changes |
| Security and governance | Review status, control owner, unresolved decision | Security owner + Account team | Customer or security approval |
| Data and integration | Data source, connectivity, schema, failure evidence | Data/Platform Engineering | Production data and integration changes |
| Product or support | Incident, reproduction, severity, workaround | Support + Engineering | Ticket creation or commitment |
| Enablement | Access works, target workflow, user readiness | Adoption Consultant | External scheduling or outreach |
| Ownership | Missing decision maker, counterpart, or date | Customer Success leadership | Escalation and commitments |

Low usage alone is not sufficient evidence for any blocker category.

## Sequence recovery

1. **Diagnose:** confirm target workflow, user population, last verified stage, blocker evidence, and owner.
2. **Unblock:** choose the smallest change, define approval and rollback, and record completion evidence.
3. **Validate:** execute one production-representative workflow with agreed entry and success criteria.
4. **Pilot:** measure reliability and repeat use for a narrow cohort.
5. **Scale:** expand only after the pilot meets explicit criteria and telemetry is available.

## Escalate when

- the blocker has no accountable owner;
- a security or governance decision has no due date;
- production access or data changes are required;
- support severity or customer impact is unresolved;
- renewal timing makes the current recovery sequence infeasible.

## Example

Weak: “Low adoption is caused by permissions; fix permissions and roll out broadly.”

Strong: “Low adoption does not establish a permission failure. Verify access for the supplied pilot cohort, capture the exact failure, assign the identity owner, approve the smallest role change, and validate one target workflow before expanding.”
