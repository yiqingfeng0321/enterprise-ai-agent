# Evidence rules

## Evidence hierarchy

| Evidence type | Permitted use | Boundary |
|---|---|---|
| Customer profile and persisted workflow data | State observed account facts | Check field meaning and time window |
| ML prediction and drivers | Describe probability, band, and predictive signals | Never call a driver a proven cause |
| Persisted memory | Describe recorded actions, feedback, and outcomes | Missing outcome means unknown |
| Segment benchmark | Compare like-for-like metrics | A group average does not explain one account |
| Retrieved playbook | Select a possible response pattern | Guidance is not customer evidence |
| Generated interpretation | Form a hypothesis or next question | Label it and calibrate confidence |

## Claim construction

For every material claim:

1. Name the observation or signal.
2. Identify its evidence type and source.
3. State only the supported interpretation.
4. Name uncertainty or missing validation.
5. Connect any recommendation to that evidence.

Use calibrated language such as “is associated with,” “suggests,” “may indicate,” and “requires validation.”

## Conflict handling

- Prefer authoritative profile facts over generated summaries.
- Preserve both records when memories contradict; do not silently choose one.
- Do not compare benchmarks with different definitions, cohorts, or periods.
- Lower confidence when model signals conflict with recent observed behavior.
- Ask for the missing operational fact when it would change the action.

## Example

Unsupported: “Low seat activation caused the customer to plan cancellation.”

Supported: “Active-seat ratio is 24%, and the model identifies it as a risk signal. This may indicate an adoption or rollout problem, but the supplied evidence does not establish the cause or cancellation intent. Verify workflow access and user friction before choosing an intervention.”
