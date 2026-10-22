from __future__ import annotations

import argparse
import asyncio
import json

from app.core.container import get_container
from pipelines.evaluation.run_agent_eval import evaluate_agent


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run and gate the golden-dataset evaluation")
    parser.add_argument("--min-risk-band-accuracy", type=float, default=0.95)
    parser.add_argument("--min-keyword-coverage", type=float, default=0.65)
    parser.add_argument("--min-review-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-evidence-rate", type=float, default=1.0)
    args = parser.parse_args()
    container = get_container()
    report = await evaluate_agent(
        container.workflow,
        container.settings.golden_dataset_path,
        container.settings.database_path.parent / "evaluations" / "agent_eval.json",
    )
    print(json.dumps(report["metrics"], indent=2))
    metrics = report["metrics"]
    failures = []
    checks = {
        "risk_band_accuracy": args.min_risk_band_accuracy,
        "mean_keyword_coverage": args.min_keyword_coverage,
        "review_pass_rate": args.min_review_pass_rate,
        "evidence_rate": args.min_evidence_rate,
    }
    for metric, minimum in checks.items():
        if metrics[metric] < minimum:
            failures.append(f"{metric}={metrics[metric]:.3f} < {minimum:.3f}")
    if failures:
        raise SystemExit("Evaluation quality gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
