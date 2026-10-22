from __future__ import annotations

import json
import time
from pathlib import Path

from app.workflows.customer_retention import CustomerRetentionWorkflow


def load_golden_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def evaluate_agent(
    workflow: CustomerRetentionWorkflow,
    golden_path: Path,
    output_path: Path | None = None,
) -> dict:
    cases = load_golden_cases(golden_path)
    if not cases:
        raise ValueError(f"Golden dataset is empty: {golden_path}")
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = await workflow.analyze(
            case["customer_id"],
            session_id="golden-evaluation",
            create_ticket=False,
            persist=False,
        )
        action_text = " ".join(
            action.action.lower() for action in result.recommendation.actions
        )
        keyword_hits = [
            keyword
            for keyword in case["required_action_keywords"]
            if keyword.lower() in action_text
        ]
        row = {
            "customer_id": case["customer_id"],
            "risk_band_correct": result.prediction.risk_band == case["expected_risk_band"],
            "keyword_coverage": len(keyword_hits)
            / max(len(case["required_action_keywords"]), 1),
            "review_passed": result.review.passed,
            "review_score": result.review.score,
            "review_score_meets_case_minimum": result.review.score
            >= case.get("minimum_review_score", 0.0),
            "has_evidence": len(result.recommendation.evidence) >= 2,
            "approval_alignment": result.recommendation.requires_human_approval
            == case["expected_human_approval"],
            "latency_ms": (time.perf_counter() - started) * 1000,
            "provider": result.provider,
            "model": result.model,
            "generation_mode": result.generation_mode,
            "fallback_used": result.fallback_used,
        }
        rows.append(row)
    metrics = {
        "case_count": len(rows),
        "risk_band_accuracy": sum(row["risk_band_correct"] for row in rows) / len(rows),
        "mean_keyword_coverage": sum(row["keyword_coverage"] for row in rows) / len(rows),
        "review_pass_rate": sum(row["review_passed"] for row in rows) / len(rows),
        "review_score_threshold_rate": sum(
            row["review_score_meets_case_minimum"] for row in rows
        )
        / len(rows),
        "evidence_rate": sum(row["has_evidence"] for row in rows) / len(rows),
        "approval_alignment_rate": sum(row["approval_alignment"] for row in rows)
        / len(rows),
        "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
        "fallback_rate": sum(row["fallback_used"] for row in rows) / len(rows),
    }
    provenance = {
        "providers": sorted({row["provider"] for row in rows}),
        "models": sorted({row["model"] for row in rows}),
        "generation_modes": sorted({row["generation_mode"] for row in rows}),
    }
    report = {"metrics": metrics, "provenance": provenance, "cases": rows}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
