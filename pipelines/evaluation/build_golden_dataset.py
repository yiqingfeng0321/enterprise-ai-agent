from __future__ import annotations

import json
from pathlib import Path

from app.repositories.customer_repository import CustomerRepository
from app.services.ml_service import MLService


def build_golden_dataset(
    repository: CustomerRepository,
    ml_service: MLService,
    output_path: Path,
    cases_per_band: int = 3,
) -> list[dict]:
    profiles = [repository.get(customer_id) for customer_id in repository.list_ids(limit=10_000)]
    profile_by_id = {profile.customer_id: profile for profile in profiles}
    scores = ml_service.score_many(profiles)
    scored = []
    for customer_id, probability, risk_band in scores:
        profile = profile_by_id[customer_id]
        expected_keywords = []
        scored.append(
            {
                "customer_id": customer_id,
                "expected_risk_band": risk_band,
                "required_action_keywords": expected_keywords[:3],
                "minimum_review_score": 0.8,
                "expected_human_approval": (
                    risk_band == "high" or profile.days_to_renewal <= 60
                ),
                "reference_probability": round(probability, 6),
            }
        )
    selected = []
    for band in ["high", "medium", "low"]:
        candidates = [item for item in scored if item["expected_risk_band"] == band]
        if band == "high":
            candidates.sort(key=lambda item: item["reference_probability"], reverse=True)
        elif band == "low":
            candidates.sort(key=lambda item: item["reference_probability"])
        else:
            candidates.sort(key=lambda item: abs(item["reference_probability"] - 0.55))
        selected.extend(candidates[:cases_per_band])

    for item in selected:
        profile = profile_by_id[item["customer_id"]]
        prediction = ml_service.predict(profile, top_n=5)
        risky_features = {
            driver.feature for driver in prediction.top_drivers if driver.direction == "risk"
        }
        expected_keywords = []
        if "support_tickets_90d" in risky_features or profile.support_tickets_90d >= 3:
            expected_keywords.append("support")
        if "usage_change_pct" in risky_features or profile.usage_change_pct <= -20:
            expected_keywords.append("usage")
        if not profile.onboarding_complete:
            expected_keywords.append("onboarding")
        if profile.days_to_renewal <= 60:
            expected_keywords.append("renewal")
        if profile.payment_failures_90d > 0:
            expected_keywords.append("payment")
        if not expected_keywords:
            expected_keywords.append("monitor")
        item["required_action_keywords"] = expected_keywords[:3]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item) + "\n")
    return selected
