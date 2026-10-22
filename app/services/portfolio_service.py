from __future__ import annotations

from app.repositories.customer_repository import CustomerRepository
from app.services.ml_service import MLService


class PortfolioService:
    """Read-only portfolio analytics used by dashboards and conversational tools."""

    def __init__(self, repository: CustomerRepository, ml_service: MLService):
        self.repository = repository
        self.ml_service = ml_service

    def ranked_customers(
        self,
        *,
        limit: int = 10,
        segment: str | None = None,
        renewal_within_days: int | None = None,
        risk_band: str | None = None,
    ) -> list[dict]:
        limit = min(max(int(limit), 1), 1_000)
        profiles = [
            self.repository.get(customer_id)
            for customer_id in self.repository.list_ids(limit=10_000)
        ]
        profile_by_id = {profile.customer_id: profile for profile in profiles}
        rows: list[dict] = []
        for customer_id, probability, predicted_band in self.ml_service.score_many(profiles):
            profile = profile_by_id[customer_id]
            if segment and profile.segment.lower() != segment.strip().lower():
                continue
            if renewal_within_days is not None and profile.days_to_renewal > renewal_within_days:
                continue
            if risk_band and predicted_band != risk_band.strip().lower():
                continue
            rows.append(
                {
                    "customer_id": customer_id,
                    "segment": profile.segment,
                    "region": profile.region,
                    "monthly_revenue": profile.monthly_revenue,
                    "revenue_at_risk": profile.revenue_at_risk,
                    "days_to_renewal": profile.days_to_renewal,
                    "renewal_timing": (
                        "within_30_days"
                        if profile.days_to_renewal <= 30
                        else "within_60_days"
                        if profile.days_to_renewal <= 60
                        else "more_than_60_days"
                    ),
                    "active_seat_ratio_30d": profile.active_seat_ratio_30d,
                    "usage_change_pct": profile.usage_change_pct,
                    "support_tickets_90d": profile.support_tickets_90d,
                    "churn_probability": probability,
                    "risk_band": predicted_band,
                    "ranking_basis": "churn_probability_desc",
                }
            )
        rows.sort(key=lambda item: item["churn_probability"], reverse=True)
        return rows[:limit]

    def summary(self) -> dict[str, int | float]:
        rows = self.ranked_customers(limit=1_000)
        return {
            "customer_count": len(rows),
            "high_risk_count": sum(row["risk_band"] == "high" for row in rows),
            "medium_risk_count": sum(row["risk_band"] == "medium" for row in rows),
            "renewal_within_30_days": sum(row["days_to_renewal"] <= 30 for row in rows),
            "high_risk_revenue_at_risk": round(
                sum(
                    float(row["revenue_at_risk"])
                    for row in rows
                    if row["risk_band"] == "high"
                ),
                2,
            ),
        }
