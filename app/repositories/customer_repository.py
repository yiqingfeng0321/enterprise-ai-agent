from __future__ import annotations

from app.core.exceptions import CustomerNotFoundError
from app.models.domain import CustomerProfile
from app.repositories.database import Database


class CustomerRepository:
    def __init__(self, database: Database):
        self.database = database

    def get(self, customer_id: str) -> CustomerProfile:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM customer_current_features WHERE customer_id = ?", (customer_id,)
            ).fetchone()
        if row is None:
            raise CustomerNotFoundError(f"Customer {customer_id!r} was not found")
        payload = dict(row)
        payload.setdefault("customer_notes", "")
        payload["onboarding_complete"] = bool(payload["onboarding_complete"])
        return CustomerProfile.model_validate(payload)

    def list_ids(self, limit: int = 100) -> list[str]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT customer_id FROM customer_current_features ORDER BY customer_id LIMIT ?", (limit,)
            ).fetchall()
        return [str(row["customer_id"]) for row in rows]

    def segment_benchmark(self, segment: str) -> dict[str, float | int | str]:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    segment,
                    COUNT(*) AS customer_count,
                    AVG(logins_30d) AS avg_logins_30d,
                    AVG(active_seat_ratio_30d) AS avg_active_seat_ratio_30d,
                    AVG(champion_count) AS avg_champion_count,
                    AVG(usage_change_pct) AS avg_usage_change_pct,
                    AVG(support_tickets_90d) AS avg_support_tickets_90d,
                    AVG(nps) AS avg_nps,
                    AVG(churned) AS observed_churn_rate
                FROM customer_history_features
                WHERE segment = ?
                GROUP BY segment
                """,
                (segment,),
            ).fetchone()
        return dict(row) if row else {"segment": segment, "customer_count": 0}
