from __future__ import annotations

import numpy as np
import pandas as pd


def transform_customer_data(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "customer_notes" not in result.columns:
        result["customer_notes"] = ""
    result["customer_notes"] = result["customer_notes"].fillna("").astype(str)
    if "licensed_seats" not in result.columns:
        result["licensed_seats"] = 25
    if "active_seat_ratio_30d" not in result.columns:
        result["active_seat_ratio_30d"] = 0.45
    if "champion_count" not in result.columns:
        result["champion_count"] = 1
    if "churned" not in result.columns:
        result["churned"] = pd.NA
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"]).dt.date.astype(str)
    numeric_columns = [
        "licensed_seats",
        "active_seat_ratio_30d",
        "champion_count",
        "tenure_months",
        "monthly_revenue",
        "logins_30d",
        "usage_change_pct",
        "support_tickets_90d",
        "nps",
        "days_to_renewal",
        "payment_failures_90d",
        "onboarding_complete",
        "churned",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["support_ticket_rate"] = (
        result["support_tickets_90d"] / np.maximum(result["tenure_months"], 1)
    ).round(4)
    result["engagement_score"] = (
        0.55 * np.clip(result["logins_30d"] / 40, 0, 1)
        + 0.25 * np.clip((result["usage_change_pct"] + 50) / 100, 0, 1)
        + 0.20 * result["onboarding_complete"]
    ).round(4)
    result["renewal_urgency"] = np.clip((90 - result["days_to_renewal"]) / 90, 0, 1).round(4)
    result["revenue_at_risk"] = (
        result["monthly_revenue"]
        * 12
        * (0.35 + 0.65 * result["renewal_urgency"])
    ).round(2)
    ordered_columns = [
        "customer_id",
        "segment",
        "region",
        "customer_notes",
        "licensed_seats",
        "active_seat_ratio_30d",
        "champion_count",
        "tenure_months",
        "monthly_revenue",
        "logins_30d",
        "usage_change_pct",
        "support_tickets_90d",
        "nps",
        "days_to_renewal",
        "payment_failures_90d",
        "onboarding_complete",
        "support_ticket_rate",
        "engagement_score",
        "renewal_urgency",
        "revenue_at_risk",
        "snapshot_date",
        "churned",
    ]
    return result[ordered_columns]
