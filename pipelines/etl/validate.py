from __future__ import annotations

import pandas as pd

from app.core.exceptions import DataValidationError

BASE_REQUIRED_COLUMNS = {
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
    "snapshot_date",
}


def validate_raw_data(frame: pd.DataFrame, *, requires_label: bool = True) -> dict[str, int | float | bool]:
    required_columns = set(BASE_REQUIRED_COLUMNS)
    if requires_label:
        required_columns.add("churned")
    missing = required_columns.difference(frame.columns)
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        errors.append("Dataset is empty")
    if "customer_id" in frame and frame["customer_id"].duplicated().any():
        errors.append("customer_id must be unique")
    checks = {
        "monthly_revenue": (0, None),
        "licensed_seats": (1, None),
        "active_seat_ratio_30d": (0, 1),
        "champion_count": (0, None),
        "tenure_months": (0, None),
        "logins_30d": (0, None),
        "support_tickets_90d": (0, None),
        "nps": (-100, 100),
        "days_to_renewal": (0, 366),
        "payment_failures_90d": (0, None),
        "onboarding_complete": (0, 1),
        "churned": (0, 1),
    }
    for column, (minimum, maximum) in checks.items():
        if column not in frame:
            continue
        if minimum is not None and (frame[column] < minimum).any():
            errors.append(f"{column} contains values below {minimum}")
        if maximum is not None and (frame[column] > maximum).any():
            errors.append(f"{column} contains values above {maximum}")
    null_rate = float(frame[list(required_columns.intersection(frame.columns))].isna().mean().max())
    if null_rate > 0.05:
        errors.append(f"Maximum required-column null rate {null_rate:.1%} exceeds 5%")
    if errors:
        raise DataValidationError("; ".join(errors))
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_customers": int(frame["customer_id"].duplicated().sum()),
        "max_null_rate": null_rate,
        "requires_label": requires_label,
    }
