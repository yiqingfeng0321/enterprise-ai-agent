import pandas as pd
import pytest

from app.core.exceptions import DataValidationError
from pipelines.etl.transform import transform_customer_data
from pipelines.etl.validate import validate_raw_data


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "CUST-1",
                "segment": "SMB",
                "region": "NA",
                "customer_notes": "phased rollout; needs role-based adoption training",
                "licensed_seats": 24,
                "active_seat_ratio_30d": 0.41,
                "champion_count": 1,
                "tenure_months": 12,
                "monthly_revenue": 500,
                "logins_30d": 10,
                "usage_change_pct": -20,
                "support_tickets_90d": 2,
                "nps": 10,
                "days_to_renewal": 30,
                "payment_failures_90d": 0,
                "onboarding_complete": 1,
                "churned": 0,
                "snapshot_date": "2026-07-01",
            }
        ]
    )


def test_etl_validation_and_derived_features() -> None:
    frame = valid_frame()
    report = validate_raw_data(frame)
    transformed = transform_customer_data(frame)
    assert report["rows"] == 1
    assert "engagement_score" in transformed
    assert transformed.loc[0, "revenue_at_risk"] > 0


def test_validation_rejects_out_of_range_nps() -> None:
    frame = valid_frame()
    frame.loc[0, "nps"] = 200
    with pytest.raises(DataValidationError):
        validate_raw_data(frame)


def test_validation_supports_current_unlabeled_snapshot() -> None:
    frame = valid_frame().drop(columns=["churned"])
    report = validate_raw_data(frame, requires_label=False)
    transformed = transform_customer_data(frame)
    assert report["requires_label"] is False
    assert "churned" in transformed.columns
