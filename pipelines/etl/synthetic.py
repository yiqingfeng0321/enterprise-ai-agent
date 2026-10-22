from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _build_customer_frame(rows: int, seed: int, include_labels: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segments = rng.choice(["SMB", "Mid-Market", "Enterprise"], rows, p=[0.5, 0.3, 0.2])
    regions = rng.choice(["North America", "EMEA", "APAC", "LATAM"], rows, p=[0.45, 0.25, 0.2, 0.1])
    licensed_seats = np.where(
        segments == "SMB",
        rng.integers(8, 80, rows),
        np.where(segments == "Mid-Market", rng.integers(80, 450, rows), rng.integers(450, 5000, rows)),
    )
    tenure = rng.integers(1, 73, rows)
    champion_count = np.where(
        segments == "Enterprise",
        rng.poisson(3.2, rows),
        np.where(segments == "Mid-Market", rng.poisson(2.0, rows), rng.poisson(1.0, rows)),
    ).clip(0, 10)
    onboarding = rng.binomial(1, np.clip(0.62 + tenure / 180, 0.6, 0.95), rows)
    active_seat_ratio = np.clip(
        rng.normal(0.34 + onboarding * 0.16 + champion_count * 0.03, 0.12, rows),
        0.03,
        0.98,
    )
    monthly_revenue = (licensed_seats * rng.normal(26, 4, rows) * (1.0 + champion_count * 0.03)).clip(150)
    logins = rng.poisson(np.clip(licensed_seats * active_seat_ratio * 2.3, 3, 900))
    usage_change = rng.normal(-3 + onboarding * 6, 22, rows).clip(-80, 80)
    support_tickets = rng.poisson(np.clip(1.2 + (usage_change < -20) * 1.8 + (active_seat_ratio < 0.25) * 0.7, 0.3, 12))
    nps = np.round(
        rng.normal(35 + onboarding * 15 + usage_change * 0.35 + active_seat_ratio * 20 - support_tickets * 4, 22, rows)
    ).clip(-100, 100).astype(int)
    days_to_renewal = rng.integers(1, 366, rows)
    payment_failures = rng.binomial(3, np.clip(0.04 + (segments == "SMB") * 0.04, 0.01, 0.22), rows)

    deployment_stages = [
        "running a limited pilot with selected business units",
        "preparing a phased organization-wide rollout",
        "scaling adoption across multiple departments",
        "standardizing mature usage and governance practices",
        "recovering a stalled deployment with low activation",
    ]
    business_priorities = [
        "needs clearer productivity and ROI evidence",
        "is focused on cost optimization and license utilization",
        "prefers online self-service",
        "values premium support responsiveness",
        "needs role-based enablement for end users",
        "is resolving security and integration blockers",
        "wants stronger executive sponsorship and governance",
    ]
    communication_styles = [
        "responds well to concise monthly business reviews",
        "prefers detailed ROI breakdowns before commitment",
        "likes proactive outreach on adoption milestones",
        "prefers email-first communication",
    ]
    customer_notes = [
        f"{deployment_stages[i % len(deployment_stages)]}; {business_priorities[(i * 3) % len(business_priorities)]}; {communication_styles[(i * 5) % len(communication_styles)]}."
        for i in range(rows)
    ]

    logit = (
        -0.8
        - 0.055 * logins
        - 0.023 * nps
        - 0.022 * tenure
        - 1.0 * onboarding
        - 0.025 * usage_change
        + 0.55 * support_tickets
        + 0.90 * payment_failures
        + 1.20 * (days_to_renewal < 45)
        + 0.50 * (segments == "SMB")
        + rng.normal(0, 0.3, rows)
    )
    churn_probability = 1 / (1 + np.exp(-logit - 2.0 * active_seat_ratio - 0.22 * champion_count + 0.0003 * licensed_seats))
    churned = rng.binomial(1, churn_probability) if include_labels else np.array([None] * rows)

    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:04d}" for index in range(1, rows + 1)],
            "segment": segments,
            "region": regions,
            "customer_notes": customer_notes,
            "licensed_seats": licensed_seats,
            "active_seat_ratio_30d": active_seat_ratio.round(4),
            "champion_count": champion_count,
            "tenure_months": tenure,
            "monthly_revenue": monthly_revenue.round(2),
            "logins_30d": logins,
            "usage_change_pct": usage_change.round(2),
            "support_tickets_90d": support_tickets,
            "nps": nps,
            "days_to_renewal": days_to_renewal,
            "payment_failures_90d": payment_failures,
            "onboarding_complete": onboarding,
            "churned": churned,
            "snapshot_date": pd.Timestamp.today().normalize().date().isoformat(),
        }
    )
    return frame


def generate_customer_data(output_path: Path, rows: int = 800, seed: int = 42) -> pd.DataFrame:
    """Backward-compatible helper that writes labeled historical data."""
    frame = _build_customer_frame(rows=rows, seed=seed, include_labels=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def generate_customer_datasets(
    history_output_path: Path,
    current_output_path: Path,
    *,
    history_rows: int = 1200,
    current_rows: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = _build_customer_frame(rows=history_rows, seed=seed, include_labels=True)
    current = _build_customer_frame(rows=current_rows, seed=seed + 1, include_labels=False).drop(columns=["churned"])
    history_output_path.parent.mkdir(parents=True, exist_ok=True)
    current_output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_output_path, index=False)
    current.to_csv(current_output_path, index=False)
    return history, current
