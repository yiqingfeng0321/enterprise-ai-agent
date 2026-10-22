from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


def _authorization_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def extract_current_customers(
    base_url: str,
    api_key: str,
    output_path: Path,
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Download current customer facts from the configured source endpoint."""
    response = httpx.get(
        f"{base_url.rstrip('/')}/v1/customers/current",
        headers=_authorization_headers(api_key),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    records = payload.get("customers") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("Customer endpoint must return a non-empty JSON list")

    frame = pd.DataFrame(records)
    if "customer_id" not in frame.columns:
        raise ValueError("Customer endpoint response is missing customer_id")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    return {"status": "success", "rows": len(frame), "output_path": str(output_path)}


def publish_predictions(
    base_url: str,
    api_key: str,
    predictions_path: Path,
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Publish authoritative model outputs to an explicitly approved endpoint."""
    frame = pd.read_csv(predictions_path)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "predictions": json.loads(frame.to_json(orient="records")),
    }
    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/churn-risk/results",
        headers=_authorization_headers(api_key),
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    result = response.json()
    return {
        "status": result.get("status", "accepted"),
        "published_rows": len(frame),
    }
