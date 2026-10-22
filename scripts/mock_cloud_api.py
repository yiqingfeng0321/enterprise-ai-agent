from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Mock Customer Cloud API",
    description="Local stand-in for the cloud endpoints used by the Airflow demo.",
)


class PredictionPublication(BaseModel):
    generated_at: str
    predictions: list[dict[str, Any]] = Field(min_length=1)


def _require_api_key(authorization: str | None) -> None:
    expected = os.getenv("MOCK_CLOUD_API_KEY", "demo-api-key")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/customers/current")
def current_customers(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_api_key(authorization)
    source_path = Path(
        os.getenv("MOCK_CLOUD_SOURCE_PATH", "/app/data/raw/current_customers.csv")
    )
    if not source_path.exists():
        raise HTTPException(status_code=503, detail=f"Source data not found: {source_path}")
    frame = pd.read_csv(source_path)
    return {"customers": json.loads(frame.to_json(orient="records")), "rows": len(frame)}


@app.post("/v1/churn-risk/results")
def receive_predictions(
    publication: PredictionPublication,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_api_key(authorization)
    output_path = Path(
        os.getenv(
            "MOCK_CLOUD_PUBLISHED_PATH",
            "/app/artifacts/cloud_demo/latest_predictions.json",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(publication.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    return {"status": "accepted", "received_rows": len(publication.predictions)}
