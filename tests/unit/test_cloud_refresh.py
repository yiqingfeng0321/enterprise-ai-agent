from pathlib import Path

import httpx
import pandas as pd

from pipelines.orchestration.cloud_refresh import (
    extract_current_customers,
    publish_predictions,
)


class StubResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def test_extract_current_customers_writes_csv(monkeypatch, tmp_path: Path) -> None:
    def fake_get(*args, **kwargs):
        return StubResponse(
            {"customers": [{"customer_id": "CUST-0001", "segment": "Enterprise"}]}
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    output_path = tmp_path / "current_customers.csv"

    report = extract_current_customers("https://example.test", "secret", output_path)

    assert report["rows"] == 1
    assert pd.read_csv(output_path).iloc[0]["customer_id"] == "CUST-0001"


def test_publish_predictions_posts_model_outputs(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return StubResponse({"status": "accepted"})

    monkeypatch.setattr(httpx, "post", fake_post)
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "customer_id": "CUST-0001",
                "churn_probability": 0.8,
                "risk_band": "high",
            }
        ]
    ).to_csv(predictions_path, index=False)

    report = publish_predictions("https://example.test", "secret", predictions_path)

    assert report == {"status": "accepted", "published_rows": 1}
    assert captured["url"].endswith("/v1/churn-risk/results")
    assert captured["json"]["predictions"][0]["risk_band"] == "high"
