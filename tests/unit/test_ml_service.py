from pathlib import Path

from app.services.ml_service import MLService


def test_artifact_risk_thresholds_override_service_defaults() -> None:
    service = MLService(Path("unused.joblib"), risk_high=0.90, risk_medium=0.80)
    artifact = {"risk_thresholds": {"high": 0.70, "medium": 0.40}}

    assert service._risk_band(0.39, artifact) == "low"
    assert service._risk_band(0.40, artifact) == "medium"
    assert service._risk_band(0.70, artifact) == "high"
