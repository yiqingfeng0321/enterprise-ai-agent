from __future__ import annotations

import json

from app.core.config import get_settings
from app.repositories.database import Database
from pipelines.inference.batch_inference import run_batch_inference


def main() -> None:
    settings = get_settings()
    predictions = run_batch_inference(
        Database(settings.database_path),
        settings.model_path,
        settings.database_path.parent / "predictions.csv",
        risk_high=settings.risk_high,
        risk_medium=settings.risk_medium,
    )
    high_risk = predictions[predictions["risk_band"] == "high"]
    high_risk.to_csv(settings.database_path.parent / "high_risk_customers.csv", index=False)
    print(
        json.dumps(
            {
                "status": "success",
                "scored_customers": len(predictions),
                "high_risk_customers": len(high_risk),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
