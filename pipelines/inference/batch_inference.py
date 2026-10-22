from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from app.repositories.database import Database


def run_batch_inference(
    database: Database,
    model_path: Path,
    output_path: Path,
    *,
    risk_high: float = 0.70,
    risk_medium: float = 0.40,
) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    thresholds = artifact.get("risk_thresholds", {})
    risk_high = float(thresholds.get("high", risk_high))
    risk_medium = float(thresholds.get("medium", risk_medium))
    with database.connect() as connection:
        customers = pd.read_sql_query("SELECT * FROM customer_current_features", connection)
    probabilities = artifact["pipeline"].predict_proba(customers[artifact["features"]])[:, 1]
    output = customers[["customer_id", "segment", "monthly_revenue"]].copy()
    output["churn_probability"] = probabilities
    output["risk_band"] = pd.cut(
        output["churn_probability"],
        bins=[float("-inf"), risk_medium, risk_high, float("inf")],
        labels=["low", "medium", "high"],
        right=False,
    ).astype(str)
    output = output.sort_values("churn_probability", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output
