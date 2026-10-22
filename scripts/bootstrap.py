from __future__ import annotations

import argparse
import json

from app.core.config import get_settings
from app.repositories.customer_repository import CustomerRepository
from app.repositories.database import Database
from app.services.ml_service import MLService
from pipelines.etl.pipeline import run_etl
from pipelines.etl.synthetic import generate_customer_datasets
from pipelines.evaluation.build_golden_dataset import build_golden_dataset
from pipelines.inference.batch_inference import run_batch_inference
from pipelines.training.train import train_churn_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the complete local project")
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    settings = get_settings()
    current_rows = max(150, args.rows // 4)
    generate_customer_datasets(
        settings.raw_history_data_path,
        settings.raw_current_data_path,
        history_rows=args.rows,
        current_rows=current_rows,
    )
    database = Database(settings.database_path)
    database.initialize_operational_tables()
    database.reset_operational_tables()
    etl_report = run_etl(
        settings.raw_history_data_path,
        settings.raw_current_data_path,
        database,
        report_path=settings.database_path.parent / "etl_report.json",
    )
    metadata = train_churn_model(
        database,
        settings.model_path,
        settings.model_metadata_path,
        max_iter=args.max_iter,
        risk_high=settings.risk_high,
        risk_medium=settings.risk_medium,
        mlflow_tracking_uri=settings.mlflow_tracking_uri if settings.enable_mlflow else None,
        mlflow_experiment_name=settings.mlflow_experiment_name,
    )
    predictions = run_batch_inference(
        database,
        settings.model_path,
        settings.database_path.parent / "predictions.csv",
        risk_high=settings.risk_high,
        risk_medium=settings.risk_medium,
    )
    repository = CustomerRepository(database)
    ml_service = MLService(settings.model_path, settings.risk_high, settings.risk_medium)
    golden = build_golden_dataset(
        repository, ml_service, settings.golden_dataset_path, cases_per_band=3
    )
    print(
        json.dumps(
            {
                "etl": etl_report,
                "model_metrics": metadata["metrics"],
                "highest_risk_customer": predictions.iloc[0]["customer_id"],
                "golden_cases": len(golden),
                "scenario": "B2B AI copilot subscription retention",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
