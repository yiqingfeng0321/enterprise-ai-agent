from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.sdk import BaseHook, Variable, dag, task
from airflow.sdk.exceptions import AirflowSkipException

from app.core.config import get_settings
from app.repositories.database import Database
from pipelines.etl.pipeline import run_etl
from pipelines.inference.batch_inference import run_batch_inference
from pipelines.orchestration.cloud_refresh import (
    extract_current_customers,
    publish_predictions,
)


def _cloud_connection() -> tuple[str, str]:
    connection = BaseHook.get_connection("customer_cloud_api")
    # In an Airflow URI such as http://host:9000 the "http" part becomes conn_type and schema is empty.
    if connection.schema in {"http", "https"}:
        scheme = connection.schema
    elif connection.conn_type in {"http", "https"}:
        scheme = connection.conn_type
    else:
        scheme = "https"
    port = f":{connection.port}" if connection.port else ""
    return f"{scheme}://{connection.host}{port}", connection.password or ""


@dag(
    dag_id="daily_customer_risk_refresh",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Los_Angeles"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["customer-retention", "etl", "ml"],
)
def daily_customer_risk_refresh():
    @task
    def extract() -> dict:
        settings = get_settings()
        base_url, api_key = _cloud_connection()
        return extract_current_customers(
            base_url,
            api_key,
            settings.raw_current_data_path,
        )

    @task
    def transform_and_load(extract_report: dict) -> dict:
        if extract_report.get("status") != "success":
            raise RuntimeError("Cloud extraction did not complete successfully")
        settings = get_settings()
        return run_etl(
            settings.raw_history_data_path,
            settings.raw_current_data_path,
            Database(settings.database_path),
            report_path=settings.database_path.parent / "etl_report.json",
        )

    @task
    def infer(etl_report: dict) -> dict:
        if etl_report.get("status") != "success":
            raise RuntimeError("ETL did not complete successfully")
        settings = get_settings()
        predictions_path = settings.database_path.parent / "predictions.csv"
        predictions = run_batch_inference(
            Database(settings.database_path),
            settings.model_path,
            predictions_path,
            risk_high=settings.risk_high,
            risk_medium=settings.risk_medium,
        )
        high_risk = predictions[predictions["risk_band"] == "high"]
        high_risk.to_csv(
            settings.database_path.parent / "high_risk_customers.csv",
            index=False,
        )
        return {
            "status": "success",
            "predictions_path": str(predictions_path),
            "scored_customers": len(predictions),
            "high_risk_customers": len(high_risk),
        }

    @task
    def publish(inference_report: dict) -> dict:
        publish_enabled = Variable.get(
            "customer_risk_publish_enabled",
            default="false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not publish_enabled:
            raise AirflowSkipException(
                "External publication requires customer_risk_publish_enabled=true"
            )
        base_url, api_key = _cloud_connection()
        return publish_predictions(
            base_url,
            api_key,
            inference_report["predictions_path"],
        )

    publish(infer(transform_and_load(extract())))


dag = daily_customer_risk_refresh()
