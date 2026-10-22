from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.core.config import get_settings
from app.repositories.database import Database
from app.services.model_registry import rollback_model_alias, sync_alias_to_local_artifacts
from pipelines.training.train import train_churn_model


def main() -> None:
    settings = get_settings()
    mlflow_root = Path("artifacts/mlflow")
    artifacts_root = mlflow_root / "artifacts"
    db_path = mlflow_root / "mlflow.db"

    if artifacts_root.exists():
        shutil.rmtree(artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    events_path = Path("artifacts/models/model_registry_events.jsonl")
    if events_path.exists():
        events_path.unlink()

    db = Database(settings.database_path)
    first = train_churn_model(
        db,
        settings.model_path,
        settings.model_metadata_path,
        max_iter=1000,
        regularization_c=0.5,
        risk_high=settings.risk_high,
        risk_medium=settings.risk_medium,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        mlflow_experiment_name="registry-demo",
        register_model=True,
        registered_model_name=settings.mlflow_registered_model_name,
        deployment_alias=settings.mlflow_deployment_alias,
    )
    second = train_churn_model(
        db,
        settings.model_path,
        settings.model_metadata_path,
        max_iter=600,
        regularization_c=1.0,
        risk_high=settings.risk_high,
        risk_medium=settings.risk_medium,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        mlflow_experiment_name="registry-demo",
        register_model=True,
        registered_model_name=settings.mlflow_registered_model_name,
        deployment_alias=settings.mlflow_deployment_alias,
    )
    rollback = rollback_model_alias(
        tracking_uri=settings.mlflow_tracking_uri,
        registered_model_name=settings.mlflow_registered_model_name,
        target_alias=settings.mlflow_deployment_alias,
        source_alias=f"previous_{settings.mlflow_deployment_alias}",
    )
    synced = sync_alias_to_local_artifacts(
        tracking_uri=settings.mlflow_tracking_uri,
        registered_model_name=settings.mlflow_registered_model_name,
        alias=settings.mlflow_deployment_alias,
        model_path=settings.model_path,
        metadata_path=settings.model_metadata_path,
    )
    print(
        json.dumps(
            {
                "first": first.get("registry"),
                "second": second.get("registry"),
                "rollback": rollback,
                "synced": synced,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
