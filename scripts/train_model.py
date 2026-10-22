import argparse

from app.core.config import get_settings
from app.repositories.database import Database
from pipelines.training.train import train_churn_model

parser = argparse.ArgumentParser(description="Train the churn model")
parser.add_argument(
    "--register-model",
    action="store_true",
    default=False,
    help="Register the trained model in MLflow Model Registry and move the deployment alias.",
)
args = parser.parse_args()

settings = get_settings()
print(
    train_churn_model(
        Database(settings.database_path),
        settings.model_path,
        settings.model_metadata_path,
        risk_high=settings.risk_high,
        risk_medium=settings.risk_medium,
        mlflow_tracking_uri=settings.mlflow_tracking_uri if settings.enable_mlflow else None,
        mlflow_experiment_name=settings.mlflow_experiment_name,
        register_model=args.register_model,
        registered_model_name=settings.mlflow_registered_model_name,
        deployment_alias=settings.mlflow_deployment_alias,
    )
)
