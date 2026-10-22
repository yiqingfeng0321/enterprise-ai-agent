from __future__ import annotations

import argparse
import json

from app.core.config import get_settings
from app.services.model_registry import rollback_model_alias, sync_alias_to_local_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollback the deployed MLflow model alias")
    parser.add_argument("--version", default="", help="Model Registry version to deploy to the target alias")
    parser.add_argument(
        "--source-alias",
        default="previous_champion",
        help="Source alias to roll back from when --version is not provided",
    )
    parser.add_argument(
        "--target-alias",
        default="",
        help="Alias to move during rollback. Defaults to the configured deployment alias.",
    )
    parser.add_argument(
        "--no-sync-local",
        action="store_true",
        default=False,
        help="Do not overwrite local model artifacts after alias rollback.",
    )
    args = parser.parse_args()

    settings = get_settings()
    target_alias = args.target_alias or settings.mlflow_deployment_alias
    result = rollback_model_alias(
        tracking_uri=settings.mlflow_tracking_uri,
        registered_model_name=settings.mlflow_registered_model_name,
        target_alias=target_alias,
        version=args.version or None,
        source_alias=None if args.version else args.source_alias,
    )

    synced = None
    if not args.no_sync_local:
        synced = sync_alias_to_local_artifacts(
            tracking_uri=settings.mlflow_tracking_uri,
            registered_model_name=settings.mlflow_registered_model_name,
            alias=target_alias,
            model_path=settings.model_path,
            metadata_path=settings.model_metadata_path,
        )

    print(json.dumps({"rollback": result, "synced_local_model": synced}, indent=2))


if __name__ == "__main__":
    main()
