from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

MODEL_ARTIFACT_NAME = "churn_model"
MODEL_METADATA_ARTIFACT = "retention_model_artifact.json"
REGISTRY_EVENTS_PATH = Path("artifacts/models/model_registry_events.jsonl")


def _append_registry_event(event: dict[str, Any]) -> None:
    REGISTRY_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **event,
    }
    with REGISTRY_EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def read_registry_events(limit: int = 20) -> list[dict[str, Any]]:
    if not REGISTRY_EVENTS_PATH.exists():
        return []
    lines = REGISTRY_EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    return list(reversed(events[-limit:]))


def build_registry_artifact_metadata(
    *,
    model_version: str,
    features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    baselines: dict[str, Any],
    threshold: float,
    risk_thresholds: dict[str, float],
    params: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_version": model_version,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "baselines": baselines,
        "threshold": threshold,
        "risk_thresholds": risk_thresholds,
        "params": params,
        "metrics": metrics,
    }


def register_model_version(
    *,
    tracking_uri: str,
    registered_model_name: str,
    run_id: str,
    alias: str = "champion",
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Install with: pip install -e .[observability]") from exc

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    previous_version: str | None = None
    try:
        previous_version = str(client.get_model_version_by_alias(registered_model_name, alias).version)
    except Exception:
        previous_version = None

    registration = mlflow.register_model(
        model_uri=f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}",
        name=registered_model_name,
    )

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current = client.get_model_version(registered_model_name, str(registration.version))
        status = str(current.status)
        if status == "READY":
            break
        if status == "FAILED_REGISTRATION":
            raise RuntimeError(
                f"Model registration failed for {registered_model_name} version {registration.version}"
            )
        time.sleep(1)

    current_version = str(registration.version)
    if previous_version and previous_version != current_version:
        client.set_registered_model_alias(
            registered_model_name,
            f"previous_{alias}",
            previous_version,
        )
    client.set_registered_model_alias(registered_model_name, alias, current_version)
    _append_registry_event(
        {
            "event_type": "register_and_promote",
            "registered_model_name": registered_model_name,
            "alias": alias,
            "new_version": current_version,
            "previous_version": previous_version,
            "run_id": run_id,
        }
    )
    return {
        "registered_model_name": registered_model_name,
        "version": current_version,
        "alias": alias,
        "previous_alias_version": previous_version,
        "model_uri": f"models:/{registered_model_name}@{alias}",
    }


def load_registered_model_artifact(
    *,
    tracking_uri: str,
    registered_model_name: str,
    alias: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Install with: pip install -e .[observability]") from exc

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_version = client.get_model_version_by_alias(registered_model_name, alias)
    metadata_path = Path(
        client.download_artifacts(model_version.run_id, MODEL_METADATA_ARTIFACT)
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pipeline = mlflow.sklearn.load_model(f"models:/{registered_model_name}@{alias}")

    artifact = {
        "pipeline": pipeline,
        "features": metadata["features"],
        "numeric_features": metadata["numeric_features"],
        "categorical_features": metadata["categorical_features"],
        "baselines": metadata["baselines"],
        "model_version": metadata["model_version"],
        "threshold": metadata["threshold"],
        "risk_thresholds": metadata.get("risk_thresholds", {"high": 0.70, "medium": 0.40}),
        "params": metadata.get("params", {}),
        "metrics": metadata.get("metrics", {}),
    }
    registry_info = {
        "registered_model_name": registered_model_name,
        "alias": alias,
        "version": str(model_version.version),
        "run_id": model_version.run_id,
        "source": model_version.source,
    }
    return artifact, registry_info


def sync_alias_to_local_artifacts(
    *,
    tracking_uri: str,
    registered_model_name: str,
    alias: str,
    model_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    import joblib

    artifact, registry_info = load_registered_model_artifact(
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        alias=alias,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metadata = {
        "model_version": artifact["model_version"],
        "features": artifact["features"],
        "params": artifact.get("params", {}),
        "metrics": artifact.get("metrics", {}),
        "registry": registry_info,
        "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return registry_info


def rollback_model_alias(
    *,
    tracking_uri: str,
    registered_model_name: str,
    target_alias: str,
    version: str | None = None,
    source_alias: str | None = None,
) -> dict[str, Any]:
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Install with: pip install -e .[observability]") from exc

    if not version and not source_alias:
        raise ValueError("Either version or source_alias must be provided for rollback")

    client = MlflowClient(tracking_uri=tracking_uri)
    previous_target_version: str | None = None
    try:
        previous_target_version = str(
            client.get_model_version_by_alias(registered_model_name, target_alias).version
        )
    except Exception:
        previous_target_version = None
    if source_alias:
        version = str(client.get_model_version_by_alias(registered_model_name, source_alias).version)
    assert version is not None
    client.set_registered_model_alias(registered_model_name, target_alias, version)
    _append_registry_event(
        {
            "event_type": "rollback",
            "registered_model_name": registered_model_name,
            "alias": target_alias,
            "new_version": str(version),
            "previous_version": previous_target_version,
            "source_alias": source_alias,
        }
    )
    return {
        "registered_model_name": registered_model_name,
        "target_alias": target_alias,
        "version": str(version),
    }


def get_model_registry_status(
    *,
    tracking_uri: str,
    registered_model_name: str,
    deployment_alias: str,
    event_limit: int = 20,
) -> dict[str, Any]:
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Install with: pip install -e .[observability]") from exc

    client = MlflowClient(tracking_uri=tracking_uri)
    aliases: dict[str, dict[str, Any] | None] = {}
    for alias_name in (deployment_alias, f"previous_{deployment_alias}"):
        try:
            model_version = client.get_model_version_by_alias(registered_model_name, alias_name)
            aliases[alias_name] = {
                "version": str(model_version.version),
                "run_id": model_version.run_id,
                "current_stage": getattr(model_version, "current_stage", "None"),
            }
        except Exception:
            aliases[alias_name] = None

    latest_versions = []
    try:
        registered_model = client.get_registered_model(registered_model_name)
        for model_version in sorted(
            registered_model.latest_versions,
            key=lambda item: int(item.version),
            reverse=True,
        ):
            latest_versions.append(
                {
                    "version": str(model_version.version),
                    "run_id": model_version.run_id,
                    "source": model_version.source,
                    "current_stage": getattr(model_version, "current_stage", "None"),
                }
            )
    except Exception:
        latest_versions = []

    return {
        "registered_model_name": registered_model_name,
        "deployment_alias": deployment_alias,
        "aliases": aliases,
        "versions": latest_versions,
        "events": read_registry_events(limit=event_limit),
    }
