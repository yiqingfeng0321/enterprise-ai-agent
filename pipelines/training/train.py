from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.repositories.database import Database
from app.services.model_registry import (
    MODEL_ARTIFACT_NAME,
    MODEL_METADATA_ARTIFACT,
    build_registry_artifact_metadata,
    register_model_version,
)

TARGET = "churned"
IDENTIFIERS = ["customer_id", "snapshot_date"]
CATEGORICAL_FEATURES = ["segment", "region"]
NUMERIC_FEATURES = [
    "licensed_seats",
    "active_seat_ratio_30d",
    "champion_count",
    "tenure_months",
    "monthly_revenue",
    "logins_30d",
    "usage_change_pct",
    "support_tickets_90d",
    "nps",
    "days_to_renewal",
    "payment_failures_90d",
    "onboarding_complete",
    "support_ticket_rate",
    "engagement_score",
    "renewal_urgency",
    "revenue_at_risk",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _read_training_data(database: Database) -> pd.DataFrame:
    with database.connect() as connection:
        frame = pd.read_sql_query("SELECT * FROM customer_history_features", connection)
    if frame.empty:
        raise RuntimeError("customer_history_features is empty; run the ETL pipeline first")
    return frame


def _build_pipeline(max_iter: int, regularization_c: float, random_state: int) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(
        max_iter=max_iter,
        C=regularization_c,
        class_weight="balanced",
        random_state=random_state,
        solver="liblinear",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _baseline_values(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        result[column] = float(frame[column].median())
    for column in CATEGORICAL_FEATURES:
        result[column] = str(frame[column].mode(dropna=True).iloc[0])
    return result


def _maybe_log_mlflow(
    pipeline: Pipeline,
    params: dict[str, Any],
    metrics: dict[str, float],
    registry_artifact_metadata: dict[str, Any],
    tracking_uri: str | None,
    experiment_name: str,
) -> str | None:
    if not tracking_uri:
        return None
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return None
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="logistic_regression_churn") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_NAME,
            serialization_format="pickle",
        )
        mlflow.log_dict(registry_artifact_metadata, MODEL_METADATA_ARTIFACT)
        return run.info.run_id


def train_churn_model(
    database: Database,
    model_path: Path,
    metadata_path: Path,
    *,
    max_iter: int = 1000,
    regularization_c: float = 0.5,
    random_state: int = 42,
    test_size: float = 0.25,
    risk_high: float = 0.70,
    risk_medium: float = 0.40,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "enterprise-retention-agent",
    register_model: bool = False,
    registered_model_name: str = "b2b-ai-copilot-churn-risk-model",
    deployment_alias: str = "champion",
) -> dict[str, Any]:
    frame = _read_training_data(database)
    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET],
    )
    pipeline = _build_pipeline(max_iter, regularization_c, random_state)
    pipeline.fit(train_frame[FEATURES], train_frame[TARGET])
    probabilities = pipeline.predict_proba(test_frame[FEATURES])[:, 1]
    candidate_thresholds = np.linspace(0.10, 0.80, 71)
    threshold_scores = [
        f1_score(test_frame[TARGET], (probabilities >= threshold).astype(int), zero_division=0)
        for threshold in candidate_thresholds
    ]
    best_threshold = float(candidate_thresholds[int(np.argmax(threshold_scores))])
    predictions = (probabilities >= best_threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(test_frame[TARGET], probabilities)),
        "average_precision": float(average_precision_score(test_frame[TARGET], probabilities)),
        "accuracy": float(accuracy_score(test_frame[TARGET], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_frame[TARGET], predictions)),
        "f1": float(f1_score(test_frame[TARGET], predictions, zero_division=0)),
        "positive_rate": float(test_frame[TARGET].mean()),
        "decision_threshold": best_threshold,
    }
    model_version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    params = {
        "model_type": "logistic_regression",
        "max_iter": max_iter,
        "regularization_c": regularization_c,
        "random_state": random_state,
        "test_size": test_size,
        "training_rows": len(train_frame),
    }
    baselines = _baseline_values(train_frame)
    registry_artifact_metadata = build_registry_artifact_metadata(
        model_version=model_version,
        features=FEATURES,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        baselines=baselines,
        threshold=best_threshold,
        risk_thresholds={"high": risk_high, "medium": risk_medium},
        params=params,
        metrics=metrics,
    )
    run_id = _maybe_log_mlflow(
        pipeline,
        params,
        metrics,
        registry_artifact_metadata,
        mlflow_tracking_uri,
        mlflow_experiment_name,
    )
    registry = None
    if register_model:
        if not mlflow_tracking_uri or not run_id:
            raise RuntimeError("Model registration requires MLflow tracking to be enabled")
        registry = register_model_version(
            tracking_uri=mlflow_tracking_uri,
            registered_model_name=registered_model_name,
            run_id=run_id,
            alias=deployment_alias,
        )
    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "baselines": baselines,
        "model_version": model_version,
        "threshold": best_threshold,
        "risk_thresholds": {"high": risk_high, "medium": risk_medium},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metadata = {
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "params": params,
        "features": FEATURES,
        "mlflow_run_id": run_id,
        "registry": registry,
        "risk_thresholds": {"high": risk_high, "medium": risk_medium},
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
