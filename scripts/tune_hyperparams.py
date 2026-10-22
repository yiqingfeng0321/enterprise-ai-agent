from __future__ import annotations

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import get_settings
from app.repositories.database import Database
from app.services.model_registry import register_model_version
from pipelines.training.train import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    _baseline_values,
    _read_training_data,
)


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one max_iter value is required")
    return [int(item) for item in values]


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one regularization value is required")
    return [float(item) for item in values]


def _sanitize(value: float | int) -> str:
    return str(value).replace(".", "p")


def _run_name(model_type: str, params: dict[str, Any]) -> str:
    important = []
    for key in sorted(params):
        if key in {"model_type", "random_state", "test_size", "training_rows"}:
            continue
        important.append(f"{key}={params[key]}")
    suffix = " | ".join(important[:3])
    return f"{model_type} | {suffix}" if suffix else model_type


def _parse_model_list(raw: str) -> list[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one model type is required")
    return values


def _parse_hidden_layers(raw: str) -> list[tuple[int, ...]]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    parsed: list[tuple[int, ...]] = []
    for value in values:
        layers = tuple(int(part) for part in value.split("x") if part)
        if not layers:
            continue
        parsed.append(layers)
    if not parsed:
        raise ValueError("At least one deep-learning hidden layer specification is required")
    return parsed


def _build_preprocessor() -> ColumnTransformer:
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
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def _build_model_pipeline(model_type: str, params: dict[str, Any], random_state: int) -> Pipeline:
    preprocessor = _build_preprocessor()
    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=int(params["max_iter"]),
            C=float(params["regularization_c"]),
            class_weight="balanced",
            random_state=random_state,
            solver="liblinear",
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if str(params["max_depth"]).lower() == "none" else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is not installed. Install with: .venv/bin/python -m pip install xgboost"
            ) from exc
        classifier = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
        )
    elif model_type == "deep_learning":
        classifier = MLPClassifier(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["learning_rate_init"]),
            max_iter=int(params["max_iter"]),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    candidate_thresholds = np.linspace(0.10, 0.80, 71)
    threshold_scores = [
        f1_score(y_true, (probabilities >= threshold).astype(int), zero_division=0)
        for threshold in candidate_thresholds
    ]
    best_threshold = float(candidate_thresholds[int(np.argmax(threshold_scores))])
    predictions = (probabilities >= best_threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
        "decision_threshold": best_threshold,
    }


def _maybe_log_mlflow(
    pipeline: Pipeline,
    params: dict[str, Any],
    metrics: dict[str, float],
    tracking_uri: str | None,
    experiment_name: str,
    model_type: str,
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
    with mlflow.start_run(run_name=_run_name(model_type, {"model_type": model_type, **params})) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name="churn_model",
            serialization_format="pickle",
        )
        return run.info.run_id


def _train_single_model(
    database: Database,
    model_type: str,
    model_params: dict[str, Any],
    model_path: Path,
    metadata_path: Path,
    *,
    random_state: int,
    test_size: float,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str,
    risk_high: float,
    risk_medium: float,
) -> dict[str, Any]:
    frame = _read_training_data(database)
    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET],
    )

    pipeline = _build_model_pipeline(model_type, model_params, random_state)
    pipeline.fit(train_frame[FEATURES], train_frame[TARGET])

    probabilities = pipeline.predict_proba(test_frame[FEATURES])[:, 1]
    metrics = _compute_metrics(test_frame[TARGET].to_numpy(), probabilities)
    model_version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    params = {
        "model_type": model_type,
        "random_state": random_state,
        "test_size": test_size,
        "training_rows": len(train_frame),
        **model_params,
    }
    run_id = _maybe_log_mlflow(
        pipeline,
        params,
        metrics,
        mlflow_tracking_uri,
        mlflow_experiment_name,
        model_type,
    )

    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "baselines": _baseline_values(train_frame),
        "model_version": model_version,
        "threshold": metrics["decision_threshold"],
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
        "risk_thresholds": {"high": risk_high, "medium": risk_medium},
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _metric_score(metadata: dict[str, Any], metric_name: str) -> float:
    metrics = metadata.get("metrics", {})
    if metric_name not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise KeyError(f"Metric '{metric_name}' not found. Available metrics: {available}")
    return float(metrics[metric_name])


def _promote_best_model(best: dict[str, Any], final_model_path: Path, final_metadata_path: Path) -> None:
    shutil.copy2(best["model_path"], final_model_path)

    metadata = dict(best["metadata"])
    metadata["selected_from_tuning"] = {
        "score_metric": best["score_metric"],
        "score": best["score"],
        "source_model_path": str(best["model_path"]),
        "source_metadata_path": str(best["metadata_path"]),
    }
    final_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_leaderboard(results: list[dict[str, Any]], output_path: Path) -> None:
    best_per_model: dict[str, dict[str, Any]] = {}
    for item in sorted(results, key=lambda run: run["score"], reverse=True):
        if item["model_type"] in best_per_model:
            continue
        best_per_model[item["model_type"]] = item
    leaderboard = [
        {
            "model_type": item["model_type"],
            "score": item["score"],
            "score_metric": item["score_metric"],
            "mlflow_run_id": item["mlflow_run_id"],
            "params": item["params"],
        }
        for item in best_per_model.values()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search multiple churn models")
    parser.add_argument(
        "--models",
        default="logistic_regression,random_forest,xgboost,deep_learning",
        help="Comma-separated model types",
    )

    parser.add_argument("--logistic-max-iter", default="300,600,1000")
    parser.add_argument("--logistic-c", default="0.1,0.3,0.5,1.0,3.0")

    parser.add_argument("--rf-n-estimators", default="200,400")
    parser.add_argument("--rf-max-depth", default="8,12,None")
    parser.add_argument("--rf-min-samples-leaf", default="1,2")

    parser.add_argument("--xgb-n-estimators", default="200,400")
    parser.add_argument("--xgb-max-depth", default="4,6")
    parser.add_argument("--xgb-learning-rate", default="0.05,0.1")
    parser.add_argument("--xgb-subsample", default="0.8")
    parser.add_argument("--xgb-colsample-bytree", default="0.8")

    parser.add_argument("--dl-hidden-layers", default="64x32,128x64")
    parser.add_argument("--dl-alpha", default="0.0001,0.001")
    parser.add_argument("--dl-learning-rate-init", default="0.001")
    parser.add_argument("--dl-max-iter", default="300,500")

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--metric",
        default="roc_auc",
        help="Metric used to choose the best run (e.g., roc_auc, f1, average_precision)",
    )
    parser.add_argument(
        "--run-tag",
        default="grid_search",
        help="Tag folder suffix for generated tuning artifacts",
    )
    parser.add_argument(
        "--promote-best",
        action="store_true",
        default=False,
        help="Copy the best tuned model into production model paths",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        default=False,
        help="Force-enable MLflow logging for this tuning run",
    )
    parser.add_argument(
        "--register-best",
        action="store_true",
        default=False,
        help="Register the best run in MLflow Model Registry and move the deployment alias.",
    )
    parser.add_argument(
        "--registry-model-name",
        default="",
        help="Override the registered model name used when --register-best is set.",
    )
    parser.add_argument(
        "--registry-alias",
        default="",
        help="Override the deployment alias used when --register-best is set.",
    )
    args = parser.parse_args()

    settings = get_settings()
    database = Database(settings.database_path)
    models = _parse_model_list(args.models)

    use_mlflow = args.enable_mlflow or settings.enable_mlflow
    tracking_uri = settings.mlflow_tracking_uri if use_mlflow else None

    tuning_dir = settings.model_path.parent / "tuning" / args.run_tag
    tuning_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for model_type in models:
        param_grid: list[dict[str, Any]] = []
        if model_type == "logistic_regression":
            for max_iter, regularization_c in itertools.product(
                _parse_int_list(args.logistic_max_iter),
                _parse_float_list(args.logistic_c),
            ):
                param_grid.append(
                    {
                        "max_iter": max_iter,
                        "regularization_c": regularization_c,
                    }
                )
        elif model_type == "random_forest":
            for n_estimators, max_depth, min_samples_leaf in itertools.product(
                _parse_int_list(args.rf_n_estimators),
                [item.strip() for item in args.rf_max_depth.split(",") if item.strip()],
                _parse_int_list(args.rf_min_samples_leaf),
            ):
                param_grid.append(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    }
                )
        elif model_type == "xgboost":
            for n_estimators, max_depth, learning_rate, subsample, colsample_bytree in itertools.product(
                _parse_int_list(args.xgb_n_estimators),
                _parse_int_list(args.xgb_max_depth),
                _parse_float_list(args.xgb_learning_rate),
                _parse_float_list(args.xgb_subsample),
                _parse_float_list(args.xgb_colsample_bytree),
            ):
                param_grid.append(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                    }
                )
        elif model_type == "deep_learning":
            for hidden_layer_sizes, alpha, learning_rate_init, max_iter in itertools.product(
                _parse_hidden_layers(args.dl_hidden_layers),
                _parse_float_list(args.dl_alpha),
                _parse_float_list(args.dl_learning_rate_init),
                _parse_int_list(args.dl_max_iter),
            ):
                param_grid.append(
                    {
                        "hidden_layer_sizes": hidden_layer_sizes,
                        "alpha": alpha,
                        "learning_rate_init": learning_rate_init,
                        "max_iter": max_iter,
                    }
                )
        else:
            raise ValueError(
                f"Unsupported model type '{model_type}'. Use logistic_regression, random_forest, xgboost, deep_learning"
            )

        for model_params in param_grid:
            param_suffix = "_".join(
                f"{key}{_sanitize(value)}"
                for key, value in model_params.items()
            )
            suffix = f"{model_type}_{param_suffix}"
            model_path = tuning_dir / f"churn_model_{suffix}.joblib"
            metadata_path = tuning_dir / f"churn_model_{suffix}.json"
            metadata = _train_single_model(
                database,
                model_type,
                model_params,
                model_path,
                metadata_path,
                random_state=args.random_state,
                test_size=args.test_size,
                mlflow_tracking_uri=tracking_uri,
                mlflow_experiment_name=settings.mlflow_experiment_name,
                risk_high=settings.risk_high,
                risk_medium=settings.risk_medium,
            )
            score = _metric_score(metadata, args.metric)
            results.append(
                {
                    "model_type": model_type,
                    "params": metadata.get("params", {}),
                    "metrics": metadata.get("metrics", {}),
                    "score_metric": args.metric,
                    "score": score,
                    "mlflow_run_id": metadata.get("mlflow_run_id"),
                    "model_path": model_path,
                    "metadata_path": metadata_path,
                    "metadata": metadata,
                }
            )

    if not results:
        raise RuntimeError("No tuning runs were executed")

    best = max(results, key=lambda item: item["score"])
    best_by_model: dict[str, dict[str, Any]] = {}
    for item in sorted(results, key=lambda run: run["score"], reverse=True):
        model_type = item["model_type"]
        if model_type not in best_by_model:
            best_by_model[model_type] = {
                "params": item["params"],
                "score": item["score"],
                "mlflow_run_id": item["mlflow_run_id"],
                "model_path": str(item["model_path"]),
                "metadata_path": str(item["metadata_path"]),
            }

    _write_leaderboard(results, tuning_dir / "leaderboard.json")

    if args.promote_best:
        _promote_best_model(best, settings.model_path, settings.model_metadata_path)

    registry = None
    if args.register_best:
        if not use_mlflow:
            raise RuntimeError("--register-best requires MLflow to be enabled")
        if not best.get("mlflow_run_id"):
            raise RuntimeError("The best run does not have an MLflow run ID to register")
        registry = register_model_version(
            tracking_uri=settings.mlflow_tracking_uri,
            registered_model_name=args.registry_model_name or settings.mlflow_registered_model_name,
            run_id=str(best["mlflow_run_id"]),
            alias=args.registry_alias or settings.mlflow_deployment_alias,
        )

    output = {
        "mlflow_enabled": use_mlflow,
        "tracking_uri": tracking_uri,
        "experiment_name": settings.mlflow_experiment_name,
        "score_metric": args.metric,
        "run_count": len(results),
        "best": {
            "model_type": best["model_type"],
            "params": best["params"],
            "score": best["score"],
            "mlflow_run_id": best["mlflow_run_id"],
            "model_path": str(best["model_path"]),
            "metadata_path": str(best["metadata_path"]),
        },
        "best_by_model": best_by_model,
        "leaderboard_path": str(tuning_dir / "leaderboard.json"),
        "promoted_to_production": args.promote_best,
        "registry": registry,
        "all_runs": [
            {
                "model_type": item["model_type"],
                "params": item["params"],
                "score": item["score"],
                "mlflow_run_id": item["mlflow_run_id"],
            }
            for item in sorted(results, key=lambda run: run["score"], reverse=True)
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
