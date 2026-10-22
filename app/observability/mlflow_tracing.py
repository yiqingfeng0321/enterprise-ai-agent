from __future__ import annotations


def setup_agent_framework_mlflow_tracing(
    tracking_uri: str, experiment_name: str, enable_sensitive_data: bool = False
) -> bool:
    """Configure the official Agent Framework -> OpenTelemetry -> MLflow trace path."""
    try:
        import mlflow
        from agent_framework.observability import setup_observability
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        return False

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    exporter = OTLPSpanExporter(
        endpoint=f"{tracking_uri.rstrip('/')}/v1/traces",
        headers={"x-mlflow-experiment-id": experiment.experiment_id},
    )
    setup_observability(
        enable_sensitive_data=enable_sensitive_data,
        exporters=[exporter],
    )
    return True
