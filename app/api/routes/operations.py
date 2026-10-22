from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import container_dependency
from app.core.container import ApplicationContainer
from app.services.model_registry import get_model_registry_status

router = APIRouter(prefix="/v1/operations", tags=["operations"])


@router.get("/dashboard")
def dashboard(
    limit: int = Query(default=100, ge=1, le=1000),
    container: ApplicationContainer = Depends(container_dependency),
) -> dict:
    profiles = [
        container.customer_repository.get(customer_id)
        for customer_id in container.customer_repository.list_ids(limit=10_000)
    ]
    profile_by_id = {profile.customer_id: profile for profile in profiles}
    scored: list[dict] = []
    for customer_id, probability, risk_band in container.ml_service.score_many(profiles):
        profile = profile_by_id[customer_id]
        scored.append(
            {
                "customer_id": customer_id,
                "segment": profile.segment,
                "region": profile.region,
                "monthly_revenue": profile.monthly_revenue,
                "revenue_at_risk": profile.revenue_at_risk,
                "days_to_renewal": profile.days_to_renewal,
                "usage_change_pct": profile.usage_change_pct,
                "support_tickets_90d": profile.support_tickets_90d,
                "payment_failures_90d": profile.payment_failures_90d,
                "churn_probability": probability,
                "risk_band": risk_band,
            }
        )

    scored.sort(key=lambda item: item["churn_probability"], reverse=True)
    high_risk = [item for item in scored if item["risk_band"] == "high"]
    medium_risk = [item for item in scored if item["risk_band"] == "medium"]
    renewal_30d = [item for item in scored if item["days_to_renewal"] <= 30]

    with container.database.connect() as connection:
        pending_approvals = connection.execute(
            "SELECT COUNT(*) AS count FROM retention_tickets WHERE status = 'pending_approval'"
        ).fetchone()["count"]

    return {
        "summary": {
            "total_customers": len(scored),
            "high_risk_customers": len(high_risk),
            "medium_risk_customers": len(medium_risk),
            "renewal_within_30_days": len(renewal_30d),
            "pending_approvals": int(pending_approvals),
            "revenue_at_risk": round(sum(float(item["revenue_at_risk"]) for item in high_risk), 2),
        },
        "customers": scored[:limit],
    }


@router.get("/evaluation")
def evaluation(container: ApplicationContainer = Depends(container_dependency)) -> dict:
    evaluation_path = Path("artifacts/evaluations/agent_eval.json")
    model_path = container.settings.model_metadata_path
    evaluation_payload = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.exists()
        else {"metrics": {}, "cases": []}
    )
    model_payload = (
        json.loads(model_path.read_text(encoding="utf-8")) if model_path.exists() else {}
    )
    return {"agent_evaluation": evaluation_payload, "model": model_payload}


@router.get("/traces")
def traces(
    limit: int = Query(default=50, ge=1, le=500),
    container: ApplicationContainer = Depends(container_dependency),
) -> dict:
    with container.database.connect() as connection:
        predictions = [
            dict(row)
            for row in connection.execute(
                """
                SELECT p.analysis_id, p.customer_id, p.probability, p.risk_band,
                       p.model_version, p.created_at,
                       t.id AS ticket_id, t.status AS ticket_status
                FROM predictions p
                LEFT JOIN retention_tickets t ON t.analysis_id = p.analysis_id
                ORDER BY p.created_at DESC, p.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        ]
        tickets = [
            dict(row)
            for row in connection.execute(
                """
                SELECT id, analysis_id, customer_id, title, status, created_at
                FROM retention_tickets
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        ]
        feedback = [
            dict(row)
            for row in connection.execute(
                """
                SELECT id, analysis_id, rating, comment, created_at
                FROM feedback
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        ]
        memory_count = connection.execute(
            "SELECT COUNT(*) AS count FROM customer_memories"
        ).fetchone()["count"]
        session_message_count = connection.execute(
            "SELECT COUNT(*) AS count FROM session_messages"
        ).fetchone()["count"]

    provider_status = container.provider_status
    retrieval_status = container.knowledge_service.status
    provider_available = bool(
        provider_status.get("provider_available", provider_status.get("ollama_available", False))
    )
    return {
        "predictions": predictions,
        "tickets": tickets,
        "feedback": feedback,
        "runtime": {
            "prediction_count": len(predictions),
            "ticket_count": len(tickets),
            "feedback_count": len(feedback),
            "memory_count": int(memory_count),
            "session_message_count": int(session_message_count),
            "agent_provider": container.settings.agent_provider,
            "model": provider_status.get("model", container.settings.ollama_model),
            "requested_provider": provider_status.get("requested_provider", container.settings.agent_provider),
            "generation_mode": provider_status.get("generation_mode", "deterministic"),
            "model_ready": container.ml_service.ready,
            "provider_available": provider_available,
            "ollama_available": bool(provider_status.get("ollama_available", False)),
            "fallback_used": bool(provider_status.get("fallback_used", False)),
            "fallback_reason": provider_status.get("fallback_reason"),
            "last_successful_llm_call": provider_status.get("last_successful_llm_call"),
            "retrieval_requested_backend": retrieval_status["requested_backend"],
            "retrieval_active_backend": retrieval_status["active_backend"],
            "retrieval_degraded": retrieval_status["degraded"],
            "retrieval_error": retrieval_status["error"],
        },
    }


@router.get("/model-registry")
def model_registry(container: ApplicationContainer = Depends(container_dependency)) -> dict:
    settings = container.settings
    if not settings.mlflow_use_model_registry:
        return {
            "enabled": False,
            "reason": "MLFLOW_USE_MODEL_REGISTRY is disabled.",
        }

    try:
        status = get_model_registry_status(
            tracking_uri=settings.mlflow_tracking_uri,
            registered_model_name=settings.mlflow_registered_model_name,
            deployment_alias=settings.mlflow_deployment_alias,
        )
        return {"enabled": True, **status}
    except Exception as exc:  # noqa: BLE001
        return {
            "enabled": True,
            "error": str(exc),
            "registered_model_name": settings.mlflow_registered_model_name,
            "deployment_alias": settings.mlflow_deployment_alias,
            "aliases": {},
            "versions": [],
            "events": [],
        }
