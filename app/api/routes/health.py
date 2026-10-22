from fastapi import APIRouter, Depends

from app.api.dependencies import container_dependency
from app.core.container import ApplicationContainer
from app.models.api import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(container: ApplicationContainer = Depends(container_dependency)) -> HealthResponse:
    database_ready = container.settings.database_path.exists()
    model_ready = container.ml_service.ready
    provider_status = container.provider_status
    provider_available = bool(
        provider_status.get("provider_available", provider_status.get("ollama_available", False))
    )
    ollama_available = bool(provider_status.get("ollama_available", False))
    retrieval_status = container.knowledge_service.status
    status = (
        "ok"
        if database_ready
        and model_ready
        and provider_available
        and not retrieval_status["degraded"]
        else "degraded"
    )
    return HealthResponse(
        status=status,
        model_ready=model_ready,
        database_ready=database_ready,
        agent_provider=container.settings.agent_provider,
        model=provider_status.get("model", container.settings.ollama_model),
        requested_provider=provider_status.get("requested_provider", container.settings.agent_provider),
        generation_mode=provider_status.get("generation_mode", "deterministic"),
        provider_available=provider_available,
        ollama_available=ollama_available,
        fallback_used=bool(provider_status.get("fallback_used", False)),
        fallback_reason=provider_status.get("fallback_reason"),
        last_successful_llm_call=provider_status.get("last_successful_llm_call"),
        retrieval_requested_backend=str(retrieval_status["requested_backend"]),
        retrieval_active_backend=str(retrieval_status["active_backend"]),
        retrieval_degraded=bool(retrieval_status["degraded"]),
        retrieval_error=(
            str(retrieval_status["error"]) if retrieval_status["error"] else None
        ),
    )
