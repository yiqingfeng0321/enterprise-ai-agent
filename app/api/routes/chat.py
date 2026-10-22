from fastapi import APIRouter, Depends

from app.api.dependencies import container_dependency
from app.core.container import ApplicationContainer
from app.core.logging import audit_event
from app.models.api import ChatApiResponse, ChatRequest

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("", response_model=ChatApiResponse)
async def chat(
    request: ChatRequest,
    container: ApplicationContainer = Depends(container_dependency),
) -> ChatApiResponse:
    result = await container.chat_agent.run(request.message, request.history)
    audit_event(
        "retention_chat_completed",
        session_id=request.session_id,
        provider=result.provider,
        tools_used=result.tools_used,
        fallback_used=result.fallback_used,
        latency_ms=result.latency_ms,
    )
    return ChatApiResponse(data=result)
