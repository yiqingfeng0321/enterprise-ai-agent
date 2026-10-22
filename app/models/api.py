from pydantic import BaseModel, Field

from app.models.domain import AnalysisResponse, ChatHistoryMessage, ChatResult, Feedback


class AnalysisRequest(BaseModel):
    customer_id: str = Field(min_length=3)
    session_id: str = Field(default="default", min_length=1)
    create_ticket: bool = False


class AnalysisApiResponse(BaseModel):
    data: AnalysisResponse


class FeedbackRequest(Feedback):
    pass


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    session_id: str = Field(default="streamlit-chat", min_length=1, max_length=100)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=12)


class ChatApiResponse(BaseModel):
    data: ChatResult


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    database_ready: bool
    agent_provider: str
    model: str = ""
    requested_provider: str = ""
    generation_mode: str = "deterministic"
    provider_available: bool = False
    ollama_available: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    last_successful_llm_call: str | None = None
    retrieval_requested_backend: str = "lexical"
    retrieval_active_backend: str = "tfidf"
    retrieval_degraded: bool = False
    retrieval_error: str | None = None
