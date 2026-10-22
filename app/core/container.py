from __future__ import annotations

from functools import lru_cache

from app.agents.chat_agent import RetentionChatAgent
from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.ml_agent import MLAgent
from app.agents.providers.azure_openai import AzureOpenAIRecommendationProvider
from app.agents.providers.base import infer_generation_mode
from app.agents.providers.deterministic import DeterministicRecommendationProvider
from app.agents.providers.deterministic_chat import DeterministicRetentionChatProvider
from app.agents.providers.langgraph_ollama import LangGraphOllamaRecommendationProvider
from app.agents.providers.microsoft_ollama import MicrosoftOllamaRecommendationProvider
from app.agents.providers.microsoft_ollama_chat import MicrosoftOllamaChatProvider
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.report_agent import ReportAgent
from app.agents.reviewer_agent import ReviewerAgent
from app.agents.sql_agent import SQLAgent
from app.core.config import Settings, get_settings
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.customer_repository import CustomerRepository
from app.repositories.database import Database
from app.repositories.memory_repository import MemoryRepository
from app.services.agent_skill_service import AgentSkillService
from app.services.knowledge_service import KnowledgeService
from app.services.memory_service import MemoryService
from app.services.ml_service import MLService
from app.services.notification_service import NotificationService
from app.services.portfolio_service import PortfolioService
from app.services.recommendation_service import RecommendationService
from app.services.sql_service import SQLService
from app.tools.retention_tools import RetentionToolRegistry
from app.workflows.customer_retention import CustomerRetentionWorkflow


class ApplicationContainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.database = Database(settings.database_path)
        self.database.initialize_operational_tables()
        self.customer_repository = CustomerRepository(self.database)
        self.memory_repository = MemoryRepository(self.database)
        self.analysis_repository = AnalysisRepository(self.database)
        self.sql_service = SQLService(self.customer_repository)
        self.ml_service = MLService(
            settings.model_path,
            settings.risk_high,
            settings.risk_medium,
            tracking_uri=settings.mlflow_tracking_uri,
            use_model_registry=settings.mlflow_use_model_registry,
            registered_model_name=settings.mlflow_registered_model_name,
            deployment_alias=settings.mlflow_deployment_alias,
        )
        self.knowledge_service = KnowledgeService(
            settings.knowledge_dir,
            settings.retrieval_min_score,
            backend=settings.retrieval_backend,
            chroma_persist_path=settings.retrieval_chroma_persist_path,
            chroma_collection=settings.retrieval_chroma_collection,
            embedding_provider=settings.retrieval_embedding_provider,
            azure_openai_endpoint=settings.retrieval_azure_openai_endpoint,
            azure_openai_api_key=settings.retrieval_azure_openai_api_key,
            azure_openai_deployment=settings.retrieval_azure_openai_deployment,
            azure_openai_api_version=settings.retrieval_azure_openai_api_version,
            azure_openai_timeout_seconds=settings.retrieval_azure_openai_timeout_seconds,
        )
        self.memory_service = MemoryService(
            self.memory_repository,
            settings.max_session_messages,
            settings.max_customer_memories,
        )
        self.recommendation_service = RecommendationService()
        self.portfolio_service = PortfolioService(
            self.customer_repository,
            self.ml_service,
        )
        self.notification_service = NotificationService(self.analysis_repository)
        self.agent_skill_service = AgentSkillService(
            settings.skills_dir,
            settings.skill_max_resource_bytes,
        )
        self.tools = RetentionToolRegistry(
            self.sql_service,
            self.ml_service,
            self.knowledge_service,
            self.agent_skill_service,
            self.memory_service,
            self.portfolio_service,
        )
        fallback = DeterministicRecommendationProvider(self.recommendation_service)
        if settings.agent_provider == "microsoft_ollama":
            provider = MicrosoftOllamaRecommendationProvider(
                settings.ollama_base_url,
                settings.ollama_model,
                self.tools,
                settings=settings,
                skills_dir=settings.skills_dir,
            )
        elif settings.agent_provider == "langgraph_ollama":
            provider = LangGraphOllamaRecommendationProvider(
                settings.ollama_base_url,
                settings.ollama_model,
                settings=settings,
                tools=self.tools,
            )
        elif settings.agent_provider == "azure_openai":
            provider = AzureOpenAIRecommendationProvider(
                settings.azure_openai_endpoint,
                settings.azure_openai_api_key,
                settings.azure_openai_deployment,
                api_version=settings.azure_openai_api_version,
                settings=settings,
            )
        else:
            provider = fallback
        self.provider = provider
        chat_fallback = DeterministicRetentionChatProvider(self.tools)
        if settings.agent_provider == "microsoft_ollama":
            chat_provider = MicrosoftOllamaChatProvider(
                settings.ollama_base_url,
                settings.ollama_model,
                self.tools,
                settings,
            )
        else:
            chat_provider = chat_fallback
        self.chat_agent = RetentionChatAgent(chat_provider, chat_fallback)
        initial_generation_mode = "deterministic" if provider is fallback else infer_generation_mode(provider)
        self.provider_status = {
            "provider": settings.agent_provider,
            "requested_provider": settings.agent_provider,
            "model": getattr(provider, "model_id", settings.ollama_model),
            "generation_mode": initial_generation_mode,
            "provider_available": bool(
                getattr(provider, "service_available", getattr(provider, "ollama_available", provider is fallback))
            ),
            "ollama_available": bool(getattr(provider, "ollama_available", False)),
            "fallback_used": False,
            "fallback_reason": None,
            "llm_latency_ms": None,
            "last_successful_llm_call": getattr(provider, "last_successful_llm_call", None),
        }
        self.recommendation_agent = RecommendationAgent(
            provider,
            fallback,
            allow_fallback=settings.allow_deterministic_fallback,
        )
        self.reviewer_agent = ReviewerAgent(
            llm_provider=provider if hasattr(provider, "review_recommendation") else None,
            use_llm_review=settings.agent_provider == "microsoft_ollama",
        )
        self.workflow = CustomerRetentionWorkflow(
            sql_agent=SQLAgent(self.sql_service),
            ml_agent=MLAgent(self.ml_service),
            knowledge_agent=KnowledgeAgent(
                self.knowledge_service, top_k=settings.retrieval_top_k
            ),
            recommendation_agent=self.recommendation_agent,
            reviewer_agent=self.reviewer_agent,
            report_agent=ReportAgent(),
            memory_service=self.memory_service,
            analysis_repository=self.analysis_repository,
            notification_service=self.notification_service,
            provider_status=self.provider_status,
        )


@lru_cache(maxsize=1)
def get_container() -> ApplicationContainer:
    return ApplicationContainer(get_settings())
