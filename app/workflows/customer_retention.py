from __future__ import annotations

import asyncio
import uuid

from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.ml_agent import MLAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.report_agent import ReportAgent
from app.agents.reviewer_agent import ReviewerAgent
from app.agents.sql_agent import SQLAgent
from app.core.logging import audit_event
from app.models.domain import AnalysisResponse
from app.observability.timing import timer
from app.repositories.analysis_repository import AnalysisRepository
from app.services.memory_service import MemoryService
from app.services.notification_service import NotificationService


class CustomerRetentionWorkflow:
    def __init__(
        self,
        sql_agent: SQLAgent,
        ml_agent: MLAgent,
        knowledge_agent: KnowledgeAgent,
        recommendation_agent: RecommendationAgent,
        reviewer_agent: ReviewerAgent,
        report_agent: ReportAgent,
        memory_service: MemoryService,
        analysis_repository: AnalysisRepository,
        notification_service: NotificationService,
        provider_status: dict | None = None,
    ):
        self.sql_agent = sql_agent
        self.ml_agent = ml_agent
        self.knowledge_agent = knowledge_agent
        self.recommendation_agent = recommendation_agent
        self.reviewer_agent = reviewer_agent
        self.report_agent = report_agent
        self.memory_service = memory_service
        self.analysis_repository = analysis_repository
        self.notification_service = notification_service
        self.provider_status = provider_status if provider_status is not None else {}

    async def analyze(
        self,
        customer_id: str,
        session_id: str = "default",
        create_ticket: bool = False,
        *,
        persist: bool = True,
    ) -> AnalysisResponse:
        if create_ticket and not persist:
            raise ValueError("create_ticket requires persist=True")
        analysis_id = str(uuid.uuid4())
        with timer() as elapsed:
            if persist:
                self.memory_service.record_request(
                    session_id, customer_id, f"Analyze churn risk for {customer_id}"
                )
            profile = await self.sql_agent.get_profile(customer_id)
            prediction_task = asyncio.create_task(self.ml_agent.run(profile))
            benchmark_task = asyncio.create_task(self.sql_agent.get_benchmark(profile.segment))
            if persist:
                session_messages, memories = self.memory_service.get_context(
                    session_id, customer_id
                )
            else:
                session_messages, memories = [], []
            prediction, benchmark = await asyncio.gather(prediction_task, benchmark_task)
            knowledge = await self.knowledge_agent.run(profile, prediction)
            recommendation = await self.recommendation_agent.run(
                profile, prediction, benchmark, knowledge, memories
            )
            review = await self.reviewer_agent.run(profile, prediction, recommendation)
            report = await self.report_agent.run(
                profile, prediction, benchmark, knowledge, recommendation, review
            )
            if persist:
                self.analysis_repository.save_prediction(analysis_id, prediction)
                self.memory_service.record_result(session_id, prediction, recommendation)
            ticket_id = None
            if create_ticket:
                ticket_id = self.notification_service.create_retention_ticket(
                    analysis_id, customer_id, recommendation
                )

        recommendation_meta = getattr(self.recommendation_agent, "last_run_metadata", {})
        self.provider_status.update(
            {
                "provider": recommendation_meta.get("provider", ""),
                "requested_provider": recommendation_meta.get("requested_provider", ""),
                "model": recommendation_meta.get("model", ""),
                "generation_mode": recommendation_meta.get("generation_mode", "deterministic"),
                "provider_available": recommendation_meta.get(
                    "provider_available",
                    recommendation_meta.get("ollama_available", True),
                ),
                "ollama_available": recommendation_meta.get("ollama_available", True),
                "fallback_used": recommendation_meta.get("fallback_used", False),
                "fallback_reason": recommendation_meta.get("fallback_reason"),
                "llm_latency_ms": recommendation_meta.get("llm_latency_ms"),
                "last_successful_llm_call": recommendation_meta.get("last_successful_llm_call"),
            }
        )
        audit_event(
            "retention_analysis_completed",
            analysis_id=analysis_id,
            customer_id=customer_id,
            risk_band=prediction.risk_band,
            provider=recommendation_meta.get("provider", recommendation.generated_by),
            latency_ms=round(elapsed["elapsed_ms"], 2),
            session_context_messages=len(session_messages),
        )
        return AnalysisResponse(
            analysis_id=analysis_id,
            session_id=session_id,
            customer=profile,
            prediction=prediction,
            benchmark=benchmark,
            knowledge=knowledge,
            memories_used=memories,
            recommendation=recommendation,
            review=review,
            report_markdown=report,
            ticket_id=ticket_id,
            latency_ms=elapsed["elapsed_ms"],
            provider=recommendation_meta.get("provider", ""),
            model=recommendation_meta.get("model", ""),
            requested_provider=recommendation_meta.get("requested_provider", ""),
            generation_mode=recommendation_meta.get("generation_mode", "deterministic"),
            provider_available=recommendation_meta.get(
                "provider_available",
                recommendation_meta.get("ollama_available", True),
            ),
            ollama_available=recommendation_meta.get("ollama_available", True),
            fallback_used=recommendation_meta.get("fallback_used", False),
            fallback_reason=recommendation_meta.get("fallback_reason"),
            llm_latency_ms=recommendation_meta.get("llm_latency_ms"),
            last_successful_llm_call=recommendation_meta.get("last_successful_llm_call"),
        )
