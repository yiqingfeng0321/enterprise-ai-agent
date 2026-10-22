from __future__ import annotations

import logging
from typing import Any

from app.agents.providers.base import RecommendationProvider, infer_generation_mode
from app.agents.providers.deterministic import DeterministicRecommendationProvider
from app.core.exceptions import AgentProviderError
from app.models.domain import (
    CustomerProfile,
    KnowledgeHit,
    MemoryItem,
    PredictionResult,
    Recommendation,
)

logger = logging.getLogger(__name__)


class RecommendationAgent:
    def __init__(
        self,
        provider: RecommendationProvider,
        fallback: DeterministicRecommendationProvider | None,
        allow_fallback: bool = False,
    ):
        self.provider = provider
        self.fallback = fallback
        self.allow_fallback = allow_fallback
        self.last_run_metadata: dict[str, Any] = {}

    def _build_metadata(self, *, provider_name: str, model_name: str | None, fallback_used: bool, generation_mode: str, fallback_reason: str | None, llm_latency_ms: float | None, provider_available: bool, ollama_available: bool) -> dict[str, Any]:
        return {
            "provider": provider_name,
            "requested_provider": getattr(self.provider, "name", type(self.provider).__name__),
            "model": model_name or "rules-v1",
            "generation_mode": generation_mode,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "llm_latency_ms": llm_latency_ms,
            "provider_available": provider_available,
            "ollama_available": ollama_available,
            "last_successful_llm_call": getattr(self.provider, "last_successful_llm_call", None),
        }

    @staticmethod
    def _should_auto_fallback(exc: AgentProviderError) -> bool:
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "404",
                "connection",
                "connect",
                "timed out",
                "timeout",
                "refused",
                "transport",
                "service not found",
                "invalid json",
                "validation failed",
                "payload validation",
                "schema",
            )
        )

    async def run(
        self,
        profile: CustomerProfile,
        prediction: PredictionResult,
        benchmark: dict,
        knowledge: list[KnowledgeHit],
        memories: list[MemoryItem],
    ) -> Recommendation:
        self.last_run_metadata = self._build_metadata(
            provider_name=getattr(self.provider, "name", type(self.provider).__name__),
            model_name=getattr(self.provider, "model_id", None),
            fallback_used=False,
            generation_mode=infer_generation_mode(self.provider),
            fallback_reason=None,
            llm_latency_ms=getattr(self.provider, "last_latency_ms", None),
            provider_available=bool(
                getattr(self.provider, "service_available", getattr(self.provider, "ollama_available", False))
            ),
            ollama_available=bool(getattr(self.provider, "ollama_available", False)),
        )
        try:
            recommendation = await self.provider.generate(
                profile, prediction, benchmark, knowledge, memories
            )
            self.last_run_metadata.update(
                self._build_metadata(
                    provider_name=getattr(self.provider, "name", type(self.provider).__name__),
                    model_name=getattr(self.provider, "model_id", None),
                    fallback_used=False,
                    generation_mode=infer_generation_mode(self.provider),
                    fallback_reason=None,
                    llm_latency_ms=getattr(self.provider, "last_latency_ms", None),
                    provider_available=bool(
                        getattr(self.provider, "service_available", getattr(self.provider, "ollama_available", False))
                    ),
                    ollama_available=bool(getattr(self.provider, "ollama_available", False)),
                )
            )
            return recommendation
        except AgentProviderError as exc:
            if self.allow_fallback and self.fallback is not None:
                logger.warning("Agent provider failed; using deterministic fallback: %s", exc)
                result = await self.fallback.generate(
                    profile, prediction, benchmark, knowledge, memories
                )
                result.generated_by = f"deterministic_fallback_after:{type(exc).__name__}"
                self.last_run_metadata.update(
                    self._build_metadata(
                        provider_name="deterministic",
                        model_name=getattr(self.provider, "model_id", None),
                        fallback_used=True,
                        generation_mode="deterministic",
                        fallback_reason=str(exc),
                        llm_latency_ms=None,
                        provider_available=bool(
                            getattr(self.provider, "service_available", getattr(self.provider, "ollama_available", False))
                        ),
                        ollama_available=bool(getattr(self.provider, "ollama_available", False)),
                    )
                )
                return result
            if self._should_auto_fallback(exc):
                logger.warning("Agent provider unavailable; using deterministic fallback for local demo path: %s", exc)
                result = await self.fallback.generate(
                    profile, prediction, benchmark, knowledge, memories
                )
                result.generated_by = f"deterministic_fallback_after:{type(exc).__name__}"
                self.last_run_metadata.update(
                    self._build_metadata(
                        provider_name="deterministic",
                        model_name=getattr(self.provider, "model_id", None),
                        fallback_used=True,
                        generation_mode="deterministic",
                        fallback_reason=str(exc),
                        llm_latency_ms=None,
                        provider_available=bool(
                            getattr(self.provider, "service_available", getattr(self.provider, "ollama_available", False))
                        ),
                        ollama_available=bool(getattr(self.provider, "ollama_available", False)),
                    )
                )
                return result
            logger.error("Agent provider failed and deterministic fallback is disabled: %s", exc)
            raise
