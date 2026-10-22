from __future__ import annotations

from typing import Protocol

from app.models.domain import (
    CustomerProfile,
    KnowledgeHit,
    MemoryItem,
    PredictionResult,
    Recommendation,
)


class RecommendationProvider(Protocol):
    async def generate(
        self,
        profile: CustomerProfile,
        prediction: PredictionResult,
        benchmark: dict,
        knowledge: list[KnowledgeHit],
        memories: list[MemoryItem],
    ) -> Recommendation: ...


def infer_generation_mode(provider: object) -> str:
    mode = getattr(provider, "generation_mode", "")
    if mode:
        return str(mode)
    provider_name = str(getattr(provider, "name", type(provider).__name__)).lower()
    if "deterministic" in provider_name or "fallback" in provider_name:
        return "deterministic"
    if "ollama" in provider_name:
        return "ollama"
    return provider_name or type(provider).__name__.lower()
