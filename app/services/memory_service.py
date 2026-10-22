from __future__ import annotations

import re

from app.models.domain import MemoryItem, PredictionResult, Recommendation
from app.repositories.memory_repository import MemoryRepository


class MemoryService:
    def __init__(
        self,
        repository: MemoryRepository,
        max_session_messages: int = 12,
        max_customer_memories: int = 8,
    ):
        self.repository = repository
        self.max_session_messages = max_session_messages
        self.max_customer_memories = max_customer_memories

    def get_context(self, session_id: str, customer_id: str) -> tuple[list[dict], list[MemoryItem]]:
        session = self.repository.get_session_messages(
            session_id, customer_id, self.max_session_messages
        )
        long_term = self.repository.get_customer_memories(
            customer_id, self.max_customer_memories
        )
        return session, long_term

    def get_customer_memories(self, customer_id: str) -> list[MemoryItem]:
        return self.repository.get_customer_memories(
            customer_id,
            self.max_customer_memories,
        )

    def record_request(self, session_id: str, customer_id: str, content: str) -> None:
        self.repository.add_session_message(session_id, customer_id, "user", content)

    def record_result(
        self,
        session_id: str,
        prediction: PredictionResult,
        recommendation: Recommendation,
    ) -> None:
        summary = (
            f"Risk={prediction.risk_band} ({prediction.churn_probability:.1%}). "
            f"Recommended: {recommendation.actions[0].action if recommendation.actions else 'No action'}."
        )
        self.repository.add_session_message(
            session_id, prediction.customer_id, "assistant", summary
        )
        importance = 0.9 if prediction.risk_band == "high" else 0.6
        self.repository.add_customer_memory(
            MemoryItem(
                customer_id=prediction.customer_id,
                memory_type="analysis_summary",
                content=summary,
                importance=importance,
            )
        )

        # Keep lightweight intervention history so future recommendations can avoid repeats.
        for action in recommendation.actions[:3]:
            self.repository.add_customer_memory(
                MemoryItem(
                    customer_id=prediction.customer_id,
                    memory_type="intervention_recommendation",
                    content=f"{action.action} | owner={action.owner} | priority={action.priority}",
                    importance=0.7 if action.approval_required else 0.55,
                )
            )

    @staticmethod
    def _memory_type_from_feedback(comment: str, rating: int) -> str:
        text = comment.lower()
        negative_tokens = (
            "no effect",
            "didn't work",
            "did not work",
            "ineffective",
            "failed",
            "没效果",
            "无效",
            "失败",
        )
        if any(token in text for token in negative_tokens) or rating <= 2:
            return "intervention_outcome_negative"
        if rating >= 4:
            return "intervention_outcome_positive"
        return "intervention_outcome_mixed"

    @staticmethod
    def _importance_from_feedback(rating: int) -> float:
        if rating <= 2:
            return 0.95
        if rating >= 4:
            return 0.8
        return 0.7

    def record_feedback(self, customer_id: str, rating: int, comment: str) -> None:
        cleaned_comment = re.sub(r"\s+", " ", (comment or "")).strip() or "(no comment)"
        self.repository.add_customer_memory(
            MemoryItem(
                customer_id=customer_id,
                memory_type=self._memory_type_from_feedback(cleaned_comment, rating),
                content=f"feedback_rating={rating}; note={cleaned_comment}",
                importance=self._importance_from_feedback(rating),
            )
        )
