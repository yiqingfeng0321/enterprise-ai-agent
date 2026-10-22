from __future__ import annotations

import logging
from typing import Any

from app.core.exceptions import AgentProviderError
from app.models.domain import ChatHistoryMessage, ChatResult

logger = logging.getLogger(__name__)


class RetentionChatAgent:
    def __init__(self, provider: Any, fallback: Any):
        self.provider = provider
        self.fallback = fallback

    async def run(
        self,
        message: str,
        history: list[ChatHistoryMessage] | None = None,
    ) -> ChatResult:
        try:
            return await self.provider.chat(message, history or [])
        except AgentProviderError as exc:
            logger.warning("Chat provider failed; using deterministic fallback: %s", exc)
            result = await self.fallback.chat(message, history or [])
            result.fallback_used = True
            result.fallback_reason = str(exc)
            return result
