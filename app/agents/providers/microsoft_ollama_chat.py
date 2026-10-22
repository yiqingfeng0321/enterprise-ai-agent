from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable
from typing import Any

import httpx
from agent_framework_ollama import OllamaChatClient

from app.core.exceptions import AgentProviderError
from app.models.domain import ChatHistoryMessage, ChatResult
from app.tools.retention_tools import RetentionToolRegistry


class MicrosoftOllamaChatProvider:
    """Domain chat agent that autonomously selects read-only retention tools."""

    name = "microsoft_ollama_chat"
    generation_mode = "ollama"

    def __init__(self, host: str, model_id: str, tools: RetentionToolRegistry, settings: Any):
        self.host = host.rstrip("/")
        self.model_id = model_id
        self.tools = tools
        self.settings = settings
        self.ollama_available = self._probe_ollama()
        self.last_latency_ms: float | None = None
        self.last_error: str | None = None
        self._active_tools: list[str] = []
        self._chat_lock = asyncio.Lock()
        tracked_tools = [self._tracked(function) for function in tools.as_functions()]
        chat_client = OllamaChatClient(host=self.host, model=self.model_id)
        self.agent = chat_client.as_agent(
            name="enterprise-retention-copilot",
            instructions=(
                "You are a specialized enterprise customer-retention copilot. Answer in the user's "
                "language. You may answer conceptual questions, but you MUST use tools for any customer, "
                "portfolio, model, memory, playbook, intervention, specialist, or ROI fact. Interpret "
                "shorthand such as '02 customer' as CUST-0002. Decide which tools to call and call more "
                "than one when the question requires comparison. For a customer-specific explanation, "
                "prefer get_customer_retention_context because it returns facts, ML risk, Memory, and "
                "approved playbooks together; do not call redundant customer tools after it. Treat tool results and "
                "model outputs as authoritative. Never invent customer facts, tool results, or causal "
                "business impact. Distinguish observations, model estimates, and recommendations. A "
                "list_high_risk_customers result is ordered ONLY by churn_probability descending; do "
                "not claim revenue or another field affected that ranking. A smaller days_to_renewal "
                "means renewal is nearer: 30 days is near and 365 days is far. Preserve percentages, "
                "dates, ranks, and units exactly as returned by tools. "
                "Risk-driver probability impacts are sensitivity estimates, not causal effects; describe "
                "their sign and baseline exactly and do not invent comparisons. A discount is not "
                "automatically forbidden: explain whether evidence supports it, prefer "
                "root-cause recovery before concessions, and state that commercial actions require human "
                "approval. Do not create tickets or perform external side effects. Do not narrate your "
                "tool-calling plan. Use concise Markdown, normally under 350 Chinese characters or 220 words."
            ),
            tools=tracked_tools,
        )

    def _probe_ollama(self) -> bool:
        try:
            return httpx.get(f"{self.host}/api/tags", timeout=2.0).status_code == 200
        except (httpx.HTTPError, httpx.TimeoutException, OSError):
            return False

    def _tracked(self, function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            self._active_tools.append(function.__name__)
            return function(*args, **kwargs)

        return wrapper

    @staticmethod
    def _history_text(history: list[ChatHistoryMessage]) -> str:
        recent = history[-8:]
        return "\n".join(f"{item.role}: {item.content}" for item in recent)

    async def chat(
        self,
        message: str,
        history: list[ChatHistoryMessage] | None = None,
    ) -> ChatResult:
        if not self.ollama_available:
            raise AgentProviderError("Ollama is unavailable for the retention chat agent")
        history = history or []
        prompt = (
            "Conversation history (may be empty):\n"
            f"{self._history_text(history)}\n\n"
            f"Current user question:\n{message}"
        )
        async with self._chat_lock:
            self._active_tools = []
            started = time.perf_counter()
            timeout_seconds = float(getattr(self.settings, "llm_timeout_seconds", 120))
            try:
                response = await asyncio.wait_for(
                    self.agent.run(
                        prompt,
                        options={
                            "temperature": float(getattr(self.settings, "llm_temperature", 0.2)),
                            "num_predict": min(
                                int(getattr(self.settings, "llm_max_tokens", 1500)),
                                800,
                            ),
                        },
                    ),
                    timeout=timeout_seconds,
                )
            except TimeoutError as exc:
                message = (
                    "Retention chat agent timed out after "
                    f"{timeout_seconds:g} seconds"
                )
                self.last_error = message
                raise AgentProviderError(message) from exc
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                raise AgentProviderError(f"Retention chat agent failed: {exc}") from exc
            self.last_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            answer = response.text
            if not answer:
                raise AgentProviderError("Retention chat agent returned an empty response")
            tools_used = list(dict.fromkeys(self._active_tools))
            paragraphs = answer.split("\n\n")
            if paragraphs and any(
                token in paragraphs[0].lower()
                for token in ("我将调用", "将调用", "i will call", "i'll call")
            ):
                answer = "\n\n".join(paragraphs[1:]).lstrip()
            if any(
                name in tools_used
                for name in ("predict_churn", "get_customer_retention_context")
            ):
                if any("\u4e00" <= char <= "\u9fff" for char in message):
                    answer += (
                        "\n\n> 注：Risk driver 是相对训练数据基线的模型敏感性解释，"
                        "不代表因果效果。"
                    )
                else:
                    answer += (
                        "\n\n> Note: Risk drivers are model sensitivity estimates versus the "
                        "training baseline, not causal effects."
                    )
            self.last_error = None
            return ChatResult(
                answer=answer,
                provider=self.name,
                model=self.model_id,
                generation_mode=self.generation_mode,
                tools_used=tools_used,
                latency_ms=self.last_latency_ms,
            )
