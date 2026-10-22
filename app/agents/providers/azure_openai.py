from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from app.agents.providers.microsoft_ollama import MicrosoftOllamaRecommendationProvider
from app.core.exceptions import AgentProviderError
from app.models.domain import (
    CustomerProfile,
    KnowledgeHit,
    MemoryItem,
    PredictionResult,
    Recommendation,
)


class AzureOpenAIRecommendationProvider(MicrosoftOllamaRecommendationProvider):
    """Azure OpenAI-backed provider that uses raw HTTP calls instead of an SDK."""

    name = "azure_openai"
    generation_mode = "azure_openai"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-06-01",
        settings: Any | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model_id = deployment
        self.api_version = api_version
        self.settings = settings
        self.service_available = self._probe_azure_openai()
        self.ollama_available = False
        self.last_latency_ms: float | None = None
        self.last_successful_llm_call: str | None = None
        self.last_error: str | None = None
        self.last_prompt: str | None = None

    def _probe_azure_openai(self) -> bool:
        if not self.endpoint or not self.api_key or not self.model_id:
            return False
        url = f"{self.endpoint}/openai/deployments?api-version={self.api_version}"
        headers = {"api-key": self.api_key}
        try:
            response = httpx.get(url, headers=headers, timeout=3.0)
            return response.status_code == 200
        except (httpx.HTTPError, httpx.TimeoutException, OSError):
            return False

    @staticmethod
    def _extract_content(response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            raise AgentProviderError("Azure OpenAI returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if not content:
            raise AgentProviderError("Azure OpenAI returned an empty response")
        return str(content)

    @staticmethod
    def _build_repair_prompt(raw_text: str, validation_error: str, risk_band: str) -> str:
        extra = (
            "Remember: for high-risk accounts, requires_human_approval must be true and one action must require approval."
            if risk_band == "high"
            else ""
        )
        return (
            "The previous response was invalid or incomplete. Return one valid JSON object only using this schema: "
            "{"
            "\"executive_summary\": string,"
            "\"risk_assessment\": string,"
            "\"recommended_actions\": ["
            "{\"action\": string, \"owner\": string, \"rationale\": string, \"expected_impact\": string, \"approval_required\": boolean}"
            "],"
            "\"rationale\": string,"
            "\"evidence\": [string],"
            "\"do_not_do\": [string],"
            "\"confidence\": number,"
            "\"requires_human_approval\": boolean"
            "}. "
            f"Validation error: {validation_error}. {extra} Do not reuse prior wording. "
            f"Previous response: {raw_text}. Keep the revised response concise."
        )

    async def _chat(self, prompt: str, *, retry: bool = False) -> str:
        if not self.service_available:
            raise AgentProviderError(
                "Azure OpenAI is unavailable. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT."
            )
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful retention-planning assistant. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(getattr(self.settings, "llm_temperature", 0.2)),
            "max_tokens": int(getattr(self.settings, "llm_max_tokens", 1500)),
        }
        url = (
            f"{self.endpoint}/openai/deployments/{self.model_id}"
            f"/chat/completions?api-version={self.api_version}"
        )
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=float(getattr(self.settings, "llm_timeout_seconds", 120))) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self.last_error = str(exc)
            raise AgentProviderError(f"Azure OpenAI agent execution failed: {exc}") from exc
        except httpx.HTTPError as exc:
            self.last_error = str(exc)
            raise AgentProviderError(f"Azure OpenAI agent execution failed: {exc}") from exc

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        self.last_latency_ms = latency_ms
        try:
            response_json = response.json()
        except ValueError as exc:
            self.last_error = str(exc)
            raise AgentProviderError("Azure OpenAI returned an invalid JSON payload") from exc
        content = self._extract_content(response_json)
        self.last_successful_llm_call = datetime.now(UTC).isoformat()
        if retry:
            self.last_error = None
        return content

    async def generate(
        self,
        profile: CustomerProfile,
        prediction: PredictionResult,
        benchmark: dict,
        knowledge: list[KnowledgeHit],
        memories: list[MemoryItem],
    ) -> Recommendation:
        context = self._build_context(profile, prediction, benchmark, knowledge, memories)
        prompt = self._build_prompt(context)
        self.last_prompt = prompt
        raw_text = ""
        try:
            raw_text = await self._chat(prompt)
        except AgentProviderError as exc:
            self.last_error = str(exc)
            raise

        try:
            parsed = json.loads(self._extract_json(raw_text))
            return self._normalize_payload(parsed)
        except (json.JSONDecodeError, AgentProviderError):
            try:
                repaired_text = await self._chat(self._build_repair_prompt(raw_text, "invalid JSON", prediction.risk_band), retry=True)
                parsed = json.loads(self._extract_json(repaired_text))
                return self._normalize_payload(parsed)
            except (json.JSONDecodeError, AgentProviderError) as repair_exc:
                raise AgentProviderError(
                    "Azure OpenAI returned invalid JSON after a repair attempt"
                ) from repair_exc
