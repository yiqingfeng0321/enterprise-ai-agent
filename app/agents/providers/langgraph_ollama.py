from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime
from typing import Any, TypedDict

import httpx

from app.core.exceptions import AgentProviderError
from app.models.domain import (
    ActionItem,
    CustomerProfile,
    KnowledgeHit,
    MemoryItem,
    PredictionResult,
    Recommendation,
)
from app.services.action_catalog import approved_interventions
from app.services.agent_skill_router import recommended_agent_skills
from app.services.specialist_catalog import available_specialists, recommended_specialist_focus
from app.tools.retention_tools import RetentionToolRegistry


class RetentionState(TypedDict, total=False):
    prompt: str
    raw_text: str
    payload: dict[str, Any]
    validation_error: str
    repair_attempts: int


class LangGraphOllamaRecommendationProvider:
    """LangGraph-based provider that composes LangChain prompt/model with stateful orchestration."""

    name = "langgraph_ollama"
    generation_mode = "ollama"

    def __init__(
        self,
        host: str,
        model_id: str,
        settings: Any | None = None,
        tools: RetentionToolRegistry | None = None,
    ):
        self.host = host.rstrip("/")
        self.model_id = model_id
        self.settings = settings
        self.tools = tools
        self.ollama_available = self._probe_ollama()
        self.last_latency_ms: float | None = None
        self.last_successful_llm_call: str | None = None
        self.last_error: str | None = None

    def _probe_ollama(self) -> bool:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=2.0)
            return response.status_code == 200
        except (httpx.HTTPError, httpx.TimeoutException, OSError):
            return False

    @staticmethod
    def _extract_json(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        raise AgentProviderError("The LangGraph/Ollama path did not return a JSON object")

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.5
        if confidence > 1.0 and confidence <= 100.0:
            confidence = confidence / 100.0
        if confidence < 0.0:
            return 0.0
        if confidence > 1.0:
            return 1.0
        return confidence

    @staticmethod
    def _coerce_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            blocks = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    blocks.append(str(item["text"]))
                else:
                    blocks.append(str(item))
            return "\n".join(blocks)
        return str(content)

    def _build_context(
        self,
        profile: CustomerProfile,
        prediction: PredictionResult,
        benchmark: dict,
        knowledge: list[KnowledgeHit],
        memories: list[MemoryItem],
    ) -> dict[str, Any]:
        intervention_history = [
            {
                "type": memory.memory_type,
                "content": memory.content[:220],
                "importance": memory.importance,
            }
            for memory in memories[:3]
            if memory.memory_type.startswith("intervention_")
        ]
        recommended_skills = recommended_agent_skills(profile, prediction, memories)
        loaded_skills = (
            [
                self.tools.agent_skill_service.load_skill(item["skill"])
                for item in recommended_skills
            ]
            if self.tools
            else []
        )
        return {
            "customer_id": profile.customer_id,
            "customer_profile": {
                "segment": profile.segment,
                "region": profile.region,
                "customer_notes": profile.customer_notes,
                "licensed_seats": profile.licensed_seats,
                "active_seat_ratio_30d": profile.active_seat_ratio_30d,
                "champion_count": profile.champion_count,
                "monthly_revenue": profile.monthly_revenue,
                "days_to_renewal": profile.days_to_renewal,
                "onboarding_complete": profile.onboarding_complete,
                "support_tickets_90d": profile.support_tickets_90d,
                "payment_failures_90d": profile.payment_failures_90d,
                "engagement_score": profile.engagement_score,
                "revenue_at_risk": profile.revenue_at_risk,
            },
            "prediction": {
                "customer_id": prediction.customer_id,
                "churn_probability": prediction.churn_probability,
                "risk_band": prediction.risk_band,
                "model_version": prediction.model_version,
            },
            "risk_drivers": [driver.model_dump() for driver in prediction.top_drivers[:3]],
            "segment_benchmark": {
                "segment": benchmark.get("segment"),
                "observed_churn_rate": benchmark.get("observed_churn_rate"),
                "avg_nps": benchmark.get("avg_nps"),
                "avg_logins_30d": benchmark.get("avg_logins_30d"),
                "avg_active_seat_ratio_30d": benchmark.get("avg_active_seat_ratio_30d"),
                "avg_support_tickets_90d": benchmark.get("avg_support_tickets_90d"),
            },
            "retrieved_playbooks": [
                {
                    "document": hit.document,
                    "section": hit.section,
                    "score": hit.score,
                }
                for hit in knowledge[:2]
            ],
            "long_term_memory": [
                {
                    "memory_type": memory.memory_type,
                    "content": memory.content[:160],
                    "importance": memory.importance,
                }
                for memory in memories[:2]
            ],
            "intervention_history": intervention_history,
            "available_interventions": [
                {
                    "name": item["name"],
                    "owner": item["owner"],
                    "approval_required": item["approval_required"],
                }
                for item in approved_interventions()[:5]
            ],
            "available_agent_skills": (
                self.tools.agent_skill_service.list_skills() if self.tools else []
            ),
            "recommended_agent_skills": recommended_skills,
            "loaded_agent_skills": loaded_skills,
            "available_specialists": [
                {
                    "name": item["name"],
                    "best_for": item["best_for"],
                }
                for item in available_specialists()[:4]
            ],
            "recommended_specialist_focus": recommended_specialist_focus(profile, prediction)[:3],
            "business_constraints": [
                "Use only retrieved customer facts and model outputs.",
                "Do not invent customer facts or model metrics.",
                "Use customer_profile.customer_notes only when present to personalize communication and interventions.",
                "Choose from or adapt the approved available_interventions list before inventing a new tactic.",
                "Follow the loaded_agent_skills selected by the application-side skill router.",
                "Use available_specialists and recommended_specialist_focus to assign internal ownership.",
                "Avoid repeating interventions that were previously marked ineffective in memory.",
                "The evidence array must include a bullet that starts with 'Model risk:' and cites churn probability and risk band.",
                "Return valid JSON only.",
                "Human approval is required before external side effects.",
            ],
        }

    @staticmethod
    def _build_prompt(context_json: str, risk_band: str) -> str:
        approval_instruction = (
            "Because this account is high risk, requires_human_approval must be true and at least one recommended action must have approval_required=true."
            if risk_band == "high"
            else "Set requires_human_approval according to the action plan."
        )
        return (
            "You are an enterprise retention planner. Return one valid JSON object only. "
            "Use only supplied facts. Do not invent customer statements, product facts, or model metrics. "
            "The JSON schema is: "
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
            "Return 2 to 4 recommended_actions. Include at least 3 evidence bullets, and one evidence bullet must start with 'Model risk:' and cite the churn probability and risk band. "
            f"{approval_instruction} "
            "Keep each field concise. Prefer operational recovery, adoption improvement, and value realization before commercial concessions. "
            f"Context: {context_json}"
        )

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
            f"Validation error: {validation_error}. {extra} Do not reuse prior wording. Keep the revised response concise."
        )

    @staticmethod
    def _validate_payload(payload: dict[str, Any], risk_band: str) -> None:
        required = [
            "executive_summary",
            "risk_assessment",
            "recommended_actions",
            "rationale",
            "evidence",
            "do_not_do",
            "confidence",
            "requires_human_approval",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise AgentProviderError(f"Missing required keys: {', '.join(missing)}")
        recommended_actions = payload.get("recommended_actions")
        if not isinstance(recommended_actions, list) or len(recommended_actions) < 2:
            raise AgentProviderError("recommended_actions must contain at least 2 actions")
        evidence = payload.get("evidence")
        if not isinstance(evidence, list) or len(evidence) < 3:
            raise AgentProviderError("evidence must contain at least 3 bullets")
        if not any(
            "model prediction" in str(item).lower()
            or "model risk" in str(item).lower()
            or "churn probability" in str(item).lower()
            for item in evidence
        ):
            raise AgentProviderError("evidence must cite the model prediction and churn probability")
        if risk_band == "high":
            if payload.get("requires_human_approval") is not True:
                raise AgentProviderError("high-risk accounts must require human approval")
            if not any(
                isinstance(action, dict) and bool(action.get("approval_required"))
                for action in recommended_actions
            ):
                raise AgentProviderError("high-risk accounts must include at least one approval-required action")

    def _normalize_payload(self, payload: dict[str, Any]) -> Recommendation:
        actions: list[ActionItem] = []
        recommended_actions = payload.get("recommended_actions") or []
        if isinstance(recommended_actions, list) and recommended_actions:
            for index, action in enumerate(recommended_actions, start=1):
                if isinstance(action, dict):
                    action_text = str(action.get("action", action))
                    rationale = str(action.get("rationale", payload.get("rationale", "Generated by LangGraph")))
                    owner = str(action.get("owner", "Customer Success"))
                    expected_impact = str(
                        action.get(
                            "expected_impact",
                            "Address churn risk with evidence-backed next steps.",
                        )
                    )
                    approval_required = bool(
                        action.get("approval_required", payload.get("requires_human_approval", False))
                    )
                else:
                    action_text = str(action)
                    rationale = str(payload.get("rationale", "Generated by LangGraph"))
                    owner = "Customer Success"
                    expected_impact = "Address churn risk with evidence-backed next steps."
                    approval_required = bool(payload.get("requires_human_approval", False))
                actions.append(
                    ActionItem(
                        priority=index,
                        action=action_text,
                        owner=owner,
                        rationale=rationale,
                        expected_impact=expected_impact,
                        approval_required=approval_required,
                    )
                )
        if not actions:
            actions.append(
                ActionItem(
                    priority=1,
                    action="Schedule an evidence-based retention review with the account owner.",
                    owner="Customer Success",
                    rationale=str(payload.get("rationale", "Generated by LangGraph")),
                    expected_impact="Address churn risk with evidence-backed next steps.",
                    approval_required=bool(payload.get("requires_human_approval", False)),
                )
            )

        evidence = payload.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        do_not_do = payload.get("do_not_do") or []
        if not isinstance(do_not_do, list):
            do_not_do = [str(do_not_do)]

        return Recommendation(
            executive_summary=str(payload.get("executive_summary", "")),
            risk_assessment=str(payload.get("risk_assessment", "")),
            recommended_actions=[
                str(item.get("action", item)) if isinstance(item, dict) else str(item)
                for item in recommended_actions
            ]
            if isinstance(recommended_actions, list)
            else [],
            rationale=str(payload.get("rationale", "")),
            actions=actions,
            do_not_do=[str(item) for item in do_not_do],
            evidence=[str(item) for item in evidence],
            confidence=self._normalize_confidence(payload.get("confidence", 0.5)),
            generated_by=self.name,
            requires_human_approval=bool(payload.get("requires_human_approval", False)),
        )

    async def generate(
        self,
        profile: CustomerProfile,
        prediction: PredictionResult,
        benchmark: dict,
        knowledge: list[KnowledgeHit],
        memories: list[MemoryItem],
    ) -> Recommendation:
        if not self.ollama_available:
            raise AgentProviderError(
                "Ollama is unavailable. Start the local service and ensure the configured model is installed."
            )

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_ollama import ChatOllama
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise AgentProviderError(
                "LangChain/LangGraph dependencies are missing. Install with: pip install -e '.[dev,ui]'"
            ) from exc

        context = self._build_context(profile, prediction, benchmark, knowledge, memories)
        context_json = json.dumps(context, default=str, separators=(",", ":"))
        prompt_template = ChatPromptTemplate.from_messages([("human", "{prompt}")])
        llm = ChatOllama(
            model=self.model_id,
            base_url=self.host,
            format="json",
            temperature=0.0,
            num_predict=min(int(getattr(self.settings, "llm_max_tokens", 1500)), 512),
        )

        async def llm_node(state: RetentionState) -> RetentionState:
            chain = prompt_template | llm
            start = time.perf_counter()
            response = await chain.ainvoke({"prompt": state["prompt"]})
            self.last_latency_ms = round((time.perf_counter() - start) * 1000, 2)
            self.last_successful_llm_call = datetime.now(UTC).isoformat()
            return {"raw_text": self._coerce_text(response.content)}

        def parse_node(state: RetentionState) -> RetentionState:
            try:
                payload = json.loads(self._extract_json(state["raw_text"]))
                self._validate_payload(payload, prediction.risk_band)
                return {"payload": payload, "validation_error": ""}
            except Exception as exc:  # noqa: BLE001
                attempts = int(state.get("repair_attempts", 0))
                if attempts >= 1:
                    raise AgentProviderError(
                        f"LangGraph/Ollama payload validation failed after repair: {exc}"
                    ) from exc
                return {
                    "validation_error": str(exc),
                    "repair_attempts": attempts + 1,
                    "prompt": self._build_repair_prompt(
                        state["raw_text"],
                        str(exc),
                        prediction.risk_band,
                    ),
                }

        def route_after_parse(state: RetentionState) -> str:
            return "end" if isinstance(state.get("payload"), dict) else "repair"

        graph = StateGraph(RetentionState)
        graph.add_node("llm", llm_node)
        graph.add_node("parse", parse_node)
        graph.add_node("repair", llm_node)
        graph.add_edge(START, "llm")
        graph.add_edge("llm", "parse")
        graph.add_conditional_edges(
            "parse",
            route_after_parse,
            {"repair": "repair", "end": END},
        )
        graph.add_edge("repair", "parse")

        try:
            app = graph.compile()
            timeout = float(getattr(self.settings, "llm_timeout_seconds", 90))
            try:
                state = await asyncio.wait_for(
                    app.ainvoke(
                        {
                            "prompt": self._build_prompt(context_json, prediction.risk_band),
                            "repair_attempts": 0,
                        }
                    ),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                raise AgentProviderError(
                    f"LangGraph/Ollama timed out after {timeout:.0f}s — Ollama may be busy or the model too large"
                ) from exc
            payload = state.get("payload")
            if not isinstance(payload, dict):
                raise AgentProviderError("LangGraph flow did not produce a recommendation payload")
            self.last_error = None
            return self._normalize_payload(payload)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            raise AgentProviderError(f"LangGraph/Ollama agent execution failed: {exc}") from exc
