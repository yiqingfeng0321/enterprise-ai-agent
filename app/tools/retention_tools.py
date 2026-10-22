from __future__ import annotations

import json

from app.services.action_catalog import approved_interventions
from app.services.agent_skill_service import AgentSkillService
from app.services.knowledge_service import KnowledgeService
from app.services.memory_service import MemoryService
from app.services.ml_service import MLService
from app.services.portfolio_service import PortfolioService
from app.services.specialist_catalog import available_specialists
from app.services.sql_service import SQLService


class RetentionToolRegistry:
    """Tool boundary exposed to LLM agents. Tools return JSON-serializable text."""

    def __init__(
        self,
        sql_service: SQLService,
        ml_service: MLService,
        knowledge_service: KnowledgeService,
        agent_skill_service: AgentSkillService,
        memory_service: MemoryService,
        portfolio_service: PortfolioService,
    ):
        self.sql_service = sql_service
        self.ml_service = ml_service
        self.knowledge_service = knowledge_service
        self.agent_skill_service = agent_skill_service
        self.memory_service = memory_service
        self.portfolio_service = portfolio_service

    @staticmethod
    def _customer_id(value: str) -> str:
        cleaned = value.strip().upper().replace(" ", "")
        if cleaned.startswith("CUST-"):
            suffix = cleaned.removeprefix("CUST-")
        elif cleaned.startswith("CUST"):
            suffix = cleaned.removeprefix("CUST")
        else:
            suffix = cleaned
        return f"CUST-{int(suffix):04d}" if suffix.isdigit() else cleaned

    def get_customer_profile(self, customer_id: str) -> str:
        """Get the complete customer profile for a customer ID."""
        return self.sql_service.customer_profile(self._customer_id(customer_id)).model_dump_json()

    def predict_churn(self, customer_id: str) -> str:
        """Predict churn risk and return model explanations for a customer ID."""
        profile = self.sql_service.customer_profile(self._customer_id(customer_id))
        return self.ml_service.predict(profile).model_dump_json()

    def get_customer_memories(self, customer_id: str) -> str:
        """Get prior analyses and intervention feedback for one customer."""
        normalized = self._customer_id(customer_id)
        memories = self.memory_service.get_customer_memories(normalized)
        return json.dumps([item.model_dump(mode="json") for item in memories], default=str)

    def get_customer_retention_context(
        self,
        customer_id: str,
        knowledge_query: str = "retention adoption support renewal discount value realization",
    ) -> str:
        """Get one customer's facts, ML risk, memories, and relevant approved playbooks."""
        normalized = self._customer_id(customer_id)
        profile = self.sql_service.customer_profile(normalized)
        prediction = self.ml_service.predict(profile)
        memories = self.memory_service.get_customer_memories(normalized)
        hits = self.knowledge_service.search(knowledge_query, top_k=3)
        return json.dumps(
            {
                "customer": profile.model_dump(mode="json"),
                "prediction": prediction.model_dump(mode="json"),
                "interpretation_rules": {
                    "risk_driver": (
                        "A model sensitivity estimate versus the training-data baseline; "
                        "it is not a causal effect."
                    ),
                    "renewal_urgency": (
                        "A derived 0-to-1 urgency score; higher means renewal is nearer. "
                        "It is not a renewal rate."
                    ),
                    "days_to_renewal": "Smaller means renewal is nearer.",
                },
                "memories": [item.model_dump(mode="json") for item in memories],
                "approved_playbooks": [hit.model_dump(mode="json") for hit in hits],
            },
            default=str,
        )

    def list_high_risk_customers(
        self,
        limit: int = 10,
        segment: str = "",
        renewal_within_days: int = 366,
    ) -> str:
        """Rank customers by model churn probability with optional portfolio filters."""
        rows = self.portfolio_service.ranked_customers(
            limit=limit,
            segment=segment or None,
            renewal_within_days=(
                renewal_within_days if 0 <= renewal_within_days < 366 else None
            ),
        )
        return json.dumps(rows, default=str)

    def get_portfolio_summary(self) -> str:
        """Return current portfolio risk and renewal summary metrics."""
        return json.dumps(self.portfolio_service.summary(), default=str)

    def get_segment_benchmark(self, segment: str) -> str:
        """Get aggregated engagement and churn benchmarks for a customer segment."""
        return json.dumps(self.sql_service.segment_benchmark(segment), default=str)

    def search_retention_playbooks(self, query: str) -> str:
        """Search approved retention and customer-success playbooks."""
        hits = self.knowledge_service.search(query, top_k=3)
        return json.dumps([hit.model_dump() for hit in hits], default=str)

    @staticmethod
    def list_available_interventions() -> str:
        """List approved intervention types that the agent may recommend."""
        return json.dumps(approved_interventions(), default=str)

    @staticmethod
    def list_available_specialists() -> str:
        """List internal specialist roles that can own parts of a retention plan."""
        return json.dumps(available_specialists(), default=str)

    def list_agent_skills(self) -> str:
        """Advertise formal Agent Skills by name and description without loading them."""
        return json.dumps(self.agent_skill_service.list_skills(), default=str)

    def load_agent_skill(self, name: str) -> str:
        """Load instructions and resource names for one formal Agent Skill."""
        return json.dumps(self.agent_skill_service.load_skill(name), default=str)

    def read_agent_skill_resource(self, name: str, resource_path: str) -> str:
        """Read one text resource belonging to a loaded formal Agent Skill."""
        return json.dumps(
            self.agent_skill_service.read_skill_resource(name, resource_path),
            default=str,
        )

    @staticmethod
    def calculate_offer_roi(
        annual_revenue: float, discount_pct: float, save_probability_lift: float
    ) -> str:
        """Estimate expected retained revenue and discount cost for a proposed offer."""
        discount_pct = min(max(discount_pct, 0), 0.5)
        save_probability_lift = min(max(save_probability_lift, 0), 1)
        cost = annual_revenue * discount_pct
        expected_retained_revenue = annual_revenue * save_probability_lift - cost
        return json.dumps(
            {
                "annual_revenue": annual_revenue,
                "discount_cost": round(cost, 2),
                "expected_net_retained_revenue": round(expected_retained_revenue, 2),
                "warning": "This is a scenario estimate, not a causal treatment-effect model.",
            }
        )

    def as_functions(self) -> list:
        return [
            self.get_customer_profile,
            self.predict_churn,
            self.get_customer_memories,
            self.get_customer_retention_context,
            self.list_high_risk_customers,
            self.get_portfolio_summary,
            self.get_segment_benchmark,
            self.search_retention_playbooks,
            self.list_available_interventions,
            self.list_available_specialists,
            self.list_agent_skills,
            self.load_agent_skill,
            self.read_agent_skill_resource,
            self.calculate_offer_roi,
        ]
