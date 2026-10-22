from app.models.domain import Recommendation
from app.repositories.analysis_repository import AnalysisRepository


class NotificationService:
    def __init__(self, repository: AnalysisRepository):
        self.repository = repository

    def create_retention_ticket(
        self, analysis_id: str, customer_id: str, recommendation: Recommendation
    ) -> int:
        title = f"Retention review required: {customer_id}"
        body = recommendation.executive_summary + "\n\n" + "\n".join(
            f"{item.priority}. {item.action}" for item in recommendation.actions
        )
        return self.repository.create_ticket(analysis_id, customer_id, title, body)
