from app.models.domain import CustomerProfile, KnowledgeHit, PredictionResult
from app.services.knowledge_service import KnowledgeService


class KnowledgeAgent:
    def __init__(self, service: KnowledgeService, top_k: int = 3):
        self.service = service
        self.top_k = top_k

    async def run(
        self, profile: CustomerProfile, prediction: PredictionResult
    ) -> list[KnowledgeHit]:
        drivers = " ".join(driver.feature for driver in prediction.top_drivers)
        query = (
            f"customer retention {profile.segment} risk {prediction.risk_band} "
            f"renewal support usage onboarding payment {drivers}"
        )
        return self.service.search(query, top_k=self.top_k)
