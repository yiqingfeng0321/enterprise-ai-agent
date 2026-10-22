from app.models.domain import CustomerProfile, PredictionResult
from app.services.ml_service import MLService


class MLAgent:
    def __init__(self, service: MLService):
        self.service = service

    async def run(self, profile: CustomerProfile) -> PredictionResult:
        return self.service.predict(profile)
