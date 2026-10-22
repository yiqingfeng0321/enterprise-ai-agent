from app.models.domain import CustomerProfile
from app.services.sql_service import SQLService


class SQLAgent:
    def __init__(self, service: SQLService):
        self.service = service

    async def get_profile(self, customer_id: str) -> CustomerProfile:
        return self.service.customer_profile(customer_id)

    async def get_benchmark(self, segment: str) -> dict:
        return self.service.segment_benchmark(segment)
