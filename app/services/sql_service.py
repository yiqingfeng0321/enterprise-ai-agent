from app.models.domain import CustomerProfile
from app.repositories.customer_repository import CustomerRepository


class SQLService:
    """Read-only, allow-listed analytical queries used by the SQL agent."""

    def __init__(self, repository: CustomerRepository):
        self.repository = repository

    def customer_profile(self, customer_id: str) -> CustomerProfile:
        return self.repository.get(customer_id)

    def segment_benchmark(self, segment: str) -> dict[str, float | int | str]:
        benchmark = self.repository.segment_benchmark(segment)
        return {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in benchmark.items()
        }
