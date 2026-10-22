from app.core.config import get_settings
from app.repositories.customer_repository import CustomerRepository
from app.repositories.database import Database
from app.services.ml_service import MLService
from app.services.portfolio_service import PortfolioService


def test_portfolio_service_returns_ranked_filtered_customers() -> None:
    settings = get_settings()
    database = Database(settings.database_path)
    repository = CustomerRepository(database)
    ml_service = MLService(settings.model_path, settings.risk_high, settings.risk_medium)
    service = PortfolioService(repository, ml_service)

    rows = service.ranked_customers(limit=5)

    assert len(rows) == 5
    assert rows[0]["churn_probability"] >= rows[-1]["churn_probability"]
    assert all("customer_id" in row for row in rows)
    assert all(row["ranking_basis"] == "churn_probability_desc" for row in rows)
