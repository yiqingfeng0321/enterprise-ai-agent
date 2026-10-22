from fastapi import APIRouter, Depends, Query

from app.api.dependencies import container_dependency
from app.core.container import ApplicationContainer
from app.models.domain import CustomerProfile

router = APIRouter(prefix="/v1/customers", tags=["customers"])


@router.get("/high-risk")
def high_risk_customers(
    limit: int = Query(default=10, ge=1, le=100),
    container: ApplicationContainer = Depends(container_dependency),
) -> list[dict]:
    profiles = [
        container.customer_repository.get(customer_id)
        for customer_id in container.customer_repository.list_ids(limit=10_000)
    ]
    profile_by_id = {profile.customer_id: profile for profile in profiles}
    results = []
    for customer_id, probability, risk_band in container.ml_service.score_many(profiles):
        profile = profile_by_id[customer_id]
        results.append(
            {
                "customer_id": customer_id,
                "segment": profile.segment,
                "monthly_revenue": profile.monthly_revenue,
                "churn_probability": probability,
                "risk_band": risk_band,
            }
        )
    return sorted(results, key=lambda item: item["churn_probability"], reverse=True)[:limit]


@router.get("/{customer_id}", response_model=CustomerProfile)
def get_customer(
    customer_id: str,
    container: ApplicationContainer = Depends(container_dependency),
) -> CustomerProfile:
    return container.customer_repository.get(customer_id)
