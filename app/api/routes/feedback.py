from fastapi import APIRouter, Depends

from app.api.dependencies import container_dependency
from app.core.container import ApplicationContainer
from app.models.api import FeedbackRequest

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


@router.post("")
def submit_feedback(
    request: FeedbackRequest,
    container: ApplicationContainer = Depends(container_dependency),
) -> dict[str, int | str]:
    feedback_id = container.analysis_repository.save_feedback(request)
    customer_id = container.analysis_repository.get_customer_id_by_analysis(request.analysis_id)
    if customer_id:
        container.memory_service.record_feedback(
            customer_id=customer_id,
            rating=request.rating,
            comment=request.comment,
        )
    return {"status": "saved", "feedback_id": feedback_id}
