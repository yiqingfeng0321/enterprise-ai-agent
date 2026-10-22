from fastapi import APIRouter, Depends

from app.api.dependencies import container_dependency
from app.core.container import ApplicationContainer
from app.models.api import AnalysisApiResponse, AnalysisRequest

router = APIRouter(prefix="/v1/analysis", tags=["analysis"])


@router.post("", response_model=AnalysisApiResponse)
async def analyze_customer(
    request: AnalysisRequest,
    container: ApplicationContainer = Depends(container_dependency),
) -> AnalysisApiResponse:
    result = await container.workflow.analyze(
        request.customer_id,
        session_id=request.session_id,
        create_ticket=request.create_ticket,
    )
    return AnalysisApiResponse(data=result)
