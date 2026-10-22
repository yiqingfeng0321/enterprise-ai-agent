from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import analysis, chat, customers, feedback, health, operations
from app.core.config import get_settings
from app.core.exceptions import ApplicationError, CustomerNotFoundError
from app.core.logging import configure_logging

configure_logging()
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Enterprise customer-retention decision intelligence API",
)
app.include_router(health.router)
app.include_router(customers.router)
app.include_router(analysis.router)
app.include_router(chat.router)
app.include_router(feedback.router)
app.include_router(operations.router)


@app.exception_handler(CustomerNotFoundError)
async def customer_not_found_handler(
    request: Request, exc: CustomerNotFoundError
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ApplicationError)
async def application_error_handler(request: Request, exc: ApplicationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
