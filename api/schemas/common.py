"""通用响应模型"""
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int

    model_config = {"from_attributes": True}
