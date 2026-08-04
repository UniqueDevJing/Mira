"""文档管理相关模型"""
from enum import Enum
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    processing = "processing"
    ready = "ready"
    failed = "failed"
    not_found = "not_found"


class DocumentUploadResponse(BaseModel):
    doc_id: str
    status: DocumentStatus
    estimated_time: int = 0
    error: str | None = Field(default=None, description="失败时的错误信息，最多 200 字符")


class DocumentStatusResponse(BaseModel):
    doc_id: str
    filename: str
    status: DocumentStatus
    page_count: int | None = None
    chunk_count: int | None = None


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    status: DocumentStatus


class DocumentListResponse(BaseModel):
    items: list[DocumentListItem]
    total: int
