"""文档管理 API"""
import uuid
import asyncio
import logging
import os
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException

from api.schemas.documents import (
    DocumentUploadResponse, DocumentStatusResponse,
    DocumentListItem, DocumentListResponse,
)
from api.core.document_store import get_document_store
from engines.parsing.registry import get_parser, SUPPORTED_MIME

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def _process_document_background(doc_id: str, filename: str, content: bytes):
    """后台异步处理文档（不阻塞事件循环）"""
    doc_store = get_document_store()
    try:
        result = await asyncio.to_thread(_process_document_pipeline, doc_id, filename, content)
        status = "empty" if result.get("chunks", 0) == 0 else "ready"
        doc_store.update_status(
            doc_id, status,
            page_count=result.get("pages", 0),
            chunk_count=result.get("chunks", 0),
        )
    except Exception as e:
        doc_store.update_status(doc_id, "failed", error=str(e)[:500])
        logger.error("[%s] 文档处理失败: %s", doc_id, str(e)[:500], exc_info=True)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    department: str = Form(""),
    tags: str = Form(""),
    access_level: str = Form("internal"),
):
    doc_id = str(uuid.uuid4())[:12]
    content = await file.read()

    # 扩展名校验: 未知格式直接 400, 不落库
    ext = os.path.splitext(file.filename or "")[1].lower()
    if get_parser(ext) is None:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {ext or '未知'}")
    # MIME 软校验: 与扩展名不符仅告警, 扩展名仍权威
    if file.content_type:
        expected = SUPPORTED_MIME.get(ext, set())
        if expected and file.content_type.lower() not in expected:
            logger.warning("[%s] MIME 与扩展名不符: filename=%s, content_type=%s",
                           doc_id, file.filename, file.content_type)

    # 保存到 SQLite
    doc_store = get_document_store()
    doc_store.save(doc_id, file.filename, status="processing")

    # 异步后台处理，不阻塞请求
    background_tasks.add_task(_process_document_background, doc_id, file.filename, content)

    return DocumentUploadResponse(doc_id=doc_id, status="processing", estimated_time=5)


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(doc_id: str):
    doc_store = get_document_store()
    doc = doc_store.get(doc_id)
    if doc:
        return DocumentStatusResponse(
            doc_id=doc["doc_id"], filename=doc["filename"],
            status=doc["status"],
            page_count=doc.get("page_count"),
            chunk_count=doc.get("chunk_count"),
        )
    return DocumentStatusResponse(doc_id=doc_id, filename="", status="not_found", chunk_count=0)


@router.get("", response_model=DocumentListResponse)
async def list_documents(page: int = 1, size: int = 20):
    doc_store = get_document_store()
    result = doc_store.list_all(page=page, size=size)
    return DocumentListResponse(
        items=[DocumentListItem(
            doc_id=d["doc_id"], filename=d["filename"], status=d["status"]
        ) for d in result["items"]],
        total=result["total"],
    )


def _process_document_pipeline(doc_id: str, filename: str, content: bytes):
    """文档处理流水线（在后台线程中运行）"""
    import tempfile, os
    from engines.parsing.registry import get_parser
    from engines.chunking.structure_chunker import StructureChunker
    from engines.embedding.embedder import EmbeddingService
    from api.config import settings

    tmp_path = None
    ext = os.path.splitext(filename)[1].lower()
    parser = get_parser(ext)
    if parser is None:
        raise ValueError(f"不支持的格式: {ext}")
    try:
        # 用真实后缀写临时文件, 保证解析器能按文件类型识别
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_path = tmp.name
        tmp.write(content)
        tmp.close()

        uir = parser.parse(tmp_path)
        logger.info("[%s] 解析完成: %d 页", doc_id, len(uir.pages))

        chunker = StructureChunker(max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap)
        chunks = chunker.chunk(uir)
        logger.info("[%s] 分块完成: %d chunks", doc_id, len(chunks))

        if not chunks:
            return {"pages": len(uir.pages), "chunks": 0}

        embedder = EmbeddingService()
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        try:
            from api.state import get_vector_store
            store = get_vector_store()
            store.insert(chunks)
        except Exception as e:
            logger.error("[%s] 向量存储失败: %s", doc_id, str(e)[:200])

        # 构建知识图谱
        try:
            from api.state import get_graph_rag
            graph_rag = get_graph_rag()
            graph_result = graph_rag.build_from_chunks(chunks)
            logger.info(
                "[%s] 图谱构建: %d 实体, %d 关系",
                doc_id, graph_result['entities'], graph_result['relations']
            )
        except Exception as e:
            logger.error("[%s] 图谱构建失败: %s", doc_id, str(e)[:200])

        logger.info("[%s] 处理完成", doc_id)
        return {"pages": len(uir.pages), "chunks": len(chunks)}
    except Exception as e:
        logger.error("[%s] 处理失败: %s", doc_id, str(e)[:200])
        raise
    finally:
        # 确保临时文件被清理
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning("[%s] 临时文件清理失败: %s", doc_id, str(e)[:100])
