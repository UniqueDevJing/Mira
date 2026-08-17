"""文档管理 API"""

import asyncio
import logging
import os
import re
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from api.core.document_store import get_document_store
from api.core.metrics import document_uploads_total
from api.schemas.documents import (
    DocumentListItem,
    DocumentListResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from api.state import get_bm25_index, get_vector_store
from engines.parsing.registry import SUPPORTED_MIME, get_parser

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def _process_document_background(doc_id: str, filename: str, content: bytes, kb: str = "documents"):
    """后台异步处理文档（不阻塞事件循环）"""
    doc_store = get_document_store()
    try:
        result = await asyncio.to_thread(_process_document_pipeline, doc_id, filename, content, kb)
        status = "empty" if result.get("chunks", 0) == 0 else "ready"
        # sqlite 同步 IO 卸载到线程池, 避免阻塞事件循环 (背景任务仍在 loop 上调度)
        await asyncio.to_thread(
            doc_store.update_status,
            doc_id,
            status,
            page_count=result.get("pages", 0),
            chunk_count=result.get("chunks", 0),
        )
    except Exception as e:
        await asyncio.to_thread(doc_store.update_status, doc_id, "failed", error=str(e)[:500])
        logger.exception("[%s] 文档处理失败", doc_id)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008 — FastAPI 依赖注入的惯用写法
    knowledge_base: str = Form("documents"),
):
    from api.config import settings

    doc_id = str(uuid.uuid4())[:12]

    # 知识库名白名单: 防空名/路径字符 (kb 用于 LanceDB 表名 rag_<kb>)
    if not knowledge_base or not re.fullmatch(r"[A-Za-z0-9_-]+", knowledge_base):
        raise HTTPException(status_code=400, detail="非法知识库名称")

    # 大小限制: 优先按 Content-Length 头预检, 再分块读入边读边限 (防绕过头全量读入内存)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content_length = getattr(file, "size", None)
    if content_length is not None and content_length > max_bytes:
        document_uploads_total.labels(status="rejected").inc()
        raise HTTPException(status_code=413, detail=f"文件超过大小限制 ({settings.max_upload_mb}MB)")
    content = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > max_bytes:
            document_uploads_total.labels(status="rejected").inc()
            raise HTTPException(status_code=413, detail=f"文件超过大小限制 ({settings.max_upload_mb}MB)")
    content = bytes(content)

    # 扩展名校验: 未知格式直接 400, 不落库
    ext = os.path.splitext(file.filename or "")[1].lower()
    if get_parser(ext) is None:
        document_uploads_total.labels(status="rejected").inc()
        raise HTTPException(status_code=400, detail=f"不支持的格式: {ext or '未知'}")
    # MIME 软校验: 与扩展名不符仅告警, 扩展名仍权威
    if file.content_type:
        expected = SUPPORTED_MIME.get(ext, set())
        if expected and file.content_type.lower() not in expected:
            logger.warning(
                "[%s] MIME 与扩展名不符: filename=%s, content_type=%s", doc_id, file.filename, file.content_type
            )

    # 保存到 SQLite (同步 IO 卸载到线程池, 不阻塞事件循环)
    doc_store = get_document_store()
    await asyncio.to_thread(doc_store.save, doc_id, file.filename, status="processing", knowledge_base=knowledge_base)

    # 异步后台处理，不阻塞请求
    background_tasks.add_task(_process_document_background, doc_id, file.filename, content, knowledge_base)

    document_uploads_total.labels(status="success").inc()
    return DocumentUploadResponse(doc_id=doc_id, status="processing", estimated_time=5)


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(doc_id: str):
    doc_store = get_document_store()
    doc = await asyncio.to_thread(doc_store.get, doc_id)
    if doc:
        return DocumentStatusResponse(
            doc_id=doc["doc_id"],
            filename=doc["filename"],
            status=doc["status"],
            page_count=doc.get("page_count"),
            chunk_count=doc.get("chunk_count"),
        )
    return DocumentStatusResponse(doc_id=doc_id, filename="", status="not_found", chunk_count=0)


@router.get("", response_model=DocumentListResponse)
async def list_documents(page: int = 1, size: int = 20):
    doc_store = get_document_store()
    result = await asyncio.to_thread(doc_store.list_all, page=page, size=size)
    return DocumentListResponse(
        items=[
            DocumentListItem(doc_id=d["doc_id"], filename=d["filename"], status=d["status"]) for d in result["items"]
        ],
        total=result["total"],
    )


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档: SQLite 记录 + 向量 + BM25 索引同步清理, 防检索残留。"""
    doc_store = get_document_store()
    doc = await asyncio.to_thread(doc_store.get, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    kb = doc.get("knowledge_base") or "documents"

    try:
        get_vector_store(kb).delete_by_doc_id(doc_id)
    except Exception as e:  # noqa: BLE001 — 删除兜底: 向量清理失败不阻断删除
        logger.warning("[%s] 删除文档 %s 向量失败: %s", kb, doc_id, str(e)[:120])
    try:
        get_bm25_index(kb).remove_doc(doc_id)
    except Exception as e:  # noqa: BLE001 — 删除兜底: BM25 清理失败不阻断删除
        logger.warning("[%s] 删除文档 %s BM25 索引失败: %s", kb, doc_id, str(e)[:120])

    ok = await asyncio.to_thread(doc_store.delete, doc_id)
    return {"detail": "已删除", "doc_id": doc_id, "deleted": ok}


def _process_document_pipeline(doc_id: str, filename: str, content: bytes, kb: str = "documents"):
    """文档处理流水线（在后台线程中运行）"""
    import os
    import tempfile

    from api.config import settings
    from engines.chunking.structure_chunker import StructureChunker
    from engines.embedding.embedder import EmbeddingService
    from engines.parsing.registry import get_parser

    tmp_path = None
    ext = os.path.splitext(filename)[1].lower()
    parser = get_parser(ext)
    if parser is None:
        raise ValueError(f"不支持的格式: {ext}")
    try:
        # 用真实后缀写临时文件, 保证解析器能按文件类型识别
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        uir = parser.parse(tmp_path)
        uir.doc_id = doc_id  # 统一用上传 uuid, 避免内容哈希跨格式冲突
        logger.info("[%s] 解析完成: %d 页", doc_id, len(uir.pages))

        if uir.tables:
            logger.warning("[%s] 解析出 %d 个表格, 表格内容当前未入库 (仅段落文本入向量库)", doc_id, len(uir.tables))

        chunker = StructureChunker(max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap)
        chunks = chunker.chunk(uir)
        logger.info("[%s] 分块完成: %d chunks", doc_id, len(chunks))

        if not chunks:
            return {"pages": len(uir.pages), "chunks": 0}

        embedder = EmbeddingService(model_name=settings.embedding_model, device=settings.embedding_device)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        try:
            from api.state import get_bm25_index, get_vector_store

            store = get_vector_store(kb)
            store.insert(chunks)
            # 同步维护该库的 BM25 稀疏索引
            bm25_docs = [
                {"id": c.chunk_id, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.content} for c in chunks
            ]
            get_bm25_index(kb).add_documents(bm25_docs)
        except Exception as e:  # noqa: BLE001 — 降级边界: 向量入库失败仅记日志
            logger.error("[%s] 向量存储失败: %s", doc_id, str(e)[:200])

        # 构建知识图谱（按库隔离）
        try:
            from api.state import get_graph_rag

            graph_rag = get_graph_rag(kb)
            graph_result = graph_rag.build_from_chunks(chunks)
            logger.info("[%s] 图谱构建: %d 实体, %d 关系", doc_id, graph_result["entities"], graph_result["relations"])
        except Exception as e:  # noqa: BLE001 — 降级边界: 图谱失败不影响入库结果
            logger.error("[%s] 图谱构建失败: %s", doc_id, str(e)[:200])

        logger.info("[%s] 处理完成 (kb=%s)", doc_id, kb)
        return {"pages": len(uir.pages), "chunks": len(chunks)}
    except Exception as e:
        logger.error("[%s] 处理失败: %s", doc_id, str(e)[:200])
        raise
    finally:
        # 确保临时文件被清理
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning("[%s] 临时文件清理失败: %s", doc_id, str(e)[:100])
