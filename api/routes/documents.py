"""文档管理 API"""
import uuid
import asyncio
import logging
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks

from api.schemas.documents import (
    DocumentUploadResponse, DocumentStatusResponse,
    DocumentListItem, DocumentListResponse,
)
from api.core.document_store import get_document_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def _process_document_background(doc_id: str, filename: str, content: bytes):
    """后台异步处理文档（不阻塞事件循环）"""
    doc_store = get_document_store()
    try:
        result = await asyncio.to_thread(_process_document_pipeline, doc_id, filename, content)
        doc_store.update_status(
            doc_id, "ready",
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
    from engines.parsing.pdf_parser import PDFParser
    from engines.chunking.structure_chunker import StructureChunker
    from engines.embedding.embedder import EmbeddingService

    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        tmp.write(content)
        tmp.close()

        parser = PDFParser()
        uir = parser.parse(tmp_path)
        logger.info("[%s] 解析完成: %d 页", doc_id, len(uir.pages))

        chunker = StructureChunker()
        chunks = chunker.chunk(uir)
        logger.info("[%s] 分块完成: %d chunks", doc_id, len(chunks))

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
