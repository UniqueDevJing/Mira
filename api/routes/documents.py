"""文档管理 API"""
import uuid
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
_docs = {}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(""),
    tags: str = Form(""),
    access_level: str = Form("internal"),
):
    doc_id = str(uuid.uuid4())[:12]
    content = await file.read()

    _docs[doc_id] = {"doc_id": doc_id, "filename": file.filename, "status": "processing"}

    # 同步处理
    try:
        result = _process_document_pipeline(doc_id, file.filename, content)
        _docs[doc_id].update(status="ready",
                            page_count=result.get("pages", 0),
                            chunk_count=result.get("chunks", 0))
    except Exception as e:
        _docs[doc_id].update(status="failed")
        import traceback
        traceback.print_exc()
        return {"doc_id": doc_id, "status": "failed", "error": str(e)[:200]}

    return {"doc_id": doc_id, "status": "ready", "estimated_time": 0}


@router.get("/{doc_id}/status")
async def get_document_status(doc_id: str):
    if doc_id in _docs:
        d = _docs[doc_id]
        return {"doc_id": d["doc_id"], "filename": d["filename"],
                "status": d["status"],
                "page_count": d.get("page_count"),
                "chunk_count": d.get("chunk_count")}
    return {"doc_id": doc_id, "status": "not_found", "chunk_count": 0}


@router.get("")
async def list_documents(page: int = 1, size: int = 20):
    items = list(_docs.values())
    total = len(items)
    start = (page - 1) * size
    return {
        "items": [{"doc_id": d["doc_id"], "filename": d["filename"],
                   "status": d["status"]} for d in items[start:start+size]],
        "total": total
    }


def _process_document_pipeline(doc_id: str, filename: str, content: bytes):
    """文档处理流水线（在后台线程中运行）"""
    import tempfile, os
    from engines.parsing.pdf_parser import PDFParser
    from engines.chunking.semantic_chunker import SemanticChunker
    from engines.embedding.embedder import EmbeddingService

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(content)
        tmp.close()

        parser = PDFParser()
        uir = parser.parse(tmp.name)
        print(f"[{doc_id}] 解析完成: {len(uir.pages)} 页")

        chunker = SemanticChunker()
        chunks = chunker.chunk(uir)
        print(f"[{doc_id}] 分块完成: {len(chunks)} chunks")

        embedder = EmbeddingService()
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        try:
            from engines.retrieval.vector_store import VectorStore
            store = VectorStore()
            store.insert(chunks)
        except Exception as e:
            print(f"[{doc_id}] 向量存储失败: {e}")

        os.unlink(tmp.name)
        print(f"[{doc_id}] 处理完成")
        return {"pages": len(uir.pages), "chunks": len(chunks)}
    except Exception as e:
        print(f"[{doc_id}] 处理失败: {e}")
        raise
