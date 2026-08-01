"""文档管理 API"""
import uuid
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    department: str = Form(""),
    tags: str = Form(""),
    access_level: str = Form("internal"),
):
    doc_id = str(uuid.uuid4())[:12]
    content = await file.read()
    background_tasks.add_task(_process_document_pipeline, doc_id, file.filename, content)
    return {"doc_id": doc_id, "status": "processing", "estimated_time": 15}


@router.get("/{doc_id}/status")
async def get_document_status(doc_id: str):
    return {"doc_id": doc_id, "status": "completed", "chunk_count": 0}


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
    except Exception as e:
        print(f"[{doc_id}] 处理失败: {e}")
