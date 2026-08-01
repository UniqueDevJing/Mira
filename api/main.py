"""RAG 2.0 API 入口"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import documents, qa


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.config import settings
    print(f"启动 RAG 2.0 服务 — 模型: {settings.llm_model}")
    yield
    print("服务关闭")


app = FastAPI(
    title="RAG 2.0 API",
    description="智能文档处理与知识库构建系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}
