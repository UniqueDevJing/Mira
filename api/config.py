"""应用配置"""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_base_url: str = "https://tokenhub.itcast.cn/v1"
    llm_model: str = "deepseek-v4-flash"
    llm_api_key: str = ""  # 通过环境变量 RAG_LLM_API_KEY 或 .env 文件注入

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""  # 必须通过 RAG_NEO4J_PASSWORD 环境变量设置

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""  # 必须通过环境变量设置
    minio_secret_key: str = ""  # 必须通过环境变量设置

    # Database
    database_url: str = "postgresql+asyncpg://raguser:ragpass@localhost:5432/rag20"

    # OCR
    ocr_lang: str = "ch"

    # 分块
    chunk_max_chars: int = 800
    chunk_overlap: int = 128

    # CORS — 开发环境默认 *，生产环境通过 RAG_CORS_ORIGINS 配置（逗号分隔）
    cors_origins: List[str] = ["*"]

    class Config:
        env_prefix = "RAG_"


settings = Settings()
