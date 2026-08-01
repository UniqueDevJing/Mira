"""应用配置"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_base_url: str = "https://tokenhub.itcast.cn/v1"
    llm_model: str = "deepseek-v4-flash"
    llm_api_key: str = "sk-46be1cbfe489e032287846c7a894a3b32e46319c6e26d8a6"

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"

    # Database
    database_url: str = "postgresql+asyncpg://raguser:ragpass@localhost:5432/rag20"

    # OCR
    ocr_lang: str = "ch"

    class Config:
        env_prefix = "RAG_"


settings = Settings()
