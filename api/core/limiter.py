"""限流器 — slowapi 单例。

启用 (RAG_RATE_LIMIT_ENABLED=1) 时构建; 存储后端由 shared_state_backend 决定:
- memory (默认): 单进程有效
- redis (RAG_SHARED_STATE_BACKEND=redis + RAG_REDIS_URL): 多 worker 共享, slowapi 原生 storage_uri 接入

main.py 注册 app.state.limiter + 异常处理, qa.py 装饰端点, 两端共用同一实例。
"""

from api.config import settings

limiter = None

if settings.rate_limit_enabled:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    # Redis 共享态: 走 slowapi 原生 storage_uri; 内存则省略 (默认 MemoryStorage)
    storage_uri = settings.redis_url if settings.shared_state_backend == "redis" else None
    limiter = Limiter(key_func=get_remote_address, storage_uri=storage_uri)
