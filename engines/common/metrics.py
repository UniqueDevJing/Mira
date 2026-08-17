"""共享 LLM 指标 — Prometheus Counter 定义。

下沉自 api/core/metrics.py: engines 层 (同步 LLM 客户端/实体抽取) 需要 token/error 统计,
原从 engines 导入 api.core.metrics 会形成 engines → api 反向依赖。api 层 re-export 保持兼容。
"""

from prometheus_client import Counter

llm_tokens_total = Counter(
    "rag_llm_tokens_total",
    "LLM Token 消耗累计",
    ["type"],  # prompt / completion
)

llm_errors_total = Counter(
    "rag_llm_errors_total",
    "LLM 调用错误总数",
    ["error_type"],  # timeout / 5xx / rate_limit
)
