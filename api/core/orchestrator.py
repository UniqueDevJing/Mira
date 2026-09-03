"""编排层 facade。

原 1840 行单文件已按职责拆分为:
  - api.core.degradation : 降级阶段维度辅助 (_deg_*)
  - api.core.guardrails  : 忠实度护栏 / 生成前幻觉护栏
  - api.core.routing     : 路由 / 消息组装 / 候选 KB 收敛
  - api.core.retrieval   : 检索管线 / 跨库兜底 / 上下文组装 / 检索工具
  - api.core.skills      : Skill 执行 + 编排入口 (ask / ask_stream 等)

本文件仅做符号重导出, 保证 `from api.core.orchestrator import X` 与
`import api.core.orchestrator as oc; oc.X` 的既有调用方(路由层 + 测试)零改动。
所有阈值/超时读 Settings (env RAG_* 可调)。

降级等级: 0 正常 / 1 rerank 跳过 / 2 向量失败仅 BM25 / 3 LLM 失败返回摘要。
各阶段降级另由 rag_degradation_stage_total{stage} 独立计数 (一次查询可同时命中多阶段,
等级维度只记最高级, 会丢失中间阶段; 阶段维度与等级维度互补, 见 degradation 模块)。
"""

# 测试直接访问的模块级符号 (原 orchestrator 顶层 import)
from api.config import settings  # noqa: F401
from api.core.degradation import (
    _deg_bump,
    _deg_flush,
    _deg_mark,
    _deg_record_level,
    _deg_record_stage_only,
    _deg_reset,
    _deg_stages,
)
from api.core.guardrails import (
    _faithfulness_guard,
    _guard_faithfulness,
    _pregeneration_hallucination_guard,
    _pregeneration_hallucination_guard_async,
    _preguard_embed_fn,
)
from api.core.llm_client import get_llm_client  # noqa: F401
from api.core.retrieval import (
    _apply_deferred_rerank,
    _build_context,
    _build_retrieval_result,
    _cache_stats_last,
    _cross_kb_fallback,
    _dedupe_docs,
    _embed_query,
    _embed_safe,
    _graph_retrieve_safe,
    _merge_docs,
    _parallel_retrieve,
    _preprocess_query,
    _reorder_for_attention,
    _report_embed_cache,
    _rerank_safe,
    _retrieve_context,
    _retrieve_fanout,
    _retrieve_graph_only,
    _retrieve_kb,
    _retrieve_self,
)
from api.core.routing import (
    _candidate_kbs,
    _chat_messages,
    _direct_messages,
    _history_to_messages,
    _remaining,
    _route,
    _should_fanout,
)
from api.core.skills import (
    _generate,
    _record_qa_quality,
    _replay_cache_stream,
    _run_stream_llm,
    _skill_direct,
    _skill_rag,
    _stream_direct,
    _stream_rag,
    ask,
    ask_stream,
)

__all__ = [
    "_apply_deferred_rerank",
    "_build_context",
    "_build_retrieval_result",
    "_cache_stats_last",
    "_candidate_kbs",
    "_chat_messages",
    "_cross_kb_fallback",
    "_dedupe_docs",
    "_deg_bump",
    "_deg_flush",
    "_deg_mark",
    "_deg_record_level",
    "_deg_record_stage_only",
    "_deg_reset",
    "_deg_stages",
    "_direct_messages",
    "_embed_query",
    "_embed_safe",
    "_faithfulness_guard",
    "_generate",
    "_graph_retrieve_safe",
    "_guard_faithfulness",
    "_history_to_messages",
    "_merge_docs",
    "_parallel_retrieve",
    "_pregeneration_hallucination_guard",
    "_pregeneration_hallucination_guard_async",
    "_preguard_embed_fn",
    "_preprocess_query",
    "_record_qa_quality",
    "_remaining",
    "_reorder_for_attention",
    "_replay_cache_stream",
    "_report_embed_cache",
    "_rerank_safe",
    "_retrieve_context",
    "_retrieve_fanout",
    "_retrieve_graph_only",
    "_retrieve_kb",
    "_retrieve_self",
    "_route",
    "_run_stream_llm",
    "_should_fanout",
    "_skill_direct",
    "_skill_rag",
    "_stream_direct",
    "_stream_rag",
    "ask",
    "ask_stream",
]
