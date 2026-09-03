#!/usr/bin/env python
"""前端联调集成测试 · 后端桩服务。

真实启动 uvicorn (127.0.0.1:8911)，注入 FakeLLM 绕过外部 LLM 额度(与 bench_stream_sources 同手法:
打 orchestrator.get_llm_client 模块名)。返回一段**富 markdown 答案(含故意 XSS 载荷)**，用于联调验证
前端 SSE 消费 + 增量渲染 + markdown 渲染 + XSS 转义。

测试专用覆盖:
  - 离线环境(HF_HUB_OFFLINE)避免权重下载
  - 免鉴权(RAG_API_KEY_ENABLED=false) + CORS 全开, 前端可匿名直连
  - answer_confidence_floor=0 / fidelity_threshold=0: 避免"置信度拒答/忠实度拒答"干扰正常路径验证
该桩仅用于本地联调, 不触碰生产 .env。
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

# ── 离线 / 免鉴权 / 防限流 ──
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RAG_RATE_LIMIT_ENABLED", "0")
os.environ["RAG_API_KEY_ENABLED"] = "false"
os.environ["RAG_CORS_ORIGINS"] = '["*"]'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

from api import config as _cfg
from api.core import orchestrator as _orch

FAKE_ANSWER = (
    "# RAG 2.0 架构概览\n\n"
    "**检索增强生成（RAG）** 通过混合检索提升召回质量。核心流程包括：\n\n"
    "- 向量检索（dense embedding）\n"
    "- BM25 关键词检索\n"
    "- RRF 分数融合排序\n\n"
    "示例代码片段：\n\n"
    "```python\nretriever.fuse(bm25, dense, method=\"rrf\")\n```\n\n"
    "安全说明：<script>alert('xss')</script> 与 <img src=x onerror=alert(1)> "
    "这类载荷不应被执行；合法外链 [官方文档](https://example.com/docs) 应正常渲染。\n"
)


class FakeLLM:
    base_url = "fake://local"
    model = "fake-llm"
    api_key = "fake"

    async def chat(self, messages, **kwargs):
        return types.SimpleNamespace(content=FAKE_ANSWER, latency_ms=1.0)

    async def stream_chat(self, messages, **kwargs):
        # 分块以触发前端的"增量 appendData"路径
        for chunk in [
            "# RAG 2.0 架构概览\n\n",
            "**检索增强生成（RAG）** 通过混合检索提升召回质量。核心流程包括：\n\n",
            "- 向量检索（dense embedding）\n",
            "- BM25 关键词检索\n",
            "- RRF 分数融合排序\n\n",
            "示例代码片段：\n\n",
            "```python\nretriever.fuse(bm25, dense, method=\"rrf\")\n```\n\n",
            "安全说明：<script>alert('xss')</script> 与 <img src=x onerror=alert(1)> "
            "这类载荷不应被执行；合法外链 [官方文档](https://example.com/docs) 应正常渲染。\n",
        ]:
            # 块间留 15ms, 让前端的"流式卡片 + 增量 appendData"窗口可被稳定观测
            await asyncio.sleep(0.015)
            yield {"type": "delta", "content": chunk}


# 必须打 orchestrator 模块名(get_llm_client 被直接绑定到名字)
_orch.get_llm_client = lambda *a, **k: FakeLLM()  # type: ignore[assignment]

# 测试专用: 桩掉真实检索(embedding/BM25/lancedb 在 CPU 上偶发慢且不确定, 会让联调 flaky),
# 改为秒回固定 docs。注意 docs 直接按"前端 sources 消费的字段"构造(source_file/score/content),
# 不经 _build_context —— 后者会丢弃 source_file/score(见报告中的生产链路缺陷)。
async def _fake_retrieve_context(
    question, routing, top_k, start,
    enable_self_retrieval: bool = False,
    mode: str = "hybrid",
    candidate_kbs=None,
    defer_rerank: bool = False,
):
    docs = [{
        "content": "测试来源：RAG 系统通过向量（dense）与 BM25 混合检索，经 RRF 融合后送交大模型生成答案。",
        "source_file": "rag_architecture.md",
        "score": 0.92,
        "kb": "rag_tech",
    }]
    return {
        "docs": docs,
        "context": docs[0]["content"],
        "degradation": 0,
        "retrieval_ms": 1.0,
        "rerank_ms": 0.0,
        "top1_score": 0.92,
        "cross_kb_kbs": [],
        "retrieval_rounds": 1,
        "rewritten_queries": [],
        "graph_context": None,
    }


_orch._retrieve_context = _fake_retrieve_context  # type: ignore[assignment]

# 测试专用: 关闭答案缓存。命中缓存时走 _replay_cache_stream, 整个答案只发 1 个 delta
# (orchestrator.py:883), 前端的"多次增量 appendData"流式路径就覆盖不到。
_cfg.settings.qa_cache_enabled = False

# 测试专用: 关闭拒答护栏, 保证走"正常答案 + sources"路径
_cfg.settings.answer_confidence_floor = 0.0
_cfg.settings.fidelity_threshold = 0.0
_cfg.settings.fidelity_use_embedding = False
_cfg.settings.fidelity_check_numbers = False

from api.main import app


def main() -> None:
    port = int(os.environ.get("IT_PORT", "8911"))
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    print(f"IT_BACKEND_READY port={port}", flush=True)
    server.run()


if __name__ == "__main__":
    main()
