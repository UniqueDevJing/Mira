#!/usr/bin/env python
"""验证生产链路 sources 字段完整性(真实检索 + FakeLLM)。

背景: _build_context 曾丢弃 source_file/score, 导致前端来源面板显示空文件名与 0.000 分。
本脚本走**真实检索**(不打 _retrieve_context 桩), 只替换 LLM, 检查 SSE sources 事件
是否带齐 source_file / score —— 即修复在真实链路上是否生效。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

# 不强制离线: 真实检索需要本地缓存的 embedding 权重, 强制 OFFLINE 会让模型加载失败
os.environ["RAG_API_KEY_ENABLED"] = "false"
os.environ["RAG_CORS_ORIGINS"] = '["*"]'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import config as _cfg
from api.core import orchestrator as _orch


class FakeLLM:
    base_url = "fake://local"
    model = "fake-llm"
    api_key = "fake"

    async def stream_chat(self, messages, **kwargs):
        yield {"type": "delta", "content": "（验证用答案）"}


_orch.get_llm_client = lambda *a, **k: FakeLLM()  # type: ignore[assignment]
# 关缓存, 避免命中重放路径(重放只发 1 个 delta 且 sources 来自缓存)
_cfg.settings.qa_cache_enabled = False


async def main() -> int:
    sources = []
    async for ev in _orch.ask_stream(
        question="RAG 系统的检索流程是怎样的？",
        top_k=5,
        temperature=0.3,
        mode="hybrid",
    ):
        if ev.get("type") == "sources" and ev.get("final", True):
            sources = ev.get("sources") or []

    print("sources 条数:", len(sources))
    if sources:
        print("首条 sources:", json.dumps(sources[0], ensure_ascii=False)[:400])
    missing = [k for k in ("source_file", "score") if sources and any(k not in s for s in sources)]
    nonempty_file = bool(sources) and any(str(s.get("source_file", "")).strip() for s in sources)
    print("缺字段:", missing or "无")
    print("source_file 非空:", nonempty_file)
    ok = bool(sources) and not missing and nonempty_file
    print("RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
