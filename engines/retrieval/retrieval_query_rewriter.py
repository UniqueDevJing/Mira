"""单轮 LLM 查询改写 — P1-1 检索质量优化。

诊断结论: 当前重排/融合/参数调优均无法突破 context_precision≈0.318,
根本原因是约 14% 的问题向量与 BM25 都把 golden 排在第 6-10 位。
改写查询(补同义/领域词、消歧义、纠正实体指代)可让 golden 排更靠前, 是突破天花板
的唯一杠杆。

与 self-retrieval 的多轮 QueryRewriter 区分: 本类只做单轮、返回单个改写后查询(字符串),
由 config.query_rewrite_enabled 门控; 改写查询仅用于检索(向量+BM25), 原问题仍交 LLM 生成答案。

设计:
  - 无状态: 仅在 rewrite() 内懒取 LLM 客户端, 不在构造时加载模型。
  - 超时/异常安全: 任何失败都回退原查询, 绝不阻断检索主链路。
"""
from __future__ import annotations

import asyncio
import logging

from api.config import settings

logger = logging.getLogger(__name__)

_SYSTEM = (
    "你是检索查询优化器。用户会给你一个中文知识库问答问题, 请改写成一条"
    "最适合向量检索与关键词检索的查询: 补全同义/近义词与领域术语, 明确歧义实体的指代,"
    "纠正明显的口语化表述, 删除寒暄与冗余。只输出改写后的查询本身, 不要解释, 不要加引号,"
    "长度不超过原问题的 1.5 倍。"
)


class RetrievalQueryRewriter:
    def __init__(self, enabled: bool | None = None, timeout_s: float | None = None):
        self.enabled = settings.query_rewrite_enabled if enabled is None else enabled
        self.timeout_s = settings.query_rewrite_timeout_s if timeout_s is None else timeout_s

    async def rewrite(self, question: str) -> str:
        """返回改写后的检索查询; 任何失败/超时回退原查询。"""
        if not self.enabled or not question or not question.strip():
            return question
        try:
            from api.core.llm_client import get_llm_client

            client = get_llm_client()
            if not getattr(client, "api_key", None):
                return question  # 无 key 直接回退, 不浪费一次失败调用
            resp = await asyncio.wait_for(
                client.chat(
                    [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.0,
                    max_tokens=64,
                ),
                timeout=self.timeout_s,
            )
            out = (getattr(resp, "content", "") or "").strip()
            # 回退条件: 空 / 过长 / 与原文几乎相同(改写无效)
            if not out or len(out) > len(question) * 2 + 20:
                return question
            return out
        except TimeoutError:
            logger.warning("[query_rewrite] 超时(%.2fs), 回退原查询", self.timeout_s)
            return question
        except Exception as e:  # noqa: BLE001 — 改写是增强项, 失败绝不阻断检索
            logger.warning("[query_rewrite] 失败, 回退原查询: %s", str(e)[:120])
            return question
