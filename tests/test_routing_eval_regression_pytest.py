"""路由回归门禁 — 纯规则、零外部依赖（CI 友好）。

固定 data/_eval_p1_subset.json 这 12 道「黄金路由」题, 用 IntentRouter(llm_client=None)
的纯规则路径断言 gold_kb ∈ 路由候选集。防止路由/关键词/数据归属的改动把已知正确的路由打回。

设计要点:
- llm_client=None → 完全确定性, 不调 LLM、不碰向量库/服务, 秒级跑完, 适合常跑/CI。
- 覆盖 #1 数据归属修正后的关键题: Q7/Q8(doc c3dbffff6a960cc1 已从 service 迁到 tech) 等。
- 若某人改坏 doc_types 关键词或误标数据, 此题集会立刻红, 阻断合并。
"""
import asyncio
import json
from pathlib import Path

import pytest

from engines.router.intent_router import IntentRouter

SUBSET = Path(__file__).resolve().parents[1] / "data" / "_eval_p1_subset.json"
CASES = json.loads(SUBSET.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['kb']}:{c['question'][:18]}")
def test_routing_gold_kb_in_candidates(case):
    """gold_kb 必须出现在纯规则路由的候选 KB 集合里。"""
    router = IntentRouter(llm_client=None)
    cands = asyncio.run(router.route_multi(case["question"]))
    kb_set = {c.kb for c in cands}
    assert case["kb"] in kb_set, (
        f"路由未命中 gold={case['kb']} | 候选={[ (c.skill, c.confidence) for c in cands ]} | "
        f"问题={case['question']}"
    )


def test_subset_is_nonempty_and_covers_all_types():
    """护栏: 评测子集不为空, 且覆盖 policy/service/tech 三类。"""
    assert len(CASES) >= 12
    covered = {c["kb"] for c in CASES}
    assert {"policy", "service", "tech"} <= covered
