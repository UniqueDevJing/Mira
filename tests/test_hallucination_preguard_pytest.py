"""#3 跨库 rerank + #4 幻觉前置守卫 — 回归门禁 (无 LLM/向量/服务依赖)。

- #3 (_retrieve_fanout): 各候选 KB 只出 RRF 融合候选(defer_rerank=True, **不各自 rerank**),
  跨库按 chunk_id 去重合并成单一候选池, 对整池**仅一次** _rerank_safe 全局重排
  (Cross-Encoder 在同一语义空间比较不同库片段)。mock fused 验证: 调用恰一次、
  入参为跨库合并池、返回序跟随全局重排输出。
- #4 (_pregeneration_hallucination_guard): 生成前幻觉护栏双链路 (_skill_rag / _stream_rag)
  接线一致, 通过 precomputed_retr 注入跳过真实检索, 验证拒答 reason 透传与不误拒。
  low_confidence 始终生效; low_relevance 是 feature-flag (answerability_preguard_enabled,
  默认关闭) 分支 —— 用例显式开启 flag 验证行为, 并锁定默认关闭语义(防误拒同义改写)。
"""
import asyncio

import api.core.guardrails as guardrails_mod
import api.core.retrieval as retrieval_mod
import api.core.skills as skills_mod
from api.config import settings
from api.core import orchestrator
from engines.router.intent_router import RoutingResult


def _routing(kb: str = "service") -> RoutingResult:
    return RoutingResult(skill=kb, kb=kb, confidence=1.0, source="rule")


def _precomputed(docs: list[dict], top1: float) -> dict:
    return {
        "docs": docs,
        "context": "ctx",
        "degradation": 0,
        "retrieval_ms": 1.0,
        "rerank_ms": 0.0,
        "top1_score": top1,
        "cross_kb_kbs": [],
        "retrieval_rounds": 1,
        "rewritten_queries": [],
        "graph_context": None,
    }


def _src(content: str, score: float, cid: str) -> dict:
    return {"content": content, "score": score, "doc_id": cid, "chunk_id": cid}


# 显式开启 low_relevance 分支 (生产已默认开, 此处保证用例与默认解耦)
def _enable_preguard(monkeypatch) -> None:
    monkeypatch.setattr(settings, "answerability_preguard_enabled", True)


# ── 语义兜底可控 embed: 按文本查表返回固定向量, 使测试可精确构造 cos 相似度 ──
# 组1=[1,0](退款时序话题: q 改写 + 其真实 ctx), 组2=[0,1](物流干扰话题)。
# 查表外的文本(零向量)cos=0 → 视为低相似。
_EMBED_TABLE = {
    "钱通常多久能到我的卡上呀": [1.0, 0.0],                       # 等价改写(与 ctx 零词重合, 但同话题)
    "退款审核通过后会原路退回您的支付账户，一般一到三个工作日到账": [1.0, 0.0],  # 其真实 ctx → cos=1 放行
    "如何申请退款": [1.0, 0.0],                                   # 退款话题
    "物流配送时效与运费计算说明": [0.0, 1.0],                       # 物流话题 → cos=0 触发
    "物流配送说明": [0.0, 1.0],
}


def _table_embed(text: str):
    return _EMBED_TABLE.get(text, [0.0, 0.0])


# ───────────────────────── #3 跨库 rerank ─────────────────────────
def test_fanout_global_rerank_exactly_once(monkeypatch):
    """扇出应把多库融合候选合并成整池后只跑一次全局 _rerank_safe, 而非逐库各自 rerank。"""
    captured = {"calls": [], "fused_ids": [], "defer_seen": []}
    kb_a = [_src("service 退款政策", 0.30, "a1")]
    kb_b = [_src("tech 部署架构", 0.25, "b1")]

    async def fake_retrieve(question, routing, top_k, start, *args, **kwargs):
        # #3 契约: 各库必须 defer_rerank=True —— 只出融合候选, 重排留给合并后的全局一次
        captured["defer_seen"].append(kwargs.get("defer_rerank"))
        fused = kb_a if routing.kb == "service" else kb_b
        return _precomputed([], 0.0) | {"fused": fused}

    async def fake_rerank(kb, question, fused, top_k, start, degradation):
        captured["calls"].append(kb)
        captured["fused_ids"] = [d.get("chunk_id") for d in fused]
        return fused[:top_k], 5.0, 0

    monkeypatch.setattr(skills_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_rerank_safe", fake_rerank)

    routings = [_routing("service"), _routing("tech")]
    asyncio.run(orchestrator._retrieve_fanout("q", routings, 5, 0.0))

    assert captured["defer_seen"] == [True, True]  # 各库都不各自 rerank
    assert captured["calls"] == [routings[0].kb]  # 全局重排恰一次
    assert set(captured["fused_ids"]) == {"a1", "b1"}  # 入参是跨库合并池


def test_fanout_final_order_follows_global_rerank(monkeypatch):
    """全局 rerank 的输出序即最终 docs 序 —— 证明跨库一次性重排(非各库 top 拼接)。"""
    captured = {}
    kb_a = [_src("service 片段A", 0.90, "a1"), _src("service 片段B", 0.80, "a2")]
    kb_b = [_src("tech 片段C", 0.70, "b1")]

    async def fake_retrieve(question, routing, top_k, start, *args, **kwargs):
        return _precomputed([], 0.0) | {"fused": kb_a if routing.kb == "service" else kb_b}

    async def fake_rerank(kb, question, fused, top_k, start, degradation):
        captured["in_ids"] = [d.get("chunk_id") for d in fused]
        # 模拟 CE 判定: 把 tech 片段 C 判为最相关 → 全局重排跨库把 C 提到最前
        reordered = sorted(fused, key=lambda d: d.get("chunk_id") != "b1")
        return reordered[:top_k], 5.0, 0

    monkeypatch.setattr(skills_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_rerank_safe", fake_rerank)

    res = asyncio.run(orchestrator._retrieve_fanout("q", [_routing("service"), _routing("tech")], 5, 0.0))
    # 候选池跨库轮转交织(round-robin): service a1 → tech b1 → service a2 ...
    # (非旧逻辑的库序 concat ["a1","a2","b1"]) —— 保证后序库候选不被 rerank_candidate_k 截断排除
    assert captured["in_ids"] == ["a1", "b1", "a2"]
    # top1_score 取自全局重排返回的首个片段(tech C) → 最终序跟随全局重排而非库内 RRF 序
    assert res["top1_score"] == 0.70


def test_fanout_interleave_keeps_secondary_kb_inside_budget(monkeypatch):
    """截断偏差回归锁: 主库候选占满预算时, 次库 top 候选仍须进入全局重排池(修复前会被截掉)。"""
    from api.config import settings as cfg

    monkeypatch.setattr(cfg, "rerank_candidate_k", 10)
    captured = {}
    kb_a = [_src(f"service 片段{i}", round(0.9 - i * 0.01, 3), f"a{i}") for i in range(25)]
    kb_b = [_src("tech 真答案", 0.88, "b1")]

    async def fake_retrieve(question, routing, top_k, start, *args, **kwargs):
        return _precomputed([], 0.0) | {"fused": kb_a if routing.kb == "service" else kb_b}

    async def fake_rerank(kb, question, fused, top_k, start, degradation):
        captured["ids"] = [d.get("chunk_id") for d in fused]
        return fused[:top_k], 0.0, 0

    monkeypatch.setattr(skills_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_rerank_safe", fake_rerank)

    asyncio.run(orchestrator._retrieve_fanout("q", [_routing("service"), _routing("tech")], 5, 0.0))
    assert len(captured["ids"]) <= 10  # 不突破 CE 候选预算
    assert "b1" in captured["ids"]  # 次库候选仍在池内(旧逻辑 a1..a10 会把它截掉)
    assert captured["ids"][0] == "a0" and captured["ids"][1] == "b1"  # 轮转: 次库紧随主库首名


def test_fanout_dedup_cross_kb_keeps_best_rrf(monkeypatch):
    """同 chunk_id 跨库去重, 保留 _rrf 最高的一份后再全局重排。"""
    captured = {}

    async def fake_retrieve(question, routing, top_k, start, *args, **kwargs):
        if routing.kb == "service":
            return _precomputed([], 0.0) | {"fused": [_src("service 旧副本", 0.40, "x1") | {"_rrf": 0.4}]}
        return _precomputed([], 0.0) | {"fused": [_src("tech 新副本", 0.60, "x1") | {"_rrf": 0.6}]}

    async def fake_rerank(kb, question, fused, top_k, start, degradation):
        captured["fused"] = fused
        return fused[:top_k], 0.0, 0

    monkeypatch.setattr(skills_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_rerank_safe", fake_rerank)

    asyncio.run(orchestrator._retrieve_fanout("q", [_routing("service"), _routing("tech")], 5, 0.0))
    assert len(captured["fused"]) == 1  # 去重后仅一份
    assert captured["fused"][0]["_rrf"] == 0.6  # 保留的是更高 _rrf 的那份
    assert captured["fused"][0]["content"] == "tech 新副本"


def test_fanout_empty_pool_safe(monkeypatch):
    """全库均无候选时安全返回空结果(空池仍进 _rerank_safe, 其空输入早退, 不抛异常)。"""
    captured = []

    async def fake_retrieve(question, routing, top_k, start, *args, **kwargs):
        return _precomputed([], 0.0) | {"fused": []}

    async def fake_rerank(kb, question, fused, top_k, start, degradation):
        captured.append(fused)
        return fused[:top_k], 0.0, 0

    monkeypatch.setattr(skills_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_rerank_safe", fake_rerank)

    res = asyncio.run(orchestrator._retrieve_fanout("q", [_routing("service")], 5, 0.0))
    assert captured == [[]]  # 空池透传给全局重排, 由其空输入早退保证安全
    assert res["docs"] == []
    assert res["top1_score"] == 0.0


# ───────────────────────── #4 守卫核心 (纯单测) ─────────────────────────
def test_guard_low_confidence_below_floor(monkeypatch):
    _enable_preguard(monkeypatch)
    docs = [_src("退款政策说明", 0.10, "d1")]
    assert orchestrator._pregeneration_hallucination_guard("如何申请退款", docs, 0.10) == "low_confidence"


def test_guard_low_relevance_zero_overlap(monkeypatch):
    # flag 开启: 分数达标但上下文与问题零内容词重合 → 强无关, 前置拒答
    _enable_preguard(monkeypatch)
    docs = [_src("物流配送时效与运费计算说明", 0.95, "d2")]
    assert orchestrator._pregeneration_hallucination_guard("如何申请退款", docs, 0.95) == "low_relevance"


def test_guard_low_relevance_disabled_switchable(monkeypatch):
    """flag 可整体关闭: 关闭后零重合不触发 low_relevance(交由后置护栏), 但 low_confidence 不受影响。"""
    monkeypatch.setattr(settings, "answerability_preguard_enabled", False)
    docs = [_src("物流配送时效与运费计算说明", 0.95, "d2")]
    assert orchestrator._pregeneration_hallucination_guard("如何申请退款", docs, 0.95) is None
    # low_confidence 独立于本开关, 始终生效
    docs_low = [_src("物流配送说明", 0.10, "d2b")]
    assert orchestrator._pregeneration_hallucination_guard("如何申请退款", docs_low, 0.10) == "low_confidence"


def test_settings_default_preguard_enabled():
    """锁定生产默认: 语义兜底(低误拒)就绪后 low_relevance 前置护栏默认开启。"""
    from api.config import settings as cfg

    assert cfg.answerability_preguard_enabled is True
    assert cfg.low_relevance_min_sim == 0.5


# ── #4 语义兜底: 零词重合不再等于无关 (离线评估 scripts/eval_preguard.py 驱动) ──
def test_guard_semantic_rescue_rewrite(monkeypatch):
    """零词重合 + 语义相似(等价改写) → 放行, 不误拒(评估: 改写误拒 90%→0)。"""
    _enable_preguard(monkeypatch)
    docs = [_src("退款审核通过后会原路退回您的支付账户，一般一到三个工作日到账", 0.95, "d11")]
    assert orchestrator._pregeneration_hallucination_guard(
        "钱通常多久能到我的卡上呀", docs, 0.95, embed_fn=_table_embed
    ) is None


def test_guard_semantic_low_sim_still_refuses(monkeypatch):
    """零词重合 + 语义弱(真无关) → 仍触发 low_relevance(评估: 无关高分干扰 100% 拦截)。"""
    _enable_preguard(monkeypatch)
    docs = [_src("物流配送时效与运费计算说明", 0.95, "d12")]
    assert orchestrator._pregeneration_hallucination_guard(
        "如何申请退款", docs, 0.95, embed_fn=_table_embed
    ) == "low_relevance"


def test_guard_semantic_embed_failure_fails_open(monkeypatch):
    """embed 故障 → 保守放行(零重合信号本身不可靠), 不因护栏故障误伤。"""
    _enable_preguard(monkeypatch)

    def _boom(text):
        raise RuntimeError("embed down")

    docs = [_src("物流配送时效与运费计算说明", 0.95, "d13")]
    assert orchestrator._pregeneration_hallucination_guard(
        "如何申请退款", docs, 0.95, embed_fn=_boom
    ) is None


def test_guard_semantic_none_fn_keeps_legacy(monkeypatch):
    """不注入 embed_fn 时保持原语义(零重合即触发) —— 向后兼容纯词重合路径。"""
    _enable_preguard(monkeypatch)
    docs = [_src("物流配送说明", 0.95, "d14")]
    assert orchestrator._pregeneration_hallucination_guard("如何申请退款", docs, 0.95) == "low_relevance"


def test_guard_none_when_relevant(monkeypatch):
    _enable_preguard(monkeypatch)
    docs = [_src("退款需在订单页点击申请退款，7个工作日到账", 0.92, "d3")]
    assert orchestrator._pregeneration_hallucination_guard("如何申请退款", docs, 0.92) is None


def test_guard_none_when_empty_docs(monkeypatch):
    _enable_preguard(monkeypatch)
    assert orchestrator._pregeneration_hallucination_guard("任何问题", [], 0.9) is None


def test_guard_low_relevance_yields_to_low_confidence(monkeypatch):
    # 零重合但分数低于下限 → 归为 low_confidence (不重复判 low_relevance)
    _enable_preguard(monkeypatch)
    docs = [_src("物流配送说明", 0.10, "d4")]
    assert orchestrator._pregeneration_hallucination_guard("如何申请退款", docs, 0.10) == "low_confidence"


# ───────────────────────── #4 双链路接线 ─────────────────────────
async def _fake_generate(question, context, llm, start, temperature=0.1, history=None, kb=None):
    return "已生成答案", None, 10.0, True


async def _fake_guard(answer, docs):
    return 1.0


async def _collect_stream(question, docs, top1):
    out = []
    async for ev in orchestrator._stream_rag(
        question, _routing(), object(), 5, 0.0, precomputed_retr=_precomputed(docs, top1)
    ):
        out.append(ev)
    return out


def test_skill_rag_wiring_low_confidence(monkeypatch):
    monkeypatch.setattr(skills_mod, "_generate", _fake_generate)
    docs = [_src("退款政策", 0.10, "d5")]
    res = asyncio.run(orchestrator._skill_rag(
        "如何申请退款", _routing(), object(), 5, 0.0, precomputed_retr=_precomputed(docs, 0.10)
    ))
    assert res["refusal"]["reason"] == "low_confidence"


def test_skill_rag_wiring_low_relevance(monkeypatch):
    _enable_preguard(monkeypatch)
    monkeypatch.setattr(skills_mod, "_generate", _fake_generate)
    # 语义兜底 embed 打桩: 物流 ctx 与退款 q 零词重合且低相似 → 触发 low_relevance
    monkeypatch.setattr(guardrails_mod, "_preguard_embed_fn", lambda: _table_embed)
    docs = [_src("物流配送时效与运费计算说明", 0.95, "d6")]
    res = asyncio.run(orchestrator._skill_rag(
        "如何申请退款", _routing(), object(), 5, 0.0, precomputed_retr=_precomputed(docs, 0.95)
    ))
    assert res["refusal"]["reason"] == "low_relevance"


def test_skill_rag_wiring_low_relevance_disabled_no_refusal(monkeypatch):
    """flag 关 + 零重合高分 → 护栏不拒, 正常生成且不误挂 refusal。"""
    monkeypatch.setattr(settings, "answerability_preguard_enabled", False)
    monkeypatch.setattr(skills_mod, "_generate", _fake_generate)
    monkeypatch.setattr(skills_mod, "_guard_faithfulness", _fake_guard)
    docs = [_src("物流配送时效与运费计算说明", 0.95, "d10")]
    res = asyncio.run(orchestrator._skill_rag(
        "如何申请退款", _routing(), object(), 5, 0.0, precomputed_retr=_precomputed(docs, 0.95)
    ))
    assert res.get("refusal") is None
    assert res["answer"] == "已生成答案"


def test_skill_rag_wiring_no_false_refusal(monkeypatch):
    _enable_preguard(monkeypatch)
    monkeypatch.setattr(skills_mod, "_generate", _fake_generate)
    monkeypatch.setattr(skills_mod, "_guard_faithfulness", _fake_guard)
    docs = [_src("退款需在订单页点击申请退款，7个工作日到账", 0.92, "d7")]
    res = asyncio.run(orchestrator._skill_rag(
        "如何申请退款", _routing(), object(), 5, 0.0, precomputed_retr=_precomputed(docs, 0.92)
    ))
    assert res.get("refusal") is None


def test_stream_rag_wiring_low_confidence(monkeypatch):
    docs = [_src("退款政策", 0.10, "d8")]
    events = asyncio.run(_collect_stream("如何申请退款", docs, 0.10))
    done = next(e for e in events if e.get("type") == "done")
    assert done["refusal"]["reason"] == "low_confidence"


def test_stream_rag_wiring_low_relevance(monkeypatch):
    _enable_preguard(monkeypatch)
    # 语义兜底 embed 打桩: 物流 ctx 与退款 q 零词重合且低相似 → 触发 low_relevance
    monkeypatch.setattr(guardrails_mod, "_preguard_embed_fn", lambda: _table_embed)
    docs = [_src("物流配送时效与运费计算说明", 0.95, "d9")]
    events = asyncio.run(_collect_stream("如何申请退款", docs, 0.95))
    done = next(e for e in events if e.get("type") == "done")
    assert done["refusal"]["reason"] == "low_relevance"
