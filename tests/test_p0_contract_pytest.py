"""P0' 契约测试 (OPT-C2) — 锁死本轮优化引入/修复的接口契约, 防回归。

覆盖五个曾出过真实 bug 的点:
  1. QARequest.skill 必须放行路由表全部技能 (否则评测/前端无法指定 policy 等库);
  2. SourceDocument.chunk_ids 必须存在且默认空列表 (评测 recall 依赖);
  3. public_sources 必须剔除 context_full (内部字段不外泄), 且不得改动原 dict
     (原列表在流式链路后续仍被护栏使用);
  4. build_sources_event 走 public_sources (SSE 出口与 HTTP 出口行为一致);
  5. 显式指定技能 (routing.source="manual") 时禁用跨库兜底 (OPT-X1),
     自动路由路径不受影响。

测试策略: 纯单元级, 不依赖真实 lancedb/LLM (检索边界全部 monkeypatch)。
"""

import asyncio

import pytest
from pydantic import ValidationError

import api.core.retrieval as retrieval_mod
import api.core.routing as routing_mod
import api.core.skills as skills_mod
from api.core import orchestrator
from api.core.response import build_sources_event, public_sources
from api.schemas.qa import QARequest, SourceDocument, to_source_document
from engines.router.intent_router import RoutingResult
from engines.router.routing_rules import SKILLS

# ── 1. skill 校验: 路由表全放行, 未知值仍拒绝 ──────────────────────────


def test_skill_pattern_accepts_all_router_skills():
    for skill in SKILLS:
        req = QARequest(question="x", skill=skill)
        assert req.skill == skill


def test_skill_pattern_still_rejects_unknown():
    with pytest.raises(ValidationError):
        QARequest(question="x", skill="hacker")


# ── 2. SourceDocument.chunk_ids 契约 ──────────────────────────────────


def test_sourcedocument_chunk_ids_default_empty_list():
    d = SourceDocument()
    assert d.chunk_ids == []


def test_sourcedocument_chunk_ids_roundtrip():
    d = SourceDocument(doc_id="doc1", chunk_ids=["doc1_chunk_0000", "doc1_chunk_0001"])
    assert d.chunk_ids == ["doc1_chunk_0000", "doc1_chunk_0001"]


# ── 3/4. public_sources / build_sources_event 契约 ────────────────────


def _doc_with_internal_fields():
    return {
        "doc_id": "d1",
        "chunk_ids": ["d1_chunk_0000"],
        "content": "200字片段",
        "context_full": "完整800字文本" * 100,
        "source_file": "a.md",
        "score": 0.9,
    }


def test_public_sources_strips_context_full():
    docs = [_doc_with_internal_fields()]
    out = public_sources(docs)
    assert len(out) == 1
    assert "context_full" not in out[0]
    assert out[0]["chunk_ids"] == ["d1_chunk_0000"]  # 其余字段原样保留


def test_public_sources_does_not_mutate_original():
    """原 dict 不能被改动: 流式链路发完 sources 后还要用 context_full 算护栏。"""
    docs = [_doc_with_internal_fields()]
    public_sources(docs)
    assert "context_full" in docs[0]


def test_build_sources_event_strips_internal_fields():
    ev = build_sources_event([_doc_with_internal_fields()], 0.9, [], 0)
    assert ev["type"] == "sources"
    assert all("context_full" not in s for s in ev["sources"])
    assert ev["retrieval_meta"]["result_count"] == 1


# ── 5. OPT-X1: manual 路由禁用跨库兜底 ────────────────────────────────


def _patch_retrieval_to_empty(monkeypatch):
    """主库检索返回空 → 触发跨库兜底条件 (not fused)。"""
    async def _fake_embed_safe(embedder, vq, kb, start):
        return None

    async def _fake_retrieve_safe(kb, q_emb, bq, top_k, start, mode="hybrid"):
        return [], [], 0

    async def _fake_graph(kb, question, top_k, start):
        return None

    def _fake_fuse(vector_docs, bm25_docs, method=None, w_bm25=0.5):
        return []

    fallback_calls = {"n": 0, "kbs": None}

    async def _fake_cross_kb_fallback(fused, vq, bq, kb, top_k, start, candidate_kbs):
        fallback_calls["n"] += 1
        return [], ["service"]

    async def _fake_rerank_safe(kb, question, fused, top_k, start, degradation):
        return list(fused), 0.0, degradation

    monkeypatch.setattr(retrieval_mod, "_embed_safe", _fake_embed_safe)
    monkeypatch.setattr(retrieval_mod, "_retrieve_safe", _fake_retrieve_safe)
    monkeypatch.setattr(retrieval_mod, "_graph_retrieve_safe", _fake_graph)
    monkeypatch.setattr(retrieval_mod, "fuse", _fake_fuse)
    monkeypatch.setattr(retrieval_mod, "_cross_kb_fallback", _fake_cross_kb_fallback)
    monkeypatch.setattr(retrieval_mod, "_rerank_safe", _fake_rerank_safe)
    monkeypatch.setattr(orchestrator.settings, "query_rewrite_enabled", False)
    return fallback_calls


def test_manual_skill_skips_cross_kb_fallback(monkeypatch):
    """显式指定技能时: 主库无结果 → 如实空结果, 绝不静默混入其他库。"""
    calls = _patch_retrieval_to_empty(monkeypatch)
    routing = RoutingResult("tech", "tech", 1.0, "manual")
    retr = asyncio.run(
        orchestrator._retrieve_context("q", routing, 5, 0.0, False, mode="hybrid", candidate_kbs=None)
    )
    assert calls["n"] == 0
    assert retr["docs"] == []
    assert retr["cross_kb_kbs"] == []


def test_auto_route_still_uses_cross_kb_fallback(monkeypatch):
    """自动路由路径行为不变: 主库空仍走跨库兜底 (保持既有救援能力)。"""
    calls = _patch_retrieval_to_empty(monkeypatch)
    routing = RoutingResult("tech", "tech", 0.9, "llm")
    retr = asyncio.run(
        orchestrator._retrieve_context("q", routing, 5, 0.0, False, mode="hybrid", candidate_kbs=None)
    )
    assert calls["n"] == 1
    assert retr["cross_kb_kbs"] == ["service"]


def test_route_returns_manual_source_for_explicit_skill():
    """_route 契约: 显式 skill → source=manual, 置信度 1.0。"""
    from api.config import settings  # noqa: F401 — 确保配置加载

    class _NoLLM:
        pass

    routing, _cands, _ms = asyncio.run(orchestrator._route("q", "policy", _NoLLM(), 0.0, None))
    assert routing.source == "manual"
    assert routing.skill == "policy"
    assert routing.kb == "policy"
    assert routing.confidence == 1.0
    assert len(_cands) == 1 and _cands[0] is routing  # P1': 返回候选列表, 主候选为 candidates[0]


# ── 6. OPT-C1: sources 序列化统一 (to_source_document) ──────────────────


def test_to_source_document_maps_all_fields():
    """内部 source dict → SourceDocument 字段映射完整且正确。"""
    d = {"id": "c1", "chunk_id": "c1", "doc_id": "d1", "content": "片段",
         "score": 0.8, "chunk_ids": ["c1", "c2"]}
    sd = to_source_document(d)
    assert sd.id == "c1"
    assert sd.chunk_id == "c1"
    assert sd.doc_id == "d1"
    assert sd.content == "片段"
    assert sd.score == 0.8
    assert sd.chunk_ids == ["c1", "c2"]


def test_to_source_document_score_none_safe():
    """score 为 None / 缺失时不得抛错, 归一为 0.0 (防 None/字符串污染 HTTP 响应)。"""
    assert to_source_document({"score": None}).score == 0.0
    assert to_source_document({}).score == 0.0


def test_build_context_preserves_id_and_chunk_id():
    """C1 核心修复: _build_context 分组后不得丢弃 id/chunk_id,
    否则经 to_source_document 映射后 HTTP 响应的 SourceDocument.id/chunk_id 恒为空。"""
    docs = [{
        "doc_id": "d1",
        "id": "d1_chunk_0000",
        "chunk_id": "d1_chunk_0000",
        "title_chain": ["第一章"],
        "doc_title": "退款政策",
        "source_file": "policy.md",
        "score": 0.9,
        "parent_content": "父块完整内容",
        "content": "子块片段",
    }]
    _ctx, sources = orchestrator._build_context(docs, 5)
    assert len(sources) == 1
    assert sources[0]["id"] == "d1_chunk_0000"
    assert sources[0]["chunk_id"] == "d1_chunk_0000"
    assert sources[0]["doc_id"] == "d1"
    # 经映射后 HTTP 响应字段确实非空
    sd = to_source_document(sources[0])
    assert sd.id == "d1_chunk_0000"
    assert sd.chunk_id == "d1_chunk_0000"


# ── 7. OPT-U1: 拒答分级 (RefusalInfo) ──────────────────────────────────


def _refusal_docs():
    """模拟 _build_context 产出的分组来源(含 title_chain/source_file/score/chunk_ids)。"""
    return [
        {
            "doc_id": "d1", "title_chain": ["退款", "时效"], "doc_title": "退款政策",
            "source_file": "policy.md", "score": 0.55, "chunk_ids": ["d1_chunk_0000", "d1_chunk_0001"],
        },
        {
            "doc_id": "d2", "title_chain": [], "doc_title": "", "source_file": "faq.docx",
            "score": 0.42, "chunk_ids": ["d2_chunk_0000"],
        },
    ]


def test_build_refusal_info_low_confidence_carries_candidates_and_kb():
    """低置信度拒答: 附候选来源, 候选必须带 doc_id + kb(U2 展开全文的查询键)。"""
    from api.core.response import build_refusal_info

    info = build_refusal_info("low_confidence", _refusal_docs(), "怎么退款", kb="policy")
    assert info["is_refusal"] is True
    assert info["reason"] == "low_confidence"
    assert len(info["candidates"]) == 2
    # 候选按 doc_id 去重 + 含 U2 所需键
    c0 = info["candidates"][0]
    assert c0["doc_id"] == "d1"
    assert c0["kb"] == "policy"
    assert c0["title"] == "退款 > 时效"
    assert c0["score"] == 0.55
    assert c0["chunk_ids"] == ["d1_chunk_0000", "d1_chunk_0001"]
    # 引导追问确定性生成
    assert any("展开来源全文" in q for q in info["suggested_questions"])


def test_build_refusal_info_title_fallback_to_source_file():
    """候选标题缺失 title_chain/doc_title 时, 回退 source_file, 不得为空字符串候选。"""
    from api.core.response import build_refusal_info

    info = build_refusal_info("low_fidelity", _refusal_docs(), "faq", kb="policy")
    d2 = next(c for c in info["candidates"] if c["doc_id"] == "d2")
    assert d2["title"] == "faq.docx"  # 回退文件名


def test_build_refusal_info_no_docs_empty_candidates():
    """无检索结果(no_docs): 候选为空(真拒答语义), 仍产出 is_refusal 便于前端判断。"""
    from api.core.response import build_refusal_info

    info = build_refusal_info("no_docs", [], "随便问", kb="service")
    assert info["is_refusal"] is True
    assert info["reason"] == "no_docs"
    assert info["candidates"] == []


def test_build_refusal_info_defensive_on_bad_docs():
    """防御性: docs 为 None / 字段缺失不得抛异常。"""
    from api.core.response import build_refusal_info

    info = build_refusal_info("low_confidence", None, "q", kb="policy")
    assert info["candidates"] == []
    info2 = build_refusal_info("low_confidence", [{"doc_id": "x"}], "q", kb="policy")
    assert info2["candidates"][0]["doc_id"] == "x"


def test_qa_response_refusal_field_optional():
    """QAResponse.refusal 默认 None(正常回答), 赋值时校验结构。"""
    from api.schemas.qa import QAResponse, RefusalCandidate, RefusalInfo

    ok = QAResponse(answer="正常回答")
    assert ok.refusal is None

    refused = QAResponse(
        answer="未找到",
        refusal=RefusalInfo(
            reason="low_confidence",
            candidates=[RefusalCandidate(doc_id="d1", kb="policy", title="退款", score=0.5)],
        ),
    )
    assert refused.refusal.is_refusal is True
    assert refused.refusal.candidates[0].doc_id == "d1"


# ── 8. OPT-U2: 来源展开全文端点 (GET /api/v1/qa/sources/{doc_id}) ──────────


def _fake_principal(allowed_kbs):
    from api.core.auth import Principal

    return Principal(key_id="k", name="tester", role="admin" if allowed_kbs is None else "reader", allowed_kbs=allowed_kbs)


def _patch_source_endpoint(monkeypatch, *, chunks, raise_exc=None):
    """monkeypatch 端点的 get_principal + get_vector_store, 隔离真实 lancedb/LLM。"""
    from api.routes import qa as qa_route

    monkeypatch.setattr(qa_route, "get_principal", lambda req: _fake_principal(None))

    class _FakeStore:
        def get_by_doc_id(self, doc_id):
            if raise_exc:
                raise raise_exc
            return chunks

    monkeypatch.setattr(qa_route, "get_vector_store", lambda kb: _FakeStore())


def test_source_detail_requires_kb(monkeypatch):
    """kb 缺失 → 400 (来源所属库必须显式传入, 用于 RBAC 与定位向量表)。"""
    from fastapi import HTTPException

    from api.routes import qa as qa_route

    monkeypatch.setattr(qa_route, "get_principal", lambda req: _fake_principal(None))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(qa_route.get_source_detail("d1", kb=None, request=None))
    assert exc.value.status_code == 400


def test_source_detail_rbac_forbidden(monkeypatch):
    """reader 无权访问 service → 403 (RBAC 在查询前拦截, 不泄露跨库存在性)。"""
    from fastapi import HTTPException

    from api.routes import qa as qa_route

    monkeypatch.setattr(qa_route, "get_principal", lambda req: _fake_principal(["policy"]))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(qa_route.get_source_detail("d1", kb="service", request=None))
    assert exc.value.status_code == 403


def test_source_detail_returns_chunks(monkeypatch):
    """正常: 返回该 doc 全部 chunk 完整内容(含 parent 全文), 响应体结构正确。"""
    from api.routes import qa as qa_route

    chunks = [
        {"id": "c1", "chunk_id": "c1", "doc_id": "d1", "content": "父块完整代码", "doc_title": "退款", "parent_id": ""},
        {"id": "c2", "chunk_id": "c2", "doc_id": "d1", "content": "子块切片", "doc_title": "退款", "parent_id": "c1"},
    ]
    _patch_source_endpoint(monkeypatch, chunks=chunks)
    result = asyncio.run(qa_route.get_source_detail("d1", kb="policy", request=None))
    assert result["doc_id"] == "d1"
    assert result["kb"] == "policy"
    assert result["chunk_count"] == 2
    assert result["chunks"][0]["content"] == "父块完整代码"


def test_source_detail_404_when_empty(monkeypatch):
    """无内容 → 404 (不泄露是否存在)。"""
    from fastapi import HTTPException

    from api.routes import qa as qa_route

    _patch_source_endpoint(monkeypatch, chunks=[])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(qa_route.get_source_detail("d1", kb="policy", request=None))
    assert exc.value.status_code == 404


def test_source_detail_404_on_query_error(monkeypatch):
    """查询异常 → 统一降级 404 (不暴露内部错误堆栈)。"""
    from fastapi import HTTPException

    from api.routes import qa as qa_route

    _patch_source_endpoint(monkeypatch, chunks=[], raise_exc=RuntimeError("lancedb down"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(qa_route.get_source_detail("d1", kb="policy", request=None))
    assert exc.value.status_code == 404


# ── P1' 选择性扇出 + early-exit 契约 ────────────────────────────────────


def test_route_returns_three_tuple():
    """P1' 后 _route 必须返回 (main, candidates, ms) 三元组, 且 candidates[0] == main。"""
    from api.config import settings  # noqa: F401 — 确保配置加载

    class _NoLLM:
        pass

    routing, candidates, ms = asyncio.run(
        orchestrator._route("q", "policy", _NoLLM(), 0.0, None)
    )
    assert isinstance(routing, RoutingResult)
    assert isinstance(candidates, list) and candidates
    assert candidates[0] is routing
    assert isinstance(ms, float)


def test_fanout_merges_and_dedups(monkeypatch):
    """_retrieve_fanout 跨多 KB 检索, 按 chunk_id 去重, 合并取 top_k, 结构对齐 _retrieve_context。"""
    # 契约只锁扇出合并/去重/结构; rerank 打桩为 identity —— 测试环境 .env 开着真实
    # bge-reranker, 若跑真实 _rerank_safe, CE 会把语义垃圾片段 score 改写/乱序 → 时序性假失败。
    async def _fake_rerank_identity(kb, question, fused, top_k, start, degradation):
        return fused[:top_k], 0.0, degradation

    async def _fake_retrieve(question, routing, top_k, start, *a, **k):
        # 每个 KB 返回该库命中片段 (chunk_id 以 kb 区分)
        cid = f"{routing.kb}_c0"
        doc = {
            "doc_id": f"d_{routing.kb}",
            "id": cid,
            "chunk_id": cid,
            "title_chain": [routing.kb],
            "doc_title": routing.kb,
            "source_file": f"{routing.kb}.md",
            "content": f"{routing.kb} 内容",
            "score": 0.9 if routing.kb == "policy" else 0.7,
        }
        return {
            "docs": [doc],
            "fused": [doc],
            "context": "ctx",
            "degradation": 0,
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "top1_score": 0.9,
            "cross_kb_kbs": [],
            "retrieval_rounds": 1,
            "rewritten_queries": [],
            "graph_context": None,
        }

    monkeypatch.setattr(skills_mod, "_retrieve_context", _fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", _fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_rerank_safe", _fake_rerank_identity)

    routings = [
        RoutingResult("policy", "policy", 0.9, "rule"),
        RoutingResult("service", "service", 0.6, "rule"),
    ]
    out = asyncio.run(orchestrator._retrieve_fanout("q", routings, 10, 0.0))
    # 两个库各 1 条, 无重复 → docs 含 2 条
    assert len(out["docs"]) == 2
    assert out["top1_score"] == 0.9  # 降序: policy 在前
    # 结构键对齐 _retrieve_context
    for key in ("docs", "context", "top1_score", "cross_kb_kbs", "retrieval_rounds"):
        assert key in out


def test_fanout_dedups_same_chunk_id(monkeypatch):
    """同 chunk_id 在多个 KB 命中时只保留最高分一条 (不重复计数)。"""

    async def _fake_retrieve(question, routing, top_k, start, *a, **k):
        doc = {
            "doc_id": "d1", "id": "c1", "chunk_id": "c1",
            "title_chain": [], "doc_title": "x", "source_file": "x.md",
            "content": "c", "score": 0.8 if routing.kb == "policy" else 0.5,
        }
        return {
            "docs": [doc], "fused": [doc],
            "context": "ctx", "degradation": 0, "retrieval_ms": 0.0, "rerank_ms": 0.0,
            "top1_score": 0.8, "cross_kb_kbs": [], "retrieval_rounds": 1,
            "rewritten_queries": [], "graph_context": None,
        }

    monkeypatch.setattr(skills_mod, "_retrieve_context", _fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", _fake_retrieve)
    # 同上: rerank 打桩 identity, 锁去重契约而非真实 CE 行为
    async def _fake_rerank_identity(kb, question, fused, top_k, start, degradation):
        return fused[:top_k], 0.0, degradation

    monkeypatch.setattr(retrieval_mod, "_rerank_safe", _fake_rerank_identity)
    routings = [
        RoutingResult("policy", "policy", 0.9, "rule"),
        RoutingResult("service", "service", 0.6, "rule"),
    ]
    out = asyncio.run(orchestrator._retrieve_fanout("q", routings, 10, 0.0))
    assert len(out["docs"]) == 1  # 去重
    assert out["top1_score"] == 0.8  # 保留最高分


# ── 扇出判据 + 候选收敛契约 (修复: 空壳库路由 / 跨库主题单路漏召回) ──────────
# 背景: 路由候选原先取自 SKILLS 全量(含 product/finance 等 0 行的"空壳库"),
# 且扇出只在主候选 conf<0.8 时触发, 导致退换货这类横跨 policy/service 的主题
# 只检索单库 →答案在另一库时 recall_doc=0。以下用例锁死修复后的行为。


def test_should_fanout_on_ambiguous_main():
    """主候选置信度低于 early_exit → 扇出 (P1' 原判据)。"""
    main = RoutingResult("policy", "policy", 0.5, "llm")
    assert orchestrator._should_fanout(main, [RoutingResult("service", "service", 0.4, "llm")])


def test_should_fanout_on_close_contender():
    """主候选虽高, 但次选旗鼓相当 → 仍扇出 (跨库主题单路会漏掉答案所在库)。

    回归用例: 退换货同时命中 policy(0.9) 与 service(0.9), 主候选"看似确定",
    单路只检索 policy → 答案落在 service 时 recall_doc=0。置信度接近本身就说明
    路由分不清归属, 此时扇出并行检索才是正确取舍。
    """
    main = RoutingResult("policy", "policy", 0.9, "rule")
    assert orchestrator._should_fanout(main, [RoutingResult("service", "service", 0.9, "rule")])


def test_should_not_fanout_when_main_clear():
    """主候选明确、次选明显更弱 → 单路, 不白增一倍检索延迟。"""
    main = RoutingResult("tech", "tech", 0.9, "rule")
    assert not orchestrator._should_fanout(main, [RoutingResult("service", "service", 0.3, "rule")])


def test_should_not_fanout_without_contender():
    """无次选候选 → 不扇出。"""
    assert not orchestrator._should_fanout(RoutingResult("tech", "tech", 0.9, "rule"), [])


def test_route_filters_candidates_to_mounted_kbs(monkeypatch):
    """路由候选收敛到已挂载库: 剔除 product 等"注册了类型但无数据"的空壳库。

    回归用例: 收敛前「你们的搜索功能是怎么…」被路由到 product(该库 0 行) → recall=0。
    """
    from engines.router.intent_router import IntentRouter

    monkeypatch.setattr(routing_mod, "mounted_kbs", lambda: ["policy", "service", "tech"])

    async def _fake_route_multi(self, question, top_n=3):
        return [
            RoutingResult("product", "product", 0.9, "llm"),
            RoutingResult("tech", "tech", 0.8, "llm"),
        ]

    monkeypatch.setattr(IntentRouter, "route_multi", _fake_route_multi)

    class _NoLLM:
        pass

    routing, candidates, _ms = asyncio.run(
        orchestrator._route("你们的搜索功能是怎么实现的", None, _NoLLM(), 0.0, None)
    )
    assert all(c.kb != "product" for c in candidates)
    assert routing.kb == "tech"


def test_route_fallback_stays_inside_mounted_kbs(monkeypatch):
    """候选全被收敛剔除时, 兜底库必须落在已挂载范围内 (默认 fallback=tech 可能不在)。

    同时守护兜底路径的 FALLBACK_SKILL 引用 (该分支曾因漏 import 而是个 NameError)。
    """
    from engines.router.intent_router import IntentRouter

    monkeypatch.setattr(routing_mod, "mounted_kbs", lambda: ["policy", "service"])

    async def _fake_route_multi(self, question, top_n=3):
        return [RoutingResult("product", "product", 0.9, "llm")]

    monkeypatch.setattr(IntentRouter, "route_multi", _fake_route_multi)

    class _NoLLM:
        pass

    routing, candidates, _ms = asyncio.run(orchestrator._route("q", None, _NoLLM(), 0.0, None))
    assert routing.source == "fallback"
    assert routing.kb in ("policy", "service")  # 不得兜回 tech 等未挂载库
    assert [c.kb for c in candidates] == [routing.kb]

