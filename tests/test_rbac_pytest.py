"""RBAC + API Key 白名单测试。

- 单元: 白名单解析 / Principal 授权判断 / candidate_kbs 收窄 / ask 抛 KBForbiddenError / list_all(kb_in) 过滤
- 集成(TestClient): 中间件校验(401/403) / 路由层按 API Key 做 KB 级 403 / 授权放行
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from api.core import auth as auth_mod
from api.core.auth import KBForbiddenError, Principal
from api.core.orchestrator import _candidate_kbs


# ───────────────────────── 单元: 白名单解析 ─────────────────────────


def test_whitelist_parse_admin_and_reader(monkeypatch):
    monkeypatch.setenv(
        "RAG_API_KEY_WHITELIST",
        json.dumps(
            {
                "k_admin": {"name": "admin", "kbs": "*", "role": "admin"},
                "k_reader": {"name": "r", "kbs": ["service"], "role": "reader"},
                "k_null": {"name": "n", "kbs": None},
            }
        ),
    )
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    auth_mod._KEYS = None
    keys = auth_mod.load_api_keys()
    assert keys["k_admin"].allowed_kbs is None and keys["k_admin"].is_admin()
    assert keys["k_reader"].allowed_kbs == ["service"] and not keys["k_reader"].is_admin()
    assert keys["k_null"].allowed_kbs is None  # null => 全部知识库


def test_whitelist_invalid_json_raises(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY_WHITELIST", "{not json")
    auth_mod._KEYS = None
    with pytest.raises(ValueError):
        auth_mod.load_api_keys()


def test_legacy_single_key_is_admin(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY_WHITELIST", raising=False)
    monkeypatch.setenv("RAG_API_KEY", "legacy-key")
    auth_mod._KEYS = None
    assert auth_mod.load_api_keys()["legacy-key"].is_admin()


def test_authenticate_lookup(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.setenv("RAG_API_KEY_WHITELIST", json.dumps({"good": {"kbs": ["service"]}}))
    auth_mod._KEYS = None
    assert auth_mod.authenticate("good").key_id == "good"
    assert auth_mod.authenticate("bad") is None
    assert auth_mod.authenticate(None) is None


def test_principal_can_access_kb():
    p = Principal(key_id="x", name="x", role="reader", allowed_kbs=["service", "tech"])
    assert p.can_access_kb("service")
    assert not p.can_access_kb("docs")
    assert Principal(key_id="a", name="a", role="admin", allowed_kbs=None).can_access_kb("anything")


def test_candidate_kbs_narrows():
    assert _candidate_kbs(None)  # 默认 RAG_KBS 非空
    assert _candidate_kbs(["service"]) == ["service"]
    assert _candidate_kbs(["nope"]) == []


# ───────────────────────── 单元: ask 抛 KBForbiddenError ─────────────────────────


def test_ask_raises_forbidden_on_disallowed_kb(monkeypatch):
    import api.core.orchestrator as oc

    async def _fake_route(q, skill, llm, start, candidate_kbs=None):
        from engines.router.intent_router import RoutingResult

        return RoutingResult("tech", "tech", 1.0, "manual"), 1.0

    async def _go():
        monkeypatch.setattr(oc, "_route", _fake_route)
        monkeypatch.setattr(oc, "load_session", lambda sid: [])
        monkeypatch.setattr(oc, "save_session", lambda *a, **k: None)
        with pytest.raises(KBForbiddenError):
            await oc.ask("hi", allowed_kbs=["service"])

    asyncio.run(_go())


# ───────────────────────── 单元: list_all(kb_in) 过滤 ─────────────────────────


def test_list_all_kb_in_filter(tmp_path):
    from api.core.document_store import DocumentStore

    ds = DocumentStore(db_path=str(tmp_path / "d.db"))
    ds.save("1", "a.txt", status="ready", knowledge_base="service")
    ds.save("2", "b.txt", status="ready", knowledge_base="tech")
    assert ds.list_all()["total"] == 2
    svc = ds.list_all(kb_in=["service"])
    assert svc["total"] == 1 and svc["items"][0]["doc_id"] == "1"
    assert ds.list_all(kb_in=["nope"])["total"] == 0


# ───────────────────────── 集成: 中间件 + 路由层 RBAC ─────────────────────────


@pytest.fixture
def rbac_client(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY_ENABLED", "true")
    monkeypatch.setenv(
        "RAG_API_KEY_WHITELIST",
        json.dumps(
            {
                "admin": {"kbs": "*"},
                "reader_svc": {"kbs": ["service"], "role": "reader"},
            }
        ),
    )
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    auth_mod._KEYS = None

    # 避免真实模型加载 / 真实 LLM
    import engines.embedding.embedder as emb_mod

    class _FE:
        def __init__(self, *a, **k):
            pass

        def embed_query(self, q):
            return [0.0] * 8

        def embed_batch(self, t):
            return [[0.0] * 8 for _ in t]

    monkeypatch.setattr(emb_mod, "EmbeddingService", _FE)

    async def _fake_ask(*a, **k):
        return {
            "answer": "ok",
            "sources": [],
            "skill": "direct",
            "kb_id": None,
            "degradation_level": 0,
            "latency_breakdown": {},
            "retrieval_meta": {},
            "qa_metrics": {},
        }

    async def _fake_stream(*a, **k):
        yield {"type": "done", "answer": "ok"}

    # qa.py 以 `ask as orchestrate` 别名绑定, 必须 patch qa 模块里的别名
    import api.routes.qa as qa_mod

    monkeypatch.setattr(qa_mod, "orchestrate", _fake_ask)
    monkeypatch.setattr(qa_mod, "orchestrate_stream", _fake_stream)

    from api.main import app

    return TestClient(app)


def test_no_key_401(rbac_client):
    r = rbac_client.post("/api/v1/qa/ask", json={"question": "x"})
    assert r.status_code == 401


def test_bad_key_403(rbac_client):
    r = rbac_client.post("/api/v1/qa/ask", json={"question": "x"}, headers={"X-API-Key": "nope"})
    assert r.status_code == 403


def test_admin_key_allowed(rbac_client):
    r = rbac_client.post("/api/v1/qa/ask", json={"question": "x"}, headers={"X-API-Key": "admin"})
    assert r.status_code == 200


def test_reader_service_skill_allowed(rbac_client):
    r = rbac_client.post(
        "/api/v1/qa/ask", json={"question": "x", "skill": "service"}, headers={"X-API-Key": "reader_svc"}
    )
    assert r.status_code == 200


def test_reader_tech_skill_forbidden(rbac_client):
    r = rbac_client.post("/api/v1/qa/ask", json={"question": "x", "skill": "tech"}, headers={"X-API-Key": "reader_svc"})
    assert r.status_code == 403


def test_reader_upload_disallowed_kb_403(rbac_client):
    # 上传目标知识库不在 reader_svc 授权范围(service) → 403 (路由层先于处理拦截)
    r = rbac_client.post(
        "/api/v1/documents/upload",
        files={"file": ("b.txt", b"hello", "text/plain")},
        data={"knowledge_base": "tech"},
        headers={"X-API-Key": "reader_svc"},
    )
    assert r.status_code == 403


def test_admin_upload_allowed_kb(rbac_client):
    r = rbac_client.post(
        "/api/v1/documents/upload",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"knowledge_base": "service"},
        headers={"X-API-Key": "admin"},
    )
    # 200 = 通过 RBAC 校验进入处理流程(后台任务可能异步失败, 但路由授权已放行)
    assert r.status_code == 200
