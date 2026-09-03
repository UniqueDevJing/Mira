"""Router + Skill 集成测试 — 通过 /qa/ask 验证路由与响应元信息。

注: 依赖空库（未上传文档），RAG skill 返回"未找到"，direct skill LLM 无 Key 降级。
均不依赖真实 LLM/embedding 成功。
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_manual_skill_service():
    r = client.post("/api/v1/qa/ask", json={"question": "退货流程是什么", "skill": "service"})
    d = r.json()
    assert r.status_code == 200
    assert d["skill"] == "service"
    assert d["kb_id"] == "service"
    assert d["routing_source"] == "manual"
    assert d["answer"] != ""


def test_rule_route_service():
    """纯客服领域词 (物流) 仍规则命中 service —— 不受退换货改归 policy 影响。"""
    d = client.post("/api/v1/qa/ask", json={"question": "物流多久能到"}).json()
    assert d["skill"] == "service" and d["routing_source"] == "rule"


def test_rule_route_refund_goes_to_policy():
    """退换货主题主路由 policy, service 作为次选由 P1' 扇出覆盖。

    行为变更说明: 退换货的"能否退 / 期限 / 退款比例"属制度规范, 评测集亦将退货标为
    policy, 因此给 policy 补了 退货/退款/退换货/七天无理由 等关键词。客服话术库同样
    含流程与到账时效内容, 故两者同权重 (0.9) 命中 —— 主路由取 policy, service 作为
    次选进入扇出 (置信度差 0.0 ≤ FANOUT_MARGIN), 端到端召回不丢失。
    """
    d = client.post("/api/v1/qa/ask", json={"question": "退款怎么处理"}).json()
    assert d["skill"] == "policy" and d["routing_source"] == "rule"


def test_rule_route_tech():
    d = client.post("/api/v1/qa/ask", json={"question": "系统架构如何部署"}).json()
    assert d["skill"] == "tech" and d["routing_source"] == "rule"
    assert d["kb_id"] == "tech"


def test_rule_route_direct():
    d = client.post("/api/v1/qa/ask", json={"question": "你好"}).json()
    assert d["skill"] == "direct"
    assert d["routing_source"] == "rule"
    assert d["sources"] == []
    assert d["kb_id"] is None


def test_response_meta_fields():
    d = client.post("/api/v1/qa/ask", json={"question": "退货流程", "skill": "tech"}).json()
    # 路由/降级/耗时/检索元信息都在
    assert "degradation_level" in d
    assert "latency_breakdown" in d and "router_ms" in d["latency_breakdown"]
    assert "retrieval_meta" in d and "top1_score" in d["retrieval_meta"]
    assert "token_usage" in d


def test_invalid_skill_rejected():
    r = client.post("/api/v1/qa/ask", json={"question": "x", "skill": "hacker"})
    assert r.status_code == 422
