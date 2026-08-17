"""路由规则配置化测试 — 验证环境变量/JSON 文件覆盖词表与阈值, 及回退与容错。"""

import importlib
import json

from engines.router import routing_rules
from engines.router.routing_rules import load_rules_from_file


def _reload():
    importlib.reload(routing_rules)


def test_load_rules_from_file(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps({"service": [["退款", 0.9]], "tech": [], "direct": []}), encoding="utf-8")
    rules = load_rules_from_file(p)
    assert rules["service"] == [["退款", 0.9]]


def test_module_picks_up_rules_file(monkeypatch, tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps({"service": [["定制词", 0.8]], "tech": [], "direct": []}), encoding="utf-8")
    monkeypatch.setenv("RAG_ROUTING_RULES_FILE", str(p))
    _reload()
    try:
        assert routing_rules.SKILL_RULES["service"] == [["定制词", 0.8]]
    finally:
        monkeypatch.delenv("RAG_ROUTING_RULES_FILE", raising=False)
        _reload()


def test_invalid_rules_file_falls_back(monkeypatch, tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("RAG_ROUTING_RULES_FILE", str(p))
    _reload()
    try:
        assert routing_rules.SKILL_RULES is routing_rules.DEFAULT_SKILL_RULES  # 回退内置, 非崩溃
    finally:
        monkeypatch.delenv("RAG_ROUTING_RULES_FILE", raising=False)
        _reload()


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("RAG_ROUTE_THRESHOLD", "0.9")
    monkeypatch.setenv("RAG_LLM_TIMEOUT_S", "2.0")
    monkeypatch.setenv("RAG_FALLBACK_SKILL", "service")
    _reload()
    try:
        assert routing_rules.ROUTE_THRESHOLD == 0.9
        assert routing_rules.LLM_TIMEOUT_S == 2.0
        assert routing_rules.FALLBACK_SKILL == "service"
    finally:
        for n in ("RAG_ROUTE_THRESHOLD", "RAG_LLM_TIMEOUT_S", "RAG_FALLBACK_SKILL"):
            monkeypatch.delenv(n, raising=False)
        _reload()


def test_threshold_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("RAG_ROUTE_THRESHOLD", "not-a-float")
    _reload()
    try:
        assert routing_rules.ROUTE_THRESHOLD == 0.85  # 非数值回退默认
    finally:
        monkeypatch.delenv("RAG_ROUTE_THRESHOLD", raising=False)
        _reload()
