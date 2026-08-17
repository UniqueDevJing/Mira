"""export_qa_logs 单测: sources 落库 + 导出结构 + faithfulness 映射。"""

import json
import sqlite3

import scripts.export_qa_logs as mod
from api.core.document_store import DocumentStore


def test_faithfulness_overlap_and_empty():
    s = mod._faithfulness("北京天气晴", ["北京天气晴转多云"])
    assert 0.0 <= s <= 1.0
    assert s > 0.0
    # 无上下文 -> 0
    assert mod._faithfulness("abc", []) == 0.0
    # 空答案 -> 0
    assert mod._faithfulness("", ["x"]) == 0.0


def test_log_qa_stores_sources_and_full_answer(tmp_path):
    db = str(tmp_path / "documents.db")
    store = DocumentStore(db)
    # answer 超过 500 字也不应被截断 (修复前 answer[:500])
    long_answer = "结论一。" * 300
    store.log_qa(
        "q1",
        long_answer,
        skill="weather",
        kb_id="kb1",
        routing_source="rule",
        sources=[{"content": "北京天气晴转多云"}, {"content": "下午有风"}],
    )
    # 校验落库字段
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT answer, sources FROM qa_logs ORDER BY id DESC LIMIT 1").fetchone()
    assert row["answer"] == long_answer  # 未截断
    assert json.loads(row["sources"])[0]["content"] == "北京天气晴转多云"


def test_export_writes_both_artifacts(tmp_path):
    db = str(tmp_path / "documents.db")
    store = DocumentStore(db)
    store.log_qa("q1", "北京今天晴", skill="weather", kb_id="kb1", routing_source="rule",
                 sources=[{"content": "北京天气晴转多云"}, {"content": "下午有风"}])
    store.log_qa("q2", "支持退款", kb_id="kb2", sources=[{"content": "7天无理由退款"}])

    out = tmp_path / "qa_export.json"
    labeled = tmp_path / "labeled_production.json"
    res = mod.export_qa_logs(db, str(out), str(labeled), None, None)
    assert res["count"] == 2

    data = json.loads(out.read_text(encoding="utf-8"))
    assert {c["question"] for c in data["cases"]} == {"q1", "q2"}
    q1 = next(c for c in data["cases"] if c["question"] == "q1")
    assert isinstance(q1["sources"], list)
    assert q1["faithfulness"] > 0.0

    lab = json.loads(labeled.read_text(encoding="utf-8"))
    lab_q1 = next(c for c in lab["cases"] if c["question"] == "q1")
    assert lab_q1["score"] == q1["faithfulness"]
    assert lab_q1["is_bad"] is None
    assert lab_q1["contexts"][0] == "北京天气晴转多云"


def test_export_limit(tmp_path):
    db = str(tmp_path / "documents.db")
    store = DocumentStore(db)
    store.log_qa("q1", "a1", sources=[{"content": "x"}])
    store.log_qa("q2", "a2", sources=[{"content": "y"}])
    out = tmp_path / "qa_export.json"
    res = mod.export_qa_logs(db, str(out), None, limit=1, since=None)
    assert res["count"] == 1


def test_export_missing_db_errors(tmp_path):
    try:
        mod.export_qa_logs(str(tmp_path / "nope.db"), str(tmp_path / "o.json"), None, None, None)
        assert False, "应抛 FileNotFoundError"
    except FileNotFoundError:
        pass
