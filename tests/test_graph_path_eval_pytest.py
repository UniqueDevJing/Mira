"""M4 GraphRAG 可证伪评测的单测（纯离线，不依赖 LLM / 向量库 / FastAPI 启动）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import Settings
from scripts.eval_graph_path import evaluate, keyword_baseline, load_dataset


def test_multihop_questions_show_positive_gain():
    """核心可证伪判据：多跳问题必须靠图谱 2 跳才能召回证据 chunk（增量 > 0）。"""
    result = evaluate(load_dataset())
    assert result["multihop_gain"] > 0, "多跳问题应存在图谱增量，否则图谱该被关掉"
    assert result["path_precision"] >= 0.5, "路径正确率应达标（预期三元组被遍历还原）"
    assert result["verdict"]["keep_graph_default_on"] is True


def test_singlehop_gain_is_zero():
    """诚实对照：单跳同 chunk 共现本就够用，图谱增量应为 0——测不出来说明夹具没失真。"""
    result = evaluate(load_dataset())
    assert result["singlehop_gain"] == 0.0


def test_baseline_misses_multihop_target():
    """夹具性质：q2 的目标证据 chunk 与问题词面零重叠，关键词基线必须召回不到它。"""
    ds = load_dataset()
    q2 = next(q for q in ds["questions"] if q["id"] == "q2")
    hits = keyword_baseline(q2["question"], ds["corpus"], top_k=3)
    assert "c2" not in hits, "夹具失真：基线不应能词面命中多跳目标 chunk"


def test_negative_control_produces_no_graph_output():
    """负控制：与语料无关的问题，图谱不应有实体/chunk 产出（多管闲事按 0 计）。"""
    result = evaluate(load_dataset())
    q4 = next(r for r in result["questions"] if r["id"] == "q4")
    assert q4["entities_found"] == []
    assert q4["graph_chunks"] == []


def test_graph_enabled_switch():
    """总开关可证伪落点：RAG_GRAPH_ENABLED=false 时可整体关闭图谱腿，零代码改动。"""
    assert Settings(graph_enabled=True).graph_enabled is True
    assert Settings(graph_enabled=False).graph_enabled is False
