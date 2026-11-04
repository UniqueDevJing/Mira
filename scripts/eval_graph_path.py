"""GraphRAG 关系路径正确率评测（M4 可证伪化）。

目的：关系推理最容易变玄学——"图谱增强了检索"如果不拿数字证明，就该被关掉。
本脚本构造一个规则抽取友好、零 LLM 成本的小评测集（scripts/graph_path_set.json），
离线回答三个可证伪的问题：

1. 路径正确率：图谱遍历产出的三元组，有多少命中预期关系路径（精确匹配）？
2. 图谱增量收益：graph 开（多跳召回）vs 关（纯关键词基线），证据 chunk 召回差多少？
3. 收益集中在哪：多跳问题 vs 单跳问题分开统计（单跳同 chunk 共现本就够用，
   图谱收益应为 0——测不出来就承认，这正是"可证伪"的含义）。

结论规则（写死在 verdict 里，不拍脑袋）：
- 多跳问题增量召回 > 0 且路径正确率 >= 0.5 → 图谱默认保留（RAG_GRAPH_ENABLED=true）
- 否则 → 建议 RAG_GRAPH_ENABLED=false（开关已落在 api/config.py，关掉零代码改动）

用法（纯离线，不依赖 LLM / 向量库 / FastAPI）：

    python scripts/eval_graph_path.py                # 打印摘要
    python scripts/eval_graph_path.py --out out.json # 同时落盘
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.graph_rag.entity_extractor import EntityExtractor, RelationExtractor
from engines.graph_rag.graph_retriever import GraphRAGRetriever
from engines.graph_rag.graph_store import GraphStore

DATA_PATH = Path(__file__).resolve().parent / "graph_path_set.json"


class _Chunk:
    """最小 chunk 适配：build_from_chunks 只关心 content/chunk_id 两个属性。"""

    def __init__(self, chunk_id: str, content: str):
        self.chunk_id = chunk_id
        self.content = content


def load_dataset(path: Path = DATA_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"corpus": data["corpus"], "questions": data["questions"]}


def build_graph(corpus: list[dict]) -> GraphRAGRetriever:
    """零 LLM 建图：抽取器不配 Key 时自动走规则兜底（实体=技术词表，关系=动词模板）。"""
    store = GraphStore(persist_path=None)
    retriever = GraphRAGRetriever(EntityExtractor(), RelationExtractor(), store)
    retriever.build_from_chunks([_Chunk(c["chunk_id"], c["content"]) for c in corpus])
    return retriever


def _normalize_triples(graph_context: list[str]) -> set[tuple[str, str, str]]:
    out = set()
    for line in graph_context or []:
        parts = line.split()
        if len(parts) >= 3:
            out.add((parts[0], parts[1], " ".join(parts[2:])))
    return out


def _tokens(text: str) -> set[str]:
    """轻量分词：CJK 二元组 + ASCII 词。评测专用基线，不代表线上 BM25。"""
    text = text.lower()
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.update(f"{a}{b}" for a, b in zip(cjk, cjk[1:]))
    return tokens


def keyword_baseline(question: str, corpus: list[dict], top_k: int = 3) -> list[str]:
    """graph 关闭时的替代路径：纯词面重叠检索（vector/BM25 的离线近似下界）。"""
    qt = _tokens(question)
    scored = []
    for c in corpus:
        dt = _tokens(c["content"])
        if not qt:
            scored.append((c["chunk_id"], 0.0))
            continue
        overlap = len(qt & dt)
        scored.append((c["chunk_id"], overlap / len(qt)))
    scored.sort(key=lambda x: -x[1])
    return [cid for cid, s in scored[:top_k] if s > 0]


def evaluate(dataset: dict, top_k: int = 5) -> dict:
    corpus, questions = dataset["corpus"], dataset["questions"]
    retriever = build_graph(corpus)

    rows = []
    for q in questions:
        ctx = retriever.retrieve(q["question"], top_k=top_k, rule_only=True)
        got_triples = _normalize_triples(ctx.get("graph_context"))
        expected = {tuple(t) for t in q.get("expected_triples") or []}

        if expected:
            path_precision = len(got_triples & expected) / len(expected)
        else:
            # 负控制：预期无路径。图谱若仍产出实体/上下文，按 0 计（多管闲事）
            path_precision = 0.0 if (ctx.get("entities") or got_triples) else 1.0

        graph_chunks = set(ctx.get("source_chunks") or [])
        base_chunks = set(keyword_baseline(q["question"], corpus, top_k=top_k))
        expected_chunks = set(q.get("expected_chunks") or [])

        if expected_chunks:
            recall_graph = len(graph_chunks & expected_chunks) / len(expected_chunks)
            recall_base = len(base_chunks & expected_chunks) / len(expected_chunks)
        else:
            # 负控制：图谱不应带出任何 chunk（纯噪声 = 收益为负记 0）
            recall_graph = 1.0 if not graph_chunks else 0.0
            recall_base = 1.0

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "needs_multihop": q.get("needs_multihop", False),
            "entities_found": ctx.get("entities") or [],
            "graph_context": ctx.get("graph_context") or [],
            "graph_chunks": sorted(graph_chunks),
            "baseline_chunks": sorted(base_chunks),
            "expected_chunks": sorted(expected_chunks),
            "path_precision": round(path_precision, 3),
            "recall_graph": round(recall_graph, 3),
            "recall_baseline": round(recall_base, 3),
            "gain": round(recall_graph - recall_base, 3),
        })

    multi = [r for r in rows if r["needs_multihop"]]
    single = [r for r in rows if not r["needs_multihop"]]

    def _macro(rows_, key):
        vals = [r[key] for r in rows_]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    multi_gain = _macro(multi, "gain")
    path_precision = _macro([r for r in rows if r["expected_chunks"]], "path_precision")
    # 保留判据：多跳问题有正增量 且 路径正确率达标（0.5 = 一半预期路径能被遍历还原）
    keep = multi_gain > 0 and path_precision >= 0.5

    return {
        "questions": rows,
        "path_precision": path_precision,
        "macro_recall_graph": _macro([r for r in rows if r["expected_chunks"]], "recall_graph"),
        "macro_recall_baseline": _macro([r for r in rows if r["expected_chunks"]], "recall_baseline"),
        "multihop_gain": multi_gain,
        "singlehop_gain": _macro(single, "gain"),
        "verdict": {
            "keep_graph_default_on": keep,
            "recommendation": (
                "RAG_GRAPH_ENABLED 保持 true：多跳问题存在正增量且路径正确率达标"
                if keep else
                "建议 RAG_GRAPH_ENABLED=false：图谱增量收益未达标（api/config.py 已备开关，关掉零代码改动）"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="GraphRAG 关系路径正确率评测（M4）")
    parser.add_argument("--dataset", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=None, help="评测摘要 JSON 落盘路径")
    args = parser.parse_args()

    result = evaluate(load_dataset(args.dataset))

    print("=== GraphRAG 可证伪评测 ===")
    print(f"路径正确率        : {result['path_precision']}")
    print(f"macro 召回 图开/基线: {result['macro_recall_graph']} / {result['macro_recall_baseline']}")
    print(f"多跳增量 gain     : {result['multihop_gain']}")
    print(f"单跳增量 gain     : {result['singlehop_gain']}")
    print(f"结论              : {result['verdict']['recommendation']}")
    for r in result["questions"]:
        print(f"  [{r['id']}] path={r['path_precision']} graph={r['recall_graph']} "
              f"base={r['recall_baseline']} gain={r['gain']:+.3f}  {r['question']}")

    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
