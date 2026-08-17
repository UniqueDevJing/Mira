"""导出生产 qa_logs 为可标注样本。

每次 /ask 与 /ask/stream 都已异步写 qa_logs (api/core/document_store.py),
含 question/answer/sources/kb/routing 等。本脚本把它导出为:
  - qa_export.json: 原始全字段 + 离线算的 faithfulness (词重叠, 与 evaluate 口径一致)
  - labeled_production.json: 可直接喂 calibrate_fidelity 的骨架 (score=faithfulness, is_bad=null 待人工标)

score 语义: 高=忠实。词重叠 faithfulness = |答案词 ∩ 上下文词| / |答案词|。
(与 scripts/evaluate.py 的 hallucination_rate=1-faith 互补; 无 LLM 依赖, 离线可算)

运行:
  python scripts/export_qa_logs.py [--db data/documents.db]
    [--out data/qa_export.json] [--labeled data/labeled_production.json]
    [--limit N] [--since YYYY-MM-DD]
"""

import argparse
import datetime
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jieba

from api.core.document_store import DEFAULT_DB_PATH


def _faithfulness(answer: str, contexts: list[str]) -> float:
    """词重叠忠实度 (高=答案事实多由上下文覆盖), 与 evaluate.py 口径一致。"""
    if not answer or not contexts:
        return 0.0
    ans_tokens = {t for t in jieba.cut(answer) if len(t.strip()) >= 2}
    ctx_tokens: set[str] = set()
    for c in contexts:
        ctx_tokens |= {t for t in jieba.cut(c or "") if len(t.strip()) >= 2}
    if not ans_tokens:
        return 0.0
    return round(len(ans_tokens & ctx_tokens) / len(ans_tokens), 4)


def export_qa_logs(db_path: str, out_path: str, labeled_path: str | None, limit: int | None, since: str | None) -> dict:
    """导出 qa_logs 为 JSON。返回统计 {count, out, labeled}。"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"找不到数据库: {db_path}")

    conds, params = [], []
    if since:
        conds.append("created_at >= ?")
        params.append(since)
    where = f"WHERE {' AND '.join(conds)}" if conds else ""
    limit_sql = f"LIMIT {int(limit)}" if limit else ""

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT id, question, answer, skill, kb_id, routing_source, degradation_level, "
            f"latency_ms, tokens_total, sources, created_at FROM qa_logs {where} "
            f"ORDER BY id DESC {limit_sql}",
            params,
        ).fetchall()

    cases: list[dict] = []
    labeled: list[dict] = []
    for r in rows:
        try:
            sources = json.loads(r["sources"]) if r["sources"] else []
        except (json.JSONDecodeError, TypeError):
            sources = []
        contexts = [s.get("content", "") if isinstance(s, dict) else str(s) for s in sources]
        faith = _faithfulness(r["answer"] or "", contexts)
        cases.append(
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "sources": sources,
                "skill": r["skill"],
                "kb_id": r["kb_id"],
                "routing_source": r["routing_source"],
                "degradation_level": r["degradation_level"],
                "latency_ms": r["latency_ms"],
                "tokens_total": r["tokens_total"],
                "created_at": r["created_at"],
                "faithfulness": faith,
            }
        )
        labeled.append(
            {
                "question": r["question"],
                "kb": r["kb_id"] or "",
                "score": faith,
                "answer": r["answer"] or "",
                "contexts": contexts,
                "is_bad": None,
            }
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "exported_at": datetime.datetime.now().astimezone().isoformat(),
        "count": len(cases),
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if labeled_path:
        os.makedirs(os.path.dirname(labeled_path) or ".", exist_ok=True)
        with open(labeled_path, "w", encoding="utf-8") as f:
            json.dump({"cases": labeled}, f, ensure_ascii=False, indent=2)

    return {"count": len(cases), "out": out_path, "labeled": labeled_path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--out", default="data/qa_export.json")
    ap.add_argument("--labeled", default="data/labeled_production.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--since", default=None)
    args = ap.parse_args()

    try:
        res = export_qa_logs(args.db, args.out, args.labeled, args.limit, args.since)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return
    print(f"已导出 {res['count']} 条 → {res['out']}")
    if res["labeled"]:
        print(f"已生成可标注骨架 → {res['labeled']} (score=faithfulness 已填, is_bad 待人工标)")
        print(f"人工标 is_bad 后: python scripts/calibrate_fidelity.py {res['labeled']}")


if __name__ == "__main__":
    main()
