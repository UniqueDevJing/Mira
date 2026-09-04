"""KB 级召回评测 runner（生产对齐，离线，无需 LLM）。

消费 scripts/gen_kb_eval.py 生成的 tests/eval_dataset_kb.json（问题 + 所属 KB +
expected_chunk_ids），用与生产一致的路径（api.state.get_vector_store 按名打开 KB 表 →
真实 Embedding → VectorStore.search）逐条检索，计算 Recall@K / MRR / 延迟。

用法:
  python scripts/eval_kb_retrieval.py                # 全量
  python scripts/eval_kb_retrieval.py --limit 20     # 每库抽样上限
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DATASET = os.path.join("tests", "eval_dataset_kb.json")
DEFAULT_OUT = os.path.join("data", "eval", "kb_retrieval_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="每库抽样上限, 0=全部")
    args = ap.parse_args()

    from api.state import get_embedder, get_vector_store

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    embedder = get_embedder()
    ks = [1, 3, 5, 10]
    per_kb: dict[str, dict] = {}
    details = []
    errors = 0

    items = dataset
    if args.limit:
        seen: dict[str, int] = {}
        items = []
        for x in dataset:
            kb = x["kb"]
            if seen.get(kb, 0) >= args.limit:
                continue
            seen[kb] = seen.get(kb, 0) + 1
            items.append(x)

    for item in items:
        kb = item["kb"]
        question = item["question"]
        expected = set(item.get("expected_chunk_ids") or [])
        if kb not in per_kb:
            per_kb[kb] = {"n": 0, "hits": {k: 0 for k in ks}, "rr_sum": 0.0, "lat_sum": 0.0, "errs": 0}
        stat = per_kb[kb]

        try:
            vec = embedder.embed_query(question)
            t0 = time.time()
            res = get_vector_store(kb).search(vec, top_k=args.topk)
            lat = (time.time() - t0) * 1000
        except Exception as e:  # noqa: BLE001 — 单条失败不阻断整体评测
            stat["errs"] += 1
            errors += 1
            details.append({"kb": kb, "question": question[:40], "error": str(e)[:120]})
            continue

        ids = [r.get("id") or r.get("chunk_id") for r in res]
        rr = 0.0
        for rank, cid in enumerate(ids, start=1):
            if cid in expected:
                rr = 1.0 / rank
                break
        stat["n"] += 1
        stat["rr_sum"] += rr
        stat["lat_sum"] += lat
        for k in ks:
            if any(cid in expected for cid in ids[:k]):
                stat["hits"][k] += 1
        details.append({
            "kb": kb,
            "question": question[:50],
            "top1": ids[0] if ids else None,
            "hit1": bool(ids and ids[0] in expected),
            "rr": round(rr, 4),
            "score_top1": res[0]["score"] if res else None,
            "latency_ms": round(lat, 1),
        })

    summary = {"dataset": args.dataset, "total": len(items), "errors": errors, "per_kb": {}, "overall": {}}
    all_n, all_hits, all_rr = 0, {k: 0 for k in ks}, 0.0
    for kb, s in sorted(per_kb.items()):
        n = s["n"] or 1
        row = {
            "n": s["n"],
            "mrr": round(s["rr_sum"] / n, 4),
            "latency_ms_avg": round(s["lat_sum"] / n, 1),
            "errors": s["errs"],
        }
        for k in ks:
            row[f"recall@{k}"] = round(s["hits"][k] / n, 4)
        summary["per_kb"][kb] = row
        all_n += s["n"]
        for k in ks:
            all_hits[k] += s["hits"][k]
        all_rr += s["rr_sum"]
    n = all_n or 1
    summary["overall"] = {"n": all_n, "mrr": round(all_rr / n, 4)}
    for k in ks:
        summary["overall"][f"recall@{k}"] = round(all_hits[k] / n, 4)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n报告已写入 {args.out}")


if __name__ == "__main__":
    main()
