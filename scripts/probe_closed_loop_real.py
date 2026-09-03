"""真·闭环检索探针 — 对生产环境每张真实 LanceDB 表实测可检索性。

不依赖离线 eval_dataset, 直接拿已入库的真实 chunk 做自检索 (self-retrieval):
  用 chunk 正文 embed → VectorStore.search → 检查该 chunk 是否命中 top-k。
输出 data/eval/closed_loop_probe_real.json + 控制台摘要。

用法: venv/Scripts/python.exe scripts/probe_closed_loop_real.py  (需 HF_HUB_OFFLINE=1)
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lancedb import connect

from api.state import get_embedder
from engines.retrieval.vector_store import VectorStore, get_vector_search_error_count

URI = os.path.join(ROOT, "lancedb_data")
DIM = 512
SAMPLE_N = 5
TOP_K = 5


def _has_vec_index(vs: VectorStore) -> bool:
    try:
        idxs = vs.table.list_indices()
    except Exception:
        return False
    return any("embedding" in (getattr(ic, "columns", None) or []) for ic in idxs)


def main():
    db = connect(URI)
    # 枚举全部 .lance 目录 (含 table_names() 未登记但数据仍在盘的表, 如 rag_service/rag_tech)
    import glob as _glob

    table_names = sorted(
        os.path.basename(p)[: -len(".lance")]
        for p in _glob.glob(os.path.join(URI, "*.lance"))
    )
    listed = set(db.table_names())
    embedder = get_embedder()

    tables = {}
    for tname in table_names:
        entry = {"rows": None, "has_vector_index": None, "in_registry": tname in listed,
                 "sampled": 0,
                 "self_hits": 0, "self_hit_rate": None, "latency_ms": None, "error": None}
        try:
            vs = VectorStore(uri=URI, dim=DIM, table_name=tname)
            n = vs.table.count_rows()
            entry["rows"] = n
            entry["has_vector_index"] = _has_vec_index(vs)
            if n == 0:
                tables[tname] = entry
                continue
            rows = vs.table.search().limit(min(n, SAMPLE_N)).to_list()
            hits, lat = 0, []
            for r in rows:
                cid = r.get("id")
                content = (r.get("content") or "")[:200]
                if not content:
                    continue
                t0 = time.time()
                try:
                    emb = embedder.embed_query(content)
                except Exception as e:
                    entry["error"] = f"embed: {e}"
                    break
                docs = vs.search(emb, top_k=TOP_K)
                lat.append((time.time() - t0) * 1000)
                ids = [d.get("chunk_id") or d.get("id") for d in docs]
                entry["sampled"] += 1
                if cid in ids:
                    hits += 1
            entry["self_hits"] = hits
            entry["self_hit_rate"] = (hits / entry["sampled"]) if entry["sampled"] else None
            entry["latency_ms"] = round(sum(lat) / len(lat), 1) if lat else None
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        tables[tname] = entry

    empty = [t for t, e in tables.items() if e.get("rows") == 0]
    non_empty = [t for t, e in tables.items() if e.get("rows")]
    broken = [t for t, e in tables.items()
              if e.get("rows") and e.get("self_hit_rate") == 0.0 and not e.get("error")]
    errs = [t for t, e in tables.items() if e.get("error")]

    summary = {
        "total_tables": len(tables),
        "empty_tables": empty,
        "non_empty_tables": non_empty,
        "empty_count": len(empty),
        "broken_retrieval_tables": broken,
        "error_tables": errs,
        "search_error_count": get_vector_search_error_count(),
    }
    out = {"tables": tables, "summary": summary}
    os.makedirs(os.path.join(ROOT, "data/eval"), exist_ok=True)
    path = os.path.join(ROOT, "data/eval/closed_loop_probe_real.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print(f"{'TABLE':<24}{'ROWS':>7}{'IDX':>5}{'SAMP':>5}{'HIT':>5}{'RATE':>7}{'LAT(ms)':>9}")
    for t in table_names:
        e = tables[t]
        print(f"{t:<24}{e.get('rows')!s:>7}{e.get('has_vector_index')!s:>5}"
              f"{e.get('sampled')!s:>5}{e.get('self_hits')!s:>5}"
              f"{e.get('self_hit_rate')!s:>7}{e.get('latency_ms')!s:>9}")
    print("\n=== SUMMARY ===")
    print(f"tables={summary['total_tables']}  empty={summary['empty_count']}  "
          f"broken={len(broken)}  errors={len(errs)}  search_errs={summary['search_error_count']}")
    if empty:
        print("EMPTY :", empty)
    if broken:
        print("BROKEN:", broken)
    if errs:
        print("ERRORS:", errs)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
