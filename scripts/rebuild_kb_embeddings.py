"""10 知识库向量重建 — embedding 模型切换 (bge-small 512维 → bge-base 768维)。

对每张含 embedding 列的 LanceDB 表: 读全部行 → 用新模型重嵌 content → 整表重建。
rag_memory 同样重建 (旧记忆向量与新查询向量必须同模型)。

用法: python scripts/rebuild_kb_embeddings.py [--db lancedb_data]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lancedb  # noqa: E402

from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL_PATH = "models/bge-base-zh-v1.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="lancedb_data")
    args = ap.parse_args()

    db = lancedb.connect(args.db)
    tables = [t for t in db.table_names()]
    print(f"[rebuild] 表: {tables}")

    emb = SentenceTransformer(MODEL_PATH, device="cpu")

    for name in tables:
        t = db.open_table(name)
        if "embedding" not in t.schema.names:
            print(f"[rebuild] 跳过 {name} (无 embedding 列)")
            continue
        rows = t.to_arrow().to_pylist()
        if not rows:
            print(f"[rebuild] 跳过 {name} (空表)")
            continue
        t0 = time.time()
        texts = [r.get("content") or "" for r in rows]
        embs = emb.encode(texts, batch_size=32, show_progress_bar=False)
        for r, e in zip(rows, embs):
            r["embedding"] = list(map(float, e))
        db.drop_table(name)
        db.create_table(name, data=rows)
        print(f"[rebuild] {name}: {len(rows)} 行重嵌完成 ({time.time()-t0:.0f}s), 维度 {len(embs[0])}")

    print("[rebuild] 全部完成")


if __name__ == "__main__":
    main()
