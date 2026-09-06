"""为指定 embedding 模型生成重嵌入语料文件 (供 eval_retrieval --chunks 对比评测)。

用法: python scripts/build_reembed.py --model BAAI/bge-m3 --tag bgem3 [--limit 0]
输出: data/eval/corpus_chunks_<tag>.json
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.embedding.embedder import EmbeddingService  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chunks", default="data/eval/corpus_chunks.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    chunks = json.load(open(args.chunks, encoding="utf-8"))
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"[reembed] 模型 {args.model}, 块数 {len(chunks)}")

    emb = EmbeddingService(model_name=args.model, backend="local")
    texts = [c["content"] for c in chunks]
    t0 = time.time()
    embs = emb.embed_batch(texts)
    print(f"[reembed] 嵌入完成, 耗时 {time.time()-t0:.0f}s, 维度 {len(embs[0])}")

    out_path = f"data/eval/corpus_chunks_{args.tag}.json"
    out = []
    for c, e in zip(chunks, embs):
        nc = dict(c)
        nc["embedding"] = list(map(float, e))
        out.append(nc)
    json.dump(out, open(out_path, "w", encoding="utf-8"))
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
