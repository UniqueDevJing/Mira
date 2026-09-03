"""离线重嵌入 — 用指定 embedding 模型重算 corpus_chunks.json 的向量。

不改动原文件, 输出到独立文件(默认 corpus_chunks_<model>.json), 供 eval_retrieval.py
的 --chunks / --embedding-model 做 A/B 对比。复用 EmbeddingService.embed_batch,
保证与线上 embed_query 同前缀/截断/归一化。

用法:
  python scripts/reembed.py --model BAAI/bge-m3 [--eval-dir data/eval] [--out corpus_chunks_bgem3.json]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import settings
from engines.embedding.embedder import EmbeddingService


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--model", default=settings.embedding_model)
    ap.add_argument("--in", dest="in_file", default="corpus_chunks.json")
    ap.add_argument("--out", default=None, help="输出文件(默认 corpus_chunks_<model短名>.json)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    in_path = os.path.join(args.eval_dir, args.in_file)
    out_name = args.out or f"corpus_chunks_{args.model.split('/')[-1].replace('-', '_')}.json"
    out_path = os.path.join(args.eval_dir, out_name)

    chunks = json.load(open(in_path, encoding="utf-8"))
    print(f"[reembed] {len(chunks)} chunks, model={args.model}")

    svc = EmbeddingService(model_name=args.model)
    svc.batch_size = args.batch_size
    t0 = time.time()
    contents = [c["content"] for c in chunks]
    embs = svc.embed_batch(contents)
    print(f"[reembed] embedded {len(embs)} in {time.time()-t0:.1f}s, dim={len(embs[0])}")

    for c, e in zip(chunks, embs):
        c["embedding"] = e
    json.dump(chunks, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[reembed] written -> {out_path}")


if __name__ == "__main__":
    main()
