"""评测集构建 — 把公开 CRUD-RAG 的 questanswer_* 转成本项目可用的带 golden_chunk_ids 评测集。

设计要点：
- 走真实 StructureChunker 分块（产出 chunk_id 与线上一致：{doc_id}_chunk_{index:04d}）。
- 每个问题标注 golden_chunk_ids：在该问题所属源文档的 chunks 中，答案词覆盖最高的 chunk(s)。
- 同时导出可离线索引的语料（含 embedding），交给 eval_retrieval.py 做 rerank 关/开 对比。
- 全程离线（bge-small-zh-v1.5 本地缓存），不联网、不起服务、不调 LLM。

用法:
  python scripts/build_eval_set.py [--seed 7] [--n1 150] [--n2 120] [--n3 120] [--out data/eval]
"""
import argparse
import json
import os
import random
import re
import sys
import time

# 离线优先：避免 sentence-transformers 联网校验 adapter_config（本环境 SSL 不通）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jieba

from api.config import settings
from engines.chunking.structure_chunker import StructureChunker
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index


def _tok(s: str) -> set[str]:
    return {t for t in jieba.lcut(s or "") if t.strip()}


class _Doc:
    """最小 UIR doc 壳，供 StructureChunker 使用。"""
    def __init__(self, doc_id: str, text: str):
        self.doc_id = doc_id
        self.source = {"path": doc_id}
        self.pages = [{"blocks": [{"type": "paragraph", "content": text, "metadata": {}, "page_num": 1}]}]


def chunk_doc(doc_id: str, text: str, chunker: StructureChunker) -> list[dict]:
    if not text or not text.strip():
        return []
    chunks = chunker.chunk(_Doc(doc_id, text))
    return [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.content}
        for c in chunks
    ]


def golden_chunks(answer: str, chunks: list[dict]) -> list[str]:
    """答案词覆盖最高的 chunk 视为 golden；放宽到覆盖 >= 0.3 倍答案词量。"""
    ans_tok = _tok(answer)
    if not ans_tok:
        return [c["chunk_id"] for c in chunks[:1]]
    scored = []
    for c in chunks:
        cov = len(ans_tok & _tok(c["content"])) / len(ans_tok)
        scored.append((cov, c["chunk_id"]))
    best = max(s[0] for s in scored)
    gold = [cid for cov, cid in scored if cov >= max(0.3, 0.5 * best)]
    return gold or [max(scored, key=lambda x: x[0])[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external/crud_rag_split.json")
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n1", type=int, default=150, help="questanswer_1doc 取样数")
    ap.add_argument("--n2", type=int, default=120, help="questanswer_2docs 取样数")
    ap.add_argument("--n3", type=int, default=120, help="questanswer_3docs 取样数")
    ap.add_argument("--distractors", type=int, default=0,
                    help="注入纯干扰文档数(来自 event_summary/continuing_writing 中文新闻, 无 golden 标注), 制造词汇竞争使评测更难")
    ap.add_argument("--hard-negatives", type=int, default=0,
                    help="每问题生成的实体替换型硬负例数(高词汇相似/错语义, 专测 rerank 语义辨别力)")
    args = ap.parse_args()

    random.seed(args.seed)
    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)

    chunker = StructureChunker(max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap)
    emb = EmbeddingService()

    corpus_docs: dict[str, str] = {}      # doc_id -> content（去重）
    eval_items: list[dict] = []
    chunk_cache: dict[str, list[dict]] = {}  # doc_id -> chunks

    def ensure_doc(doc_id: str, text: str):
        if doc_id in corpus_docs:
            return
        corpus_docs[doc_id] = text
        chunk_cache[doc_id] = chunk_doc(doc_id, text, chunker)

    plan = [
        ("questanswer_1doc", args.n1, 1),
        ("questanswer_2docs", args.n2, 2),
        ("questanswer_3docs", args.n3, 3),
    ]
    for key, n, ndoc in plan:
        items = data.get(key, [])
        sample = random.sample(items, min(n, len(items)))
        for idx, it in enumerate(sample):
            news_fields = [it.get(f"news{i}") for i in range(1, ndoc + 1)]
            doc_ids = []
            for j, txt in enumerate(news_fields, 1):
                did = f"{key}_{idx:04d}_doc{j}"
                ensure_doc(did, txt or "")
                doc_ids.append(did)
            q = it.get("questions", "").strip()
            a = it.get("answers", "").strip()
            if not q or not a:
                continue
            # golden 在所有源文档的 chunks 中找
            item_chunks = [c for did in doc_ids for c in chunk_cache.get(did, [])]
            gold = golden_chunks(a, item_chunks)
            eval_items.append({
                "id": f"{key}_{idx:04d}",
                "kb": key,
                "category": f"{ndoc}doc",
                "question": q,
                "reference_answer": a,
                "golden_doc_ids": doc_ids,
                "expected_chunk_ids": gold,
            })

    # ---- 纯干扰文档（无 golden 标注）：制造词汇竞争，逼出 rerank 的翻盘场景 ----
    # 仅来自与评测问题同源的中文新闻语料，保证"干扰项与问题共享用词但语义无关"，
    # 这正是 RRF/BM25 易排错、Cross-Encoder rerank 最可能翻盘的真实场景。
    if args.distractors:
        es = data.get("event_summary", [])
        cw = data.get("continuing_writing", [])
        for i in range(args.distractors):
            if i % 2 == 0 and es:
                it = random.choice(es)
                txt = it.get("text") or it.get("title") or ""
            else:
                it = random.choice(cw)
                txt = (it.get("beginning") or "") + "\n" + (it.get("continuing") or "")
            ensure_doc(f"distractor_{i:05d}", txt or "")
        print(f"[build] distractors added: {args.distractors}")

    # ---- 实体替换型硬负例：把 golden chunk 的年份/《标题》换成其他实体，制造
    #      "高词汇相似、错语义" 的近重复干扰项 —— 这正是 Cross-Encoder rerank 的试金石。
    #      这类负例 BM25/向量 会排得极高（与 golden 几乎同词），只能靠语义辨别翻盘。
    if args.hard_negatives:
        _year_re = re.compile(r"\d{4}年")
        _title_re = re.compile(r"《[^》]{2,30}》")
        pool_text = " ".join(corpus_docs.values())
        years = [y for y in dict.fromkeys(_year_re.findall(pool_text)) if y] or ["2023年"]
        titles = [t for t in dict.fromkeys(_title_re.findall(pool_text)) if t] or ["《通知》"]
        random.seed((args.seed ^ 0x9E3779B9) & 0xFFFFFFFF)
        neg_count = 0
        for it in eval_items:
            for gid in it["expected_chunk_ids"]:
                gc = next((c for c in chunk_cache.get(gid.split("_chunk_")[0], []) if c["chunk_id"] == gid), None)
                if gc is None:
                    # fallback: 在全部 chunk 中找
                    gc = next((c for ds in chunk_cache.values() for c in ds if c["chunk_id"] == gid), None)
                if gc is None:
                    continue
                for _ in range(args.hard_negatives):
                    txt = gc["content"]
                    for y in set(_year_re.findall(txt)):
                        txt = txt.replace(y, random.choice([x for x in years if x != y] or years), 1)
                    for t in set(_title_re.findall(txt)):
                        txt = txt.replace(t, random.choice([x for x in titles if x != t] or titles), 1)
                    ensure_doc(f"hardneg_{neg_count:05d}", txt)
                    neg_count += 1
        print(f"[build] synthetic hard-negatives added: {neg_count}")

    # 批量嵌入所有 chunk（离线，本地缓存）
    all_chunks: list[dict] = [c for c in (chunk_cache[d] for d in corpus_docs) for c in c]
    print(f"[build] corpus docs={len(corpus_docs)} chunks={len(all_chunks)} questions={len(eval_items)}")
    t = time.time()
    texts = [c["content"] for c in all_chunks]
    vectors = emb.embed_batch(texts)
    for c, v in zip(all_chunks, vectors):
        c["embedding"] = [float(x) for x in v]
    print(f"[build] embedded {len(all_chunks)} chunks in {time.time()-t:.1f}s")

    # ---- 硬负例挖掘 (BM25)：识别"高相似但非答案"的干扰 chunk，并标记难样本 ----
    # 目的：当前评测集每源文档≈1 chunk，RRF 已封顶，rerank 提升被压扁。
    # 难样本 = BM25 把 golden 排到 top-10 之外的问题（说明存在强词汇干扰）；
    # 这些正是 rerank（Cross-Encoder 语义）最可能翻盘的场景，量化它才有意义。
    bm = Bm25Index()
    bm.add_documents([{"id": c["chunk_id"], "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"]} for c in all_chunks])
    for it in eval_items:
        gold = set(it["expected_chunk_ids"])
        hits = bm.search(it["question"], top_k=20)
        gold_rank = None
        for r, h in enumerate(hits, 1):
            if h["chunk_id"] in gold:
                gold_rank = r
                break
        neg = [h["chunk_id"] for h in hits if h["chunk_id"] not in gold][:10]
        it["bm25_golden_rank"] = gold_rank
        it["hard_negatives"] = neg
        it["hard"] = (gold_rank is None or gold_rank > 10)
    n_hard = sum(1 for it in eval_items if it["hard"])
    print(f"[build] hard questions (BM25 漏召 golden): {n_hard}/{len(eval_items)}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "corpus_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    with open(os.path.join(args.out, "eval_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(eval_items, f, ensure_ascii=False, indent=1)

    # 兼容既有 evaluate.py（需起服务 + LLM；此处只填纯检索可测字段）
    compat = [{
        "kb": it["kb"], "question": it["question"],
        "reference_answer": it["reference_answer"],
        "expected_chunk_ids": it["expected_chunk_ids"],
    } for it in eval_items]
    with open(os.path.join(args.out, "eval_dataset.compat.json"), "w", encoding="utf-8") as f:
        json.dump(compat, f, ensure_ascii=False, indent=1)

    print(f"[build] done -> {args.out}/corpus_chunks.json + eval_dataset.json ({len(eval_items)} questions)")


if __name__ == "__main__":
    main()
