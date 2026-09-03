"""实体歧义型多文档评测集 — 构造"同名异义"检索压力测试。

动机：
  CRUD-RAG 原评测集每问只挂自己源文档的 chunk，BM25/向量已能轻松命中，压不出
  Cross-Encoder rerank 的"语义辨别力"。真实难点是：同名实体出现在不同主题的文档
  （如两个都叫"张伟"的人、两个同名机构），检索时二者都被排很高，只能靠语义把
  "真正对应"的那篇挑出来。

构造（纯离线，复用已嵌入语料，无需重嵌）：
  1. jieba 抽每 chunk 的专名实体（人名/机构/地名/其他专名），建倒排索引
  2. 对每个出现在 >=2 个文档的实体 E，对每个含 E 的源 chunk ca：
     - query 只提 E（不提消歧短语），保证检索端 ca/cb 都排很高（最大化竞争）
     - 竞争者 cb：不同文档、含 E、且在 向量+BM25 融合候选池 top-15 内、
       且与 ca 的"非 E 实体重叠度"最低（= 同名但不同主题，避免近重复新闻）
     - 从 ca 挑 cb 没有的消歧短语 P 作为 golden 的语义依据（仅元数据 + 答案）
  3. 自过滤：cb 必须真实进入融合候选池（rerank 实际输入），且主题确实不同
  4. 复用 build_eval_set 的 BM25 后处理：填 bm25_golden_rank / hard / hard_negatives

输出 data/eval/entity_ambig_dataset.json（同 schema，可被 eval_retrieval.py --dataset 直接吃）。

用法:
  python scripts/build_entity_ambig_set.py [--eval-dir data/eval] [--target 200] [--max-pairs-per-entity 3]
"""
import argparse
import json
import os
import random
import re
import sys
import time

# 离线优先：避免 sentence-transformers 联网校验（本环境 SSL 不通）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jieba.posseg as pseg
import numpy as np

from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.fusion import rrf_fuse

# 过于通用的实体（几乎每篇都有，无区分度），直接丢弃
_GENERIC = {
    "中国", "北京", "上海", "广州", "深圳", "美国", "日本", "公司", "集团", "企业",
    "项目", "活动", "大学", "学校", "学生", "记者", "工作", "发展", "社会", "国家",
    "政府", "技术", "数据", "服务", "用户", "系统", "平台", "医院", "城市", "地区",
    "部门", "会议", "研究", "团队", "产品", "市场", "经济", "文化", "教育", "社区",
    "人员", "组织", "中心", "委员会", "协会", "基金会", "研究院", "实验室", "股份",
    "有限", "责任", "科技", "网络", "信息", "时间", "问题", "情况", "方面", "结果",
}
_YEAR_RE = re.compile(r"\d{4}年")
_SENT_SPLIT = re.compile(r"[。！？\n;；]")


def _entities(text: str) -> list[tuple[str, str]]:
    """返回 (surface, flag) 列表，仅保留专名与年份。

    注意：jieba 会把 "2023年" 切成 "2023"+"年" 两个 token，单 token 正则匹配不到年份，
    因此年份必须从【原文】用正则抽取（而非依赖分词结果）。年份是最强消歧符，必须可用。
    """
    out = []
    seen = set()
    for w, flag in pseg.cut(text or ""):
        w = w.strip()
        if not w:
            continue
        if flag in ("nr", "nt", "ns", "nz") and len(w) >= 2 and w not in _GENERIC and w not in seen:
            out.append((w, flag))
            seen.add(w)
    for y in _YEAR_RE.findall(text or ""):
        if y not in seen:
            out.append((y, "year"))
            seen.add(y)
    return out


def _sent_with(text: str, ent: str) -> str:
    for seg in _SENT_SPLIT.split(text or ""):
        if ent in seg:
            return seg.strip()[:150]
    return (text or "").strip()[:150]


def _pick_disambiguator(ent: str, ca_ents: set, ca_years: set,
                        cb_text: str, cb_ents: set, cb_years: set):
    """从 ca 选一个 cb 的【原文】里没有的消歧短语：优先年份，其次专名。

    必须用 cb 原始正文（而非 cb 实体集合）判定缺失 —— 否则像"两国"这种
    未被 jieba 标为专名的短语会漏判，导致 golden 的消歧短语其实也出现在 trap 里。
    """
    for y in sorted(ca_years):
        if y not in cb_years and y not in cb_text:
            return y
    for pe in sorted(ca_ents - {ent}):
        if pe in _GENERIC:
            continue
        if pe not in cb_ents and pe not in cb_text:
            return pe
    return None


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--target", type=int, default=200, help="目标问题数")
    ap.add_argument("--max-pairs-per-entity", type=int, default=3)
    ap.add_argument("--pool-k", type=int, default=15, help="融合候选池判定阈值(rerank 真实输入)")
    ap.add_argument("--max-ent-overlap", type=float, default=0.5,
                    help="ca 与 cb 非-E 实体 Jaccard 上限(超过视为近重复同主题, 跳过)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    with open(os.path.join(args.eval_dir, "corpus_chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[build] corpus chunks={len(chunks)}")

    t0 = time.time()
    emb = EmbeddingService()
    embs = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.clip(norms, 1e-9, None)
    print(f"[build] vector index built in {time.time()-t0:.1f}s")

    # 1. 实体倒排 + 每 chunk 实体/年份缓存
    inv: dict[str, dict[str, list[int]]] = {}
    chunk_entities: list[set] = [set() for _ in chunks]
    year_by_chunk: list[set] = [set() for _ in chunks]
    for i, c in enumerate(chunks):
        ents = _entities(c["content"])
        chunk_entities[i] = {e for e, _ in ents}
        year_by_chunk[i] = {e for e, fl in ents if fl == "year"}
        for e, _ in ents:
            inv.setdefault(e, {}).setdefault(c["doc_id"], []).append(i)
    print(f"[build] distinct entities={len(inv)}")

    cand = [e for e, docs in inv.items() if 2 <= len(docs) <= 40]
    cand.sort(key=lambda e: len(inv[e]))  # 优先稀有实体（更具区分度）
    print(f"[build] candidate entities (2~40 docs)={len(cand)}")

    bm = Bm25Index()
    bm.add_documents([{"id": c["chunk_id"], "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"]} for c in chunks])

    # query -> 融合候选池 缓存（同一实体只算一次）
    fused_cache: dict[str, list[dict]] = {}

    def get_fused(ent: str) -> list[dict]:
        if ent in fused_cache:
            return fused_cache[ent]
        q = f"{ent}的相关具体情况是怎样的？"
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        sims = embs_n @ q_emb
        v_docs = []
        for idx in np.argsort(-sims)[:40]:
            c = chunks[int(idx)]
            v_docs.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"], "score": float(sims[idx])})
        b_docs = bm.search(q, 40)
        fused = rrf_fuse(v_docs, b_docs)
        fused_cache[ent] = fused
        return fused

    items: list[dict] = []
    seen_pairs = set()

    for ent in cand:
        if len(items) >= args.target:
            break
        docs = inv[ent]
        doc_ids = list(docs.keys())
        pairs_made = 0
        random.shuffle(doc_ids)
        fused = get_fused(ent)
        pool_ids = {d.get("chunk_id") for d in fused[:args.pool_k]}

        for i in range(len(doc_ids)):
            if pairs_made >= args.max_pairs_per_entity:
                break
            ca_idx = max(docs[doc_ids[i]], key=lambda idx: len(chunks[idx]["content"]))
            ca_text = chunks[ca_idx]["content"]
            ca_ents = chunk_entities[ca_idx] - {ent}
            ca_years = year_by_chunk[ca_idx]

            # 竞争者候选：不同文档、含 E、且在融合候选池内
            cands = []
            for ob_did in doc_ids:
                if ob_did == doc_ids[i]:
                    continue
                for cb_idx in docs[ob_did]:
                    cb_cid = chunks[cb_idx]["chunk_id"]
                    if cb_cid not in pool_ids:
                        continue
                    cands.append(cb_idx)
            if not cands:
                continue

            # 选与 ca 主题最不同（非-E 实体 Jaccard 最低）者；要求 < 上限（避免近重复同主题）
            best_cb = None
            best_overlap = 1.1
            best_p = None
            for cb_idx in cands:
                cb_ents = chunk_entities[cb_idx] - {ent}
                ov = _jaccard(ca_ents, cb_ents)
                if ov >= args.max_ent_overlap:
                    continue
                p = _pick_disambiguator(ent, ca_ents, ca_years,
                                        chunks[cb_idx]["content"],
                                        cb_ents, year_by_chunk[cb_idx])
                if p is None:
                    continue
                # 越小越不同主题；并列时选实体更丰富的 cb（更有区分信息）
                score = ov - 0.01 * len(cb_ents)
                if score < best_overlap:
                    best_overlap, best_cb, best_p = score, cb_idx, p
            if best_cb is None:
                continue

            pair_key = (ent, ca_idx, best_cb)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            q = f"{ent}的相关具体情况是怎样的？"
            ref = _sent_with(ca_text, ent) or ca_text[:150]
            items.append({
                "id": f"entamb_{len(items):05d}",
                "kb": "entity_ambiguity",
                "category": "disambig",
                "question": q,
                "reference_answer": ref,
                "golden_doc_ids": [chunks[ca_idx]["doc_id"]],
                "expected_chunk_ids": [chunks[ca_idx]["chunk_id"]],
                "entity": ent,
                "disambiguator": best_p,
                "competitor_chunk_ids": [chunks[best_cb]["chunk_id"]],
                "ca_cb_cosine": round(float(embs_n[ca_idx] @ embs_n[best_cb]), 4),
                "ca_cb_ent_overlap": round(best_overlap, 3),
            })
            pairs_made += 1

    print(f"[build] raw ambiguous questions={len(items)}")

    # 2. 后处理：golden 排名 / hard / hard_negatives（基于融合序）
    kept = []
    for it in items:
        gold = set(it["expected_chunk_ids"])
        fused = get_fused(it["entity"])
        gold_rank = None
        for r, h in enumerate(fused[:10], 1):
            if h["chunk_id"] in gold:
                gold_rank = r
                break
        neg = [h["chunk_id"] for h in fused[:10] if h["chunk_id"] not in gold][:10]
        it["bm25_golden_rank"] = gold_rank
        it["hard_negatives"] = neg
        it["hard"] = (gold_rank is None or gold_rank > 10)
        kept.append(it)
    final = kept
    print(f"[build] final questions={len(final)}")

    if final:
        ovs = [it["ca_cb_ent_overlap"] for it in final]
        sims = [it["ca_cb_cosine"] for it in final]
        print(f"[build] ca-cb cosine : min={min(sims):.3f} max={max(sims):.3f} mean={sum(sims)/len(sims):.3f}")
        print(f"[build] ca-cb ent-overlap: min={min(ovs):.3f} max={max(ovs):.3f} mean={sum(ovs)/len(ovs):.3f}")

    os.makedirs(args.eval_dir, exist_ok=True)
    out_path = os.path.join(args.eval_dir, "entity_ambig_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=1)
    compat = [{
        "kb": it["kb"], "question": it["question"],
        "reference_answer": it["reference_answer"],
        "expected_chunk_ids": it["expected_chunk_ids"],
    } for it in final]
    with open(os.path.join(args.eval_dir, "entity_ambig_dataset.compat.json"), "w", encoding="utf-8") as f:
        json.dump(compat, f, ensure_ascii=False, indent=1)
    print(f"[build] done -> {out_path} ({len(final)} questions)")


if __name__ == "__main__":
    main()
