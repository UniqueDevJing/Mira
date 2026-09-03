"""low_relevance 幻觉前置守卫 — 误拒/拦截离线量化 (无 LLM, 本地 bge 仅对少量零重合样本)。

背景 (#4 幻觉前置): `_pregeneration_hallucination_guard` 在 top1 分数达标但
question 与检索上下文 **零词重合**(jieba ≥2 字 token 无交集) 时触发 low_relevance 拒答,
防"高分局外片段"幻觉。本脚本量化三种样本组, 为"语义兜底 + 默认开启"提供依据:

  G1-Gold   正样本(检索完美命中 golden): zero-overlap 占比 = 误拒上界
  G2-Hard   负样本(检索跑偏命中 hard_negative): zero-overlap 占比 = 潜在拦截收益
  G3-Para   产品语料口语改写对: zero-overlap 占比 = 改写误拒实证; embedding 相似度可救回比例
  G3-Neg    产品语料无关高分干扰对(应拦截): 同上, 但须保持拦截

对比触发逻辑(θ 扫描):
  逻辑A(纯词重合): overlap==0 → 触发
  逻辑B(语义兜底): overlap==0 且 emb_sim < θ → 触发  (emb_sim 高 = 语义等价改写, 放行)

用法:
  python scripts/eval_preguard.py
输出: 控制台表 + data/eval/preguard_summary.json。
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from api.core.qa_metrics import _word_overlap_faithfulness
from engines.embedding.embedder import EmbeddingService

THETAS = [0.45, 0.50, 0.55, 0.60, 0.65]


def _l2(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def _sim(q_emb: np.ndarray, ctx_vec: np.ndarray) -> float:
    """已归一化点积。ctx_vec 为归一化向量, 失败返回 0。"""
    try:
        if q_emb is None or len(q_emb) == 0 or ctx_vec is None or len(ctx_vec) == 0:
            return 0.0
        return float(np.dot(q_emb, _l2(ctx_vec)))
    except Exception:  # noqa: BLE001
        return 0.0


def _max_sim(q_emb, ctx_embs: list) -> float:
    return max((_sim(q_emb, c) for c in ctx_embs), default=0.0)


def _triggers(overlap: float, sim: float) -> dict:
    """逻辑A(纯词重合)与逻辑B(θ 语义兜底) 的触发判定。"""
    a = 1 if overlap == 0.0 else 0
    return {"A": a, **{f"B@{t}": (1 if (a and sim < t) else 0) for t in THETAS}}


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    t0 = time.time()
    # ── 加载语料 ──────────────────────────────────────────────
    corpus = _load_json("data/eval/corpus_chunks.json")
    cmap = {c["chunk_id"]: c for c in corpus}
    bm25 = {}
    for kb in ("service", "policy", "tech"):
        bm25[kb] = _load_json(f"data/bm25_{kb}.json")["docs"]
    para = _load_json("data/eval/preguard_paraphrase.json")["paraphrase"]

    emb = None  # 惰性加载: 仅零重合样本需要语义相似度

    def get_emb():
        nonlocal emb
        if emb is None:
            emb = EmbeddingService()
        return emb

    def embed_q(q: str):
        try:
            return np.asarray(get_emb().embed_query(q), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] embed_query 失败: {e}")
            return None

    # ── 汇总容器 ──────────────────────────────────────────────
    groups = {k: {"n": 0, "A": 0, "B": {t: 0 for t in THETAS},
                  "zero_sims": [], "ctx_examples": []} for k in
              ("G1-eval390", "G1-entity221", "G2-hard", "G3-para", "G3-neg")}

    def accum(g, q, ctx_texts, ctx_embs, ctx_preview=None):
        ov = _word_overlap_faithfulness(q, ctx_texts)
        groups[g]["n"] += 1
        if ov == 0.0:
            sim = _max_sim(embed_q(q), ctx_embs) if ctx_embs else 0.0
            tr = _triggers(ov, sim)
            groups[g]["A"] += tr["A"]
            for t in THETAS:
                groups[g]["B"][t] += tr[f"B@{t}"]
            groups[g]["zero_sims"].append(sim)
            if ctx_preview and len(groups[g]["ctx_examples"]) < 3:
                groups[g]["ctx_examples"].append({"q": q, "ctx": ctx_preview[:90], "sim": round(sim, 3)})

    # ── G1 正样本: question vs golden 上下文 ───────────────────
    for ds_name, gname in (("data/eval/eval_dataset.json", "G1-eval390"),
                           ("data/eval/entity_ambig_dataset.json", "G1-entity221")):
        ds = _load_json(ds_name)
        for it in ds:
            ids = it.get("expected_chunk_ids") or []
            ctx_texts, ctx_embs = [], []
            for cid in ids:
                c = cmap.get(cid)
                if c:
                    ctx_texts.append(c["content"])
                    ctx_embs.append(np.asarray(c["embedding"], dtype=np.float32))
            accum(gname, it["question"], ctx_texts, ctx_embs,
                  ctx_preview=(ctx_texts[0] if ctx_texts else ""))

    # ── G2 负样本: question vs hard_negative 干扰块 ────────────
    ds = _load_json("data/eval/eval_dataset.json")
    for it in ds:
        ctx_texts, ctx_embs = [], []
        for hid in (it.get("hard_negatives") or [])[:3]:
            c = cmap.get(hid)
            if c:
                ctx_texts.append(c["content"])
                ctx_embs.append(np.asarray(c["embedding"], dtype=np.float32))
        if ctx_texts:
            accum("G2-hard", it["question"], ctx_texts, ctx_embs,
                  ctx_preview=ctx_texts[0])

    # ── G3 改写/无关压力对 ────────────────────────────────────
    for p in para:
        docs = bm25[p["kb"]]
        if p["idx"] >= len(docs):
            print(f"  [warn] {p['kb']} idx {p['idx']} 越界, 跳过")
            continue
        ctx_text = docs[p["idx"]].get("content") or ""
        if not ctx_text.strip():
            print(f"  [warn] {p['kb']} idx {p['idx']} 空内容, 跳过")
            continue
        # bm25 快照无 embedding → 现场嵌入 ctx
        ctx_emb = None
        try:
            ctx_emb = np.asarray(get_emb().embed_query(ctx_text[:2000]), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] ctx embed 失败: {e}")
        g = "G3-neg" if p.get("negative") else "G3-para"
        accum(g, p["question"], [ctx_text], [ctx_emb] if ctx_emb is not None else [],
              ctx_preview=ctx_text)

    # ── 输出 ──────────────────────────────────────────────────
    def pct(x, d):
        return f"{(100.0 * x / d):6.1f}%" if d else "   n/a"

    print("=" * 80)
    print("low_relevance 前置守卫离线评估 — 触发逻辑 A(纯词重合) vs B(θ: 零重合 且 语义相似度<θ)")
    print("=" * 80)
    print(f"{'组':<13}{'n':>5}{'A触发':>8}{'B@.45':>8}{'B@.50':>8}{'B@.55':>8}{'B@.60':>8}{'B@.65':>8}  零重合样本语义相似度")
    for g, v in groups.items():
        zs = v["zero_sims"]
        sim_desc = (f"med={np.median(zs):.2f} ≥0.50:{(np.mean(np.array(zs) >= 0.50) * 100):.0f}%" if zs else "-")
        print(f"{g:<13}{v['n']:>5}{pct(v['A'], v['n']):>8}"
              + "".join(pct(v['B'][t], v['n']).rjust(8) for t in THETAS)
              + f"  {sim_desc} (n={len(zs)})")
    print("-" * 80)
    print("解读: A触发率 = 零重合率。G1(应放行)越低越好; G2/G3-neg(应拦截)越高越好;")
    print("      G3-para(等价改写)越低越好。B@θ 相对 A 的下降 = 语义相似度救回的误拒。")
    for g, v in groups.items():
        if v["ctx_examples"]:
            print(f"\n[{g}] 零重合样本示例:")
            for e in v["ctx_examples"]:
                print(f"  q  : {e['q'][:60]}")
                print(f"  ctx: {e['ctx']}")
                print(f"  sim: {e['sim']:.3f}")

    summary = {"_note": "A=纯词重合触发; B@t=零重合且 emb_sim<t。G1 误拒上界 / G2+G3-neg 拦截 / G3-para 改写误拒。",
               "thetas": THETAS,
               "groups": {g: {"n": v["n"], "A": v["A"],
                              "B": {str(t): v["B"][t] for t in THETAS},
                              "zero_n": len(v["zero_sims"]),
                              "zero_sim_median": round(float(np.median(v["zero_sims"])), 3) if v["zero_sims"] else None,
                              "zero_sim_ge050": round(float(np.mean(np.array(v["zero_sims"]) >= 0.50)), 3) if v["zero_sims"] else None}
                          for g, v in groups.items()}}
    with open("data/eval/preguard_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入 data/eval/preguard_summary.json  (耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
