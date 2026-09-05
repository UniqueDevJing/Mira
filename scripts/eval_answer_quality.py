"""端到端答案质量评测 — LLM-as-judge 量化准确率/幻觉率。

链路: 评测集问题 → 与线上一致的检索(复用 eval_retrieval 的向量+BM25+RRF+重排)
      → LLM 生成(qwen-plus) → LLM-as-judge 对比标准答案判分。

判分: correct / partial / incorrect / refusal + hallucination(无中生有的具体事实)。
输出: data/eval/answer_quality_summary.json

用法:
  python scripts/eval_answer_quality.py [--limit 0] [--concurrency 6] [--judge-model qwen-plus]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import eval_retrieval as er  # noqa: E402 — 复用同款检索链路

GEN_SYS = (
    "你是企业知识库问答助手。仅依据【知识库片段】回答【问题】; "
    "片段不足以回答时回复\"知识库中未找到相关信息\", 不得编造。回答简洁(100字内), 保留具体数字与名称。"
)
JUDGE_SYS = (
    "你是严格的问答质量评审员。对比【标准答案】与【系统回答】, 只输出 JSON: "
    '{"verdict":"correct|partial|incorrect|refusal","hallucination":true|false,"reason":"15字内"}。'
    "判定规则: correct=事实与标准答案一致(表述可不同); partial=答对部分要点; "
    "incorrect=矛盾或答非所问; refusal=系统拒答/称未找到。"
    "hallucination=true 仅当系统回答含标准答案及常识之外的具体事实断言(数字/名称/日期)。"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--limit", type=int, default=0, help="评测题数上限 (0=全量 390)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--judge-model", default="qwen-plus")
    args = ap.parse_args()

    from api.core.llm_client import get_sync_llm_client

    gen_client = get_sync_llm_client()
    judge_client = get_sync_llm_client()
    judge_client.model = args.judge_model
    if not getattr(gen_client, "api_key", None):
        print("[eval] 未配置 LLM API Key, 无法评测")
        return

    eval_dir = args.eval_dir
    dataset = json.load(open(os.path.join(eval_dir, "eval_dataset.json"), encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    chunks = json.load(open(os.path.join(eval_dir, "corpus_chunks.json"), encoding="utf-8"))
    print(f"[eval] 题数 {len(dataset)}, 语料块 {len(chunks)}")

    embs_n, bm25 = er.build_index(chunks)

    def retrieve(question: str) -> list[dict]:
        q_emb = er.EmbeddingService().embed_query(question) if hasattr(er, "EmbeddingService") else None
        # 复用 eval_retrieval 的 embedder 获取方式
        from engines.embedding.embedder import EmbeddingService

        q_vec = EmbeddingService().embed_query(question)
        import numpy as np

        q_n = np.array(q_vec, dtype=np.float32)
        q_n = q_n / max(np.linalg.norm(q_n), 1e-9)
        vec_hits = er.vector_topk(q_n, embs_n, chunks, 20)
        bm25_hits = bm25.search(question, top_k=20)
        fused = er.fuse(vec_hits, bm25_hits, k=20)
        reranked = er.Reranker().rerank(question, fused, top_k=5) if er.Reranker else fused[:5]
        return reranked[:5]

    def generate(question: str, ctx: list[dict]) -> str:
        ctx_text = "\n\n".join(f"[片段{i+1}] {(c.get('content') or '')[:400]}" for i, c in enumerate(ctx))
        resp = gen_client.chat(
            [{"role": "system", "content": GEN_SYS},
             {"role": "user", "content": f"【知识库片段】\n{ctx_text}\n\n【问题】{question}"}],
            temperature=0.1, max_tokens=300,
        )
        return (getattr(resp, "content", "") or "").strip()

    def judge(question: str, ref: str, answer: str) -> dict:
        try:
            resp = judge_client.chat(
                [{"role": "system", "content": JUDGE_SYS},
                 {"role": "user", "content": f"【标准答案】{ref}\n\n【系统回答】{answer or '(空)'}"}],
                temperature=0.0, max_tokens=120, json_mode=True,
            )
            raw = (getattr(resp, "content", "") or "").strip()
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001 — 单题评审失败不阻断
            return {"verdict": "judge_error", "hallucination": False, "reason": str(e)[:40]}

    def one(item: dict) -> dict:
        q = item["question"]
        t0 = time.time()
        try:
            ctx = retrieve(q)
            answer = generate(q, ctx)
        except Exception as e:  # noqa: BLE001
            return {**item, "answer": "", "verdict": "error", "hallucination": False, "latency_s": round(time.time() - t0, 1)}
        verdict = judge(q, item.get("reference_answer", ""), answer)
        return {**{k: item.get(k) for k in ("id", "kb", "category")},
                "question": q, "answer": answer, "latency_s": round(time.time() - t0, 1), **verdict}

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, it) for it in dataset]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 50 == 0:
                print(f"[eval] {i}/{len(dataset)} 完成, 耗时 {time.time()-t0:.0f}s")

    n = len(results)
    by = {}
    for r in results:
        by[r["verdict"]] = by.get(r["verdict"], 0) + 1
    halluc = sum(1 for r in results if r.get("hallucination"))
    summary = {
        "sample_count": n,
        "accuracy": round(by.get("correct", 0) / n, 4),
        "partial_rate": round(by.get("partial", 0) / n, 4),
        "incorrect_rate": round(by.get("incorrect", 0) / n, 4),
        "refusal_rate": round(by.get("refusal", 0) / n, 4),
        "error_rate": round((by.get("error", 0) + by.get("judge_error", 0)) / n, 4),
        "hallucination_rate": round(halluc / n, 4),
        "verdicts": by,
        "avg_latency_s": round(sum(r.get("latency_s", 0) for r in results) / n, 2),
        "judge_model": args.judge_model,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = os.path.join(eval_dir, "answer_quality_summary.json")
    json.dump({"summary": summary, "results": results}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
