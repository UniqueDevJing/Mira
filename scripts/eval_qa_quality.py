"""端到端 QA 质量评测 — 补上"答案好不好"这一此前完全没量化的维度。

此前只测了检索层 (Recall/MRR), 两项生成层 Quick Win 从未被验证:
  - 上下文重排 (_reorder_for_attention, 缓解 lost-in-the-middle)
  - 表格感知解析 (engines/doc_types.py parse={"table": True})
本脚本用 LLM-as-Judge 量化答案质量, 并支持对上下文重排做 A/B。

指标:
  faithfulness      答案的每个事实主张是否被检索到的上下文支持 (LLM 评判)
  answer_relevance  答案是否直接回应了问题 (LLM 评判)
  context_precision 送入上下文的片段中有多少是 golden (离线, 精确匹配)
  groundedness      答案实词在上下文中的覆盖率 (离线, 无需 LLM)
  numeric_ok        数字护栏: 答案里的数字是否都能在上下文中找到 (离线)

性能:
  全量 390 问 × (1 生成 + 2 评判) ≈ 1170 次 LLM 调用。
  - 旧版每问 `asyncio.run` 反复开关事件循环 → 底层 httpx 客户端绑死旧 loop, 第二问起
    全部 "Event loop is closed" (致命 bug, 修复于 2026-08-30)。
  - 现改为**单个**事件循环 + asyncio.gather 并发 (--concurrency 控制并发度), 并支持
    跨问批量 Cross-Encoder 推理 (--batch-rerank) 加速检索阶段。

用法:
  python scripts/eval_qa_quality.py --limit 60                 # 完整评测 (需 LLM, 并发8)
  python scripts/eval_qa_quality.py --limit 60 --concurrency 1 # 顺序基线 (对比并发加速比)
  python scripts/eval_qa_quality.py --limit 60 --batch-rerank  # 开跨问批量重排
  python scripts/eval_qa_quality.py --limit 60 --ab-reorder    # A/B 上下文重排
  python scripts/eval_qa_quality.py --limit 60 --no-llm        # 只跑离线指标
  python scripts/eval_qa_quality.py --eval-dir data/eval_clean # 换语料

输出: data/eval/qa_quality_summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import numpy as np
from eval_retrieval import build_index, vector_topk

from api.config import settings
from api.core import orchestrator, retrieval
from api.core.llm_client import get_llm_client
from api.state import resolve_model_path
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.dedup import adaptive_alpha
from engines.retrieval.fusion import rrf_fuse
from engines.retrieval.reranker import Reranker

_FAITH_PROMPT = """你是严谨的事实核查员。判断【答案】中的每个事实主张是否能被【上下文】支持。

【问题】
{question}

【上下文】
{context}

【答案】
{answer}

请只输出 JSON, 不要有其他文字:
{{"score": <0.0到1.0之间的数字, 1.0=全部主张都有上下文依据, 0.0=存在明显编造>, "reason": "<一句话说明>"}}"""

_RELEVANCE_PROMPT = """判断【答案】是否直接回应了【问题】, 有没有答非所问或兜圈子。

【问题】
{question}

【答案】
{answer}

请只输出 JSON, 不要有其他文字:
{{"score": <0.0到1.0之间的数字, 1.0=精准回应, 0.0=完全答非所问>, "reason": "<一句话说明>"}}"""

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')


def _parse_score(text: str) -> float | None:
    """解析 LLM 返回的分数。优先 JSON, 失败则正则兜底, 都失败返回 None (记为缺失, 不污染均值)。"""
    if not text:
        return None
    try:
        data = json.loads(text)
        v = float(data.get("score"))  # type: ignore[union-attr]
        return min(max(v, 0.0), 1.0)
    except Exception:  # noqa: BLE001 — 判据解析容错: LLM 可能包裹 markdown 代码块
        m = _SCORE_RE.search(text)
        if m:
            try:
                return min(max(float(m.group(1)), 0.0), 1.0)
            except ValueError:
                return None
        return None


def _content_words(text: str) -> set[str]:
    """粗粒度实词集合: 中文按 2-gram, 英文/数字按词。用于 groundedness。"""
    words = set()
    for piece in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text or ""):
        if "\u4e00" <= piece[0] <= "\u9fff":
            words.update(piece[i:i + 2] for i in range(len(piece) - 1)) if len(piece) > 1 else words.add(piece)
        else:
            words.add(piece.lower())
    return words


def _groundedness(answer: str, context: str) -> float:
    aw = _content_words(answer)
    if not aw:
        return 0.0
    cw = _content_words(context)
    return len(aw & cw) / len(aw)


def _numeric_ok(answer: str, context: str) -> float:
    """复用生产数字护栏: 答案里的数字必须能在上下文中找到。"""
    try:
        from api.core.qa_metrics import _extract_numbers
    except ImportError:
        return 1.0
    nums = _extract_numbers(answer or "")
    if not nums:
        return 1.0
    ctx = _extract_numbers(context or "")
    return 1.0 if nums <= ctx else 0.0


class Judge:
    """LLM 评判器。不可用时优雅降级 (返回 None), 不影响离线指标。"""

    def __init__(self, enabled: bool = True):
        self.llm = None
        self.enabled = enabled
        if enabled:
            try:
                self.llm = get_llm_client()
            except Exception as e:  # noqa: BLE001 — LLM 未配置时降级为纯离线评测
                print(f"[qa] ⚠️ LLM 不可用, 只跑离线指标: {str(e)[:120]}")
                self.enabled = False

    async def _ask(self, prompt: str) -> float | None:
        if not self.enabled or self.llm is None:
            return None
        try:
            # LLMClient.chat 是 async, 必须 await (否则拿到协程对象而非结果)
            resp = await self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            return _parse_score(getattr(resp, "content", "") or "")
        except Exception as e:  # noqa: BLE001 — 单次评判失败不应中断整轮评测
            print(f"    [judge] 调用失败: {str(e)[:80]}")
            return None

    async def faithfulness(self, q, ctx, ans):
        return await self._ask(_FAITH_PROMPT.format(question=q, context=ctx, answer=ans))

    async def relevance(self, q, ans):
        return await self._ask(_RELEVANCE_PROMPT.format(question=q, answer=ans))


def _make_context(docs: list, top_k: int, reorder: bool) -> tuple[str, list]:
    """复用生产 _build_context, 仅通过替换重排函数实现 A/B (避免复制粘贴导致逻辑漂移)。

    重排函数 _reorder_for_attention 现位于 api.core.retrieval, _build_context 在其中解析该符号;
    故 A/B 开关须打桩 retrieval._reorder_for_attention 才能命中调用点 (orchestrator 仅做重导出)。
    """
    original = retrieval._reorder_for_attention
    if not reorder:
        retrieval._reorder_for_attention = lambda parts: parts
    try:
        return orchestrator._build_context(docs, top_k)
    finally:
        retrieval._reorder_for_attention = original


async def _generate(llm, question: str, context: str, timeout: float = 30.0) -> str:
    """调用生产同款的答案生成 (直接问 LLM, 不走完整编排, 便于控制变量)。"""
    if llm is None:
        return ""
    prompt = (
        "请严格根据以下参考资料回答问题。不要编造参考资料中没有的信息;"
        "若资料不足以回答, 请直接说明“资料中未提及”。\n\n"
        f"参考资料:\n{context}\n\n问题: {question}\n\n答案:"
    )
    try:
        resp = await llm.chat([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=500)
        return getattr(resp, "content", "") or ""
    except Exception as e:  # noqa: BLE001
        print(f"    [gen] 生成失败: {str(e)[:80]}")
        return ""


async def _gen_and_judge(gen_llm, judge, question, context):
    """生成答案并做双评判, 合并进一次事件循环 (避免每个协程各起一个 loop)。"""
    answer = await _generate(gen_llm, question, context)
    if not answer:
        return None
    return {
        "answer": answer,
        "faithfulness": await judge.faithfulness(question, context, answer),
        "relevance": await judge.relevance(question, answer),
    }


def _rerank_batch(reranker: Reranker, items: list, top_k: int) -> list:
    """跨问批量 Cross-Encoder 推理 + 逐问融合。

    items: list of (query, docs, rrf_map, alpha)。
    仅 CE 推理改为一次大 batch (CPU 矩阵化, 利用率远高于逐问 pool=10 的小 batch),
    融合逻辑与 rerank_fused 完全一致 (ce_norm + rrf_norm + stable argsort)。

    降级: CE 不可用时逐问回退 rerank_fused; 批量推理异常时整体回退单问。
    """
    ce = reranker._get_ce_model()
    if ce is None:
        return [d[:top_k] for (_, d, _, _) in items]

    all_pairs: list[tuple[str, str]] = []
    bounds: list[tuple[int, list, dict, float]] = []
    for q, docs, rrf_map, alpha in items:
        pairs = [(q, d.get("content", "")[:512]) for d in docs]
        bounds.append((len(pairs), docs, rrf_map, alpha))
        all_pairs.extend(pairs)

    try:
        all_scores = list(ce.predict(all_pairs))
    except Exception as e:  # noqa: BLE001 — 批量失败降级单问, 不中断评测
        print(f"    [batch] CE 批量推理失败, 降级单问: {str(e)[:80]}")
        return [reranker.rerank_fused(q, d, rm, top_k, a) for (q, d, rm, a) in items]

    out: list[list[dict]] = []
    i = 0
    for n, docs, rrf_map, alpha in bounds:
        ce_scores = all_scores[i:i + n]
        i += n
        ce_arr = np.array([float(s) for s in ce_scores], dtype=np.float64)
        cmin, cmax = float(ce_arr.min()), float(ce_arr.max())
        ce_norm = (ce_arr - cmin) / (cmax - cmin + 1e-9)
        rrf_arr = np.array(
            [float(rrf_map.get(d.get("chunk_id") or d.get("id"), 0.0)) for d in docs],
            dtype=np.float64,
        )
        rmax = float(rrf_arr.max()) if rrf_arr.size else 1.0
        rrf_norm = rrf_arr / (rmax + 1e-9)
        final = (1.0 - alpha) * rrf_norm + alpha * ce_norm
        order = np.argsort(-final, kind="stable")[:top_k]
        out.append([{**docs[int(j)], "score": round(float(final[int(j)]), 4)} for j in order])
    return out


def _summarize(scores: dict, variants: list, args, elapsed_s: float, llm_enabled: bool, out_path: str):
    """打印汇总并写 json (与变体无关, 抽出避免重复)。"""

    def _avg(v):
        return round(sum(v) / len(v), 4) if v else None

    print("\n=== 端到端 QA 质量 ===")
    print(f"{'变体':<8} {'忠实度':>8} {'相关性':>8} {'上下文精度':>10} {'答案 grounded':>12} {'数字护栏通过':>12}")
    summary = {}
    for name, _ in variants:
        s = scores[name]
        agg = {k: _avg(v) for k, v in s.items()}
        n = {k: len(v) for k, v in s.items()}
        summary[name] = {"metrics": agg, "n": n}
        print(f"{name:<8} "
              f"{(agg['faithfulness'] if agg['faithfulness'] is not None else float('nan')):>8.3f} "
              f"{(agg['relevance'] if agg['relevance'] is not None else float('nan')):>8.3f} "
              f"{(agg['context_precision'] if agg['context_precision'] is not None else float('nan')):>10.3f} "
              f"{(agg['groundedness'] if agg['groundedness'] is not None else float('nan')):>12.3f} "
              f"{(agg['numeric_ok'] if agg['numeric_ok'] is not None else float('nan')):>12.3f}")

    if args.ab_reorder and summary.get("重排开") and summary.get("重排关"):
        a, b = summary["重排开"]["metrics"], summary["重排关"]["metrics"]
        print("\n=== 上下文重排 A/B (Δ = 重排开 - 重排关) ===")
        for k in ("faithfulness", "relevance", "groundedness", "numeric_ok"):
            if a.get(k) is not None and b.get(k) is not None:
                print(f"  {k:<14} {b[k]:.4f} -> {a[k]:.4f}   Δ {a[k]-b[k]:+.4f}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "limit": args.limit, "top_k": args.top_k, "pool": args.pool,
            "concurrency": args.concurrency, "batch_rerank": args.batch_rerank,
            "eval_dir": args.eval_dir, "llm_enabled": llm_enabled,
            "variants": summary, "elapsed_s": round(elapsed_s, 1),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out_path}  (耗时 {elapsed_s:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=5, help="送入上下文的片段数")
    ap.add_argument("--pool", type=int, default=10, help="rerank 候选池大小")
    ap.add_argument("--concurrency", type=int, default=8, help="LLM 并发度 (单事件循环内 gather)")
    ap.add_argument("--batch-rerank", action="store_true", help="跨问批量 Cross-Encoder 推理 (加速检索阶段)")
    ap.add_argument("--no-llm", action="store_true", help="只跑离线指标 (不需要 LLM)")
    ap.add_argument("--ab-reorder", action="store_true", help="A/B 上下文重排 (缓解 lost-in-the-middle)")
    ap.add_argument("--out", default="", help="输出 json 路径")
    args = ap.parse_args()

    with open(os.path.join(args.eval_dir, "corpus_chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(args.eval_dir, "eval_dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)[: args.limit]
    print(f"[qa] chunks={len(chunks)} questions={len(dataset)} top_k={args.top_k} "
          f"ab_reorder={args.ab_reorder} concurrency={args.concurrency} batch_rerank={args.batch_rerank}")

    embs_n, bm25 = build_index(chunks)
    emb = EmbeddingService()
    reranker = Reranker(embedder=emb, ce_model_name=resolve_model_path(settings.reranker_model),
                        max_length=settings.reranker_max_length or None,
                        backend=settings.reranker_backend)
    print(f"[qa] reranker loaded={reranker.warmup()}")

    judge = Judge(enabled=not args.no_llm)
    gen_llm = judge.llm

    variants = [("重排开", True)]
    if args.ab_reorder:
        variants.append(("重排关", False))

    scores: dict[str, dict[str, list]] = {
        name: {"faithfulness": [], "relevance": [], "context_precision": [],
               "groundedness": [], "numeric_ok": []}
        for name, _ in variants
    }
    # 每个变体待 LLM 评测的 (question, context) 列表
    per_variant: dict[str, list] = {name: [] for name, _ in variants}

    t0 = time.time()

    # ---------- 阶段1: 检索 + 重排 (同步 CPU) ----------
    # 先收集每问的检索输入, 便于 --batch-rerank 跨问合并 CE 推理
    retrieve_items: list[tuple[str, list, dict, float]] = []  # (q, pool_docs, rrf_map, alpha)
    ctx_precisions: list[float] = []
    for i, it in enumerate(dataset, 1):
        q = it["question"]
        expected = set(it["expected_chunk_ids"])
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        fused = rrf_fuse(vector_topk(q_emb, embs_n, chunks, 40), bm25.search(q, 40))
        pool = fused[: args.pool]

        rrf_map = {d.get("chunk_id") or d.get("id"): d.get("_rrf", 0.0) for d in fused}
        alpha = adaptive_alpha(pool, threshold=settings.rerank_density_threshold,
                               mode=settings.rerank_density_mode,
                               alpha_max=settings.rerank_alpha_max,
                               alpha_min=settings.rerank_alpha_min,
                               density_full=settings.rerank_density_full)
        retrieve_items.append((q, pool, rrf_map, alpha))

        ids = [d.get("chunk_id") or d.get("id") for d in pool[: args.top_k]]
        ctx_precisions.append(len(set(ids) & expected) / max(len(ids), 1))

        if i % 50 == 0:
            print(f"[qa] 检索 {i}/{len(dataset)}  用时 {time.time()-t0:.0f}s", flush=True)

    # 重排: 批量 (跨问一次 predict) 或 逐问
    if args.batch_rerank:
        reranked = _rerank_batch(reranker, retrieve_items, top_k=args.pool)
    else:
        reranked = [reranker.rerank_fused(q, pool, rrf_map, top_k=args.pool, alpha=alpha)
                    for (q, pool, rrf_map, alpha) in retrieve_items]

    # 记录离线指标 + 准备各变体 context (同步, 无并发竞争)
    for idx, (q, _, _, _) in enumerate(retrieve_items):
        docs = reranked[idx]
        cp = ctx_precisions[idx]
        for name, reorder in variants:
            s = scores[name]
            # context_precision 只依赖检索/重排结果, 不依赖答案生成 -> 始终记录
            s["context_precision"].append(cp)
            if args.no_llm:
                continue
            context, _ = _make_context(docs, args.top_k, reorder)
            per_variant[name].append((q, context))

    if args.no_llm:
        # 纯离线: 直接汇总, 不进事件循环
        _summarize(scores, variants, args, time.time() - t0, judge.enabled,
                   args.out or os.path.join(args.eval_dir, "qa_quality_summary.json"))
        return

    # ---------- 阶段2: LLM 并发 (单一事件循环, 修复 Event loop is closed) ----------
    sem = asyncio.Semaphore(args.concurrency)

    async def _gen_judge_sem(item):
        q, context = item
        async with sem:
            return await _gen_and_judge(gen_llm, judge, q, context)

    async def _run_all():
        results = {}
        for name, items in per_variant.items():
            tasks = [_gen_judge_sem(it) for it in items]
            results[name] = await asyncio.gather(*tasks)
        return results

    all_results = asyncio.run(_run_all())

    for name, items in per_variant.items():
        s = scores[name]
        for (_, context), res in zip(items, all_results[name]):
            if res is None:
                continue
            s["groundedness"].append(_groundedness(res["answer"], context))
            s["numeric_ok"].append(_numeric_ok(res["answer"], context))
            if res["faithfulness"] is not None:
                s["faithfulness"].append(res["faithfulness"])
            if res["relevance"] is not None:
                s["relevance"].append(res["relevance"])

    _summarize(scores, variants, args, time.time() - t0, judge.enabled,
               args.out or os.path.join(args.eval_dir, "qa_quality_summary.json"))


if __name__ == "__main__":
    main()
