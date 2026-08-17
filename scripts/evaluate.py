"""离线评估 — 单次 LLM 批量判定 RAGAS 4 指标 + 检索近似指标。

用法 (先启动 RAG 服务, LLM Key 走 RAG_LLM_API_KEY):
  python scripts/evaluate.py [--url http://127.0.0.1:8000] [--key xxx] [--dataset tests/eval_dataset.json]

每个样本 2 次 LLM 调用 (1 次问答 + 1 次 RAGAS 判定)。
产出: data/eval-summary.json — 前端 GET /api/v1/qa/eval-summary 展示。

RAGAS 指标 (单次 LLM 批量判定, 0-1):
- faithfulness:        答案事实能否由检索上下文支持
- context_precision:   检索上下文相关 chunk 占比 + 排名
- context_recall:      标注答案关键信息被检索覆盖比例
- answer_relevancy:    答案是否切题完整
"""

import argparse
import asyncio
import datetime
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import jieba

from engines.embedding.embedder import EmbeddingService

SIM_THRESHOLD = 0.5
LLM_BASE = os.environ.get("RAG_LLM_BASE_URL", "https://tokenhub.itcast.cn/v1")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "deepseek-v4-flash")

RAGAS_PROMPT = """你是严谨的 RAG 评估器。基于检索上下文评估回答质量，逐项打分 0.0-1.0（保留 2 位小数）。

问题: {question}

检索上下文:
{context}

模型回答:
{answer}

标注参考答案:
{reference}

评估四项:
1. faithfulness — 模型回答的每个事实能否由检索上下文支持？支持的占比。
2. context_precision — 检索上下文里与问题相关的片段占比，且相关片段是否靠前（排名加权）？
3. context_recall — 标注参考答案中的关键信息有多少被检索上下文覆盖？
4. answer_relevancy — 模型回答是否切题、完整回答问题的核心？

只输出 JSON，不要任何其他文字：
{{"faithfulness":0.0,"context_precision":0.0,"context_recall":0.0,"answer_relevancy":0.0}}
"""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


RAGAS_KEYS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")


def _extract_ragas_json(text: str) -> dict | None:
    """从文本提取含 4 个指标键的 JSON 对象。

    DeepSeek reasoning 模型把推理写进 reasoning_content, 最终 JSON 常夹杂其中。
    逐个尝试简单 JSON 对象, 取含全部 4 键的第一个。
    """
    for cand in re.findall(r"\{[^{}]*\}", text):
        try:
            d = json.loads(cand)
            if all(k in d for k in RAGAS_KEYS):
                return d
        except json.JSONDecodeError:
            continue
    return None


def _llm_ragas(question: str, contexts: list[str], answer: str, reference: str, llm_key: str) -> dict | None:
    """单次 LLM 调用批量判定 4 个 RAGAS 指标。失败返回 None。"""
    if not llm_key:
        return None
    prompt = RAGAS_PROMPT.format(
        question=question[:500],
        context="\n".join(f"[{i + 1}] {c[:400]}" for i, c in enumerate(contexts)) or "(空)",
        answer=answer[:800] or "(空)",
        reference=reference[:500] or "(空)",
    )
    try:
        r = httpx.post(
            f"{LLM_BASE}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 4000,
            },
            headers={"Authorization": f"Bearer {llm_key}"},
            timeout=60,
            trust_env=False,
        )  # 禁系统代理 (DevSidecar), 与 llm_client 一致
        r.raise_for_status()
        data = r.json()
        if isinstance(data, str):
            data = json.loads(data)  # TokenHub 响应外层是 JSON 字符串 (双重编码)
        message = data["choices"][0]["message"]
        # reasoning 模型推理可能吃满 token 导致 content 空 → 从 reasoning_content 兜底提取
        content = message.get("content") or message.get("reasoning_content") or ""
        ragas = _extract_ragas_json(content)
        if not ragas:
            return None
        return {k: round(float(ragas.get(k, 0)), 4) for k in RAGAS_KEYS}
    except Exception as e:  # noqa: BLE001 — RAGAS 判定失败仅跳过该指标
        print(f"  [RAGAS 判定失败] {str(e)[:80]}")
        return None


async def _evaluate_one(client, url, key, item, embed, llm_key) -> dict:
    headers = {"X-API-Key": key} if key else {}
    r = await client.post(
        f"{url}/api/v1/qa/ask", json={"question": item["question"], "top_k": 5}, headers=headers, timeout=60
    )
    d = r.json()
    answer = d.get("answer", "")
    sources = [s.get("content", "") for s in d.get("sources", [])][:5]
    ref = item.get("reference_answer", "")

    # 检索近似指标 (embedding, 快)
    ref_emb = embed.embed_query(ref)
    rel = [1 for s in sources if s and _cosine(ref_emb, embed.embed_query(s[:512])) >= SIM_THRESHOLD]
    recall = len(rel) / len(sources) if sources else 0.0
    acc = max(0.0, _cosine(embed.embed_query(answer[:512]), ref_emb)) if answer else 0.0
    ans_tokens = {t for t in jieba.cut(answer) if len(t.strip()) >= 2}
    ctx_tokens = set()
    for s in sources:
        ctx_tokens |= {t for t in jieba.cut(s) if len(t.strip()) >= 2}
    faith = len(ans_tokens & ctx_tokens) / len(ans_tokens) if ans_tokens else 0.0

    # 标准 RAGAS 4 指标 (单次 LLM 批量判定)
    ragas = _llm_ragas(item["question"], sources, answer, ref, llm_key)

    return {
        "question": item["question"],
        "kb": item.get("kb", ""),
        "accuracy": round(acc, 4),
        "recall": round(recall, 4),
        "hallucination_rate": round(1.0 - faith, 4),
        "result_count": len(sources),
        "ragas": ragas,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default=os.environ.get("RAG_ACCESS_KEY", ""))
    ap.add_argument("--llm-key", default=os.environ.get("RAG_LLM_API_KEY", ""))
    ap.add_argument("--dataset", default="tests/eval_dataset.json")
    args = ap.parse_args()

    if not args.llm_key:
        print("[警告] 未提供 RAG_LLM_API_KEY, RAGAS 指标将跳过 (仅近似指标)")

    def _load_dataset():
        with open(args.dataset, encoding="utf-8") as f:
            return json.load(f)

    dataset = await asyncio.to_thread(_load_dataset)
    embed = EmbeddingService()
    results = []
    async with httpx.AsyncClient(trust_env=False) as client:
        for item in dataset:
            try:
                res = await _evaluate_one(client, args.url, args.key, item, embed, args.llm_key)
                results.append(res)
                rag = res.get("ragas") or {}
                print(
                    f"[{res['kb']}] {res['question'][:20]:<22} "
                    f"faith={rag.get('faithfulness', 0):.2f} cprec={rag.get('context_precision', 0):.2f} "
                    f"crec={rag.get('context_recall', 0):.2f} arel={rag.get('answer_relevancy', 0):.2f} | "
                    f"hal(近似)={res['hallucination_rate']:.2f}"
                )
            except Exception as e:  # noqa: BLE001 — 单条评估失败跳过, 不影响整体
                print(f"[跳过] {item['question'][:20]}: {str(e)[:80]}")

    if not results:
        print("无有效评估结果")
        return

    def _avg(field):
        vals = [r[field] for r in results if field in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    ragas_ok = [r["ragas"] for r in results if r.get("ragas")]
    summary = {
        "accuracy": _avg("accuracy"),
        "recall": _avg("recall"),
        "hallucination_rate": _avg("hallucination_rate"),
        "sample_count": len(results),
        "updated_at": datetime.datetime.now().astimezone().isoformat(),
    }
    if ragas_ok:
        summary["ragas"] = {
            k: round(sum(r[k] for r in ragas_ok) / len(ragas_ok), 4)
            for k in ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
        }
        summary["ragas_sample_count"] = len(ragas_ok)

    os.makedirs("data", exist_ok=True)

    def _dump_summary():
        with open("data/eval-summary.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "cases": results}, f, ensure_ascii=False, indent=2)

    await asyncio.to_thread(_dump_summary)
    print(f"\n=== 整体 (n={len(results)}) ===")
    print(
        f"近似: 准确率 {summary['accuracy'] * 100:.1f}%  召回率 {summary['recall'] * 100:.1f}%  幻觉率 {summary['hallucination_rate'] * 100:.1f}%"
    )
    if summary.get("ragas"):
        r = summary["ragas"]
        print(
            f"RAGAS: faithfulness {r['faithfulness'] * 100:.1f}%  context_precision {r['context_precision'] * 100:.1f}%  "
            f"context_recall {r['context_recall'] * 100:.1f}%  answer_relevancy {r['answer_relevancy'] * 100:.1f}%"
        )
    print("报告已写入 data/eval-summary.json")


if __name__ == "__main__":
    asyncio.run(main())
