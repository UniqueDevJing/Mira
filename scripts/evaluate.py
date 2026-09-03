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
import hashlib
import json
import os
import re
import sys

# 阻断 sentence_transformers / transformers 向 HF 发起网络探测（SSL 超时污染 async client）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from engines.common.quality import cosine as _cosine_shared
from engines.common.quality import is_refusal, word_overlap_faithfulness

# pydantic Settings 的 env_file 只填进 settings 对象、不写入 os.environ;
# 本脚本用 os.environ.get("RAG_LLM_API_KEY"/"RAG_LLM_BASE_URL"/...) 读 LLM 配置,
# 必须显式 load_dotenv 才能从 .env 注入 (否则 RAGAS 判定永远跳过)。
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001, S110 — dotenv 缺失则退回纯环境变量 / 默认值
    pass

from engines.embedding.embedder import EmbeddingService

SIM_THRESHOLD = 0.5
LLM_BASE = os.environ.get("RAG_LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "deepseek-chat")

RAGAS_PROMPT = """你是 RAG 评估器。基于检索上下文对回答逐项打分 0.0-1.0（保留 2 位小数）。

问题: {question}

检索上下文:
{context}

模型回答:
{answer}

参考答案:
{reference}

只输出 JSON:
{{"faithfulness":0.0,"context_precision":0.0,"context_recall":0.0,"answer_relevancy":0.0}}
"""


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度 — 统一实现见 engines/common/quality.py (OPT-C4)。"""
    return _cosine_shared(a, b)


def _first_relevant_rank(source_dicts: list[dict], expected_ids: set[str]) -> int:
    """排序 sources 中首个命中 expected 的排名 (1-based); 无命中返回 0。

    用于 MRR (Mean Reciprocal Rank): 排序质量的核心指标, 报告 P2#10 要求补全
    (此前仅用 context_precision 近似)。sources 为检索返回的**有序**列表。
    """
    for rank, s in enumerate(source_dicts, start=1):
        sids = set()
        if s.get("chunk_id"):
            sids.add(s["chunk_id"])
        elif s.get("chunk_ids"):
            sids.update(s["chunk_ids"])
        if sids & expected_ids:
            return rank
    return 0


RAGAS_KEYS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")

# 拒答识别 (护栏/置信度下限触发的标准拒答话术 + LLM 主动声明无法回答)。
# 拒答不是幻觉: 把它混进 hallucination_rate 会虚高 (8 条拒答可把幻觉率从 1.5% 抬到 9.3%),
# 且掩盖真正的问题 —— 过度拒答 (检索已命中却拒答) 与路由失败 (检索根本没命中才拒答)。
# 拒答识别统一走 engines/common/quality.is_refusal (OPT-C4, 与运行时共享单一实现)。

# 运行时缓存: question_hash -> ragas_result，避免同问题重复判分。
_ragas_cache: dict[str, dict] = {}

# chunk_id -> 完整原文 (lancedb 三表全量, 共 <100 chunk, 启动时一次性加载)。
# 关键: /api/v1/qa/ask 的 sources[].content 只是给前端面板显示的 **200 字片段**,
# 而 LLM 实际看到的上下文是每段最长 800 字的父块合并文本。若拿 200 字片段去判
# faithfulness, 答案中超出片段的内容会被误判为幻觉 → faithfulness 假 0。
# 因此判定上下文必须用 chunk_id 反查的完整原文。
_chunk_full_map: dict[str, str] | None = None


def _load_chunk_full_map(db_path: str = "lancedb_data") -> dict[str, str]:
    global _chunk_full_map
    if _chunk_full_map is not None:
        return _chunk_full_map
    mapping: dict[str, str] = {}
    try:
        import lancedb

        db = lancedb.connect(db_path)
        for tn in ("rag_policy", "rag_service", "rag_tech"):
            try:
                for row in db.open_table(tn).search().limit(20000).to_list():
                    cid = row.get("id")
                    if cid and cid not in mapping:
                        mapping[cid] = (row.get("content") or "").strip()
            except Exception:  # noqa: BLE001, S112 — 单表失败不影响其余表
                continue
    except Exception as e:  # noqa: BLE001 — lancedb 不可用时退回 200 字片段
        print(f"[warn] lancedb 全量 chunk 读取失败, 判定上下文退回片段: {str(e)[:80]}")
    _chunk_full_map = mapping
    return mapping


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


def _llm_ragas(question: str, contexts: list[str], answer: str, reference: str, llm_key: str, llm_base: str, llm_model: str) -> dict | None:
    """单次 LLM 调用批量判定 4 个 RAGAS 指标。失败返回 None。命中缓存直接返回。"""
    if not llm_key:
        return None
    # 检查缓存 (question + top_3_contexts_hash → ragas result)
    sig = f"{question[:500]}|{contexts[0][:200] if contexts else ''}"
    sig_hash = hashlib.sha256(sig.encode()).hexdigest()[:16]
    if sig_hash in _ragas_cache:
        return _ragas_cache[sig_hash]
    prompt = RAGAS_PROMPT.format(
        question=question[:500],
        # 800 字 = _build_context 喂给 LLM 的每段实际上限; 判分器看到的上下文应与
        # 生成答案时的上下文等长, 否则会把"LLM 见过但判分器没见过的内容"误判为幻觉。
        context="\n".join(f"[{i + 1}] {c[:800]}" for i, c in enumerate(contexts)) or "(空)",
        answer=answer[:800] or "(空)",
        reference=reference[:500] or "(空)",
    )
    try:
        r = httpx.post(
            f"{llm_base}/chat/completions",
            json={
                "model": llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 128,
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
        content = message.get("content") or message.get("reasoning_content") or ""
        ragas = _extract_ragas_json(content)
        if ragas:
            result = {k: round(float(ragas.get(k, 0)), 4) for k in RAGAS_KEYS}
            _ragas_cache[sig_hash] = result
            return result
        return None
    except Exception as e:  # noqa: BLE001 — RAGAS 判定失败仅跳过该指标
        print(f"  [RAGAS 判定失败] {str(e)[:80]}")
        return None


async def _evaluate_one(client, url, key, item, embed, llm_key, llm_base, llm_model, force_kb: bool = False) -> dict:
    headers = {"X-API-Key": key} if key else {}
    payload: dict = {"question": item["question"], "top_k": 5}
    if force_kb and item.get("kb"):
        # 绕过 Router 直接指定库: 隔离"路由是否命中"与"库内检索是否召回"两个质量维度
        payload["skill"] = item["kb"]
    r = await client.post(
        f"{url}/api/v1/qa/ask", json=payload, headers=headers, timeout=60
    )
    d = r.json()
    answer = d.get("answer", "")
    source_dicts = d.get("sources", [])[:5]
    # 判定上下文用 chunk_id 反查的**完整原文**; sources[].content 是 200 字显示片段,
    # 拿它判 faithfulness 会把"LLM 见过但片段没展示的内容"误判为幻觉 (见 _load_chunk_full_map)。
    cmap = _load_chunk_full_map()
    sources = []
    for s in source_dicts:
        parts = [cmap[c] for c in (s.get("chunk_ids") or []) if cmap.get(c)]
        sources.append("\n".join(parts) if parts else s.get("content", ""))
    ref = item.get("reference_answer", "")

    # 检索 recall / precision (基于 chunk_ids 精确匹配; sources 可能带 chunk_ids 列表或 chunk_id)
    expected_ids = set(item.get("expected_chunk_ids", []))
    retrieved_ids = set()
    for s in source_dicts:
        if s.get("chunk_id"):
            retrieved_ids.add(s["chunk_id"])
        elif s.get("chunk_ids"):
            retrieved_ids.update(s["chunk_ids"])
    recall = len(retrieved_ids & expected_ids) / len(expected_ids) if expected_ids else None
    precision = len(retrieved_ids & expected_ids) / len(retrieved_ids) if retrieved_ids else 0.0
    # MRR: 首个命中 expected 的排名倒数 (P2#10 补全)
    _fr = _first_relevant_rank(source_dicts, expected_ids)
    mrr = round(1.0 / _fr, 4) if _fr else 0.0

    # 文档级 recall: 期望 chunk 所属文档被召回即算命中。
    # chunk 级 recall 会把"同一文档兄弟块命中"判成 miss (生成问题时用的是 chunk_0000,
    # 检索可能命中同文档的 chunk_0003), 文档级更能反映"用户能不能拿到答案"。
    exp_docs = {cid.split("_chunk_")[0] for cid in expected_ids if "_chunk_" in cid}
    ret_docs = {s.get("doc_id", "") for s in source_dicts}
    recall_doc = len(exp_docs & ret_docs) / len(exp_docs) if exp_docs else None

    ref_emb = embed.embed_query(ref) if embed else None
    ans_emb = embed.embed_query(answer[:512]) if embed and answer else None
    acc = max(0.0, _cosine(ans_emb, ref_emb)) if ans_emb and ref_emb else 0.0
    # 词重合近似统一走 engines/common/quality (OPT-C4): 与运行时同源, 且补齐了
    # 单字否定词保留 (旧内联版"不支持"与"支持"重合率相同, 属于已漂移的旧实现)。
    faith_approx = word_overlap_faithfulness(answer, sources)

    # 标准 RAGAS 4 指标 (单次 LLM 批量判定)
    ragas = _llm_ragas(item["question"], sources, answer, ref, llm_key, llm_base, llm_model)

    # 幻觉率优先用 LLM 判定的 faithfulness (更准), 无 LLM 时退回词重合近似
    hallucination_rate = round(1.0 - (ragas["faithfulness"] if ragas else faith_approx), 4)

    return {
        "question": item["question"],
        "kb": item.get("kb", ""),
        "routed_kb": d.get("kb_id") or "",
        "routing_hit": bool(item.get("kb")) and (d.get("kb_id") == item.get("kb")),
        "is_refusal": is_refusal(answer),
        "degradation_level": d.get("degradation_level", 0),
        "answer": answer,
        "contexts": sources,
        "accuracy": round(acc, 4),
        "recall": round(recall, 4) if recall is not None else None,
        "recall_doc": round(recall_doc, 4) if recall_doc is not None else None,
        "precision": round(precision, 4),
        "mrr": mrr,
        "hallucination_rate": hallucination_rate,
        "faithfulness": round(ragas["faithfulness"], 4) if ragas else None,
        "result_count": len(sources),
        "ragas": ragas,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default=os.environ.get("RAG_ACCESS_KEY") or os.environ.get("RAG_API_KEY", ""))
    ap.add_argument("--llm-key", default=os.environ.get("RAG_LLM_API_KEY", ""),
                    help="LLM API Key (也可用 --llm-key 传入，优先于环境变量)")
    ap.add_argument("--llm-base", default=os.environ.get("RAG_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    help="LLM API Base URL (默认 DashScope OpenAI 兼容)")
    ap.add_argument("--llm-model", default=os.environ.get("RAG_LLM_MODEL", "qwen-plus"),
                    help="LLM Model (默认 qwen-plus)")
    ap.add_argument("--dataset", default="tests/eval_dataset.json")
    ap.add_argument("--force-kb", action="store_true",
                    help="用数据集的 kb 字段强制指定检索库(绕过 Router), 用于把'库内检索召回'与'路由准确率'拆开测量")
    ap.add_argument("--out", default="data/eval-summary.json",
                    help="结果输出路径 (默认 data/eval-summary.json)")
    args = ap.parse_args()

    if not args.llm_key:
        print("[警告] 未提供 RAG_LLM_API_KEY, RAGAS 指标将跳过 (仅近似指标)")
    else:
        print(f"[LLM] base={args.llm_base} model={args.llm_model} key={args.llm_key[:8]}...")

    def _load_dataset():
        with open(args.dataset, encoding="utf-8") as f:
            return json.load(f)

    dataset = await asyncio.to_thread(_load_dataset)
    # Embedding 连接 HF 可能 SSL 失败，优雅降级为纯近似指标（jieba 词重合）
    embed = None
    try:
        embed = EmbeddingService()
    except Exception as e:  # noqa: BLE001 — 嵌入不可用时优雅降级为纯近似指标
        print(f"[嵌入] 初始化失败 ({str(e)[:80]}), 使用 jieba 近似指标")
    results = []
    async with httpx.AsyncClient(trust_env=False) as client:
        for item in dataset:
            try:
                res = await _evaluate_one(client, args.url, args.key, item, embed, args.llm_key, args.llm_base, args.llm_model, args.force_kb)
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
        vals = [r[field] for r in results if field in r and r[field] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    ragas_ok = [r["ragas"] for r in results if r.get("ragas")]
    summary = {
        "accuracy": _avg("accuracy"),
        "recall": _avg("recall"),
        "recall_doc": _avg("recall_doc"),
        "precision": _avg("precision"),
        "mrr": _avg("mrr"),
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

    # 按 KB 分桶, 定位哪类业务质量弱
    by_kb: dict[str, list] = {}
    for r in results:
        by_kb.setdefault(r.get("kb") or "unknown", []).append(r)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary["routing_accuracy"] = round(
        sum(1 for r in results if r.get("routing_hit")) / len(results), 4
    )
    # 拒答单列: 全体幻觉率会被拒答虚高; 剔除拒答后的幻觉率才反映"生成了错误内容"的比例
    non_refusal = [r for r in results if not r.get("is_refusal")]
    summary["refusal_rate"] = round((len(results) - len(non_refusal)) / len(results), 4)
    if non_refusal:
        summary["hallucination_rate_non_refusal"] = round(
            sum(r["hallucination_rate"] for r in non_refusal) / len(non_refusal), 4
        )
        summary["faithfulness_non_refusal"] = round(
            sum((r.get("ragas") or {}).get("faithfulness", 0) for r in non_refusal if r.get("ragas"))
            / max(1, sum(1 for r in non_refusal if r.get("ragas"))), 4
        )
    summary["by_kb"] = {
        kb: {
            "sample_count": len(rs),
            "recall": _mean(r["recall"] for r in rs),
            "recall_doc": _mean(r.get("recall_doc") for r in rs),
            "hallucination_rate": _mean(r["hallucination_rate"] for r in rs),
            "faithfulness": _mean(r.get("faithfulness") for r in rs),
            "routing_accuracy": _mean(1.0 if r.get("routing_hit") else 0.0 for r in rs),
            "refusal_rate": _mean(1.0 if r.get("is_refusal") else 0.0 for r in rs),
        }
        for kb, rs in by_kb.items()
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def _dump_summary():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "cases": results}, f, ensure_ascii=False, indent=2)

    await asyncio.to_thread(_dump_summary)
    print(f"\n=== 整体 (n={len(results)}) ===")
    print(
        f"近似: 准确率 {summary['accuracy'] * 100:.1f}%  chunk召回 {summary['recall'] * 100:.1f}%  "
        f"文档召回 {summary['recall_doc'] * 100:.1f}%  幻觉率 {summary['hallucination_rate'] * 100:.1f}%  "
        f"MRR {summary['mrr']:.4f}"
    )
    print(f"路由准确率 {summary['routing_accuracy'] * 100:.1f}%  (期望 KB 与实际 kb_id 一致的比例)")
    if summary.get("ragas"):
        r = summary["ragas"]
        print(
            f"RAGAS: faithfulness {r['faithfulness'] * 100:.1f}%  context_precision {r['context_precision'] * 100:.1f}%  "
            f"context_recall {r['context_recall'] * 100:.1f}%  answer_relevancy {r['answer_relevancy'] * 100:.1f}%"
        )
    print(f"报告已写入 {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
