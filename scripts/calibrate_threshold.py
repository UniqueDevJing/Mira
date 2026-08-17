"""跨库兜底阈值校准 — 遍历 t∈[0.3,0.8] 画 PR 曲线, 选 F1 最大点。

方法:
1. 从 LanceDB 重建各 kb 的 BM25 索引 (与生产混合检索对齐)
2. 临时把 cross_kb_threshold 设为 -1, 用生产 `_retrieve_context` 取纯路由库 top1
   (避免校准过程中内部兜底污染 top1)
3. 每条标注 (query, true_kb): 规则路由 → 得到路由库 R; 取 top1_R
4. 遍历阈值 t: fallback = (top1_R < t); 正确 = (fallback == (R != true_kb))
   F1 = 2*P*R/(P+R), 取 F1 最大对应的 t

局限: 标注集小 (~16 条), 路由用规则近似 (生产含 LLM 路由); 结果作参考值。
运行: python scripts/calibrate_threshold.py
"""

import asyncio
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import settings
from api.core.orchestrator import _retrieve_context
from api.state import get_bm25_index, get_vector_store
from engines.router.intent_router import IntentRouter, RoutingResult

# 标注集: (query, true_kb)。true_kb = 答案实际所在库 (基于客服话术=service / 技术白皮书=tech)
LABELED = [
    # service (客服话术)
    ("退货流程是什么", "service"),
    ("如何申请退款", "service"),
    ("物流运费怎么算", "service"),
    ("会员有哪些优惠", "service"),
    ("售后支持时间是什么", "service"),
    ("发票如何开具", "service"),
    ("订单怎么跟踪", "service"),
    ("怎么联系客服", "service"),
    # tech (技术白皮书)
    ("RRF 算法如何融合", "tech"),
    ("向量检索和 BM25 怎么结合", "tech"),
    ("BGE 嵌入模型是什么", "tech"),
    ("系统如何部署", "tech"),
    ("OCR 支持哪些格式", "tech"),
    ("知识图谱检索怎么做", "tech"),
    ("自检索是什么", "tech"),
    ("系统总体架构如何", "tech"),
    # 跨域: 规则路由会偏, 考验兜底
    ("退货接口的 API 设计", "tech"),  # "退货"→service 规则, 真答案在 tech
    ("客服系统的技术架构", "tech"),  # "客服"→service 规则, 真答案在 tech
]

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30..0.80


def _build_bm25_from_lance(kb: str) -> None:
    vs = get_vector_store(kb)
    rows = vs.table.to_arrow().to_pylist()
    get_bm25_index(kb).add_documents([{"id": r.get("id", ""), "content": r.get("content", "")} for r in rows])
    print(f"  BM25[{kb}] 重建 {len(rows)} 文档")


async def _collect() -> list[dict]:
    router = IntentRouter()
    cases = []
    for query, true_kb in LABELED:
        routed = await router.route(query)  # 生产真路由 (规则 >= 0.85 直通, 否则 LLM 分类, fallback tech)
        R = routed.kb
        routing = RoutingResult(skill=routed.skill, kb=R, confidence=routed.confidence, source=routed.source)
        retr = await _retrieve_context(query, routing, top_k=5, start=time.time(), enable_self_retrieval=False)
        top1 = retr["top1_score"]
        cases.append({"query": query, "true_kb": true_kb, "R": R, "top1": round(top1, 4)})
    return cases


def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main() -> None:
    orig_threshold = settings.cross_kb_threshold
    print(f"当前配置 cross_kb_threshold={orig_threshold}")

    print("重建 BM25 索引:")
    _build_bm25_from_lance("service")
    _build_bm25_from_lance("tech")

    settings.cross_kb_threshold = -1.0  # 收集阶段禁用内部兜底, 取纯路由库 top1
    cases = asyncio.run(_collect())
    print(f"\n共 {len(cases)} 条有效标注:\n")
    for c in cases:
        mark = "OK " if c["R"] == c["true_kb"] else "MISROUTE"
        print(f"  [{mark}] {c['query'][:20]:<22} true={c['true_kb']:<8} R={c['R']:<8} top1={c['top1']}")

    best = None
    print("\n阈值扫描 (F1 = 2·P·R/(P+R)):\n  t      TP  FP  FN  P     R     F1")
    for t in THRESHOLDS:
        tp = fp = fn = 0
        for c in cases:
            fallback = c["top1"] < t
            should = c["R"] != c["true_kb"]
            if should and fallback:
                tp += 1
            elif not should and fallback:
                fp += 1
            elif should and not fallback:
                fn += 1
        f1 = _f1(tp, fp, fn)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {t:<4} {tp:<4} {fp:<4} {fn:<4} {p:<5.3f} {r:<5.3f} {f1:.3f}")
        if best is None or f1 > best[1]:
            best = (t, f1)

    print(f"\n最佳阈值: t={best[0]} (F1={best[1]:.3f})")
    print(f"当前配置: t={orig_threshold:.2f}")


if __name__ == "__main__":
    main()
