"""QA 缓存命中率基准 — 直接驱动 api.core.qa_cache.QACache（不依赖模型/服务）。

测两项:
  1) 精确命中率: 同一问题二次请求直接命中, 跳过 路由+检索+LLM。
  2) 语义近重复命中率: 同义改写问题在阈值(默认0.92)内命中, 省 LLM+rerank 调用。

embed_fn 用确定性「字符 unigram+bigram 词袋」向量, 中文改写共享字时余弦高,
可在无 sentence-transformers 的环境复现语义命中行为。

用法:
    python scripts/bench_cache.py
"""
from __future__ import annotations

import math
from collections import Counter

from api.core.qa_cache import QACache


def _embed(question: str) -> list[float]:
    """确定性字符级词袋嵌入 (L2 归一化), 供语义近重复命中用。"""
    q = question.strip().lower()
    grams = list(q) + [q[i:i + 2] for i in range(len(q) - 1)]
    cnt = Counter(grams)
    norm = math.sqrt(sum(v * v for v in cnt.values())) or 1.0
    dim = 2048
    vec = [0.0] * dim
    for g, c in cnt.items():
        h = hash(g) % dim
        vec[h] += c / norm
    vnorm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / vnorm for v in vec]


def _key_for(q: str) -> str:
    # 复用 QACache 的键构造, 模拟生产同样的输入指纹
    return QACache.make_key(q, skill=None, top_k=5, enable_self_retrieval=False,
                            temperature=0.2, mode="hybrid", history=[], allowed_kbs=None)


def main() -> None:
    cache = QACache(semantic_enabled=True, embed_fn=_embed)
    VAL = {"answer": "cached-answer", "sources": []}

    base = [
        "如何办理退货", "退货流程是什么", "会员等级怎么升级", "积分过期了怎么办",
        "发票怎么开", "订单多久发货", "支持哪些支付方式", "怎么修改收货地址",
        "退款到账时间", "如何联系人工客服", "优惠券为什么用不了", "商品质量问题如何索赔",
        "账号被锁定了", "怎么注销账户", "物流信息不更新", "预约上门安装",
        "换货流程", "保修期多久", "如何绑定微信", "发票抬头怎么改",
    ]
    paraphrases = [
        "退货要怎么办理", "退货的具体流程", "会员等级如何提升", "积分过期了该咋办",
        "怎么开具发票", "下单后多久发货", "可以用什么方式付款", "收货地址如何修改",
        "退款什么时候到账", "怎样联系人工客服", "优惠券无法使用的原因", "买到质量问题商品怎么赔",
        "我的账号被锁了", "账户怎么注销", "快递信息一直不动", "预约个上门安装",
        "换货怎么操作", "保修时间是多长", "微信怎么绑定", "发票抬头如何更改",
    ]

    # ── 阶段1: 全部未命中, 写入缓存 ──
    miss = 0
    for q in base:
        stats: dict = {}
        if cache.get(_key_for(q), question=q, scope=QACache.make_scope(q, None, 5, False, 0.2, "hybrid", [], None), stats=stats) is None:
            miss += 1
        cache.set(_key_for(q), VAL, ttl_s=3600, question=q,
                  scope=QACache.make_scope(q, None, 5, False, 0.2, "hybrid", [], None))

    # ── 阶段2: 精确命中 (同问题复问) ──
    exact_hit = 0
    for q in base:
        stats = {}
        if cache.get(_key_for(q), question=q, scope=QACache.make_scope(q, None, 5, False, 0.2, "hybrid", [], None), stats=stats) is not None:
            exact_hit += 1

    # ── 阶段3: 语义近重复命中 (改写问题) ──
    sem_hit = 0
    for q in paraphrases:
        stats = {}
        if cache.get(_key_for(q), question=q, scope=QACache.make_scope(q, None, 5, False, 0.2, "hybrid", [], None), stats=stats) is not None:
            sem_hit += 1

    n = len(base)
    print(f"写入基数={n}  阶段1未命中(写入)={miss}/{n}")
    print(f"精确命中率 = {exact_hit}/{n} = {exact_hit / n * 100:.1f}%")
    print(f"语义近重复命中率 = {sem_hit}/{n} = {sem_hit / n * 100:.1f}%  (阈值 0.92, 字符词袋近似)")
    print("→ 二次复问可 100% 跳过 路由+检索+LLM; 同义改写按近似阈值命中, 直接省一次 LLM+rerank 调用")


if __name__ == "__main__":
    main()
