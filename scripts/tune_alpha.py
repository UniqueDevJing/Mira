"""离线网格搜索 rerank 融合权重 alpha (含自适应映射) — 纯 numpy, 秒级, 不跑模型。

依赖 scripts/dump_rerank_signals.py 转储的信号 (CE 分 / RRF 分 / embedding / golden mask)。

评估策略:
  baseline  纯检索 (RRF 序, 不重排)
  pure      alpha=1.0 纯 Cross-Encoder
  fixed     固定 alpha 网格
  adaptive  alpha = amax - (amax-amin) * clamp(density / d_full, 0, 1)
            density = 候选池近重复密度 (池内 cos>=thr 的文档对占比 / 有近重复伙伴的文档占比)

目标: 在 clean(干净语料) 与 hard(密集近重复语料) 上**同时**优于 baseline, 并尽量逼近各自上限。

用法:
  python scripts/tune_alpha.py --hard data/eval/signals_hard --clean data/eval_clean/signals_clean
"""
import argparse
import itertools
import json
import os

import numpy as np

KS = (1, 3, 5, 10)


def load(prefix: str):
    """载入转储信号, 并取回每题 golden 总数 (Recall 分母需与 eval_retrieval.py 口径一致)。

    注意: gold 掩码只标记"池内"的 golden; 而 Recall 的分母应是该问题 golden_chunk_ids 的
    总数 (部分 golden 可能没进候选池)。两者不一致会让本脚本数值虚高, 故从 eval_dataset.json 取。
    """
    z = np.load(prefix + ".npz")
    ce, rrf, embs, gold = z["ce"], z["rrf"], z["embs"], z["gold"]
    n = ce.shape[0]
    ds_path = os.path.join(os.path.dirname(prefix) or ".", "eval_dataset.json")
    if os.path.exists(ds_path):
        with open(ds_path, encoding="utf-8") as f:
            ds = json.load(f)[:n]
        ngold = np.array([max(len(set(it["expected_chunk_ids"])), 1) for it in ds], dtype=np.float64)
    else:  # 退化: 用池内 golden 数 (相对比较仍有效, 但绝对值虚高)
        ngold = np.maximum(gold.sum(axis=1), 1).astype(np.float64)
    return ce, rrf, embs, gold, ngold


def density(E: np.ndarray, thr: float, mode: str) -> float:
    """候选池近重复密度。E 已归一化 (k×d)。"""
    k = E.shape[0]
    if k < 2:
        return 0.0
    sims = E @ E.T
    ge = sims >= thr
    if mode == "pairs":
        total = k * (k - 1)
        cnt = int(ge.sum() - k)  # 去掉对角
        return cnt / total
    has_partner = (ge.sum(axis=1) - 1) > 0
    return float(has_partner.mean())


def _metrics(order: np.ndarray, gold_row: np.ndarray, n_gold: float):
    hit = None
    for r, i in enumerate(order, 1):
        if gold_row[i]:
            hit = r
            break
    rec = {K: float(gold_row[order[:K]].sum()) / n_gold for K in KS}
    mrr = 1.0 / hit if hit else 0.0
    return rec, mrr


def evaluate(ce, rrf, embs, gold, ngold, alpha_of):
    """alpha_of(i, ce_row, rrf_row, E) -> float; 返回各 K 的 Recall 与 MRR 均值。"""
    n = ce.shape[0]
    rec_sum = {K: 0.0 for K in KS}
    mrr_sum = 0.0
    for i in range(n):
        cr, rr = ce[i], rrf[i]
        cmin, cmax = cr.min(), cr.max()
        ce_n = (cr - cmin) / (cmax - cmin + 1e-9)
        rmax = rr.max()
        rrf_n = rr / (rmax + 1e-9)
        a = alpha_of(i, cr, rr, embs[i])
        final = (1.0 - a) * rrf_n + a * ce_n
        order = np.argsort(-final, kind="stable")
        rec, mrr = _metrics(order, gold[i], ngold[i])
        for K in KS:
            rec_sum[K] += rec[K]
        mrr_sum += mrr
    return {f"recall@{K}": rec_sum[K] / n for K in KS} | {"mrr": mrr_sum / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", required=True)
    ap.add_argument("--clean", required=True)
    args = ap.parse_args()

    H = load(args.hard)
    C = load(args.clean)

    print("=== 基线对照 (alpha=0 纯检索 / alpha=1 纯 CE) ===")
    res = {}
    for name, (c, r, e, g, ng) in (("clean", C), ("hard", H)):
        b = evaluate(c, r, e, g, ng, lambda i, cr, rr, E: 0.0)
        p = evaluate(c, r, e, g, ng, lambda i, cr, rr, E: 1.0)
        res[name] = {"baseline": b, "pure": p}
        print(f"\n[{name}]")
        for K in KS:
            print(f"  Recall@{K:<2}  检索 {b[f'recall@{K}']*100:6.1f}%   纯CE {p[f'recall@{K}']*100:6.1f}%"
                  f"   Δ {(p[f'recall@{K}']-b[f'recall@{K}'])*100:+6.1f}%")
        print(f"  MRR        检索 {b['mrr']:.3f}     纯CE {p['mrr']:.3f}    Δ {p['mrr']-b['mrr']:+.3f}")

    print("\n=== 固定 alpha 网格 ===")
    print(f"{'alpha':>6} | {'clean R@1':>10} {'clean MRR':>10} | {'hard R@1':>10} {'hard MRR':>10} | {'联合R@1':>9}")
    fixed_rows = []
    for a in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        rc = evaluate(C[0], C[1], C[2], C[3], C[4], lambda i, cr, rr, E, a=a: a)
        rh = evaluate(H[0], H[1], H[2], H[3], H[4], lambda i, cr, rr, E, a=a: a)
        joint = (rc["recall@1"] + rh["recall@1"]) / 2
        fixed_rows.append((a, rc, rh, joint))
        print(f"{a:>6.2f} | {rc['recall@1']*100:>9.1f}% {rc['mrr']:>10.3f} | "
              f"{rh['recall@1']*100:>9.1f}% {rh['mrr']:>10.3f} | {joint*100:>8.1f}%")

    # 密度分布: 确认 clean / hard 是否可分离
    print("\n=== 候选池近重复密度分布 (分离度检查) ===")
    for thr in (0.85, 0.90, 0.95):
        for mode in ("pairs", "docs"):
            dc = np.array([density(C[2][i], thr, mode) for i in range(C[2].shape[0])])
            dh = np.array([density(H[2][i], thr, mode) for i in range(H[2].shape[0])])
            print(f"  thr={thr} mode={mode:<5}  clean mean={dc.mean():.3f} p90={np.percentile(dc,90):.3f}"
                  f"   hard mean={dh.mean():.3f} p90={np.percentile(dh,90):.3f}")

    print("\n=== 自适应 alpha 网格搜索 ===")
    best = None
    rows = []
    for thr, mode, amax, amin, dfull in itertools.product(
        (0.85, 0.90, 0.95), ("pairs", "docs"), (0.85, 0.90, 0.95), (0.3, 0.4, 0.5, 0.6), (0.10, 0.20, 0.30, 0.50)
    ):
        def mk(thr=thr, mode=mode, amax=amax, amin=amin, dfull=dfull):
            def f(i, cr, rr, E):
                d = density(E, thr, mode)
                return amax - (amax - amin) * min(max(d / dfull, 0.0), 1.0)
            return f
        rc = evaluate(C[0], C[1], C[2], C[3], C[4], mk())
        rh = evaluate(H[0], H[1], H[2], H[3], H[4], mk())
        # 目标: 双语料 Recall@1 + MRR 相对 baseline 的平均增益, 且两边都不得低于 baseline
        gc1 = rc["recall@1"] - res["clean"]["baseline"]["recall@1"]
        gh1 = rh["recall@1"] - res["hard"]["baseline"]["recall@1"]
        gcm = rc["mrr"] - res["clean"]["baseline"]["mrr"]
        ghm = rh["mrr"] - res["hard"]["baseline"]["mrr"]
        if gc1 < 0 or gh1 < 0:  # 硬约束: 两边都不能跌破 baseline
            continue
        score = (gc1 + gh1) / 2 + (gcm + ghm)
        rows.append((score, thr, mode, amax, amin, dfull, rc, rh))
        if best is None or score > best[0]:
            best = rows[-1]

    rows.sort(key=lambda x: -x[0])
    print(f"{'thr':>5} {'mode':<6} {'amax':>5} {'amin':>5} {'dfull':>6} | {'clean R@1':>10} {'MRR':>7} | "
          f"{'hard R@1':>10} {'MRR':>7} | {'score':>7}")
    for r in rows[:12]:
        score, thr, mode, amax, amin, dfull, rc, rh = r
        print(f"{thr:>5.2f} {mode:<6} {amax:>5.2f} {amin:>5.2f} {dfull:>6.2f} | "
              f"{rc['recall@1']*100:>9.1f}% {rc['mrr']:>7.3f} | {rh['recall@1']*100:>9.1f}% {rh['mrr']:>7.3f} | {score:>7.4f}")

    if best:
        score, thr, mode, amax, amin, dfull, rc, rh = best
        print("\n=== 最优自适应配置 ===")
        print(f"alpha = {amax} - ({amax}-{amin}) * clamp(density({thr},{mode}) / {dfull}, 0, 1)")
        print(f"clean: Recall@1 {rc['recall@1']*100:.1f}%  MRR {rc['mrr']:.3f}")
        print(f"hard : Recall@1 {rh['recall@1']*100:.1f}%  MRR {rh['mrr']:.3f}")
        print(f"(对照 baseline: clean R@1 {res['clean']['baseline']['recall@1']*100:.1f}% / "
              f"hard R@1 {res['hard']['baseline']['recall@1']*100:.1f}%)")
        out = {"density_threshold": thr, "density_mode": mode, "alpha_max": amax,
               "alpha_min": amin, "density_full": dfull,
               "clean": rc, "hard": rh,
               "baseline_clean": res["clean"]["baseline"], "baseline_hard": res["hard"]["baseline"]}
        with open("data/eval/alpha_tuning.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("\n已写入 data/eval/alpha_tuning.json")


if __name__ == "__main__":
    main()
