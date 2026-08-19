"""忠实度护栏阈值校准 — 遍历 t∈[0.0,1.0] 选 F1 最大点。

用途: 关闭遗留项 ③ (语义忠实度阈值 fidelity_threshold 需真实流量标定)。
方法:
1. 收集每条约的答案"忠实度得分" score (0-1) 与人工标签 is_bad (是否幻觉/不可信)。
   score 来源: 生产中导出护栏内部 _faithfulness, 或用 scripts/evaluate.py 的
   ragas.faithfulness 近似 (需先跑 evaluate.py 拿到 per-case 分数再人工标注坏样本)。
2. 护栏逻辑: score < t → 拒绝 (判高幻觉风险), 否则放行。
3. 遍历 t: reject=(score<t); bad=is_bad
   TP=bad&reject; FP=~bad&reject; FN=bad&~reject
   F1=2PR/(P+R), 取 F1 最大对应的 t (并列取较大 t, 更严格, 少误放)。
运行: python scripts/calibrate_fidelity.py labeled.json
  labeled.json: {"cases":[{"score":0.12,"is_bad":true}, ...]}
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sweep_fidelity(cases: list[dict], step: float = 0.02) -> dict:
    """cases: [{"score": float, "is_bad": bool}]。返回最佳阈值与扫描明细。

    纯逻辑、无外部依赖, 可直接单测。
    """
    scores = [(float(c["score"]), bool(c["is_bad"])) for c in cases]
    thresholds = [round(step * i, 2) for i in range(int(1.0 / step) + 1)]
    best = None
    rows = []
    for t in thresholds:
        tp = fp = fn = 0
        for s, bad in scores:
            reject = s < t
            if bad and reject:
                tp += 1
            elif (not bad) and reject:
                fp += 1
            elif bad and (not reject):
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(
            {
                "t": t,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
            }
        )
        # 并列 F1 取最小 t: 拒绝式护栏最宽松的安全边界 (只拒必要的坏样本, 不过拒好答案)
        if best is None or f1 > best["f1"] or (abs(f1 - best["f1"]) < 1e-9 and t < best["t"]):
            best = {"t": t, "f1": f1}
    return {"best_threshold": best["t"], "best_f1": best["f1"], "scan": rows}


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python scripts/calibrate_fidelity.py <labeled.json>")
        print('  labeled.json: {"cases":[{"score":0.12,"is_bad":true}]}')
        return
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", data if isinstance(data, list) else [])
    if not cases:
        print("无标注数据")
        return
    res = sweep_fidelity(cases)
    t = res["best_threshold"]
    print(f"样本数: {len(cases)}")
    print(f"推荐 fidelity_threshold = {t} (F1={res['best_f1']:.3f})")
    print("当前 config 默认 0.40。应用推荐值无需改码 — 设置环境变量即可生效:")
    print(f"  Linux/macOS: export RAG_FIDELITY_THRESHOLD={t}")
    print(f'  Windows:     $env:RAG_FIDELITY_THRESHOLD="{t}"')


if __name__ == "__main__":
    main()
