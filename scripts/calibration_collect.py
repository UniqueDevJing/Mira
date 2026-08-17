"""标定数据采集 + 自动校准一体化脚本（供 cron / Task Scheduler 周期运行）。

流程:
1. 导出生产 QA 流量 → data/labeled_production.json (score=faithfulness 已填, is_bad 待人工标)
2. 统计标注进度（好/坏样本数、待标数）
3. 当既有"好样本"又有"坏样本"且全部标注完 → 自动跑 sweep_fidelity 给出推荐阈值

人工标注: 打开 data/labeled_production.json, 把每条 is_bad 从 null 改为 true/false
         (true=幻觉/不可信, false=忠实)。可配合 scripts/build_labeled_skeleton.py 生成审核清单。

用法:
  python scripts/calibration_collect.py
  python scripts/calibration_collect.py --db data/documents.db --out data/qa_export.json --labeled data/labeled_production.json
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from calibrate_fidelity import sweep_fidelity


def run_export(db: str, out: str, labeled: str) -> None:
    export_script = os.path.join(ROOT, "scripts", "export_qa_logs.py")
    cmd = [sys.executable, export_script, "--db", db, "--out", out, "--labeled", labeled]
    print(f"[1/3] 导出生产 QA 流量: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "documents.db"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "qa_export.json"))
    ap.add_argument("--labeled", default=os.path.join(ROOT, "data", "labeled_production.json"))
    args = ap.parse_args()

    run_export(args.db, args.out, args.labeled)

    with open(args.labeled, encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])
    total = len(cases)
    labeled_cases = [c for c in cases if c.get("is_bad") is not None]
    pending = total - len(labeled_cases)
    bad = [c for c in labeled_cases if c.get("is_bad") is True]
    good = [c for c in labeled_cases if c.get("is_bad") is False]

    print(f"[2/3] 标注进度: 共 {total} 条, 已标 {len(labeled_cases)} 条, 待标 {pending} 条")
    print(f"      已标中: 好样本 {len(good)} / 坏样本 {len(bad)}")

    if not labeled_cases:
        print("[3/3] 尚无标注数据。请人工标注 is_bad 后重跑本脚本。")
        return
    if not bad:
        print("[3/3] 仅有好样本、缺坏样本(真实幻觉) → 无法分隔。请补充标注坏样本后重跑。")
        return
    if pending:
        print(f"[3/3] 仍有 {pending} 条未标注, 暂跳过自动校准。标注完重跑即可自动出阈值。")
        return

    res = sweep_fidelity(labeled_cases)
    print(f"[3/3] 自动校准完成: 推荐 fidelity_threshold = {res['best_threshold']} (F1={res['best_f1']:.3f})")
    print("      → 将 api/config.py 的 fidelity_threshold 改为该值即可生效。")


if __name__ == "__main__":
    main()
