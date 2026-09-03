"""数字型幻觉护栏量化 — 纯函数评测, 不加载模型, 秒级完成。

基于 CRUD-RAG 的 hallu_modified 子集 (同时含 realContinuation 真值与 hallucinatedMod 幻觉值):

指标:
  - 误拒率(false-reject): 正确答案被护栏拒答的比例 —— 必须≈0
  - 数字幻觉检出率: 幻觉答案引入"上下文没有的数字"时被判定拒答的比例
  - Quick Win 增量: 仅数字校验能额外拦下的(词重合漏掉的)数字幻觉占比

对比: 同一批幻觉答案, 关掉数字校验(check_numbers=False, 即改动前) vs 开启(改动后)。

用法:
  python scripts/eval_fidelity_guardrail.py [--src data/external/crud_rag_split.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import settings
from api.core.qa_metrics import _extract_numbers, _faithfulness, _number_supported

THRESH = settings.fidelity_threshold  # 0.4 — orchestrator 用此阈值决定拒答


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external/crud_rag_split.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("hallu_modified", [])
    if args.limit:
        items = items[: args.limit]

    n = len(items)
    false_reject = 0          # 正确答案被数字校验误拒
    num_halluc = 0            # 幻觉引入上下文没有的数字 的样例
    detected_with_num = 0      # 开数字校验后被判拒答(数字或词重合任一拦下)
    detected_without_num = 0   # 仅词重合(关数字校验)被判拒答
    correct_has_num = 0

    for it in items:
        ctx = (it.get("newsBeginning", "") + " " + it.get("newsRemainder", "")).strip()
        correct = it.get("realContinuation", "").strip()
        halluc = it.get("hallucinatedMod", "").strip()
        if not ctx or not correct or not halluc:
            continue

        # 正确答案: 误拒?
        correct_pass = _number_supported(correct, [ctx])
        faith_correct = _faithfulness(correct, [ctx], embed_fn=None, check_numbers=True)
        if not correct_pass or faith_correct < THRESH:
            false_reject += 1
        if _extract_numbers(correct):
            correct_has_num += 1

        # 幻觉答案: 是否引入新数字
        ctx_nums = _extract_numbers(ctx)
        halluc_nums = _extract_numbers(halluc)
        new_nums = halluc_nums - ctx_nums
        if new_nums:
            num_halluc += 1
            faith_on = _faithfulness(halluc, [ctx], embed_fn=None, check_numbers=True)
            faith_off = _faithfulness(halluc, [ctx], embed_fn=None, check_numbers=False)
            if faith_on < THRESH:
                detected_with_num += 1
            if faith_off < THRESH:
                detected_without_num += 1

    def pct(x, d):
        return round(100.0 * x / d, 1) if d else 0.0

    print(f"=== 数字型幻觉护栏量化 (hallu_modified, n={n}) ===")
    print(f"正确答案误拒率 (false-reject) : {pct(false_reject, n)}%  ({false_reject}/{n})")
    print(f"  其中含数字的正确答案        : {correct_has_num} 条")
    print()
    print(f"幻觉引入新数字的样例           : {num_halluc} 条")
    if num_halluc:
        print(f"  ├─ 开启数字校验后检出(拒答)  : {pct(detected_with_num, num_halluc)}%  ({detected_with_num}/{num_halluc})")
        print(f"  ├─ 仅词重合(关数字校验)检出  : {pct(detected_without_num, num_halluc)}%  ({detected_without_num}/{num_halluc})")
        lift = detected_with_num - detected_without_num
        print(f"  └─ 数字校验的独有增量(Quick Win): {pct(lift, num_halluc)}%  ({lift}/{num_halluc})")

    summary = {
        "dataset": "hallu_modified", "n": n,
        "false_reject_rate": pct(false_reject, n) / 100,
        "num_halluc_cases": num_halluc,
        "detected_with_numcheck": detected_with_num,
        "detected_without_numcheck": detected_without_num,
        "numcheck_lift_cases": (detected_with_num - detected_without_num),
        "fidelity_threshold": THRESH,
    }
    os.makedirs("data/eval", exist_ok=True)
    with open("data/eval/fidelity_guardrail_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n报告已写入 data/eval/fidelity_guardrail_summary.json")


if __name__ == "__main__":
    main()
