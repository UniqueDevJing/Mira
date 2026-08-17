"""从 evaluate.py 产出生成 fidelity 标注骨架 + 人工审核清单。

用途: 关闭遗留项 ③ (fidelity_threshold 需真实流量标定) 的前置摩擦。
流程:
  1) 跑 scripts/evaluate.py 拿到 data/eval-summary.json (每条约含 answer/contexts/ragas.faithfulness)
  2) 本脚本把它转成:
     - data/labeled.json: 校准器可直接消费的骨架, score 已填, is_bad 留 null 待人工标
     - data/labeled_review.md: 人类可读审核卡 (问题/答案/上下文并排, 勾 good/bad)
  3) 人工把 labeled.json 每条 is_bad 标 true/false, 再跑 calibrate_fidelity.py

score 语义: 高=忠实。优先取 ragas.faithfulness (LLM 判); 无 RAGAS 时回退
1 - hallucination_rate (词重叠近似), 两者语义一致。

运行: python scripts/build_labeled_skeleton.py [--summary data/eval-summary.json]
      [--out data/labeled.json] [--review data/labeled_review.md]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _case_score(case: dict) -> float:
    """从 evaluate case 提取忠实度 score (0-1, 高=可信)。"""
    ragas = case.get("ragas") or {}
    if isinstance(ragas, dict) and "faithfulness" in ragas:
        try:
            return round(float(ragas["faithfulness"]), 4)
        except (TypeError, ValueError):
            pass
    # 回退: 词重叠近似 faith = 1 - hallucination_rate
    return round(1.0 - float(case.get("hallucination_rate", 0.5)), 4)


def build_labeled(cases: list[dict]) -> list[dict]:
    """把 evaluate cases 映射为 labeled 骨架 (is_bad 留 None)。"""
    out = []
    for c in cases:
        out.append(
            {
                "question": c.get("question", ""),
                "kb": c.get("kb", ""),
                "score": _case_score(c),
                "answer": c.get("answer", ""),
                "contexts": c.get("contexts", []) or [],
                "is_bad": None,  # 人工标注: 真幻觉/与文档矛盾 -> true, 否则 false
            }
        )
    return out


def _render_review(labeled: list[dict], summary_path: str) -> str:
    """生成人类可读审核清单 Markdown。"""
    lines = [
        "# Fidelity 标注审核清单",
        "",
        f"> 数据源: `{summary_path}` — 共 {len(labeled)} 条",
        ">",
        "> 如何标注: 逐条读「问题 / 系统答案 / 检索上下文」, 判断答案是否**胡说或与文档矛盾**。",
        "> 然后在 `data/labeled.json` 对应条把 `is_bad` 改为 `true`(真幻觉) 或 `false`(可信)。",
        "> 全部标完后运行: `python scripts/calibrate_fidelity.py data/labeled.json`",
        "",
        "---",
        "",
    ]
    for i, item in enumerate(labeled, 1):
        lines.append(f"## [{i}] (score={item['score']:.2f}, kb={item['kb'] or '?'})")
        lines.append("")
        lines.append(f"**问题**: {item['question']}")
        lines.append("")
        lines.append(f"**系统答案**: {item['answer'] or '(空)'}")
        lines.append("")
        lines.append("**检索上下文**:")
        ctxs = item["contexts"]
        if ctxs:
            for j, cx in enumerate(ctxs, 1):
                snippet = (cx or "").strip().replace("\n", " ")
                if len(snippet) > 300:
                    snippet = snippet[:300] + "…"
                lines.append(f"{j}. {snippet}")
        else:
            lines.append("(无)")
        lines.append("")
        lines.append("- [ ] good (可信, is_bad=false)   - [ ] bad (幻觉/矛盾, is_bad=true)")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/eval-summary.json")
    ap.add_argument("--out", default="data/labeled.json")
    ap.add_argument("--review", default="data/labeled_review.md")
    args = ap.parse_args()

    if not os.path.exists(args.summary):
        print(f"[错误] 找不到 {args.summary}, 请先跑 scripts/evaluate.py 生成评估产物")
        return

    with open(args.summary, encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])
    if not cases:
        print(f"[错误] {args.summary} 无 cases")
        return

    labeled = build_labeled(cases)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cases": labeled}, f, ensure_ascii=False, indent=2)

    review = _render_review(labeled, args.summary)
    with open(args.review, "w", encoding="utf-8") as f:
        f.write(review)

    print(f"已生成 {args.out} ({len(labeled)} 条, score 已填, is_bad 待人工标)")
    print(f"已生成 {args.review} (人工审核清单)")
    print("下一步: 打开 labeled_review.md 逐条判 good/bad, 回填 labeled.json 的 is_bad, 再跑 calibrate_fidelity.py")


if __name__ == "__main__":
    main()
