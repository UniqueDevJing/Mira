"""QA 评测集构建 — 从检索评测集分层抽样，建立可重复回归测试基线。

策略:
  Path A (本期): 从 data/eval/eval_dataset.json (390条) 按 category 分层抽样
    - 1doc (单文档): 25 条 — 最简单场景，确认检索链路通畅
    - 2doc (双文档): 25 条 — 需跨文档推理，测 RRF 融合质量
    - 3doc (三文档): 25 条 — 最强干扰，测 rerank 翻盘能力
    → 核心评测集 ~75 条，覆盖检索难度递增全谱

  Path B (后续): 手工构造高价值长尾场景 (模糊提问 / 多轮对话 / 对抗性输入)
    作为"金标准锚点"，围绕现有样本扩展

产出: tests/eval_dataset.json (兼容 evaluate.py)

用法:
  python scripts/build_qa_eval_set.py                    # 默认 75 条 (每类 25)
  python scripts/build_qa_eval_set.py --per-cat 20       # 每类 20 → 60 条
  python scripts/build_qa_eval_set.py --per-cat 30       # 每类 30 → 90 条
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser(
        description="Build QA eval set: stratified sample from retrieval eval dataset.",
    )
    ap.add_argument("--src", default="data/eval/eval_dataset.json",
                    help="源文件: data/eval/eval_dataset.json (含 category 字段的 390 条评测集)")
    ap.add_argument("--out", default="tests/eval_dataset.json",
                    help="输出路径")
    ap.add_argument("--per-cat", type=int, default=25,
                    help="每类抽取条数 (默认 25 → 75 条总评测集)")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子，保证可复现")
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.src, encoding="utf-8") as f:
        pool = json.load(f)

    # 按 category 分组
    groups: dict[str, list[dict]] = {}
    for item in pool:
        cat = item.get("category", "unknown")
        groups.setdefault(cat, []).append(item)

    sampled: list[dict] = []
    for cat in sorted(groups.keys()):
        items = groups[cat]
        n = min(args.per_cat, len(items))
        picked = random.sample(items, n)
        sampled.extend(picked)
        print(f"  [{cat}] {n}/{len(items)} sampled")

    # 输出: 兼容 evaluate.py 格式 (kb, question, reference_answer)
    out_items: list[dict] = [
        {
            "kb": x.get("kb", "unknown"),
            "question": x["question"],
            "reference_answer": x.get("reference_answer", ""),
        }
        for x in sampled
    ]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    # 统计摘要
    total = len(out_items)
    print(f"\n[done] wrote {total} items -> {args.out}")
    print(f"设计: 每类 {args.per_cat} 条 × {len(groups)} 类 = {len(groups) * args.per_cat} 条")
    print(f"实际: 总量 {total} (部分类别不足 {args.per_cat} 条时自动取满)")
    print(f"用途: python scripts/evaluate.py --dataset {args.out}")


if __name__ == "__main__":
    main()
