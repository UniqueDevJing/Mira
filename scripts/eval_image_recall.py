"""图片检索召回评测（W3）：把 recall@k / MRR 落成 JSON，供阈值门禁比对。

用法:
    # 1) 先把评测集固化为 JSON（query → 期望命中的 asset_id）
    python scripts/eval_image_recall.py --index data/image-index.json \
        --cases docs/benchmarks/image-eval.json --out docs/benchmarks/image-recall.json

    # 2) 与基线比对（复用性能门禁脚本，指标名 image_recall_at_5）
    python scripts/check_bench_regression.py \
        --baseline docs/benchmarks/image-recall-baseline.json \
        --current docs/benchmarks/image-recall.json \
        --metric image_recall_at_5 --max-regression 0.10

    # 没准备评测集时，先跑冒烟（用索引里的字幕自问自答，只能证明链路通）
    python scripts/eval_image_recall.py --index data/image-index.json --smoke

退出码：0 正常 / 2 输入问题。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from engines.multimodal import ImageIndex, ImageRetriever, mrr, recall_at_k  # noqa: E402

K = 5


def load_cases(path: str) -> list[dict]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def smoke_cases(index: ImageIndex) -> list[dict]:
    """用每张图自己的字幕当查询：能命中说明入库/检索/打分链路是通的。"""
    return [
        {"query": entry.caption, "expected": [entry.asset_id], "kb": entry.kb}
        for entry in index.all()
        if entry.caption
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="ImageIndex.save() 产出的 JSON")
    ap.add_argument("--cases", default="", help="评测集 JSON：[{query, expected[], kb?}]")
    ap.add_argument("--smoke", action="store_true", help="没有评测集时用字幕自问自答冒烟")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    try:
        index = ImageIndex.load(args.index)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[EXIT 2] 读取索引失败: {exc}")
        return 2
    if len(index) == 0:
        print("[EXIT 2] 索引为空，先入库再评测")
        return 2

    if args.cases:
        cases = load_cases(args.cases)
    elif args.smoke:
        cases = smoke_cases(index)
    else:
        print("[EXIT 2] 需要 --cases 或 --smoke")
        return 2
    if not cases:
        print("[EXIT 2] 评测集为空")
        return 2

    retriever = ImageRetriever(index)
    recalls: list[float] = []
    mrrs: list[float] = []
    details: list[dict] = []
    for case in cases:
        hits = retriever.search(case["query"], k=args.k, kb=case.get("kb"))
        expected = set(case.get("expected") or [])
        r = recall_at_k(hits, expected, k=args.k)
        recalls.append(r)
        mrrs.append(mrr(hits, expected))
        details.append(
            {
                "query": case["query"],
                "recall": round(r, 4),
                "hit_ids": [h.entry.asset_id[:12] for h in hits],
            }
        )

    payload = {
        "tool": "eval_image_recall",
        "k": args.k,
        "cases": len(cases),
        "image_recall_at_5": round(sum(recalls) / len(recalls), 4),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
        "mode": "smoke" if args.smoke and not args.cases else "eval",
        "details": details,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(
        f"cases={payload['cases']}  recall@{args.k}={payload['image_recall_at_5']:.3f}  "
        f"mrr={payload['mrr']:.3f}  mode={payload['mode']}"
    )
    if args.out:
        target = pathlib.Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(f"落盘: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())