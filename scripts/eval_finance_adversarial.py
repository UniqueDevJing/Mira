# -*- coding: utf-8 -*-
"""本地复现金融校验对抗门禁（与 demo /api/run-all 同口径），用于面试前取证。"""
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, r"C:\Users\Dominion\WorkBuddy\2026-09-16-21-30-35\gh\mira-finance")

from engines.finance import Source, verify_finance_answer  # noqa: E402
from scripts.gen_finance_adversarial import build_samples  # noqa: E402

EVAL_NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)
MAX_AGE = timedelta(days=365)


def hit(expected, got):
    if expected == "fail":
        return got == "fail"
    if expected == "fail_or_warn":
        return got in ("fail", "warn")
    if expected == "pass_nowarn":
        return got == "pass"
    if expected == "warn":
        return got == "warn"
    return got == expected


samples = build_samples()
details = []
by_cat = {}
for s in samples:
    srcs = [Source(x["name"], x["text"], x["as_of"]) for x in s["sources"]]
    v = verify_finance_answer(s["answer"], srcs, now=EVAL_NOW, max_age=MAX_AGE)
    got = v.verdict
    h = hit(s["expected"], got)
    c = by_cat.setdefault(s["category"], {"n": 0, "hit": 0})
    c["n"] += 1
    c["hit"] += 1 if h else 0
    details.append({"id": s["id"], "cat": s["category"], "attack": s["attack"],
                    "expected": s["expected"], "got": got, "hit": h})

n = len(details)
hard = [d for d in details if d["expected"] == "fail"]
strict_benign = [d for d in details if d["expected"] == "pass_nowarn"]
stale = [d for d in details if d["expected"] == "warn"]

print(f"样本总数 {n}（硬攻击 {len(hard)} / 严格正常 {len(strict_benign)} / 时效类 {len(stale)}）")
print(f"整体命中率        : {sum(1 for d in details if d['hit'])}/{n} = {sum(1 for d in details if d['hit'])/n:.2%}")
if hard:
    print(f"攻击拦截率(硬)    : {sum(1 for d in hard if d['got']=='fail')}/{len(hard)} = {sum(1 for d in hard if d['got']=='fail')/len(hard):.2%}")
if strict_benign:
    print(f"正常回答误拒率    : {sum(1 for d in strict_benign if d['got']=='fail')}/{len(strict_benign)} = {sum(1 for d in strict_benign if d['got']=='fail')/len(strict_benign):.2%}")
print("\n分类明细：")
for c, v in sorted(by_cat.items()):
    print(f"  {c:<14} {v['hit']}/{v['n']}")
misses = [d for d in details if not d["hit"]]
print(f"\n未命中 {len(misses)} 条：")
for d in misses:
    print(f"  {d['id']:<18} {d['cat']:<12} 期望={d['expected']:<12} 实际={d['got']}")
