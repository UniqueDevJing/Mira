"""M2 成本-质量帕累托扫描：把（模型档位 × 召回深度 × 上下文长度）做网格实验。

为什么做这个：
    校招简历里几乎没人有「多维权衡实验」。把（质量, 成本）画成帕累托前沿，并写清
    「为什么最终选这个点」，是工程判断力的硬证据，比「我用了某某框架」高一个量级。

怎么跑（需先启动 RAG 服务 + 配置 LLM Key）：
    python scripts/pareto_sweep.py --dry-run            # 仅打印网格，不调用 LLM（离线可验证）
    python scripts/pareto_sweep.py --limit 60           # 真实扫描（需服务，约 60 问 × 档位数）
    python scripts/pareto_sweep.py --limit 60 --plot    # 扫描后生成 data/pareto/pareto.html
    python scripts/pareto_sweep.py --plot               # 仅用已有的 results.json 画图

每个网格点：设置 env（模型档位 / 召回深度 top-k / 上下文长度 MIRA_CONTEXT_CHARS），
调用既有 `eval_qa_quality.py` 跑离线 QA 集，解析 `qa_quality_summary.json` 的质量指标，
并按档位价目估算成本，写入 `data/pareto/results.json` 与帕累托前沿 `pareto_front.json`。

成本估算说明：无真实 token 计数时按「上下文字符 / 2 + 生成 200 字符」估算中文字符≈0.5 token，
再乘档位单价（每千 token 美元）。标注为估算；若 eval 输出含真实 token 则优先采用。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SCRIPT = os.path.join(ROOT, "scripts", "eval_qa_quality.py")
OUT_DIR = os.path.join(ROOT, "data", "pareto")

# 模型档位 → (LLM_MODEL 环境变量值, 每千 token 美元单价[输入/输出取均价])
TIERS = {
    "light":     ("deepseek-chat", 0.00014),   # 轻量档：低价模型
    "standard":  ("deepseek-chat", 0.00014),   # 标准档（默认）
    "flagship":  ("deepseek-reasoner", 0.00055),  # 旗舰档：高能力高单价
}
RECALL_DEPTHS = [3, 5, 8]          # 送入上下文的片段数（--top-k）
CONTEXT_CHARS = [800, 1600, 3200]  # 每段截断长度（MIRA_CONTEXT_CHARS）

# qa_quality_summary.json 里各指标的中文键名（与 eval_qa_quality.py 输出保持一致）
QUALITY_KEYS = ["faithfulness", "answer_relevance", "context_precision", "groundedness", "numeric_ok"]


def build_grid():
    grid = []
    for tier in TIERS:
        for k in RECALL_DEPTHS:
            for c in CONTEXT_CHARS:
                grid.append({"tier": tier, "top_k": k, "context_chars": c})
    return grid


def run_one(cfg: dict, limit: int, out_json: str) -> dict:
    """运行单次 eval，返回该配置的质量/成本记录。失败返回空质量。"""
    env = dict(os.environ)
    env["LLM_MODEL"] = TIERS[cfg["tier"]][0]
    env["MIRA_CONTEXT_CHARS"] = str(cfg["context_chars"])
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--limit", str(limit),
        "--top-k", str(cfg["top_k"]),
        "--out", out_json,
    ]
    rec = {"tier": cfg["tier"], "top_k": cfg["top_k"], "context_chars": cfg["context_chars"]}
    try:
        subprocess.run(cmd, env=env, cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
        metrics = _extract_metrics(summary)
    except Exception as e:  # noqa: BLE001 — 单点失败不中断整轮扫描
        print(f"  ! {cfg} 运行失败: {e}", file=sys.stderr)
        metrics = {k: None for k in QUALITY_KEYS}
    rec["quality"] = metrics
    rec["cost_usd"] = estimate_cost(cfg)
    return rec


def _extract_metrics(summary: dict) -> dict:
    """从 qa_quality_summary.json 抽取各指标均值（兼容 variants 多分支结构）。"""
    out = {k: None for k in QUALITY_KEYS}
    variants = summary.get("variants", {})
    # 取第一个 variant 的 metrics
    first = next(iter(variants.values()), {})
    m = first.get("metrics", {}) if isinstance(first, dict) else {}
    for k in QUALITY_KEYS:
        if k in m and isinstance(m[k], (int, float)):
            out[k] = float(m[k])
    # 兼容顶层直接平铺的情况
    for k in QUALITY_KEYS:
        if out[k] is None and k in summary and isinstance(summary[k], (int, float)):
            out[k] = float(summary[k])
    return out


def estimate_cost(cfg: dict) -> float:
    """按档位单价估算单问成本（美元）。中文约 0.5 token/字符。"""
    price = TIERS[cfg["tier"]][1]
    prompt_tokens = (cfg["context_chars"] * cfg["top_k"]) / 2 + 120
    completion_tokens = 200 / 2 + 80
    return round((prompt_tokens + completion_tokens) / 1000 * price, 6)


def quality_score(rec: dict) -> float:
    """综合质量分（faithfulness 与 answer_relevance 为主，缺失项忽略）。"""
    q = rec["quality"]
    vals = [v for v in (q.get("faithfulness"), q.get("answer_relevance")) if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def compute_pareto(results: list[dict]) -> list[dict]:
    """二维帕累托前沿：质量↑ 且 成本↓。返回被支配判定的前沿点。"""
    front = []
    for r in results:
        dominated = False
        for o in results:
            if o is r:
                continue
            if quality_score(o) >= quality_score(r) and o["cost_usd"] <= r["cost_usd"] \
                    and (quality_score(o) > quality_score(r) or o["cost_usd"] < r["cost_usd"]):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda x: x["cost_usd"])


def render_html(results: list[dict], front: list[dict], path: str) -> None:
    pts = []
    for r in results:
        pts.append({
            "x": r["cost_usd"], "y": round(quality_score(r), 3),
            "tier": r["tier"], "k": r["top_k"], "c": r["context_chars"],
            "front": r in front,
        })
    rows = "\n".join(
        f'  <circle cx="{p["x"]*1e6:.1f}" cy="{300 - p["y"]*260:.1f}" r="5" '
        f'fill="{ "#e5484d" if p["front"] else "#3b82f6" }">'
        f'<title>{p["tier"]} k={p["k"]} c={p["c"]} 质量={p["y"]} 成本=${p["x"]:.5f}</title></circle>'
        for p in pts)
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Mira M2 成本-质量帕累托前沿</title>
<style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;color:#222}}
svg{{background:#fafafa;border:1px solid #ddd;border-radius:8px}}
.dot-red{{fill:#e5484d}} .dot-blue{{fill:#3b82f6}} .muted{{color:#888;font-size:13px}}</style></head>
<body><h2>Mira · M2 成本-质量帕累托前沿</h2>
<svg width="640" height="340">
<line x1="40" y1="300" x2="620" y2="300" stroke="#bbb"/>
<line x1="40" y1="40" x2="40" y2="300" stroke="#bbb"/>
<text x="320" y="328" text-anchor="middle" class="muted">成本（美元/问，×1e6 放大）</text>
<text x="14" y="170" text-anchor="middle" transform="rotate(-90 14 170)" class="muted">综合质量分（faithfulness+answer_relevance）</text>
{rows}
</svg>
<p class="muted">红点=帕累托前沿（质量不更差且成本更低）；蓝点=被支配。
由 scripts/pareto_sweep.py 在部署服务上真实扫描后生成；本图为结果可视化。</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description="Mira M2 成本-质量帕累托扫描")
    ap.add_argument("--limit", type=int, default=60, help="每点评估的 QA 数")
    ap.add_argument("--dry-run", action="store_true", help="仅打印网格，不调用 LLM")
    ap.add_argument("--plot", action="store_true", help="生成 Pareto HTML（需 results.json 或本次扫描结果）")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    args = ap.parse_args()

    grid = build_grid()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"网格规模：{len(grid)} 点 = 档位({len(TIERS)}) × 召回深度({RECALL_DEPTHS}) × 上下文({CONTEXT_CHARS})")

    if args.dry_run:
        for g in grid:
            print(f"  {g['tier']:9s} top_k={g['top_k']} context_chars={g['context_chars']} "
                  f"~cost=${estimate_cost(g):.5f}")
        return

    results = []
    for i, cfg in enumerate(grid, 1):
        out_json = os.path.join(OUT_DIR, f"run_{i:02d}.json")
        print(f"[{i}/{len(grid)}] {cfg['tier']} k={cfg['top_k']} c={cfg['context_chars']} ...", flush=True)
        results.append(run_one(cfg, args.limit, out_json))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    front = compute_pareto(results)
    with open(os.path.join(OUT_DIR, "pareto_front.json"), "w", encoding="utf-8") as f:
        json.dump(front, f, ensure_ascii=False, indent=2)

    print(f"\n扫描完成 → {args.out}")
    print(f"帕累托前沿点（{len(front)}）：")
    for r in front:
        q = quality_score(r)
        print(f"  {r['tier']:9s} k={r['top_k']} c={r['context_chars']} 质量={q:.3f} 成本=${r['cost_usd']:.5f}")

    if args.plot:
        render_html(results, front, os.path.join(OUT_DIR, "pareto.html"))
        print(f"图表 → {os.path.join(OUT_DIR, 'pareto.html')}")


if __name__ == "__main__":
    main()
