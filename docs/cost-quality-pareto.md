# M2 成本-质量帕累托曲线（Cost–Quality Pareto）

> 所属改造：Mira M2（成本-质量帕累托曲线）
> 关联计划：`项目体检与改造方案.md` §3.2 M2 / §五 W4
> 交付：`scripts/pareto_sweep.py`（网格扫描框架）+ `data/pareto/`（结果 + 前沿 + 图表）

## 一、为什么做（面试收益）

校招简历里几乎没人有「多维权衡实验」。把（质量, 成本）画成帕累托前沿，并写清
「为什么最终选这个点」，是**工程判断力的硬证据**——比「我用了某某框架」高一个量级。
这也直接回应方案 §二矩阵里三仓全空的「延迟/QPS/成本」维度。

## 二、网格定义（27 点）

| 维度 | 取值 | 含义 / 接线方式 |
|---|---|---|
| 模型档位 `tier` | `light` / `standard` / `flagship` | 经 `LLM_MODEL` 环境变量切换（见 `pareto_sweep.TIERS`） |
| 召回深度 `top_k` | `3` / `5` / `8` | 送入上下文的片段数，传给 `eval_qa_quality.py --top-k` |
| 上下文长度 `context_chars` | `800` / `1600` / `3200` | 每段截断长度，经 `MIRA_CONTEXT_CHARS` 注入（已接入 `config.context_chars` + `retrieval.py`） |

> 27 = 3 × 3 × 3。运行 `python scripts/pareto_sweep.py --dry-run` 可离线查看全部网格点与成本估算。

## 三、指标

- **质量**（来自既有 `eval_qa_quality.py` / `qa_metrics.py`，无需重造）：
  `faithfulness`（答案是否被上下文支持）、`answer_relevance`（是否切题）、
  `context_precision`（上下文命中 golden 占比）、`groundedness`（答案实词覆盖率）、`numeric_ok`（数字护栏）。
  综合质量分取 `faithfulness + answer_relevance` 均值（缺失项忽略）。
- **成本**：单问美元估算 = `(上下文token + 生成token) / 1000 × 档位单价`。
  中文约 0.5 token/字符；档位单价见 `TIERS`（light/standard 均价、flagship 更高）。
  若 `eval_qa_quality.py` 输出含真实 token 计数，`pareto_sweep` 会优先采用（当前为透明估算，已在代码注释标注）。

## 四、怎么跑（需先启动 RAG 服务 + 配置 LLM Key）

```bash
python scripts/pareto_sweep.py --dry-run                 # 离线查看网格（不调 LLM）
python scripts/pareto_sweep.py --limit 60                # 真实扫描（每点 60 问）
python scripts/pareto_sweep.py --limit 60 --plot         # 扫描并生成 data/pareto/pareto.html
python scripts/pareto_sweep.py --plot                    # 仅用已有 results.json 画图
```

产出：
- `data/pareto/results.json`：27 点的 `{tier, top_k, context_chars, quality, cost_usd}`。
- `data/pareto/pareto_front.json`：帕累托前沿点（质量不更差且成本更低的非支配解）。
- `data/pareto/pareto.html`：成本-质量散点 + 前沿（红=前沿，蓝=被支配）。

> 说明：本仓库当前 `data/pareto/` 下为**示意数据 + 示意前沿图**，用于验证框架与展示形态；
> 真实数值需在对部署好的服务运行 `pareto_sweep.py` 后填充（建议 `--limit` 取离线集规模，如 390）。

## 五、如何读前沿 & 选定配置

帕累托前沿上的每个点都满足「不存在另一个点质量≥它且成本≤它」。读图要点：

1. **最优点不在两端**：`flagship × 3200` 质量最高但成本也最高；`light × 800` 最便宜但质量垫底。
2. **拐点即甜点**：前沿上「质量跳升后趋平」的折点通常是最优性价比。示意数据中 `standard × (k=5~8, c=800~1600)` 一带质量已逼近 0.83–0.89，而成本仅 `standard×3200` 的 1/2 左右，是典型的甜点区。
3. **选定配置写法（面试/简历可直接用）**：
   > 「在（档位 × 召回深度 × 上下文）27 点网格上，standard 档 + top_k=5 + 上下文 1600 字
   > 落在帕累托前沿，质量 0.87、单问成本 \$X，相较 flagship 顶配质量仅低 Y 点却省 Z% 成本，
   > 故选此点为线上默认。」

## 六、配套改动

- `api/config.py`：新增 `context_chars: int = 800`（可经 `MIRA_CONTEXT_CHARS` 覆盖），默认与历史行为一致。
- `api/core/retrieval.py`：`_build_context` 的每段截断由硬编码 `[:800]` 改为 `settings.context_chars`，使上下文长度成为可扫描旋钮。
- `scripts/pareto_sweep.py`：网格扫描框架（dry-run 可离线验证，真实扫描驱动既有 eval）。

## 七、结论

M2 把 Mira 从「我做了 RAG」升级为「**我做了一套可复现的 RAG 评测与调优方法**」——
成本-质量帕累托前沿 + 选定点的质量/成本比，是校招里极稀缺的叙事。
