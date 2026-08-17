# RAG 2.0 部署与可观测 + 标定运行手册

> 范围：① 让系统跑在生产流量上、质量可被观测（Grafana 面板）+ ③ 之前的标定数据闭环。
> 前置：已 `git push` 到 `UniqueDevJing/rag-2.0`（见 git 提交历史，含 #21 服务端会话 / #22 KB 级 RBAC）。

---

## 1. 一键起栈（API + Prometheus + Grafana）

```bash
cd infrastructure/docker
cp .env.example .env          # 填入 RAG_LLM_API_KEY 等
docker compose up -d --build  # 首次/改代码后重建镜像
docker compose ps             # 三服务应全 Up
```

| 服务 | 地址 | 说明 |
|---|---|---|
| API | http://localhost:8000 | `/health` `/metrics` `/docs`(OpenAPI) |
| Prometheus | http://localhost:9090 | 抓取 API 的 `/metrics`（15s 间隔） |
| Grafana | http://localhost:3000 | admin/admin（已预置 Prometheus 数据源 + RAG 面板）|

> 镜像构建期会预下载 `BAAI/bge-small-zh-v1.5` embedding 模型（需联网一次）；运行时 `TRANSFORMERS_OFFLINE=1` 不重复下载。
> 多实例部署：把 `RAG_SHARED_STATE_BACKEND=redis` 并起 Redis，否则服务端会话/缓存仅单进程有效。

---

## 2. 质量可观测（Grafana 面板）

仪表盘 `RAG 2.0 — 服务监控` 含 12 个面板，其中**质量相关**为本次补强：

| 面板 | 指标 | 用途 |
|---|---|---|
| 回答忠实度分布 (P50/P95) | `rag_qa_faithfulness` | 看幻觉护栏输入侧的分数趋势，**越低越危险** |
| Top1 检索得分分布 (P50/P95) | `rag_qa_top1_score` | 检索质量，低分说明召回不准 |
| 降级层级分布 | `rag_degradation_levels_total{level}` | level 越高代表越多走兜底，异常升高=链路退化 |
| QA 请求速率 / 延迟 P50·P95·P99 | `rag_qa_requests_total` / `rag_qa_latency_seconds` | 基础 SLO |
| QA 错误率 | `rag_qa_requests_total{status="error"}` | 错误占比，阈值 1% 黄 / 5% 红 |

Prometheus 即席查询示例：
```
histogram_quantile(0.95, sum(rate(rag_qa_faithfulness_bucket[5m])) by (le))
sum(rate(rag_degradation_levels_total[5m])) by (level)
```

建议告警：忠实度 P95 连续 5m < 0.4，或错误率 > 5% → 企业微信/邮件。

---

## 3. 语义忠实度阈值标定（③ 闭环，只差数据）

护栏 `fidelity_threshold` 默认 0.40 是保守初值，**需用真实流量 + 人工标注坏样本**才能标出最优值。链路已全通：

### 3.1 采集生产流量
每次 QA 会自动落 `qa_logs` 表（含 `sources` 全文，无截断）。周期性导出可标注骨架：

```bash
# 单次
python scripts/export_qa_logs.py --db data/documents.db \
    --out data/qa_export.json --labeled data/labeled_production.json

# 或一体化脚本（推荐放进 cron / Task Scheduler 周期跑）
python scripts/calibration_collect.py
```

`data/labeled_production.json` 结构：`{"cases":[{"score":<faithfulness>,"is_bad":null}, ...]}`，
`score` 已自动填好，`is_bad` 待人工标注。

### 3.2 人工标注坏样本
打开 `data/labeled_production.json`，把每条 `is_bad` 从 `null` 改为：
- `true` = 该答案是幻觉/不可信（必须人工判断，脚本无法代劳）
- `false` = 忠实

> 关键：必须**同时有"好样本"和"坏样本（真实幻觉）**才能学到分隔面。若流量里几乎无幻觉，可主动构造/注入对抗样本。

### 3.3 自动出阈值
标完重跑一体化脚本，标注齐全且好坏都有时自动校准：

```bash
python scripts/calibration_collect.py
# 输出形如: 推荐 fidelity_threshold = 0.32 (F1=1.000)
```

把该值写入 `api/config.py` 的 `fidelity_threshold` 即可生效（无需重启，配置热加载视部署方式定）。

---

## 4. 环境变量速查

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_LLM_API_KEY` | 空 | 真实问答必需 |
| `RAG_API_KEY_ENABLED` | true | API Key 鉴权开关（fail-closed）|
| `RAG_API_KEY_WHITELIST` | 空 | KB 级 RBAC 白名单 JSON |
| `RAG_SHARED_STATE_BACKEND` | memory | memory / redis（多实例共享态）|
| `GRAFANA_USER` / `GRAFANA_PASSWORD` | admin/admin | 生产务必改 |

---

## 5. 已知边界
- 镜像构建需联网拉 `python:3.12-slim` + 依赖 + BGE 模型（一次性）；纯离线环境需提前准备基础镜像。
- `infrastructure/k8s`、`infrastructure/helm` 目前为空目录，K8s 部署清单待按本 compose 推导。
- 标定标注是人工动作，脚本只负责"采集 + 计数 + 自动算阈值"，不能替代人对幻觉的判断。
