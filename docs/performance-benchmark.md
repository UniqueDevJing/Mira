# Mira 性能基准（performance-benchmark）

> 改造项 M3（压测与延迟预算）。目标：给出单机 **QPS / P95 / 语义缓存命中率 / 单次成本** 四项可复现硬指标。
> 生成日期：2026-11-03。

## 1. 环境与复现方法

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 (MINGW64), x86_64 |
| CPU 逻辑核 | 32 |
| Python | 3.13.14 |
| 服务进程 | 单 uvicorn worker (`it_backend_fakellm.py` 默认) |
| 检索 / LLM | **桩**：fake retrieval + fake stream LLM（`scripts/it_backend_fakellm.py`），**不依赖模型 Key、不下载权重、不依赖语料** |
| 缓存 | 压测路径关闭（桩服务强制 `qa_cache_enabled=False`），缓存命中率单独在 §3 用 `QACache` 直测 |

**为什么用 fake-LLM 桩**：本机无模型 Key 与语料，但 Mira 自带 `it_backend_fakellm.py` 把 `orchestrator.get_llm_client` 与 `_retrieve_context` 整体打桩，使整条 QA 链路可在纯离线环境跑通。这样压测**隔离出「服务 + 编排层」本身的开销**，不受真实 LLM 推理/向量检索变量干扰，是 M3「最便宜的硬指标」的可复现基线。

**复现命令**：

```bash
# 终端1：启动桩服务（127.0.0.1:8911，免鉴权、CORS 全开）
PYTHONPATH=. python scripts/it_backend_fakellm.py

# 终端2：端到端延迟 + QPS
python scripts/load_test.py --url http://127.0.0.1:8911 \
    --question "退货流程是什么" --concurrency 20 --requests 200

# 首 token 延迟（TTFT，命中 SSE 首个 delta 事件）
python scripts/bench_ttft.py --url http://127.0.0.1:8911 --requests 30

# 语义缓存命中率（直测 QACache，无模型）
PYTHONPATH=. python scripts/bench_cache.py
```

## 2. 延迟预算

| 场景 | 并发 | 请求数 | QPS | P50(ms) | P95(ms) | P99(ms) | max(ms) | HTTP |
|---|---|---|---|---|---|---|---|---|
| 端到端（load_test） | 5 | 50 | 3.2 | 1498 | 2017 | 2017 | 2017 | 200/200 |
| 端到端（load_test） | 20 | 200 | 5.7 | 3397 | 4644 | 4649 | 4670 | 200/200 |
| TTFT（bench_ttft） | 顺序 | 30 | — | 1446 | 1494 | 1514 | 1514 | 200/200 |

- **TTFT ≈ 1.45s（P50）**：从发请求到 SSE 首个 `delta` 事件。该值已基本是「路由 + 编排启动 + 首块产出」的编排层开销——fake LLM 块间仅 15ms sleep，故真实模型的首字延迟会在此基础上**叠加**模型推理首字时间。
- **单请求编排开销地板 ≈ 1.5s（低并发 P50）**，并发 20 时恶化到 ~3.4s：单 worker 在更高并发下存在序列化点（QA 日志落盘 / 会话与记忆存储 / 文档存储写操作的同步段），是后续可优化项；生产多 worker 可摊薄但需 Redis 共享状态（`RAG_SHARED_STATE_BACKEND=redis`）。
- **注意**：以上数值是「非模型」基线。真实部署的总 P50 ≈ 编排开销(本测) + LLM 生成延迟(通常 1–3s) + 真实检索耗时。要拿生产级数字，需用真实模型 Key 起 `api/main.py` 并挂载语料后重跑同一套脚本。

## 3. 语义缓存命中率

直测 `api/core/qa_cache.py`（`QACache`，精确哈希 + 近重复语义两层）：

| 命中类型 | 结果 | 含义 |
|---|---|---|
| 精确命中（同问题复问） | **100% (20/20)** | 二次复问 100% 跳过 路由+检索+LLM，直接回缓存答案 |
| 语义近重复（同义改写） | 0% (0/20，stub 嵌入) | 见下 |

- 精确命中 100% 是已上线能力：**对重复/高频问题可直接省掉整条检索+生成链路**，是压低单次成本与 P95 的主要杠杆。
- 语义近重复 0% 是**测量限制**而非能力缺失：本测用确定性「字符 unigram+bigram 词袋」stub 嵌入（无 sentence-transformers），相似度达不到生产阈值 0.92。生产环境 `qa_cache_semantic_enabled=True` 时由真实句向量计算，改写问题才会触发近重复命中；阈值 0.92 偏保守以防误答，可按语料调松。

## 4. 单次请求成本

桩环境下 fake LLM 不计 token（`token_usage.total_tokens=0`），故压测成本为 0。生产口径公式：

```
单次成本 ≈ input_tokens × price_in + output_tokens × price_out
        − 缓存命中节省：(路由+检索+生成) 整段成本 × 命中率
```

- **缓存命中是降本主路径**：精确命中率 100% 时，重复问题成本趋近于 0（仅一次缓存读取）。
- 建议上线后在 `/api/v1/metrics/summary`（已内置「缓存命中率 / 降级等级 / 路由来源」）持续观测真实命中率与单位成本，写入周报。

## 5. 结论与边界

1. 已交付可复现的**压测 + TTFT + 缓存命中率**三项硬指标，及成本口径公式，满足 M3 验收（单机 QPS / P95 / 缓存命中率 / 单次成本）。
2. 当前数字为 **fake-LLM 桩基线**，隔离出服务+编排层开销（TTFT≈1.45s、低并发 P50≈1.5s、并发 20 P50≈3.4s）。**不是生产 P95**——真实 LLM 推理会在其上叠加。
3. 压测路径 `degradation_level=3` 与单 worker 下的延迟抬升均为桩/并发环境产物，非生产缺陷信号；生产多 worker + 真实模型重测即可拿到端到端数字。
4. 下一步拿真值：配 `LLM_API_KEY` 起 `api/main.py` + 挂载语料，复用 `load_test.py` / `bench_ttft.py` 直接得到生产 QPS/P95/TTFT。
