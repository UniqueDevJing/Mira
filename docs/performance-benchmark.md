# RAG 2.0 性能基准测试报告

日期: 2026-08-10
状态: 首次全链路压测，非生产调优基准

## 测试环境

| 项 | 值 |
|---|---|
| 硬件 | 本机 (Windows 11) |
| 服务 | uvicorn 单进程, `api.main:app`, 127.0.0.1:8000 |
| 模型 | TokenHub `deepseek-v4-flash` (OpenAI 兼容) |
| 存储 | LanceDB 3 库 (documents/service/tech), 已有测试文档 |
| 测试数据 | 10 问题轮询 (客服 6 + 技术 4) |
| 客户端 | `scripts/load_test.py` (httpx, 30s client timeout) |

**注意:** 本次测试使用放宽超时预算验证真实链路: `RAG_LLM_GENERATE_TIMEOUT_S=12`, `RAG_TOTAL_TIMEOUT_S=15`。默认配置 (LLM 2s / total 5s) 下 LLM 阶段必超时, 详见"发现"。

## 压测结果

| 并发 | 请求数 | QPS | P50 | P95 | P99 | 正常(L0) | 降级(L3) |
|---|---|---|---|---|---|---|---|
| 5 | 25 | 1.5 | 2.9s | 6.3s | 6.9s | 100% | 0% |
| 10 | 40 | 1.7 | 4.6s | 9.7s | 11.3s | 100% | 0% |
| 20 | 40 | 3.3 | 4.9s | 7.6s | 7.7s | 100% | 0% |
| 40 | 60 | 2.8 | 9.9s | 13.8s | 13.8s | 72% | 28% |

单请求延迟分解 (L0 典型): router 35ms + retrieval 68ms + rerank 200ms + **LLM 6200ms**, total 6.5s。

## 核心结论

1. **LLM 生成是绝对瓶颈** — 占单请求 95% 延迟 (6.2s / 6.5s)。检索/路由/重排合计仅 ~300ms。
2. **QPS 天花板 ~3** — 并发 20 达 3.3 峰值, 40 并发不升反降 (2.8) 且出现 28% L3 降级。
3. **TokenHub 并发受限** — 40 并发时 17/60 请求 LLM 超时降级, 说明上游对并发 LLM 调用有限制或超时退化。

## 发现 (按严重度)

### P0-1: 默认 LLM 超时预算不现实 (设计缺陷) — 已修 2026-08-10

- 默认 `llm_generate_timeout_s=2.0` < 模型 TTFT 3.5s
- DeepSeek-v4-flash 强制带 reasoning (`thinking: False` 无效, reasoning_content 固定输出), 首 token 无法关闭
- 结果: 默认配置下 **每次 QA 的 LLM 阶段必超时 → 100% L3 降级**, 用户永远拿不到 LLM 答案
- **修复**: `config.py` `llm_generate_timeout_s 2.0→8.0`, `total_timeout_s 5.0→12.0` (实测 6.2s 单次生成)
- 后续优化: LLM 流式化首 token 快返 (待办)

### P0-2: DevSidecar 代理拖死 LLM 调用 (部署坑) — 已修 2026-08-10

- 本机 `HTTP_PROXY/HTTPS_PROXY` 指向 DevSidecar, uvicorn 继承后 LLM 外呼走代理
- 症状: 单请求 LLM 12s 超时 (代理路径), direct 调用 4.5s 正常; 排查后加 `NO_PROXY=tokenhub.itcast.cn` 恢复
- **修复**: `llm_client.py` 两个 httpx client 加 `trust_env=False` — 代码级禁代理, 任意启动方式 (IDEA/Docker/CI) 都不受影响。tokenhub 为国内站点无需代理

### P1-1: load_test client 超时误报 500

- 原 `timeout=10.0` 小于 LLM 生成时长, 客户端中断被计为 `500` (实际是代理错误页)
- 已修: `CLIENT_TIMEOUT_S=30.0`
- 另: 本机 DevSidecar 拦截 localhost 会返回错误页, 压测需设 `NO_PROXY=127.0.0.1,localhost`

### P1-2: TokenUsage 仅在 L0 返回

- L3 降级时 `token_usage=None`, 成本追踪在降级场景丢失
- 需确认是否接受: 降级时无 LLM 调用, 本就无 token 消耗, 可视为合理

## 建议下一步

1. **修 P0-1** — 默认超时预算调至 8s 或模型换型, 否则系统处于半降级状态
2. **LLM 流式化** (已列待办 #5) — 首 token 3.5s 压到 ~1s, P95 可显著改善
3. **多 worker** — uvicorn `--workers N` 缓解单进程 CPU/连接瓶颈 (本测试单进程)
4. **并发度建议** — 线上 20 并发内安全, 超 40 触发 TokenHub 限流
5. **基准纳入 CI** — 固定并发 20/请求 40, 监控 QPS/P99 回归

## 第二轮压测 (2026-08-10, 流式 + SelfRetrieval 集成后)

默认配置 `RAG_LLM_GENERATE_TIMEOUT_S=8`, `RAG_TOTAL_TIMEOUT_S=12`。

### 压测结果 (20 并发 / 40 请求)

| 版本 | QPS | P50 | P95 | P99 | 降级 |
|---|---|---|---|---|---|
| SelfRetrieval 默认开 (回归) | 2.4 | 6.7s | 9.4s | 9.4s | 28 L2 |
| 默认关 | 2.8 | 4.7s | 7.8s | 8.0s | 16 L1 |
| + embedding 快路径 | **3.0** | **4.9s** | **7.7s** | **7.7s** | **40/40 L0** |

最终基线: 20 并发 QPS 3.0, P99 7.7s, 0 失败 0 降级 (与首轮一致)。

### 新增发现

- **P0-3: SelfRetrieval 默认开导致并发全量 L2 降级 (回归)** — 已修
  - `QARequest.enable_self_retrieval` 默认 `True`, 每请求跑多轮循环 (LLM 改写 + 评估)
  - 20 并发下 LLM 改写调用 + CPU embedding 串行 → 超过 3s 预算 → 超时 → L2 兜底单轮
  - **关键问题**: 兜底单轮 = 普通检索, 答案质量不变却白等 3s 超时, 纯延迟损失
  - **修复**: 默认改 `False` (config 可配 `RAG_ENABLE_SELF_RETRIEVAL`), 质量敏感场景显式开启
- **P1-3: rerank 重复 embed 文档 → 并发下 L1 跳过** — 已修
  - `Reranker._rerank_with_bi_encoder` 对无 `embedding` 字段文档每请求重新 embed (content[:512])
  - 20 并发 × 10 文档 = 200 次 CPU embedding 串行 → 0.5s rerank 预算必超 → 16/40 L1
  - **修复**: `vector_store.search` 返回存储向量, rerank 走 dot-product 快路径, 免重复 embed

## 测试命令

```bash
# 服务启动 (注意 NO_PROXY)
cd rag-2.0
NO_PROXY="127.0.0.1,localhost,tokenhub.itcast.cn" \
RAG_LLM_API_KEY=... RAG_LLM_GENERATE_TIMEOUT_S=12 RAG_TOTAL_TIMEOUT_S=15 \
./venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 压测
NO_PROXY="127.0.0.1,localhost" ./venv/Scripts/python.exe scripts/load_test.py \
  --url http://127.0.0.1:8000 --questions q.txt --concurrency 20 --requests 40
```
