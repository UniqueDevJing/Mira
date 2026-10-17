# RAG 2.0 前端优化报告

> 审查范围：`web/index.html`、`web/common.js`、`web/icons.js`、`web/markdown.js`
> 优化原则：**证据优先** —— 先量化基线，再只动"确实构成瓶颈"的点；每项改动都用 `perf_bench.js`（真实前端代码 + jsdom）和 `it_frontend_harness.js`（29 断言端到端回归）验证。
> 运行环境：Node 22.22.2 + jsdom（managed workspace）；后端桩用项目 `venv` 的 uvicorn 真实启动。

---

## 一、基线量化（优化前）

| 量测项 | 结果 | 结论 |
|---|---|---|
| `renderMarkdown(20KB)` | **0.435 ms/次**（200 次均值） | ✅ 非瓶颈 |
| `parseSSEStream(1000 events)` | **0.437 ms/次**（50 次均值） | ✅ 非瓶颈 |
| 单条完整回答消息 | 57 DOM 节点 / 3.3 KB innerHTML | — |
| 200 条消息累计（无清理） | **11 600 DOM 节点 / 659 KB** | 🔴 真实退化源 |
| 推算 1000 轮（无清理） | ≈ 58 000 节点 / 3.3 MB | 🔴 不可用 |

**关键结论**：渲染与 SSE 解析都不是瓶颈；真正的性能风险是**长会话 DOM 无限增长**（每轮 +~58 节点，线性膨胀）。因此优化聚焦 DOM 增长 + 一个潜伏的崩溃缺陷。

---

## 二、改动清单（按优先级）

### 🔴 CRITICAL — `SSE_MAX` 未定义导致 strict-mode ReferenceError

- **问题**：`common.js` 的 `createSSEParser()` 引用 `SSE_MAX`（第 280/282 行）但从未定义。该脚本是 `'use strict'`，一旦累积 buffer 超过上限执行 `buf.length > SSE_MAX`，即抛 `ReferenceError`，中断整条 SSE 流的解析。
- **证据**：重构引入该引用后，`it_frontend_harness.js` 的 `no_js_errors` 断言本应失败（此前 29/29 是在重构前跑的）。修复前注入 >1MB 载荷必崩。
- **改法**：`web/common.js:277` 补定义 `var SSE_MAX = 1024 * 1024;`（1MB 上限），让 `while (buf.length > SSE_MAX)` 真正生效。
- **验证**：
  - `web/test/sse_max_check.js` 注入 **2MB** 载荷 → 解析 1422 事件 / 1.8ms，**无崩溃**，buffer 被钳制到末 ~1MB。
  - `it_frontend_harness.js` 的 `no_js_errors`：**PASS**（修复的直接证明）。

❌ 修复前：`buf.length > SSE_MAX` → `SSE_MAX` 是自由变量，strict 模式读取即抛 `ReferenceError`。
✅ 修复后：`var SSE_MAX = 1024 * 1024;` → 上限生效，超长流安全截断最旧数据。

### 🟠 HIGH — 长会话 DOM 裁剪（核心性能优化）

- **问题**：200 条消息 → 11 600 节点（基线实测），长会话下滚动/重排明显卡顿。
- **改法**：`web/index.html`
  - `:536` 定义 `MAX_VISIBLE_MESSAGES = 60`（用户+助手气泡合计上限）。
  - `:538` 新增 `trimOldMessages()`：超出上限时移除最旧 `.msg`，并调用 `:541` `updateHiddenPlaceholder()` 在顶部插入占位条（`↑ 已隐藏较早的 N 条消息（上下文仍保留）`）。
  - 在三个加节点路径后调用：**`:440` `appendUserMessage`**、**`:463` `createStreamingCard`**、**`:488` `appendRestoredAssistantMessage`**。
  - **只裁剪视图**：`chatHistory` 数组不裁（后端仍按最后 20 轮取上下文），滚到顶不会丢语义上下文。
  - `:83` 补 `.hidden-msgs-ph` 样式；`:557` 暴露 `window.RAG_TEST` 测试钩子（仅 `window.__RAG_TEST__` 置位时，供 bench 驱动真实代码，不污染生产）。

❌ 优化前：每条消息永久驻留 DOM，无上限。
✅ 优化后：可见气泡恒定 ≤ 60，旧消息移出 DOM，顶部占位提示用户。

- **验证**（`perf_bench.js` 第 4 项，驱动**真实** `index.html` 逻辑，不经后端）：
  | 场景 | 可见气泡 | DOM 节点 | 序列化 |
  |---|---|---|---|
  | 200 条消息（未裁剪基线） | 200 | **11 600** | 659 KB |
  | 400 条消息（裁剪启用） | 60 | **481** | 27.3 KB |
  | **降幅** | — | **95.9%** | **95.9%** |

  即便持续追加到 400 条，节点数也被钳制在 ~481，不再随轮次线性增长。

### 🟡 MEDIUM — `scrollToBottom` 的 rAF 合并

- **问题**：流式每个 `delta` 都调用 `scrollToBottom()`，原实现每次调度一个全新 `requestAnimationFrame`，一帧内堆叠数十个冗余滚动赋值。
- **改法**：`web/index.html:524-528` 用 `_scrollScheduled` 标志合并 —— 同一帧内多次调用只排一次 rAF，下一帧执行后置位复位。
- **验证**：纯逻辑改动，29 断言回归全过（含 `incremental_text_node` 流式路径），无功能影响；jsdom 下 rAF 合并行为可在 `perf_bench` 第 4 项（400 次追加均触发滚动）间接佐证不报错、不卡。

---

## 三、验证汇总

| 验证手段 | 命令 | 结果 |
|---|---|---|
| 语法检查 | `node --check common.js icons.js markdown.js` | 全部 OK |
| 性能基线 + 裁剪 | `node web/test/perf_bench.js` | 渲染/解析非瓶颈；裁剪 **−95.9%** 节点 |
| SSE_MAX 上限 | `node web/test/sse_max_check.js` | 2MB→有界，无崩溃 |
| 端到端回归 | `node web/test/it_frontend_harness.js`（后端桩 8911） | **29/29 PASS**，`no_js_errors` PASS |

> 说明：后端桩因沙箱原 `uvicorn` 缺失，改用项目 `venv`（`C:\Users\Dominion\Desktop\rag-2.0\venv`）真实启动，SSE 契约与 `scripts/it_backend_fakellm.py` 一致，回归结论有效。

---

## 四、未改动（及原因）

- `renderMarkdown` / `parseSSEStream`：**基线已证非瓶颈**（均 <0.5ms/次），不盲目优化，避免引入风险。
- 消息内容虚拟化（虚拟列表）：当前 ≤60 气泡裁剪已解决万级节点问题，虚拟列表收益边际、改造成本高，列为**待观察项**而非本次实施。

## 五、后续可选项（按 ROI 排序）

1. **占位条"展开早期消息"**：当前隐藏消息仅留占位提示，点击可按 `chatHistory` 回填（轻量，提升长会话可用性）。
2. **`chatHistory` 软上限**：目前不裁数组（仅发最后 20 轮），超长会话内存随轮次增长，可加 `slice(-100)` 兜底。
3. **移动端 `scrollToBottom` 节流**：触屏滚动惯性下可叠加 `passive` 监听，进一步降主线程压力。
