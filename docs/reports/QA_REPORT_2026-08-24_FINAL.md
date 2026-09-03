# RAG 2.0 全面质检报告（2026-08-24 终验）

> 范围：背景/需求 → 设计 → 实现 → 测试 → 维护，多轮交叉质检。
> 结论：**发现并修复 4 个真实缺陷，消除测试偶发失败；全量测试 331 通过，应用可正常启动。**

## 一、背景与需求（复核）
- 系统目标：基于上传文档的 RAG 问答，低相关度时**干净拒答**而非编造（用户核心诉求"答非所问"）。
- 生产形态：本机 Windows 运行 `scripts/start_prod.py` + Cloudflare 隧道（非历史所述的 Docker 远程服务器）。
- 关键质量线：检索命中、置信度硬下限拒答、OCR 兜底乱码 PDF、提示注入缓解。

## 二、设计评审（自洽性）
- 路由→检索→重排→生成链路自洽；跨库兜底 + 置信度护栏构成双保险。
- 置信度硬下限（0.50）在**流式(1034行)与非流式(667行)**两条路径均已实现，拒答前不调用 LLM。
- 提示注入：上下文用`【片段开始/结束】`包裹 + 系统提示禁止执行文档内指令。
- OCR 兜底：乱码检测→RapidOCR，Dockerfile 已补 `libxcb1/libgl1/libglib2.0-0/libgomp1`。
- 设计无单点失效；架构债（10 个业务库为空、文档全挤 documents 表）为数据层面，不影响正确性。

## 三、实现层缺陷与修复（全部已验证）
| # | 文件 | 缺陷 | 影响 | 修复 |
|---|------|------|------|------|
| 1 | `api/routes/documents.py` | 缺 `import time`（第259行 `time.time()`） | 上传/空文档处理 `NameError`，4 个测试失败 | 补 `import time` |
| 2 | `engines/router/intent_router.py` | `CLASSIFY_PROMPT.format(question=...)` 二次解析 f-string 花括号 → `KeyError('"skill"')` | 意图路由恒降级 fallback，2 个测试失败 | 改用 `.replace("{question}", question)` |
| 3 | `api/core/document_store.py` | `purge_old_qa_logs` 误用未定义的 `self._conn` | 一旦接入留存清理定时任务即崩溃 | 改用 `_get_conn()` 上下文管理器 |
| 4 | `tests/conftest.py` | 隔离夹具仅 session 级 → 跨测试 KB 状态泄漏 | 残留文档触发置信度护栏误拒答，造成顺序相关偶发失败 | 新增 function 级 `_reset_state_per_test`，每测试全新 vector_uri+数据目录+清空单例 |

> 注：历史《QA_REPORT_2026-08-24.md》所列部分 P0（两阶段提交、LLM 熔断降级、0.0.0.0 绑定强 403、BM25 改 JSON、embedder 512 截断）经核查**已在先前提交中修复**，本报告不予重复。

## 四、安全 / 前端（XSS）
- 答案：流式用 `textContent`，终态走 `renderMarkdown`——`markdown.js` 先整体 `esc()` 转义再格式化，仅允许 `https?://` 链接 → 投毒文档 `<img onerror>` 被转义，**无 XSS**。
- 来源/用户问题/技能/路由来源均经 `escapeHtml`（`&<>"'` 全覆盖）。
- 密钥：`.env` 已 gitignore；`start_prod.py` 从环境变量读 `RAG_API_KEY`，缺失即报错，无硬编码。
- 无裸 `except`、无 `eval/exec`、无非测试代码调试 `print`。

## 五、测试与维护
- 全量：`331 passed`（修复前 325 passed / 6 failed），且与执行顺序无关。
- 运行冒烟：`uvicorn api.main:app` 启动成功，`/health` → HTTP 200。
- 部署：Dockerfile OCR 系统库齐备；依赖隔离在 venv。

## 六、遗留项（非阻塞，建议跟进）
1. 业务库（tech/service/...）当前为空，25 份文档集中在 `documents` 表——建议按业务补录或调整路由映射，提升检索精度。
2. 乱码源 PDF（如《一次性求职补贴发放通知》）需用户重传原件；重传时 OCR 兜底自动生效。
3. 建议在 CI 中加入 `pytest` 门禁，防止回归。
