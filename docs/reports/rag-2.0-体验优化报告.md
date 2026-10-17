# RAG 2.0 前端体验优化报告（体验三连击）

> 范围：用户消息/助手消息渲染链路、复制、滚动行为。
> 原则：**先量后动、每项改动均经真实数据验证**（专项 check + 集成回归 harness）。
> 环境：前端 `web/`（原生 JS，无构建步骤）；验证用 jsdom 驱动真实前端代码 + 项目 `venv` 内真实 uvicorn 后端桩（8911）。

## 基线回顾（上一轮已确认的非瓶颈）
- `renderMarkdown(20KB)` ≈ **0.435ms/次**、`parseSSEStream(1000 events)` ≈ 0.437ms/次 → 渲染/解析**不是**瓶颈，本轮不动它们。
- 上轮已解决：200 条消息 DOM 节点 11 600 → 裁剪后 481（降幅 95.9%）。

## 本轮三项改动

### 1. 流式 Markdown 渲染（体感最明显）
- **问题**：流式中答案以**原始文本节点**逐字追加（`updateStreamingAnswer` 用 `appendData`），直到 `finalizeAnswer` 才一次性 `renderMarkdown`。长答案流式输出时是一坨无排版的纯文本，结束瞬间"啪"一下变成标题/代码块/列表，割裂感强。
- **证据**：`index.html:473` 原实现 `else _streamNode.firstChild.appendData(delta)`；`index.html:481` 区域仅在 finalize 调 `RAG.renderMarkdown`。
- **改法**：`index.html:469` 引入 `_streamFull`（单一真相源）+ rAF 合并标记；`updateStreamingAnswer`（`index.html:473`）改为每帧把已接收全文 `renderMarkdown` 一次重渲进 `_streamNode`（rAF 合并，多个 delta 同一帧只渲一次）；`startStream` 用 `_streamFull` 替代旧 `fullAnswer` 累积。
- **预期收益**：答案随流**渐进格式化**，无结尾突跳；因 `renderMarkdown` 仅 0.435ms 且每帧最多一次，成本可忽略。

### 2. 复制按钮去重（DOM/内存）
- **问题**：`finalizeAnswer` 把整段答案又塞进 `data-copy-answer="…"` 属性（`index.html:478` 旧代码 `RAG.escapeHtml(fullAnswer)`），一份正文渲染 + 一份全文属性 = **答案体积翻倍进 DOM**，吃掉上轮裁剪收益。
- **改法**：`index.html:506` 改为 `data-answer-id` 标记；全文存入 `_answerCache[ansId]`（`index.html:507`）；点击时从缓存读取（`index.html:373` 区域）；`trimOldMessages`（`index.html:582`）移除旧消息时同步 `delete _answerCache[mid]`，长会话缓存有界。
- **预期收益**：答案体积进 DOM **归零重复**；长会话内存可控（缓存随 DOM 裁剪同步清理）。

### 3. 滚动解绑 / 贴底（不被拽走）
- **问题**：`scrollToBottom` 每个 delta 都强制滚到底，用户往上翻看历史时，新 token 持续把视口拽回底部，无法阅读。
- **改法**：`index.html:556` 引入 `_autoScroll` 标志；`scrollToBottom`（`index.html:562`）仅在贴底时滚动；`chatFlow` 的 `scroll` 事件（`index.html:557` `_onChatScroll`）据距底 <80px 切换贴底态并显隐"回到底部"浮动按钮（`index.html:121` 新增按钮 + CSS）。
- **预期收益**：用户上翻时流式输出不再打扰；回到底部按钮提供一键返回，且贴底时自动隐藏。

## 验证结果（全部真实执行）

| 验证 | 命令 | 结果 |
|---|---|---|
| 语法检查 | `node --check` ×7 文件 | 全 OK |
| 专项·流式 Markdown + 复制去重 | `web/test/stream_md_check.js` | **10/10** STREAM_MD_OK |
| 专项·滚动解绑 | `web/test/scroll_behavior_check.js` | **9/9** SCROLL_OK |
| 集成回归（真实 SSE） | `web/test/it_frontend_harness.js` | **31/31** INTEGRATION_OK |

**关键断言证据**
- `streaming_markdown_rendered`：流式中途 `answer-text` 已含 `<p>/<strong>/<li>/<pre><code>`，且 `stream html === renderMarkdown(full)`（渐进渲染与最终一致）。
- `copy_no_full_text_attr`：复制按钮 `data-copy-answer` 属性长度 **0**（原代码承载全文，现已归零）；`answer_cached` 确认全文入缓存。
- `unpinned_no_yank`：上翻（`_autoScroll=false`）后 `scrollToBottom` 不改变 `scrollTop`（不再被拽走）；`unpinned_button_shown`/`click_returns_to_bottom` 确认回到底部按钮行为。
- `no_js_errors`（集成 + 专项）全 PASS → 无功能回归；XSS 转义、SSE 序列、主题/会话/停止/错误降级等原有 29 项断言不受影响。

## 改动文件清单
- `web/index.html`：流式渲染（469/473）、复制去重（373/506/507/582）、滚动解绑（121/556/557/562）、测试钩子扩展、CSS 按钮。
- `web/test/stream_md_check.js`（新增）：驱动真实 `createStreamingCard/updateStreamingAnswer/finalizeAnswer`。
- `web/test/scroll_behavior_check.js`（新增）：桩几何验证 `stick-to-bottom`。
- `web/test/it_frontend_harness.js`：原 `incremental_text_node` 断言升级为 `streaming_markdown_rendered`（流式已变元素节点），新增 `copy_button_marker_present`/`copy_no_full_text_attr`。

## 诚实说明
- 滚动解绑的"距底 80px"阈值与按钮样式为合理默认，未做多视口人工目检（jsdom 无布局）；逻辑已由几何桩全覆盖。
- 历史会话（REST 接口恢复）仍走 `appendRestoredAssistantMessage` 的 `escapeHtml` 纯文本路径——与流式最终渲染不一致，属已知小不一致，**未纳入本轮**（保持范围聚焦），建议后续单独统一为 Markdown。

## 后续可选项（未实施）
- 历史消息统一 Markdown 渲染
- 检索过程渐进可视化（meta 事件已有路由/重排字段，当前仅静态三点）
- 代码块复制按钮 + 轻量高亮
- 答案"重新生成 / 👍👎"反馈
