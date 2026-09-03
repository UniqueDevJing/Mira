# 前端全面优化说明（2026-08-25）

对 `web/` 前端（index.html / common.css / common.js / icons.js）做了一次 premium 级优化，未改动任何后端逻辑与接口契约。

## 一、主题系统：三态升级（light / dark / system）

- **原状**：仅 light/dark 二分切换，无"跟随系统"选项。
- **优化**：
  - `common.js` 主题模块重写：偏好存 `localStorage.rag_theme`（`light|dark|system`，默认 system）；`data-theme` 始终写"生效值"，CSS 仍只需两态，**与 admin 页面完全兼容**。
  - system 模式下通过 `matchMedia('(prefers-color-scheme: dark)')` 实时跟随系统切换，无需刷新。
  - 主题按钮三态循环：亮色 → 暗色 → 跟随系统 → 亮色；图标联动 sun / moon / monitor（icons.js 新增 `monitor` 图标）。
  - 按钮 title/aria-label 动态提示当前模式；`<meta name="theme-color">` 随主题同步（移动端地址栏配色）。
  - head 内 FOUC 防护脚本升级支持三态，首帧无闪烁。

## 二、视觉与动效增强（CSS）

| 项 | 效果 |
|---|---|
| 卡片悬浮 | 桌面悬停设备卡片上浮 2px + 阴影加深（`@media (hover:hover)` 限定，移动端不误触；对话/来源面板除外） |
| 主按钮流光 | `.btn-primary` hover 时一道高光扫过 |
| 管理后台链接 | hover 变主色 + 渐变底 |
| 流式打字光标 | 回答生成期间答案尾部闪烁光标，结束/停止自动消失 |
| 环形仪表入场 | 指标环缩放淡入动画 |
| 空状态图标 | 呼吸浮动动画 |
| 空输入提示 | 空问题提交时输入框抖动 + 红框聚焦 |
| toast 轻提示 | 底部浮出提示（复制成功/失败等），自动消失 |
| 平滑滚动 | 对话区/来源面板 `scroll-behavior: smooth` |
| 无障碍 | `prefers-reduced-motion: reduce` 下全局关闭动画/过渡 |

## 三、交互增强（JS）

- **复制回答**：每条回答 meta 行新增"复制"按钮（事件委托绑定，无内联 onclick）；优先 `navigator.clipboard`，旧浏览器降级 textarea 方案；复制纯文本（已渲染前的内容，天然无 XSS）。
- **空输入 shake**：`askQuestion` 空问题时抖动聚焦，不再静默返回。
- **主题按钮**：图标/title/aria 随模式动态更新。

## 四、验证结果

- `node --check`：common.js / icons.js / markdown.js / index.html 内联脚本（2 块）全部通过。
- 服务冒烟（uvicorn 8013）：`/` 200（51.6KB 完整页面）、`/web/common.css`、`/web/common.js`、`/web/icons.js`、`/web/markdown.js`、`/health` 全部 200。
- XSS 面复核：新增 DOM 操作均使用 `textContent` / `escapeHtml` / 纯文本复制，未引入新的 innerHTML 注入点（复制按钮 data-target 为内部生成的 answerId，非用户输入）。

## 五、改动文件

- `web/icons.js` — 新增 monitor/copy/check 图标与别名
- `web/common.js` — 主题三态重写（兼容 admin）
- `web/index.html` — FOUC 脚本升级、内联样式动效层、交互 JS

预览：本地 `python -m uvicorn api.main:app` 后访问 `/` 即可体验。
