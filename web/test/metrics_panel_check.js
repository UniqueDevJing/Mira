#!/usr/bin/env node
/* 系统指标面板验证 · jsdom 加载真实前端 + 真实后端(8913) 拉取三接口, 断言 4 块渲染。
 *
 * 覆盖:
 *   - 侧栏「系统指标」按钮经 initMetricsPanel 绑定后, 点击可打开面板(遮罩 hidden=false)
 *   - ① 文档类型×切分策略: 渲染 .strat-card (来自 /type-strategies)
 *   - ② 核心指标: 渲染 .metric-card 与 3 个 .stage-row (来自 /metrics/summary)
 *   - ③ 本轮耗时与降级: 注入 _ragLastDone 后渲染 .latency-viz + .deg-badge
 *   - ④ 异常与降级兜底: 历史降级 .deg-row(4 级) + 趋势 .trend-col + 路由 .chip (来自 /metrics/history)
 *   - 无 XSS: 全量渲染结果中不含任何注入的 <script> 元素
 *   - 加载过程无 JS 异常
 *
 * 运行: node web/test/metrics_panel_check.js   (后端需已在 8913 启动)
 */
'use strict';

const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const WEB_DIR = path.join(PROJECT_ROOT, 'web');
const BASE = process.env.MP_SERVER || 'http://127.0.0.1:8913';
const API = BASE + '/api/v1';

// ── 读取真实前端资源 ──
const commonJs = fs.readFileSync(path.join(WEB_DIR, 'common.js'), 'utf8');
const iconsJs = fs.readFileSync(path.join(WEB_DIR, 'icons.js'), 'utf8');
const markdownJs = fs.readFileSync(path.join(WEB_DIR, 'markdown.js'), 'utf8');
const indexHtml = fs.readFileSync(path.join(WEB_DIR, 'index.html'), 'utf8');
const inlineMatch = [...indexHtml.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const inlineJs = inlineMatch[inlineMatch.length - 1][1];

// ── 构建 jsdom (仅执行手动注入的脚本) ──
const virtualConsole = new VirtualConsole();
const jsErrors = [];
virtualConsole.on('jsdomError', (e) => {
  jsErrors.push('jsdomError: ' + (e.detail && e.detail.message ? e.detail.message : e.message));
});
const dom = new JSDOM(indexHtml, {
  url: 'http://localhost/',
  runScripts: 'outside-only',
  pretendToBeVisual: true,
  virtualConsole,
});
const { window } = dom;
const doc = window.document;

// ── 注入 Node 原生全局 + 相对 URL 解析(模拟浏览器同源) ──
const nodeFetch = globalThis.fetch;
window.fetch = (u, o) => nodeFetch(typeof u === 'string' && u.startsWith('/') ? (BASE + u) : u, o);
window.TextDecoder = globalThis.TextDecoder;
window.AbortController = globalThis.AbortController;
if (!window.crypto || !window.crypto.randomUUID) {
  window.crypto = require('node:crypto').webcrypto;
}
if (!window.matchMedia) {
  window.matchMedia = () => ({
    matches: false, media: '', onchange: null,
    addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent() { return false; },
  });
}
window.addEventListener('error', (e) => jsErrors.push('window.error: ' + (e.error && e.error.stack ? e.error.stack : e.message)));

// ── 加载真实前端脚本 ──
window.eval(commonJs);
window.RAG.API = API;        // 内联 IIFE 启动时捕获 RAG.API
window.eval(iconsJs);
window.eval(markdownJs);
window.__RAG_TEST__ = true;  // 暴露 window.RAG_TEST
window.eval(inlineJs);
doc.dispatchEvent(new window.Event('DOMContentLoaded'));  // 触发 init() → initMetricsPanel()

// ── 结果收集 ──
const results = [];
function check(name, cond, detail) {
  results.push({ name, pass: !!cond, detail: detail || '' });
}

// 注入「本轮」done 事件, 验证 ③ 单轮耗时与降级徽章
window._ragLastDone = {
  latency_breakdown: { router_ms: 12, retrieval_ms: 240, rerank_ms: 35, llm_ms: 1800, total_ms: 2120 },
  degradation_level: 2,
};

(async function run() {
  const openBtn = doc.getElementById('metricsOpenBtn');
  check('metrics_button_exists', !!openBtn, '侧栏「系统指标」按钮存在');
  if (openBtn) {
    openBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  }
  // loadMetrics 为异步(顺序拉取 3 个接口), 等待其完成
  await new Promise((r) => setTimeout(r, 2000));

  const overlay = doc.getElementById('metricsOverlay');
  check('panel_opens_on_click', overlay && overlay.hidden === false, '点击后遮罩 hidden=false, 面板打开');

  const body = doc.getElementById('metricsBody');
  const html = body ? body.innerHTML : '';
  const text = body ? body.textContent : '';

  // 4 个区块标题 (① ② ③ ④)
  const secIdx = body ? body.querySelectorAll('.sec-idx') : [];
  check('four_sections_present', secIdx.length === 4, '4 个区块(①切分策略 ②核心指标 ③本轮耗时 ④降级兜底)均渲染, 实得 ' + secIdx.length);

  // ① 切分策略
  const stratCards = body ? body.querySelectorAll('.strat-card') : [];
  check('section1_strategies', stratCards.length >= 5, '① 渲染文档类型×切分策略卡片, 实得 ' + stratCards.length + ' 张');

  // ② 核心指标 + 阶段条
  const metricCards = body ? body.querySelectorAll('.metric-card') : [];
  const stageRows = body ? body.querySelectorAll('.stage-row') : [];
  check('section2_metric_cards', metricCards.length >= 4, '② 核心指标卡片, 实得 ' + metricCards.length);
  check('section2_stage_bars', stageRows.length === 3, '② 阶段平均耗时条(检索/重排/LLM)=3, 实得 ' + stageRows.length);

  // ③ 本轮耗时与降级
  const latViz = body ? body.querySelector('.latency-viz') : null;
  const segs = body ? body.querySelectorAll('.latency-viz .lat-seg') : [];
  const degBadge = body ? body.querySelector('.latency-viz .deg-badge') : null;
  check('section3_latency_viz', !!latViz && segs.length >= 4, '③ 渲染耗时堆叠条(≥4 段), 实得 ' + segs.length);
  check('section3_degradation_badge', !!degBadge && /降级 L2/.test(text), '③ 渲染降级徽章(注入 L2)');
  check('section3_total_shown', /2\.12s/.test(text), '③ 总耗时 fmtMs 正确(2120ms→2.12s)');

  // ④ 异常与降级兜底(历史基线)
  const degRows = body ? body.querySelectorAll('.deg-row') : [];
  const trendCols = body ? body.querySelectorAll('.trend-col') : [];
  const chips = body ? body.querySelectorAll('.chip-row .chip') : [];
  check('section4_deg_levels', degRows.length === 4, '④ 降级等级分布 4 行(L0~L3), 实得 ' + degRows.length);
  check('section4_trend', trendCols.length >= 1, '④ 按天趋势柱状图, 实得 ' + trendCols.length + ' 列');
  check('section4_routing_chips', chips.length >= 1, '④ 路由来源分布 chip, 实得 ' + chips.length);
  check('section4_history_data', /1632/.test(text), '④ 含历史降级 L3=1632 条(来源于 qa_export.json)');

  // 无 XSS: 渲染结果中不得存在注入的 <script> 元素
  const injected = body ? body.querySelectorAll('script') : [];
  check('no_xss_script_injected', injected.length === 0, '面板渲染无注入 <script> 元素');

  // 无 JS 异常
  check('no_js_errors', jsErrors.length === 0, jsErrors.join(' | '));

  // ── 输出 ──
  let pass = 0;
  for (const r of results) {
    console.log((r.pass ? '  PASS ' : '  FAIL ') + r.name + (r.pass ? '' : '  → ' + r.detail));
    if (r.pass) pass++;
  }
  console.log('\n指标面板: ' + pass + '/' + results.length + (pass === results.length ? '  ALL_OK' : '  HAS_FAILURE'));
  process.exit(pass === results.length ? 0 : 1);
})();
