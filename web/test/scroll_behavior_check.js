#!/usr/bin/env node
/* 专项验证 · 滚动解绑 (stick-to-bottom) (无需后端, 直接驱动真实前端状态)
 *
 * 通过 RAG_TEST 钩子验证:
 *   1. 贴底时 _autoScroll=true, scrollToBottom 把 scrollTop 拉到 scrollHeight
 *   2. 用户上翻(_autoScroll=false)时, scrollToBottom 不改变 scrollTop(不被新 token 拽走)
 *   3. chatFlow scroll 事件正确切换 _autoScroll(顶部=false, 底部=true)
 *   4. 回到底部按钮显隐与 _autoScroll 同步
 */
'use strict';

const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const WEB_DIR = path.resolve(__dirname, '..');
const commonJs = fs.readFileSync(path.join(WEB_DIR, 'common.js'), 'utf8');
const iconsJs = fs.readFileSync(path.join(WEB_DIR, 'icons.js'), 'utf8');
const markdownJs = fs.readFileSync(path.join(WEB_DIR, 'markdown.js'), 'utf8');
const indexHtml = fs.readFileSync(path.join(WEB_DIR, 'index.html'), 'utf8');
const inlineJs = [...indexHtml.matchAll(/<script>([\s\S]*?)<\/script>/g)].pop()[1];

const virtualConsole = new VirtualConsole();
const jsErrors = [];
virtualConsole.on('jsdomError', (e) => jsErrors.push('jsdomError: ' + (e.detail && e.detail.message ? e.detail.message : e.message)));

const dom = new JSDOM(indexHtml, { url: 'http://localhost/', runScripts: 'outside-only', pretendToBeVisual: true, virtualConsole });
const { window } = dom;
const doc = window.document;

window.matchMedia = window.matchMedia || (() => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} }));
window.addEventListener('error', (e) => jsErrors.push('window.error: ' + (e.error && e.error.stack ? e.error.stack : e.message)));

window.__RAG_TEST__ = true;
window.eval(commonJs);
window.eval(iconsJs);
window.eval(markdownJs);
window.eval(inlineJs);
doc.dispatchEvent(new window.Event('DOMContentLoaded'));

const T = window.RAG_TEST;
const chatFlow = doc.getElementById('chatFlow');
const btn = doc.getElementById('scrollBottomBtn');
const results = [];
function check(name, cond, detail) { results.push({ name, pass: !!cond, detail: detail || '' }); }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── 桩几何: jsdom 不计算布局, 手动定义 scrollHeight/clientHeight/scrollTop ──
let SH = 1000, CH = 400, ST = 0;
Object.defineProperty(chatFlow, 'scrollHeight', { configurable: true, get: () => SH });
Object.defineProperty(chatFlow, 'clientHeight', { configurable: true, get: () => CH });
Object.defineProperty(chatFlow, 'scrollTop', { configurable: true, get: () => ST, set: (v) => { ST = v; } });

(async () => {
  // 1) 贴底: _autoScroll=true → scrollToBottom 拉到底
  T.setAutoScroll(true);
  ST = 600; // 当前贴底(1000-600-400=0<80)
  T.scrollToBottom();
  await sleep(30);
  check('pinned_scrolls_to_bottom', ST === SH, `scrollTop=${ST} 期望=${SH}`);

  // 2) 上翻: 派发 scroll 事件(真实触发源) → _autoScroll=false, 按钮显示, scrollToBottom 不拽
  ST = 100; // 用户在顶部附近看历史
  chatFlow.dispatchEvent(new window.Event('scroll'));
  check('unpinned_button_shown', btn && btn.hidden === false, '回到底部按钮应显示');
  T.scrollToBottom();
  await sleep(30);
  check('unpinned_no_yank', ST === 100, `scrollTop=${ST} 期望保持 100(不应被拽到底)`);

  // 3) scroll 事件切换 _autoScroll
  ST = 100; chatFlow.dispatchEvent(new window.Event('scroll'));
  check('scroll_top_unpins', T.getAutoScroll() === false, '顶部附近应 _autoScroll=false');
  ST = 600; chatFlow.dispatchEvent(new window.Event('scroll'));
  check('scroll_bottom_pins', T.getAutoScroll() === true, '底部附近应 _autoScroll=true');
  check('pinned_button_hidden', btn && btn.hidden === true, '贴底时按钮应隐藏');

  // 4) 点击回到底部: 拉到底且重新贴底
  T.setAutoScroll(false); ST = 50;
  btn.dispatchEvent(new window.Event('click'));
  await sleep(30);
  check('click_returns_to_bottom', ST === SH, `scrollTop=${ST} 期望=${SH}`);
  check('click_repins', T.getAutoScroll() === true, '点击后应重新贴底');

  check('no_js_errors', jsErrors.length === 0, jsErrors.join(' | '));

  const passed = results.filter((r) => r.pass).length;
  const failed = results.filter((r) => !r.pass);
  console.log('\n===== 滚动解绑 (stick-to-bottom) 验证 =====');
  for (const r of results) console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}${r.detail ? '  [' + r.detail + ']' : ''}`);
  console.log(`\n通过 ${passed}/${results.length}`);
  console.log(failed.length === 0 ? 'SCROLL_OK' : 'SCROLL_FAIL');
  process.exit(failed.length === 0 ? 0 : 1);
})().catch((e) => { console.error('CHECK_ERROR: ' + (e && e.stack ? e.stack : e)); process.exit(3); });
