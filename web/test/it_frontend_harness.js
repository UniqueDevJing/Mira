#!/usr/bin/env node
/* 前端联调执行化测试 · jsdom 驱动真实前端代码 + 真实后端 SSE。
 *
 * 加载 web/index.html 的真实脚本(common.js / icons.js / markdown.js / 内联 IIFE), 注入 Node 原生
 * fetch/TextDecoder/AbortController/localStorage/matchMedia, 把 RAG.API 指向本地后端桩, 发起真实
 * SSE 请求, 逐条断言前端全链路:
 *   - 用户消息 / 助手消息渲染
 *   - renderMarkdown 输出(<p>/<strong>/<li>/<h1>/<pre><code>)
 *   - 增量 appendData 文本节点(流式顺滑路径)
 *   - XSS 转义(无原始 <script>/<img> 执行)
 *   - sources 面板渲染(含注入的确定来源)
 *   - SSE 事件序列 meta→sources→delta*→done
 *   - 主题切换 / 会话持久化 / 停止生成 / 错误降级 toast
 */
'use strict';

const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const WEB_DIR = path.join(PROJECT_ROOT, 'web');
const SERVER = process.env.IT_SERVER || 'http://127.0.0.1:8911';
const API = SERVER + '/api/v1';
const QUESTION = '这个 RAG 系统的架构是怎样的？';

// ── 读取真实前端资源 ──
const commonJs = fs.readFileSync(path.join(WEB_DIR, 'common.js'), 'utf8');
const iconsJs = fs.readFileSync(path.join(WEB_DIR, 'icons.js'), 'utf8');
const markdownJs = fs.readFileSync(path.join(WEB_DIR, 'markdown.js'), 'utf8');
const indexHtml = fs.readFileSync(path.join(WEB_DIR, 'index.html'), 'utf8');
const inlineMatch = [...indexHtml.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const inlineJs = inlineMatch[inlineMatch.length - 1][1];

// ── 构建 jsdom (仅执行我们手动注入的脚本) ──
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

// ── 注入 Node 原生全局 + 桩 ──
const nodeFetch = globalThis.fetch;
window.fetch = (...args) => nodeFetch(...args);
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
window.RAG.API = API; // 必须在加载内联脚本前覆盖(内联 IIFE 启动时捕获 RAG.API)
window.eval(iconsJs);
window.eval(markdownJs);
window.__RAG_TEST__ = true; // 暴露 RAG_TEST(含 getPipelineState) 供渐进可视化断言
window.eval(inlineJs);
doc.dispatchEvent(new window.Event('DOMContentLoaded'));

// ── 拦截 SSE 事件序列(前端真实消费路径) ──
const seenTypes = [];
const seenSources = [];
const origParse = window.RAG.parseSSEStream;
const parseCalls = [];
window.RAG.parseSSEStream = (chunk) => {
  const evs = origParse(chunk);
  parseCalls.push({ chunkLen: chunk.length, types: evs.map((e) => e.type).join(',') });
  for (const e of evs) {
    seenTypes.push(e.type);
    if (e.type === 'sources') seenSources.push(e.sources || []);
  }
  return evs;
};

// ── 工具 ──
const results = [];
function check(name, cond, detail) {
  results.push({ name, pass: !!cond, detail: detail || '' });
}
function waitFor(fn, timeout = 20000, interval = 15) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const iv = setInterval(() => {
      let ok = false;
      try { ok = fn(); } catch (_) { /* ignore */ }
      if (ok) { clearInterval(iv); resolve(); }
      else if (Date.now() - t0 > timeout) { clearInterval(iv); reject(new Error('waitFor timeout')); }
    }, interval);
  });
}
// askQuestion() 从输入框读取问题(不接受参数), 必须先把问题写入 textarea 再触发
function ask(q) {
  const input = doc.getElementById('questionInput');
  if (!input) throw new Error('questionInput not found');
  input.value = q;
  window.askQuestion();
}
async function serverReady() {
  for (let i = 0; i < 60; i++) {
    try { const r = await nodeFetch(SERVER + '/openapi.json'); if (r.ok) return true; } catch (_) { /* retry */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

async function main() {
  if (!(await serverReady())) {
    console.error('SERVER_NOT_READY ' + SERVER);
    process.exit(2);
  }

  let incrementalObserved = null;
  ask(QUESTION);

  // 流式早期轮询: 抓取"增量 Markdown 渲染"证据(答案已渲染为 HTML 而非原始文本节点)
  const incIv = setInterval(() => {
    const at = doc.querySelector('.answer-text');
    if (at && at.innerHTML && at.innerHTML.includes('<') && at.textContent.length > 0) {
      if (incrementalObserved === null) {
        incrementalObserved = { hasHtml: at.innerHTML.includes('<'), len: at.textContent.length };
      }
    }
  }, 5);

  // 检索过程渐进可视化: 轮询步骤状态, 验证"路由 done / 检索 active"在流中真实推进
  let sawRouteDone = false, sawRetrieveAdvanced = false;
  const pipeIv = setInterval(() => {
    let st = null;
    try { st = window.RAG_TEST && window.RAG_TEST.getPipelineState ? window.RAG_TEST.getPipelineState() : null; } catch (_) { /* ignore */ }
    if (st) {
      if (st.route && st.route.indexOf('done') >= 0) sawRouteDone = true;
      // retrieve 在 meta→sources 同一 chunk 内 active→done 同步翻转, 轮询难抓 transient active;
      // 以"曾达到 active 或 done"(done 持久)证明该步骤确实推进过
      if (st.retrieve && (st.retrieve.indexOf('active') >= 0 || st.retrieve.indexOf('done') >= 0)) sawRetrieveAdvanced = true;
    }
  }, 10);

  try {
    await waitFor(() => !!doc.querySelector('.answer-actions'), 25000);
  } catch (e) {
    clearInterval(incIv);
    const cc = doc.getElementById('chatContainer');
    console.log('WAIT_FAILED (.answer-actions never appeared)');
    console.log('seenTypes: ' + seenTypes.join(' -> '));
    console.log('thinkingMsg present: ' + !!doc.getElementById('thinkingMsg'));
    console.log('msg-assistant count: ' + doc.querySelectorAll('.msg-assistant').length);
    console.log('.answer-text count: ' + doc.querySelectorAll('.answer-text').length);
    console.log('chatContainer HTML (first 600):');
    console.log(cc ? cc.innerHTML.slice(0, 600) : 'NO_CHAT_CONTAINER');
    console.log('jsErrors: ' + JSON.stringify(jsErrors.slice(0, 5)));
    process.exit(4);
  }
  clearInterval(incIv);
  clearInterval(pipeIv);

  // ── 断言 ──
  const userMsg = doc.querySelector('.msg-user');
  check('user_message_rendered', userMsg && userMsg.textContent.includes(QUESTION.slice(0, 6)),
    userMsg ? userMsg.textContent.slice(0, 20) : 'missing');

  // finalize 后 .answer-text 已被 renderMarkdown 结果替换, 渲染落在 .msg-assistant .msg-body
  const msgBody = doc.querySelector('.msg-assistant .msg-body');
  check('assistant_answer_rendered', !!msgBody, msgBody ? 'len=' + msgBody.innerHTML.length : 'missing');

  const html = msgBody ? msgBody.innerHTML : '';
  if (process.env.IT_DEBUG) {
    fs.writeFileSync(
      path.join(__dirname, '_debug_html.txt'),
      '===== all .msg-assistant .msg-body =====\n' +
        [...doc.querySelectorAll('.msg-assistant .msg-body')]
          .map((b, i) => `--- body[${i}] ---\n` + b.innerHTML)
          .join('\n') + '\n'
    );
  }
  check('markdown_heading', html.includes('<h1'), '');
  check('markdown_bold', html.includes('<strong>'), '');
  check('markdown_list', html.includes('<li'), '');
  check('markdown_paragraph', html.includes('<p'), '');
  check('markdown_codeblock', html.includes('<pre') && html.includes('<code'), 'code block markup missing');

  // XSS: 判据是"是否真的解析出可执行元素", 而不是 innerHTML 字符串匹配 ——
  // 属性值(如 data-copy-answer)里的 <script 文本不构成标签, HTML 序列化只转义引号,
  // 用 includes('<script') 判断会误报。
  check('xss_no_script_element', msgBody ? msgBody.querySelectorAll('script').length === 0 : false,
    'script element created');
  check('xss_no_img_element', msgBody ? msgBody.querySelectorAll('img').length === 0 : false,
    'img element created');
  check('xss_no_onerror_attr', msgBody ? msgBody.querySelectorAll('[onerror]').length === 0 : false,
    'onerror handler injected');
  check('xss_payload_as_text', msgBody ? /<img src=x onerror/.test(msgBody.textContent) : false,
    'escaped payload should survive as plain text');
  check('xss_escaped_script', html.includes('&lt;script&gt;'), 'script not escaped');
  check('xss_escaped_img', html.includes('&lt;img'), 'img not escaped');
  // 合法外链
  check('safe_link_rendered', html.includes('href="https://example.com/docs"') && html.includes('target="_blank"'),
    'external https link not rendered safely');

  // sources 面板(后端注入了确定来源 rag_architecture.md)
  const pageHtml = doc.documentElement.innerHTML;
  check('sources_panel_rendered', doc.querySelector('.retrieval-viz') && doc.querySelector('.rv-item') && doc.querySelector('.rv-fill'),
    'retrieval-viz panel (data-driven sources viz) missing');
  check('sources_injected_source', pageHtml.includes('rag_architecture.md'), 'injected source not shown');

  // 复制去重: 复制按钮仅作标记(data-copy-answer 无全文), 全文走 _answerCache, 避免答案体积翻倍进 DOM
  const copyBtn = doc.querySelector('.answer-actions [data-copy-answer]');
  check('copy_button_marker_present', !!copyBtn, 'no copy button');
  check('copy_no_full_text_attr', copyBtn && (copyBtn.getAttribute('data-copy-answer') || '').length === 0,
    copyBtn ? 'attr len=' + (copyBtn.getAttribute('data-copy-answer') || '').length : 'n/a');

  // SSE 事件序列
  const iMeta = seenTypes.indexOf('meta');
  const iSources = seenTypes.indexOf('sources');
  const iDone = seenTypes.lastIndexOf('done');
  const nDelta = seenTypes.filter((t) => t === 'delta').length;
  check('sse_has_meta', iMeta >= 0, 'no meta');
  check('sse_has_sources', iSources >= 0, 'no sources');
  check('sse_has_delta', nDelta >= 1, 'nDelta=' + nDelta);
  check('sse_has_done', iDone >= 0, 'no done');
  check('sse_order_meta_before_sources', iMeta < iSources, `meta@${iMeta} sources@${iSources}`);
  check('sse_order_done_last', iDone > iSources && iDone === seenTypes.length - 1, `done@${iDone} len=${seenTypes.length}`);

  // 增量: 流式过程中答案已渲染为 Markdown(而非原始文本节点)
  check('streaming_markdown_rendered', incrementalObserved && incrementalObserved.hasHtml && incrementalObserved.len > 0,
    incrementalObserved ? 'hasHtml=' + incrementalObserved.hasHtml + ' len=' + incrementalObserved.len : 'not observed during stream');

  // 检索过程渐进可视化: 流中"路由→done / 检索→active"真实推进
  check('pipeline_route_done', sawRouteDone, 'route never marked done during stream');
  check('pipeline_retrieve_advanced', sawRetrieveAdvanced, 'retrieve step never advanced during stream');
  check('pipeline_badge_present', !!doc.querySelector('.pipeline-badge'), 'no pipeline badge at finalize');

  // 增量(本次修复): 4 步芯片常驻于最终答案卡, 不再随首字被删除
  const finalPipe = window.RAG_TEST && window.RAG_TEST.getPipelineState ? window.RAG_TEST.getPipelineState() : null;
  const lastAns = doc.querySelector('.msg-assistant:last-of-type');
  check('pipeline_persists_in_answer', !!(lastAns && lastAns.querySelector('.pipeline-steps')), '4-step chips not present in final answer card');
  check('pipeline_final_state_visible', !!(finalPipe && finalPipe.generate && finalPipe.generate.indexOf('done') >= 0), 'generate step not shown done in final card');
  check('no_transient_thinking_card', !doc.getElementById('thinkingMsg'), 'transient thinkingMsg still present');

  // 代码块: 语言标签 + 复制按钮
  check('code_block_rendered', !!doc.querySelector('.code-block'), 'no .code-block');
  check('code_copy_button', !!doc.querySelector('.code-block .code-copy'), 'no code copy btn');
  check('code_lang_label', !!doc.querySelector('.code-block .code-lang'), 'no code lang label');

  // 主题切换
  const beforeTheme = doc.documentElement.getAttribute('data-theme');
  window.RAG.toggleTheme();
  const afterTheme = doc.documentElement.getAttribute('data-theme');
  check('theme_toggle_changes', beforeTheme !== afterTheme, `before=${beforeTheme} after=${afterTheme}`);
  check('theme_persisted', ['light', 'dark', 'system'].includes(window.localStorage.getItem('rag_theme')),
    'rag_theme=' + window.localStorage.getItem('rag_theme'));

  // 会话持久化
  let sessions = [];
  try { sessions = JSON.parse(window.localStorage.getItem('rag_sessions') || '[]'); } catch (_) { /* ignore */ }
  check('session_persisted', Array.isArray(sessions) && sessions.length >= 1 && /RAG/.test(sessions[0].title || ''),
    'sessions=' + JSON.stringify(sessions.map((s) => (s.title || '').slice(0, 12))));

  // 重新生成 端到端 (复用同一问题重新流式, 旧答案应被新答案替换)
  const regenBtn = doc.querySelector('.answer-actions [data-regenerate]');
  check('regenerate_button_present', !!regenBtn, 'no regenerate button');
  if (regenBtn) {
    const oldAns = regenBtn.closest('[data-answer-id]');
    const oldId = oldAns ? oldAns.getAttribute('data-answer-id') : '';
    regenBtn.click();
    let regenOk = true;
    try { await waitFor(() => !doc.getElementById('thinkingMsg') && doc.querySelectorAll('.answer-actions').length >= 1, 25000); } catch (_) { regenOk = false; }
    check('regenerate_produces_new_answer', regenOk && !doc.getElementById(oldId), 'regenerate did not replace answer');
  }

  // 停止生成(abc path)
  ask('第二问：用于验证停止生成是否能中断流');
  await new Promise((r) => setTimeout(r, 30));
  window.stopGeneration();
  let stopOk = true;
  try { await waitFor(() => !doc.getElementById('thinkingMsg'), 8000); } catch (_) { stopOk = false; }
  check('stop_generation_clears_thinking', stopOk && !doc.getElementById('thinkingMsg'),
    'thinkingMsg still present after stop');

  // 错误降级 toast
  window.RAG.handleApiError('/qa/ask/stream', new Error('HTTP 500'), true);
  let toastOk = true;
  try { await waitFor(() => !!doc.querySelector('.toast.toast-err'), 4000); } catch (_) { toastOk = false; }
  check('error_toast_shown', toastOk, 'no error toast');

  // 无未捕获 JS 错误
  check('no_js_errors', jsErrors.length === 0, jsErrors.join(' | '));

  // ── 汇总 ──
  const passed = results.filter((r) => r.pass).length;
  const failed = results.filter((r) => !r.pass);
  console.log('\n===== 前端联调断言结果 =====');
  for (const r of results) {
    console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}${r.detail ? '  [' + r.detail + ']' : ''}`);
  }
  console.log(`\nSSE 事件序列: ${seenTypes.join(' -> ')}`);
  console.log('parseSSEStream 调用明细: ' + JSON.stringify(parseCalls));
  console.log('答案文本长度: ' + (msgBody ? msgBody.textContent.length : 'n/a'));
  console.log('答案含末段(script 载荷): ' + (msgBody ? /script/.test(msgBody.textContent) : 'n/a'));
  console.log(`通过 ${passed}/${results.length}`);
  console.log(failed.length === 0 ? 'INTEGRATION_OK' : 'INTEGRATION_FAIL');
  process.exit(failed.length === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error('HARNESS_ERROR: ' + (e && e.stack ? e.stack : e));
  process.exit(3);
});
