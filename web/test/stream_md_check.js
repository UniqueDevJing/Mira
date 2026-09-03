#!/usr/bin/env node
/* 专项验证 · 流式 Markdown 渲染 + 复制去重 (无需后端, 直接驱动真实前端函数)
 *
 * 加载真实前端脚本, 通过 RAG_TEST 钩子调用 createStreamingCard / updateStreamingAnswer /
 * finalizeAnswer, 验证:
 *   1. 流式中途, 答案已渲染为 Markdown(含 <p>/<strong>/<li>/<pre><code>), 而非裸文本
 *   2. 流式结束渲染结果与 RAG.renderMarkdown(全文) 一致
 *   3. 复制按钮仅作标记(data-copy-answer 无全文), 全文存 _answerCache(去重, DOM 不翻倍)
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

const RAG = window.RAG;
const T = window.RAG_TEST;
const results = [];
function check(name, cond, detail) { results.push({ name, pass: !!cond, detail: detail || '' }); }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const FULL = '# RAG 架构\n\n这是 **加粗** 与 `行内代码` 段落。\n\n- 列表项一\n- 列表项二\n\n```js\nconst x = 1;\n```\n\n> 引用一句话';

  // 1) 流式 Markdown 渲染
  T.createStreamingCard();
  // 分块喂入, 模拟真实 delta 流
  for (let i = 0; i < FULL.length; i += 7) {
    T.updateStreamingAnswer(FULL.slice(i, i + 7));
  }
  await sleep(40); // 等 rAF 合并渲染落定

  const node = T.getStreamNode();
  const streamHtml = node ? node.innerHTML : '';
  const expected = RAG.renderMarkdown(FULL);
  check('stream_has_paragraph', streamHtml.includes('<p'), 'no <p> during stream');
  check('stream_has_bold', streamHtml.includes('<strong>'), 'no <strong> during stream');
  check('stream_has_list', streamHtml.includes('<li>'), 'no <li> during stream');
  check('stream_has_codeblock', streamHtml.includes('<pre>') && streamHtml.includes('<code>'), 'no <pre><code> during stream');
  check('stream_equals_renderMarkdown', streamHtml === expected, 'stream html !== renderMarkdown(full)');

  // 2) finalize + 复制去重
  const cacheBefore = T.getAnswerCacheSize();
  T.finalizeAnswer(FULL, { routing_source: 'hybrid', kb_id: 'documents' }, []);
  await sleep(20);
  const copyBtn = doc.querySelector('.answer-actions [data-copy-answer]');
  const attrVal = copyBtn ? (copyBtn.getAttribute('data-copy-answer') || '') : 'NO_BTN';
  const cacheAfter = T.getAnswerCacheSize();
  check('copy_button_present', !!copyBtn, 'no copy button after finalize');
  check('copy_attr_empty', attrVal.length === 0, 'data-copy-answer attr len=' + attrVal.length + ' (应=0, 证明未把全文塞进 DOM)');
  check('answer_cached', cacheAfter > cacheBefore, `cache ${cacheBefore} -> ${cacheAfter}`);
  // 去重量化: 复制属性承载 0 字符, 而实际全文达 80 字符 → 证明全文未塞进 DOM 属性
  check('dedup_no_fulltext_in_attr', attrVal.length === 0 && FULL.length >= 20,
    `全文 ${FULL.length} 字符, 属性承载 ${attrVal.length} 字符`);

  check('no_js_errors', jsErrors.length === 0, jsErrors.join(' | '));

  const passed = results.filter((r) => r.pass).length;
  const failed = results.filter((r) => !r.pass);
  console.log('\n===== 流式 Markdown + 复制去重 验证 =====');
  for (const r of results) console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}${r.detail ? '  [' + r.detail + ']' : ''}`);
  console.log(`\n通过 ${passed}/${results.length}`);
  console.log(failed.length === 0 ? 'STREAM_MD_OK' : 'STREAM_MD_FAIL');
  process.exit(failed.length === 0 ? 0 : 1);
})().catch((e) => { console.error('CHECK_ERROR: ' + (e && e.stack ? e.stack : e)); process.exit(3); });
