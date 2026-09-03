#!/usr/bin/env node
/* 检索过程数据可视化验证 · jsdom 驱动真实前端代码 + 合成 SSE(真实事件形状, 不依赖网络)。
 *
 * 验证点:
 *  - 渐进挂载: sources 事件到达时 .retrieval-viz 即出现在流式答案卡中(不等到 finalize)
 *  - 数据驱动: 每个召回片段渲染为分数条(.rv-fill 宽度=score*100%)+ 真实片段(.rv-snippet)
 *  - 路由/重排状态: .rv-route 显示路由方式与知识库, .rv-rerank 显示重排结果
 *  - 持久化: finalize 后面板不被 .answer-text 的 rAF 重渲覆盖
 *  - 无 JS 错误 / 无 XSS 可执行元素
 */
'use strict';

const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const WEB_DIR = path.join(PROJECT_ROOT, 'web');

const commonJs = fs.readFileSync(path.join(WEB_DIR, 'common.js'), 'utf8');
const iconsJs = fs.readFileSync(path.join(WEB_DIR, 'icons.js'), 'utf8');
const markdownJs = fs.readFileSync(path.join(WEB_DIR, 'markdown.js'), 'utf8');
const indexHtml = fs.readFileSync(path.join(WEB_DIR, 'index.html'), 'utf8');
const inlineMatch = [...indexHtml.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const inlineJs = inlineMatch[inlineMatch.length - 1][1];

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

window.TextDecoder = globalThis.TextDecoder;
window.AbortController = globalThis.AbortController;
if (!window.crypto || !window.crypto.randomUUID) window.crypto = require('node:crypto').webcrypto;
if (!window.matchMedia) window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent() { return false; } });
window.addEventListener('error', (e) => jsErrors.push('window.error: ' + (e.error && e.error.stack ? e.error.stack : e.message)));

window.eval(commonJs);
window.RAG.API = 'http://mock/api/v1';
window.eval(iconsJs);
window.eval(markdownJs);
window.__RAG_TEST__ = true;
window.eval(inlineJs);
doc.dispatchEvent(new window.Event('DOMContentLoaded'));

// ── 合成 SSE(真实事件形状) ──
const META = { type: 'meta', skill: 'service', kb_id: 'service', routing_source: 'rule', router_ms: 5.7 };
const SOURCES = {
  type: 'sources', final: true,
  retrieval_meta: { top1_score: 0.7156, result_count: 4, final: true, degradation_level: 0 },
  sources: [
    { doc_id: 'd1', content: 'Q1: 我要退货，怎么操作？\n您好，您可以在订单详情页点击申请退货，选择退货原因后提交。审核通过后会有退货地址发送给您，请将商品打包寄回。我们会在签收后的三个工作日内完成退款。', source_file: '', score: 0.7156 },
    { doc_id: 'd2', content: '退款什么时候到账？退款审核通过后会原路退回您的支付账户，一般一到三个工作日到账，请您留意账户变动。', source_file: 'refund_policy.md', score: 0.7079 },
    { doc_id: 'd3', content: '二、退款流程 2.1 申请退款 消费者在订单详情页点击申请退款，选择退款原因并填写说明。', source_file: '', score: 0.6421 },
    { doc_id: 'd4', content: '常见问题：运费由谁承担？非质量问题退货运费由买家承担，质量问题由商家承担。', source_file: 'faq.md', score: 0.5012 },
  ],
};
const DELTAS = ['退款', '流程如下', '：  \n1. **申请退款**', '…  \n2. **审核通过**', '…  \n3. **寄回商品**'];
const DONE = { type: 'done', answer: '退款流程如下：1. 申请退款 2. 审核通过 3. 寄回商品', token_usage: { total_tokens: 88 }, degradation_level: 0 };

function sse(obj) { return 'data: ' + JSON.stringify(obj) + '\n\n'; }
const part1 = sse(META) + sse(SOURCES);
const part2 = DELTAS.map(sse).join('') + sse(DONE);

// 分两段投递: 先 meta+sources(验证渐进挂载), 250ms 后再 delta+done(验证持久化)
window.fetch = () => Promise.resolve({
  ok: true, status: 200, json: () => Promise.resolve({}),
  body: new ReadableStream({
    start(controller) {
      const enc = new TextEncoder();
      controller.enqueue(enc.encode(part1));
      setTimeout(() => { controller.enqueue(enc.encode(part2)); controller.close(); }, 250);
    },
  }),
});

const results = [];
function check(name, cond, detail) { results.push({ name, pass: !!cond, detail: detail || '' }); }
function waitFor(fn, timeout = 15000, interval = 10) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const iv = setInterval(() => {
      let ok = false; try { ok = fn(); } catch (_) {}
      if (ok) { clearInterval(iv); resolve(); }
      else if (Date.now() - t0 > timeout) { clearInterval(iv); reject(new Error('waitFor timeout: ' + name)); }
    }, interval);
  });
}

(async () => {
  // 触发真实发送路径
  const input = doc.getElementById('questionInput');
  input.value = '退款流程是怎样的？';
  window.askQuestion();

  // 1) 渐进挂载: sources 到达后应已出现面板(此时 done 尚未到达)
  let progressiveViz = null;
  try {
    await waitFor(() => {
      const v = doc.querySelector('.retrieval-viz');
      if (v && v.querySelectorAll('.rv-item').length === 4) { progressiveViz = v; return true; }
      return false;
    }, 4000);
  } catch (_) {}
  check('viz_progressive_mount', !!progressiveViz, progressiveViz ? 'mounted before finalize' : 'not mounted during streaming');

  if (progressiveViz) {
    check('viz_routing_label', /规则路由/.test(progressiveViz.querySelector('.rv-route')?.textContent || ''), progressiveViz.querySelector('.rv-route')?.textContent || '');
    check('viz_routing_kb', /service/.test(progressiveViz.querySelector('.rv-route')?.textContent || ''), progressiveViz.querySelector('.rv-route')?.textContent || '');
    check('viz_rerank_done', /重排完成/.test(progressiveViz.querySelector('.rv-rerank')?.textContent || ''), progressiveViz.querySelector('.rv-rerank')?.textContent || '');
    check('viz_item_count', progressiveViz.querySelectorAll('.rv-item').length === 4, 'count=' + progressiveViz.querySelectorAll('.rv-item').length);
    const first = progressiveViz.querySelector('.rv-item .rv-fill');
    check('viz_score_bar_width', !!first && first.style.width && first.style.width.startsWith('71.6'), first ? 'width=' + first.style.width : 'no fill');
    check('viz_snippet_present', progressiveViz.querySelector('.rv-snippet') && progressiveViz.querySelector('.rv-snippet').textContent.length > 0, '');
    check('viz_tier_class', !!progressiveViz.querySelector('.rv-item.tier-high'), 'top item should be tier-high');
  }

  // 2) 等待 finalize 完成(答案操作按钮出现)
  try { await waitFor(() => !!doc.querySelector('.answer-actions'), 15000); } catch (e) {
    console.log('WAIT_FAILED .answer-actions: ' + e.message);
  }

  // 3) 持久化: finalize 后面板仍在, 且未被覆盖
  const finalViz = doc.querySelector('.msg-assistant .retrieval-viz');
  check('viz_persist_after_finalize', !!finalViz && finalViz.querySelectorAll('.rv-item').length === 4,
    finalViz ? 'items=' + finalViz.querySelectorAll('.rv-item').length : 'missing');
  const DPF = window.Node.DOCUMENT_POSITION_FOLLOWING;
  check('viz_before_answer_text', finalViz && (finalViz.compareDocumentPosition(doc.querySelector('.msg-assistant .answer-text')) & DPF) === DPF,
    'viz should precede answer text');

  // 4) 无 XSS 可执行元素
  const body = doc.querySelector('.msg-assistant .msg-body');
  check('viz_no_script_element', body ? body.querySelectorAll('script').length === 0 : false, '');
  check('viz_no_img_element', body ? body.querySelectorAll('img').length === 0 : false, '');

  // 5) 无 JS 错误
  check('no_js_errors', jsErrors.length === 0, jsErrors.slice(0, 3).join(' | '));

  const passed = results.filter((r) => r.pass).length;
  const failed = results.filter((r) => !r.pass);
  for (const r of results) console.log((r.pass ? 'PASS ' : 'FAIL ') + r.name + (r.detail ? '  [' + r.detail + ']' : ''));
  console.log('\n' + passed + '/' + results.length + (failed.length ? '  ❌ RETRIEVAL_VIZ_FAIL' : '  ✅ RETRIEVAL_VIZ_OK'));
  process.exit(failed.length ? 1 : 0);
})();
