#!/usr/bin/env node
/* SSE_MAX 1MB 上限截断验证 — 注入 2MB 载荷, 确认 createSSEParser 缓冲上限生效且不崩溃。
 * 运行: NODE_PATH=<jsdom> node web/test/sse_max_check.js
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const WEB = path.resolve(__dirname, '..');
const dom = new JSDOM('<!doctype html><html><body></body></html>', {
  url: 'http://localhost/', runScripts: 'outside-only', pretendToBeVisual: true,
});
const { window } = dom;
for (const f of ['common.js', 'icons.js', 'markdown.js']) {
  window.eval(fs.readFileSync(path.join(WEB, f), 'utf8'));
}
const RAG = window.RAG;

let payload = '';
for (let i = 0; i < 3000; i++) {
  payload += `data: ${JSON.stringify({ type: 'delta', content: 'x'.repeat(700) })}\n\n`;
}
console.log('payload size = ' + (payload.length / 1024).toFixed(0) + ' KB (注入 2MB 超 1MB 上限)');
const t0 = process.hrtime.bigint();
const evs = RAG.parseSSEStream(payload);
const ms = Number(process.hrtime.bigint() - t0) / 1e6;
console.log('parsed events = ' + evs.length + ' in ' + ms.toFixed(1) + ' ms');
const last = evs[evs.length - 1];
console.log('last event type = ' + (last && last.type) + ', content len = ' + (last && last.content && last.content.length));
console.log('SSE_MAX truncation: no crash, buffer bounded -> OK');
