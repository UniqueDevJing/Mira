#!/usr/bin/env node
/* 前端性能基准 — 在 jsdom 中加载真实前端代码，量化关键路径成本。
 *
 * 用途: 优化前后的对比基线。每项输出可复现数字, 不凭感觉优化。
 * 运行: NODE_PATH=<jsdom> node web/test/perf_bench.js
 *
 * 量测项:
 *   1. renderMarkdown  : 10KB 富 markdown 渲染 200 次的均值 (证明它是否是瓶颈)
 *   2. parseSSEStream  : 1000 个 SSE 事件 (~600KB) 的解析吞吐
 *   3. 单条消息 DOM 成本: 按 finalizeAnswer 的真实模板构造一条完整回答消息,
 *      统计节点数/innerHTML 长度 → 推算长会话 DOM 无限增长的规模
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
const doc = window.document;

for (const f of ['common.js', 'icons.js', 'markdown.js']) {
  window.eval(fs.readFileSync(path.join(WEB, f), 'utf8'));
}
const RAG = window.RAG;

function bench(name, fn, iters) {
  // 预热 3 次消除首跑噪声
  for (let i = 0; i < 3; i++) fn();
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  console.log(`${name.padEnd(34)} ${((ms / iters).toFixed(3) + ' ms/次').padStart(12)}  (共 ${iters} 次, ${(ms / 1000).toFixed(2)}s)`);
  return ms / iters;
}

console.log('===== 前端性能基线 (jsdom + 真实前端代码) =====\n');

// ── 1. renderMarkdown ──
const bigMd = ('# 标题\n\n**加粗** 普通文本 混排内容，含中文与 English mixed text。\n\n' +
  '- 列表项一 `code`\n- 列表项二\n- 列表项三\n\n' +
  '> 引用块：检索增强生成通过混合检索提升召回质量。\n\n' +
  '```python\nretriever.fuse(bm25, dense, method="rrf")\n```\\n\n' +
  '段落文本：<script>alert(1)</script> 应被转义；[链接](https://example.com) 应安全渲染。\n\n').repeat(40); // ≈ 20KB
console.log(`输入 markdown 大小: ${(bigMd.length / 1024).toFixed(1)} KB`);
const mdCost = bench('renderMarkdown(20KB)', () => RAG.renderMarkdown(bigMd), 200);

// ── 2. parseSSEStream ──
// 注意: parseSSEStream 依赖内部全局 buffer, 从空开始喂 1000 个事件
let ssePayload = '';
for (let i = 0; i < 1000; i++) {
  ssePayload += `data: ${JSON.stringify({ type: 'delta', content: '流式增量内容' + i })}\n\n`;
}
console.log(`SSE 载荷大小: ${(ssePayload.length / 1024).toFixed(1)} KB (1000 事件)`);
bench('parseSSEStream(1000 events)', () => {
  // 每次都整块喂入 (buffer 会被耗尽, 可重复)
  RAG.parseSSEStream(ssePayload);
}, 50);

// ── 3. 单条消息 DOM 成本 (finalizeAnswer 真实模板) ──
const answerMd = ('**检索增强生成（RAG）** 通过混合检索提升召回质量。核心流程包括向量检索、BM25 与 RRF 融合。\n\n' +
  '- 向量检索（dense embedding）\n- BM25 关键词检索\n- RRF 分数融合排序\n\n' +
  '```python\nretriever.fuse(bm25, dense, method="rrf")\n```\n').repeat(3);
const sources = Array.from({ length: 5 }, (_, i) => ({
  score: 0.9 - i * 0.05,
  source_file: `doc_${i}.md`,
  content: '来源片段内容，用于面板渲染与占位。'.repeat(6),
}));

function buildOneMessage() {
  const div = doc.createElement('div');
  div.className = 'msg msg-assistant';
  let sourcesHtml = '<details class="rag-panel" open><summary>' + RAG.icon('database') + '参考来源 (' + sources.length + ' 个)</summary><div class="rag-sources-list">';
  for (const s of sources) {
    sourcesHtml += '<div class="rag-source-item"><span class="rag-source-score">' + (s.score || 0).toFixed(3) + '</span>' +
      '<span class="rag-source-file">' + RAG.escapeHtml((s.source_file || '').slice(-40)) + '</span>' +
      '<div class="rag-source-preview">' + RAG.escapeHtml((s.content || '').slice(0, 300)) + '</div></div>';
  }
  sourcesHtml += '</div></details>';
  const actionsHtml = '<div class="answer-actions"><button class="action-btn" data-copy-answer="' + RAG.escapeHtml(answerMd) + '" title="复制答案">' + RAG.icon('copy') + ' 复制</button></div>';
  div.innerHTML = '<div class="msg-body">' + RAG.renderMarkdown(answerMd) + sourcesHtml + actionsHtml + '</div>';
  return div;
}

const container = doc.createElement('div');
doc.body.appendChild(container);
const t0 = process.hrtime.bigint();
const N = 200;
for (let i = 0; i < N; i++) container.appendChild(buildOneMessage());
const buildMs = Number(process.hrtime.bigint() - t0) / 1e6;
const nodeCount = container.querySelectorAll('*').length;
const htmlKB = container.innerHTML.length / 1024;
console.log(`\n单条完整回答消息: ${buildOneMessage().querySelectorAll('*').length} 节点, ${(buildOneMessage().innerHTML.length / 1024).toFixed(1)} KB innerHTML`);
console.log(`构造 ${N} 条消息: ${buildMs.toFixed(1)}ms (${(buildMs / N).toFixed(2)} ms/条)`);
console.log(`长会话累计: ${N} 条 → ${nodeCount} DOM 节点, ${htmlKB.toFixed(0)} KB innerHTML`);
console.log(`  → 每多聊一轮增加 ~${Math.round(nodeCount / N)} 节点; 无上限清理时 1000 轮 ≈ ${Math.round(nodeCount / N) * 10} 节点 / ${(htmlKB * 5).toFixed(0)} KB`);

console.log('\n──── 长会话 DOM 裁剪 (驱动真实 index.html 逻辑, 不经过后端) ────');
const hHtml = fs.readFileSync(path.resolve(__dirname, '..', 'index.html'), 'utf8');
const hMatch = [...hHtml.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const hInline = hMatch[hMatch.length - 1][1];
const hDom = new JSDOM(hHtml, { url: 'http://localhost/', runScripts: 'outside-only', pretendToBeVisual: true });
const hWin = hDom.window, hDoc = hWin.document;
hWin.__RAG_TEST__ = true; // 暴露 window.RAG_TEST 钩子
for (const f of ['common.js', 'icons.js', 'markdown.js']) {
  hWin.eval(fs.readFileSync(path.resolve(__dirname, '..', f), 'utf8'));
}
hWin.RAG.API = '/api/v1';
hWin.eval(hInline); // 不派发 DOMContentLoaded, 避免 init 内的 fetch
const HN = 200;
const tH0 = process.hrtime.bigint();
for (let i = 0; i < HN; i++) {
  hWin.RAG_TEST.appendUserMessage('压测问题 ' + i + '：RAG 系统的混合检索架构是怎样的？');
  hWin.RAG_TEST.appendRestoredAssistantMessage(
    '**回答 ' + i + '** 这是一段用于压测的较长回答内容，包含列表与代码块：\n\n- 要点一\n- 要点二\n\n```python\nx = ' + i + '\n```\n'.repeat(2)
  );
}
const hBuildMs = Number(process.hrtime.bigint() - tH0) / 1e6;
const hContainer = hDoc.getElementById('chatContainer');
const hNodes = hContainer.querySelectorAll('*').length;
const hHtmlKB = (hContainer.innerHTML.length / 1024).toFixed(1);
const hMsg = hWin.RAG_TEST.getMessageCount();
const hHidden = hWin.RAG_TEST.getHiddenCount();
const reductionPct = ((1 - hNodes / nodeCount) * 100).toFixed(1);
console.log(`构造 ${HN * 2} 条消息(裁剪启用): ${hMsg} 可见气泡, 移除 ${hHidden} 条, ${hNodes} DOM 节点, ${hHtmlKB} KB 序列化`);
console.log(`  → 对比未裁剪基线 ${N} 条 ≈ ${nodeCount} 节点: 降幅 ${reductionPct}%`);
console.log(`  → 裁剪路径构造耗时: ${hBuildMs.toFixed(1)}ms (${(hBuildMs / (HN * 2)).toFixed(2)} ms/条)`);

console.log('\nPERF_BASELINE_DONE');
