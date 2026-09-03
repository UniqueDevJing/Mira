#!/usr/bin/env node
/* 历史消息统一 Markdown + 复制/重新生成 专项验证 (无需后端)
 * 驱动真实 appendUserMessage / appendRestoredAssistantMessage / regenerate, 断言:
 *   - 历史助手消息经 renderMarkdown 渲染(<p>/<strong>/<li>/<h1>), 不再是裸文本
 *   - 含 .code-block 且带复制按钮与高亮
 *   - 带 复制 / 重新生成 按钮, 全文进 _answerCache
 *   - 点击"重新生成"移除旧气泡、新建思考卡、清理缓存 (需前序用户消息供取问题) */
'use strict';
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const WEB = path.resolve(__dirname, '..');
const commonJs = fs.readFileSync(WEB + '/common.js', 'utf8');
const iconsJs = fs.readFileSync(WEB + '/icons.js', 'utf8');
const markdownJs = fs.readFileSync(WEB + '/markdown.js', 'utf8');
const indexHtml = fs.readFileSync(WEB + '/index.html', 'utf8');
const inlineJs = [...indexHtml.matchAll(/<script>([\s\S]*?)<\/script>/g)].pop()[1];

const dom = new JSDOM(indexHtml, { url: 'http://localhost/', runScripts: 'outside-only', pretendToBeVisual: true });
const w = dom.window, doc = w.document;
w.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
w.__RAG_TEST__ = true;
w.fetch = () => new Promise(() => {}); // 阻断网络, 仅验证前端逻辑不崩
w.eval(commonJs); w.eval(iconsJs); w.eval(markdownJs); w.eval(inlineJs);
doc.dispatchEvent(new w.Event('DOMContentLoaded'));
const T = w.RAG_TEST;

const results = [];
function check(n, c, d) { results.push({ n, p: !!c, d: d || '' }); }

// 先放一个用户消息, 供"重新生成"从 DOM 前序用户消息取问题
T.appendUserMessage('RAG 系统的架构是怎样的？');
const MD = '# 标题\n这是 **加粗** 与 *斜体*。\n\n- 项目一\n- 项目二\n\n```python\ndef add(a, b):\n    return a + b  # 求和\n```\n';
T.appendRestoredAssistantMessage(MD);
const at = doc.querySelector('.answer-text');
check('history_rendered_markdown', !!at && at.innerHTML.includes('<p>'), at ? 'hasP=' + at.innerHTML.includes('<p>') : 'missing');
check('history_bold', !!at && at.innerHTML.includes('<strong>'), at ? 'hasStrong=' + at.innerHTML.includes('<strong>') : 'missing');
check('history_list', !!at && at.innerHTML.includes('<li>'), at ? 'hasLi=' + at.innerHTML.includes('<li>') : 'missing');
check('history_heading', !!at && at.innerHTML.includes('<h1>'), at ? 'hasH1=' + at.innerHTML.includes('<h1>') : 'missing');
check('history_no_raw_md', !!at && at.innerHTML.indexOf('**加粗**') < 0, at ? 'rawStillPresent=' + (at.innerHTML.indexOf('**加粗**') >= 0) : 'missing');
check('history_codeblock', !!doc.querySelector('.code-block'), 'no .code-block');
check('history_code_copy_btn', !!doc.querySelector('.code-block .code-copy'), 'no code copy btn');
check('history_code_lang', !!doc.querySelector('.code-block .code-lang'), 'no code lang');
check('history_highlight_kw', !!doc.querySelector('.code-block') && doc.querySelector('.code-block').innerHTML.includes('tok-kw'), 'no highlight');
check('history_regenerate_btn', !!doc.querySelector('.answer-actions [data-regenerate]'), 'no regen btn');
const cacheBefore = T.getAnswerCacheSize();
check('history_cache_stored', cacheBefore >= 1, 'cache=' + cacheBefore);

// 重新生成: 点击 -> 旧节点移除 + 新思考卡 + 缓存清理
const regenBtn = doc.querySelector('.answer-actions [data-regenerate]');
const oldId = regenBtn.closest('[data-answer-id]').getAttribute('data-answer-id');
regenBtn.click();
check('history_regen_removed_old', !doc.getElementById(oldId), 'old assistant not removed');
const newCard=[...doc.querySelectorAll('.msg-assistant')].find(function(c){return c.querySelector('.pipeline-steps')&&c.id!==oldId;});
check('history_regen_new_card', !!newCard, 'no new streaming card after regenerate');
check('history_regen_cache_cleared', T.getAnswerCacheSize() < cacheBefore, 'cache=' + T.getAnswerCacheSize());

const passed = results.filter(r => r.p).length;
results.forEach(r => console.log((r.p ? 'PASS' : 'FAIL') + '  ' + r.n + (r.d ? '  [' + r.d + ']' : '')));
console.log(`通过 ${passed}/${results.length}`);
console.log(passed === results.length ? 'HISTORY_MD_OK' : 'HISTORY_MD_FAIL');
process.exit(passed === results.length ? 0 : 1);
