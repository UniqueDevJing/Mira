#!/usr/bin/env node
/* 代码块 + 轻量高亮 专项验证 (无需后端)
 * 驱动真实 RAG.renderMarkdown, 断言:
 *   - 代码块含语言标签 / 复制按钮 / tok-* 高亮 span
 *   - XSS 安全: 代码块与文本中的 <script>/<img onerror> 均被转义, 不生成可执行元素 */
'use strict';
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const WEB = path.resolve(__dirname, '..');
const commonJs = fs.readFileSync(WEB + '/common.js', 'utf8');
const markdownJs = fs.readFileSync(WEB + '/markdown.js', 'utf8');
const dom = new JSDOM('<!doctype html><body></body>', { url: 'http://localhost/', runScripts: 'outside-only' });
const w = dom.window, doc = w.document;
w.eval(commonJs);
w.eval(markdownJs);
const RAG = w.RAG;

const results = [];
function check(n, c, d) { results.push({ n, p: !!c, d: d || '' }); }

const code = RAG.renderMarkdown('```python\ndef add(a, b):\n    return a + b  # 求和\nx = 42\n```');
check('code_block_present', code.includes('class="code-block"'), 'no .code-block');
check('code_copy_btn', code.includes('data-copy-code'), 'no copy btn');
check('code_lang_label', code.includes('code-lang') && code.includes('python'), 'no lang label');
check('highlight_kw', code.includes('tok-kw'), 'no tok-kw span');
check('highlight_num', code.includes('tok-num'), 'no tok-num span');
check('highlight_com', code.includes('tok-com'), 'no tok-com span');

// XSS 安全: 代码块内含 <script>
const xssCode = RAG.renderMarkdown('```\n<script>alert(1)</script>\n```');
const c2 = doc.createElement('div'); c2.innerHTML = xssCode;
check('xss_code_no_literal_script', xssCode.indexOf('<script>') < 0, 'literal <script> present (not escaped)');
check('xss_code_escaped', xssCode.includes('&lt;') && xssCode.includes('&gt;'), 'angle brackets not escaped');
check('xss_code_no_script_el', c2.querySelectorAll('script').length === 0, 'script element created');

// 全局 XSS: 文本含 <img onerror>
const xssText = RAG.renderMarkdown('看 <img src=x onerror=alert(1)> 这里');
const c3 = doc.createElement('div'); c3.innerHTML = xssText;
check('xss_text_no_img_onerror', c3.querySelectorAll('[onerror]').length === 0, 'onerror injected');
check('xss_text_escaped', xssText.includes('&lt;img'), 'img not escaped');

const passed = results.filter(r => r.p).length;
results.forEach(r => console.log((r.p ? 'PASS' : 'FAIL') + '  ' + r.n + (r.d ? '  [' + r.d + ']' : '')));
console.log(`通过 ${passed}/${results.length}`);
console.log(passed === results.length ? 'CODE_HL_OK' : 'CODE_HL_FAIL');
process.exit(passed === results.length ? 0 : 1);
