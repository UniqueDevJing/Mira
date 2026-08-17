// 前端安全/健壮性验证: 从真实 web/index.html 提取 escapeHtml / parseSSEEvent 源码,
// 经 new Function 导出后断言 XSS 转义与 SSE 容错真实生效。node 运行 (前端无 pytest 基建)。
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const htmlPath = join(__dirname, '..', 'web', 'index.html');
const html = readFileSync(htmlPath, 'utf8');
const code = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]
  .map(m => m[1])
  .find(b => b.includes('function escapeHtml')) || '';

// 括号平衡提取顶层函数源码 (跳过 ' " ` 字符串与 /正则/ 字面量内的花括号/引号)
function extractFn(src, name) {
  const start = src.indexOf('function ' + name);
  if (start < 0) return null;
  let i = src.indexOf('{', start);
  let depth = 0;
  let q = ''; // 当前字符串引号: ' " `
  let rMode = false; // 正则字面量模式
  const isRegexStart = (p) => p === '' || ' \t\n([]{};,=:!&|?+-*%^~<>)'.includes(p);
  for (; i < src.length; i++) {
    const c = src[i];
    const prev = i > 0 ? src[i - 1] : '';
    if (q) {
      if (c === '\\') { i++; continue; }
      if (c === q) q = '';
      continue;
    }
    if (rMode) {
      if (c === '\\') { i++; continue; }
      if (c === '/') {
        rMode = false;
        while (i + 1 < src.length && /[a-z]/i.test(src[i + 1])) i++;
      }
      continue;
    }
    if (c === "'" || c === '"' || c === '`') { q = c; continue; }
    if (c === '/' && isRegexStart(prev)) { rMode = true; continue; }
    if (c === '{') depth++;
    else if (c === '}') { depth--; if (depth === 0) { i++; break; } }
  }
  return src.slice(start, i);
}

const escSrc = extractFn(code, 'escapeHtml');
const sseSrc = extractFn(code, 'parseSSEEvent');
if (!escSrc || !sseSrc) { console.error('FAIL: 未提取到函数', { esc: !!escSrc, sse: !!sseSrc }); process.exit(1); }

const api = new Function(`${escSrc}\n${sseSrc}\nreturn { escapeHtml, parseSSEEvent };`)();

let failures = 0;
function assert(name, cond) {
  if (cond) console.log('PASS:', name);
  else { console.error('FAIL:', name); failures++; }
}

// 1) escapeHtml 对 XSS payload 免疫
const payload = '<script>alert(1)</script>"\'&';
const escaped = api.escapeHtml(payload);
assert('escapeHtml 转义 <script>', !escaped.includes('<script>'));
assert('escapeHtml 转义引号', escaped.includes('&quot;') && escaped.includes('&#39;'));
assert('escapeHtml 转义 &', escaped.includes('&amp;'));
assert('escapeHtml 转义 >', escaped.includes('&gt;'));

// 2) parseSSEEvent: 合法块解析
const o1 = api.parseSSEEvent('event: message\ndata: {"type":"delta","content":"hi"}\n\n');
assert('parseSSEEvent 合法块返回对象', o1 && o1.type === 'delta' && o1.content === 'hi');

// 3) parseSSEEvent: 坏 JSON 不抛异常, 返回 null (容错核心)
let threw = false;
let o2 = null;
try { o2 = api.parseSSEEvent('data: {bad json,,,\n\n'); } catch (e) { threw = true; }
assert('parseSSEEvent 坏 JSON 不抛异常', !threw);
assert('parseSSEEvent 坏 JSON 返回 null', o2 === null);

// 4) parseSSEEvent: 无 data: 行返回 null
assert('parseSSEEvent 空 data 返回 null', api.parseSSEEvent('event: ping\n\n') === null);

// 5) parseSSEEvent: 截断块返回 null
assert('parseSSEEvent 截断块返回 null', api.parseSSEEvent('data: {"type":"delta","content":"hi"') === null);

// 6) 静态校验: routingSource 已转义 (774 行修复落地)
assert('routingSource 已 escapeHtml', html.includes('路由:${escapeHtml(routingSource)}'));
assert('routingSource 旧写法已消失', !html.includes('路由:${routingSource}'));

// 7) 静态校验: SSE 主循环改用 parseSSEEvent (无裸 JSON.parse 内联)
assert('SSE 主循环调用 parseSSEEvent', html.includes('const ev = parseSSEEvent(block);'));

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
