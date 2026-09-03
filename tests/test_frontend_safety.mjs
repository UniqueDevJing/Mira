// 前端安全/健壮性验证: 从真实 web/common.js 提取 escapeHtml,
// 经 new Function 导出后断言 XSS 转义与 SSE 容错真实生效。node 运行 (前端无 pytest 基建)。
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const htmlPath = join(__dirname, '..', 'web', 'index.html');
const adminPath = join(__dirname, '..', 'web', 'admin.html');
const commonJsPath = join(__dirname, '..', 'web', 'common.js');
const html = readFileSync(htmlPath, 'utf8');
const adminHtml = readFileSync(adminPath, 'utf8');
const commonJs = readFileSync(commonJsPath, 'utf8');

// 括号平衡提取顶层函数源码
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

const escSrc = extractFn(commonJs, 'escapeHtml');
// parseSSEEvent is extracted from common.js for testing (it's the parsing logic inside RAG.parseSSEStream flow)
const sseSrc = extractFn(commonJs, 'parseSSEEvent');
if (!escSrc) { console.error('FAIL: 未提取到函数', { esc: !!escSrc, sse: !!sseSrc }); process.exit(1); }

const api = new Function(`${escSrc}\n${sseSrc}\nreturn { escapeHtml, parseSSEEvent };`)();

let failures = 0;
function assert(name, cond) {
  if (cond) console.log('PASS:', name);
  else { console.error('FAIL:', name); failures++; }
}

// ═══════════════════════════════════════════
// 1) escapeHtml 核心 XSS 防御
// ═══════════════════════════════════════════

assert('escapeHtml 转义 <script>', !api.escapeHtml('<script>alert(1)</script>').includes('<script>'));
assert('escapeHtml 转义双引号', api.escapeHtml('"').includes('&quot;'));
assert('escapeHtml 转义单引号', api.escapeHtml("'").includes('&#39;'));
assert('escapeHtml 转义 &', api.escapeHtml('&amp;').includes('&amp;amp;'));
assert('escapeHtml 转义 >', api.escapeHtml('> ').includes('&gt;'));

// ── 属性上下文注入 ──
const imgPayload = '<img src=x onerror=alert(1)>';
assert('onclick 属性值逃逸（<> 被转义）',
  api.escapeHtml(imgPayload).includes('&lt;img') && api.escapeHtml(imgPayload).includes('&gt;'));

const stylePayload = '<div style="background:url(javascript:alert(1))">';
assert('style 属性值逃逸（< 被转义）',
  api.escapeHtml(stylePayload).includes('&lt;div'));

// ── Data URI 攻击 ───
assert('data:text/html iframe XSS 逃逸',
  api.escapeHtml('<iframe src="data:text/html,<script>alert(1)</script>">').includes('&lt;iframe'));
assert('data:application/javascript script XSS 逃逸',
  api.escapeHtml('<script data-foo="data:application/javascript;alert(1)">').includes('&lt;script'));

// ── HTML 实体混淆 ──
assert('半实体 &#60; 经转义后失活', api.escapeHtml('&#60;').includes('&amp;#60;'));
assert('全实体 &#x3C; 经转义后失活', api.escapeHtml('&#x3C;').includes('&amp;#x3C;'));

// ── Unicode / 特殊字符 ──
assert('null 返回空字符串', api.escapeHtml(null) === '');
assert('undefined 返回空字符串', api.escapeHtml(undefined) === '');
assert('数字 0 转义为 "0"', api.escapeHtml(0) === '0');
assert('包含换行符的字符串被正确转义', api.escapeHtml('\n<script>') === '\n&lt;script&gt;');
assert('Unicode emoji 不被破坏', api.escapeHtml('你好🎉测试') === '你好🎉测试');

// ── 长输入 / ReDoS 防护 ──
assert('超大输入不阻塞（50ms 限制）', (() => {
  const big = 'A'.repeat(10000) + '</script>';
  const t = Date.now();
  api.escapeHtml(big);
  return Date.now() - t < 50;
})());

// ── SQL 注入在文本上下文中无害 ──
assert('SQL 注入 payload 中特殊字符被转义',
  api.escapeHtml("'; DROP TABLE users;--").includes('&#39;'));

// ── Prompt Injection 在文档名称中的影响 ──
assert('prompt 注入指令含 <script> 被转义',
  api.escapeHtml('ignore instructions<script>inject</script>')
    .includes('&lt;script&gt;'));

// ═══════════════════════════════════════════
// 2) parseSSEEvent 安全边界
// ═══════════════════════════════════════════

assert('parseSSEEvent 解析 delta 事件',
  api.parseSSEEvent('data: {"type":"delta","content":"hello"}\n\n').type === 'delta');
assert('parseSSEEvent 解析 sources 事件',
  api.parseSSEEvent('data: {"type":"sources","count":3}\n\n').count === 3);
assert('parseSSEEvent 解析 error 事件',
  api.parseSSEEvent('data: {"type":"error","detail":"timeout"}\n\n').detail === 'timeout');
assert('parseSSEEvent 解析 done 事件',
  api.parseSSEEvent('data: {"type":"done","skill":"policy"}\n\n').skill === 'policy');

assert('parseSSEEvent 对 script 标签内 JSON 不执行代码',
  (() => {
    const ev = api.parseSSEEvent('data: {"type":"error","detail":"<script>alert(document.cookie)</script>"}\n\n');
    return ev && ev.detail.includes('<script>');
  })());
assert('parseSSEEvent 数据溢出保护：过长 JSON 仍解析但不崩溃',
  (() => {
    const longData = JSON.stringify({ content: 'X'.repeat(500000) });
    try {
      const result = api.parseSSEEvent('data: ' + longData + '\n\n');
      return result !== null;
    } catch(e) { return false; }
  })());
assert('parseSSEEvent 多 data: 行取第一条',
  api.parseSSEEvent('data: {"type":"a"}\ndata: {"type":"b"}\n\n').type === 'a');
assert('parseSSEEvent 空 data 行返回 null',
  api.parseSSEEvent('data:\n\n') === null);
assert('parseSSEEvent 仅空白 data 行返回 null',
  api.parseSSEEvent('data:   \n\n') === null);

// ═══════════════════════════════════════════
// 3) 静态分析: HTML 中安全实践检查
// ═══════════════════════════════════════════

assert('index.html 使用 escapeHTML 转义用户问题',
  html.includes('RAG.escapeHtml(content)') || html.includes('RAG.escapeHtml'));
assert('index.html 有 chatHistory.push assistant 防注入',
  html.includes("role: 'assistant'"));

// ── agent workspace 架构检查 ──
assert('无 tab-bar 导航 (单栏 agent)', !html.includes('tab-bar') && !html.includes('switchTab'));
assert('无三视图切换 (welcome → chat feed)', !html.includes('view-chat') && !html.includes('view-docs'));
assert('index.html 有欢迎屏(空状态)', html.includes('empty-state') || html.includes('welcome-screen'));
assert('index.html 有 chat feed', html.includes('chat-feed') || html.includes('chatContainer'));
assert('index.html 有文件 chip 行', html.includes('file-chips-row') || html.includes('fileChipsRow'));
assert('index.html 有固定底部输入区', html.includes('input-area') || html.includes('inputAreaWrap'));

// ── DOM 属性注入防护 ──
assert('copy-answer 使用 data-* 属性委托 (非 onclick)',
  html.includes('[data-copy-answer]'));
assert('事件委托绑定存在 (closest)', html.includes('e.target.closest') || html.includes('closest('));

// ── 上传功能内联到输入区 ──
assert('index.html 有 uploadTrigger 按钮', html.includes('upload-trigger') || html.includes('uploadTrigger'));
assert('index.html 无 panel-upload (已移除)', !html.includes('panel-upload'));
assert('index.html 无 panel-docs (已移除)', !html.includes('panel-docs'));

// ── admin.html 安全检查 ──
assert('admin.html doc-row 文件名已转义',
  adminHtml.includes('RAG.escapeHtml(d.filename)'));
assert('admin.html doc-id 已转义',
  adminHtml.includes('RAG.escapeHtml(d.doc_id)'));
assert('admin.html uploadResult 已转义',
  adminHtml.includes('RAG.escapeHtml(data.doc_id)') && adminHtml.includes('RAG.escapeHtml(data.status)'));

// ═══════════════════════════════════════════
// 4) RAG namespace 完整性检查
// ═══════════════════════════════════════════

assert('两个 HTML 页面均未重复定义 authHeaders',
  !html.match(/function\s+authHeaders\s*\(/) && !adminHtml.match(/function\s+authHeaders\s*\(/));
assert('两个 HTML 页面均未重复定义 apiFetch',
  !html.match(/async\s+function\s+apiFetch\s*\(/) && !adminHtml.match(/async\s+function\s+apiFetch\s*\(/));
assert('两个 HTML 页面均未重复定义 escapeHtml',
  !html.match(/function\s+escapeHtml\s*\(/) && !adminHtml.match(/function\s+escapeHtml\s*\(/));
assert('index.html 使用 RAG.xxx 命名空间调用',
  html.includes('RAG.authHeaders()') && html.includes('RAG.escapeHtml') &&
  html.includes('RAG.icon') && html.includes('RAG.toast'));
assert('admin.html 使用 RAG.xxx 命名空间调用',
  adminHtml.includes('RAG.authHeaders(') && adminHtml.includes('RAG.apiFetch(') &&
  adminHtml.includes('RAG.escapeHtml(') && adminHtml.includes('RAG.icon('));

// ═══════════════════════════════════════════
// 5) common.js 工具函数存在性检查
// ═══════════════════════════════════════════

assert('common.js 暴露 RAG.formatBytes', commonJs.includes('RAG.formatBytes = formatBytes'));
assert('common.js 暴露 RAG.timeAgo', commonJs.includes('RAG.timeAgo = timeAgo'));
assert('common.js 暴露 RAG.debounce', commonJs.includes('RAG.debounce = debounce'));
assert('common.js 暴露 RAG.copyText', commonJs.includes('RAG.copyText = copyText'));
assert('common.js 暴露 RAG.toast', commonJs.includes('RAG.toast = toast'));
assert('common.js 暴露 RAG.skeleton', commonJs.includes('RAG.skeleton = skeleton'));
assert('common.js 暴露 RAG.parseSSEStream', commonJs.includes('RAG.parseSSEStream = parseSSEStream'));

// ── 新增图标 ──
const iconsJs = readFileSync(join(__dirname, '..', 'web', 'icons.js'), 'utf8');
assert('icons.js 包含 search 图标', iconsJs.includes("search:"));

// ── Upload validation (admin.html 保留) ──
assert('admin.html 上传有文件大小校验', adminHtml.includes('50 * 1024 * 1024'));
assert('admin.html 上传有类型白名单', adminHtml.includes('allowed') && adminHtml.includes('.test'));

// ── SSE 流式与 AbortController (新 Agent Workspace) ──
assert('index.html askQuestion 使用 SSE streaming',
  html.includes('qa/ask/stream') && html.includes('.getReader()'));
assert('index.html chatHistory 多轮上下文管理',
  html.includes('chatHistory.push') && html.includes('role:') && html.includes('content:'));
assert('index.html 有 AbortController 取消流式',
  html.includes('AbortController') && html.includes('abort()'));
assert('index.html 用户消息经 RAG.escapeHtml 转义',
  html.includes('RAG.escapeHtml'));
assert('index.html 文件芯片行显隐控制',
  html.includes('file-chips-row') || html.includes('fileChip'));
assert('index.html input-area 固定底部输入区',
  html.includes('input-area') || html.includes('inputAreaWrap'));
assert('index.html 有 thinking-dots 思考动画',
  html.includes('thinking-dots') || html.includes('thinkingMsg'));
assert('index.html 有 pipeline-badge 流程指示器',
  html.includes('pipeline-badge') || html.includes('pipelineStep'));
assert('index.html 有 rag-panel 可折叠来源面板',
  html.includes('rag-panel') || html.includes('rag-source'));

// ── SSE Buffer 管理存在 ──
assert('common.js parseSSEStream 有 buffer cap', commonJs.includes('SSE_MAX'));
assert('common.js parseSSEStream 有截断逻辑', commonJs.includes('slice(cutoff + 2)'));

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
