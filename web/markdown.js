/* markdown.js — 轻量、安全的 Markdown → HTML 渲染器
   安全策略: 先整体 HTML 转义, 再做格式化; 仅允许 http(s) 链接; 不允许原始 HTML 注入。 */
(function () {
  'use strict';

  function inline(text) {
    var t = String(text);
    // 行内代码 (先于加粗/斜体, 避免代码内符号被二次格式化)
    t = t.replace(/`([^`]+)`/g, function (_, c) { return '<code>' + c + '</code>'; });
    // 粗体
    t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    // 斜体 (*...* 与 _..._)
    t = t.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    t = t.replace(/(^|[^\w])_([^_]+)_(?=[^\w]|$)/g, '$1<em>$2</em>');
    // 删除线
    t = t.replace(/~~([^~]+)~~/g, '<del>$1</del>');
    // 链接 [text](http(s)://...)
    t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    return t;
  }

  // ── 安全代码高亮 (自包含, 无外部依赖) ──
  // 策略: 在「原始」代码上做词法切分, 每个 token 单独转义后包裹 <span>。
  // 因为所有输出都经过 ESC 转义, 不存在原始 HTML 注入风险。
  var ESC = (window.RAG && window.RAG.escapeHtml) || function (s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };
  var KW = {
    js: ['const','let','var','function','return','if','else','for','while','do','switch','case','break','continue','new','class','extends','super','this','typeof','instanceof','in','of','await','async','yield','import','export','from','default','try','catch','finally','throw','delete','void','null','true','false'],
    ts: ['const','let','var','function','return','if','else','for','while','class','extends','interface','type','enum','implements','public','private','protected','readonly','namespace','import','export','from','async','await','new','this','null','true','false'],
    python: ['def','return','if','elif','else','for','while','break','continue','import','from','as','class','try','except','finally','with','lambda','yield','global','nonlocal','pass','raise','assert','del','in','is','not','and','or','None','True','False','self'],
    json: ['true','false','null'],
    bash: ['if','then','else','fi','for','in','do','done','while','case','esac','function','export','local','return','echo','cd','exit'],
    sql: ['SELECT','FROM','WHERE','INSERT','INTO','VALUES','UPDATE','SET','DELETE','CREATE','TABLE','JOIN','LEFT','RIGHT','INNER','OUTER','ON','GROUP','BY','ORDER','HAVING','LIMIT','AS','AND','OR','NOT','NULL','COUNT','SUM','AVG','MIN','MAX'],
    default: []
  };
  function highlight(raw, lang) {
    var kws = KW[lang] || KW.default;
    var commentRe = (lang === 'python') ? /^#[^\n]*/ : /^\/\/[^\n]*|^\/\*[\s\S]*?\*\//;
    var stringRe = /^"(?:[^"\\]|\\.)*"|^'(?:[^'\\]|\\.)*'|^`(?:[^`\\]|\\.)*`/;
    var numberRe = /^\d+(?:\.\d+)?\b/;
    var kwRe = kws.length ? new RegExp('^\\b(?:' + kws.join('|') + ')\\b' + (lang === 'sql' ? 'i' : '')) : null;
    var out = '', pos = 0, n = raw.length;
    while (pos < n) {
      var rest = raw.slice(pos), m;
      if ((m = commentRe.exec(rest))) { out += '<span class="tok-com">' + ESC(m[0]) + '</span>'; }
      else if ((m = stringRe.exec(rest))) { out += '<span class="tok-str">' + ESC(m[0]) + '</span>'; }
      else if ((m = numberRe.exec(rest))) { out += '<span class="tok-num">' + ESC(m[0]) + '</span>'; }
      else if (kwRe && (m = kwRe.exec(rest))) { out += '<span class="tok-kw">' + ESC(m[0]) + '</span>'; }
      else if ((m = /^[\w$]+/.exec(rest))) { out += '<span class="tok-id">' + ESC(m[0]) + '</span>'; }
      else if ((m = /^\s+/.exec(rest))) { out += ESC(m[0]); }
      else { out += ESC(rest[0]); }
      pos += (m ? m[0].length : 1);
    }
    return out;
  }
  function renderCodeBlock(raw, lang) {
    lang = lang || '';
    var label = lang || 'text';
    var iconHtml = (window.RAG && window.RAG.icon) ? window.RAG.icon('copy') : '';
    return '<div class="code-block" data-lang="' + ESC(label) + '">' +
      '<div class="code-head"><span class="code-lang">' + ESC(label) + '</span>' +
      '<button type="button" class="code-copy" data-copy-code aria-label="复制代码">' + iconHtml + ' 复制</button></div>' +
      '<pre><code class="lang-' + ESC(label) + '">' + highlight(raw, lang) + '</code></pre></div>';
  }

  function render(src) {
    if (src == null) return '';
    var esc = (window.RAG && window.RAG.escapeHtml) || function (s) { return String(s); };
    var rawLines = String(src).replace(/\r\n?/g, '\n').split('\n');
    var lines = esc(String(src)).replace(/\r\n?/g, '\n').split('\n');
    var html = '';
    var i = 0;
    var BLOCK_RE = /^(#{1,6}\s|```|>\s?|[-*+]\s+|\d+\.\s+)/;
    while (i < lines.length) {
      var line = lines[i];
      // 代码块 (```lang ... ```) — 语言标签 + 复制按钮 + 轻量高亮
      if (/^```/.test(line)) {
        var fence = line.match(/^```\s*([\w+#.-]*)/);
        var lang = fence && fence[1] ? fence[1] : '';
        var cstart = i + 1; i++;
        while (i < lines.length && !/^```/.test(lines[i])) { i++; }
        var codeRaw = rawLines.slice(cstart, i).join('\n');
        i++; // 跳过结束围栏
        html += renderCodeBlock(codeRaw, lang);
        continue;
      }
      // 标题
      var m = line.match(/^(#{1,6})\s+(.*)$/);
      if (m) { var lvl = m[1].length; html += '<h' + lvl + '>' + inline(m[2]) + '</h' + lvl + '>'; i++; continue; }
      // 分割线
      if (/^(\*{3,}|-{3,}|_{3,})$/.test(line.trim())) { html += '<hr>'; i++; continue; }
      // 引用
      if (/^>\s?/.test(line)) {
        var qb = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) { qb.push(lines[i].replace(/^>\s?/, '')); i++; }
        html += '<blockquote>' + inline(qb.join(' ')) + '</blockquote>';
        continue;
      }
      // 无序列表
      if (/^[-*+]\s+/.test(line)) {
        var ub = [];
        while (i < lines.length && /^[-*+]\s+/.test(lines[i])) { ub.push(lines[i].replace(/^[-*+]\s+/, '')); i++; }
        html += '<ul>' + ub.map(function (x) { return '<li>' + inline(x) + '</li>'; }).join('') + '</ul>';
        continue;
      }
      // 有序列表
      if (/^\d+\.\s+/.test(line)) {
        var ob = [];
        while (i < lines.length && /^\d+\.\s+/.test(lines[i])) { ob.push(lines[i].replace(/^\d+\.\s+/, '')); i++; }
        html += '<ol>' + ob.map(function (x) { return '<li>' + inline(x) + '</li>'; }).join('') + '</ol>';
        continue;
      }
      // 空行
      if (line.trim() === '') { i++; continue; }
      // 段落
      var pb = [line]; i++;
      while (i < lines.length && lines[i].trim() !== '' && !BLOCK_RE.test(lines[i])) { pb.push(lines[i]); i++; }
      html += '<p>' + inline(pb.join(' ')) + '</p>';
    }
    return html;
  }

  var RAG = (window.RAG = window.RAG || {});
  RAG.renderMarkdown = render;
})();
