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

  function render(src) {
    if (src == null) return '';
    var esc = (window.RAG && window.RAG.escapeHtml) || function (s) { return String(s); };
    var lines = esc(String(src)).replace(/\r\n?/g, '\n').split('\n');
    var html = '';
    var i = 0;
    var BLOCK_RE = /^(#{1,6}\s|```|>\s?|[-*+]\s+|\d+\.\s+)/;
    while (i < lines.length) {
      var line = lines[i];
      // 代码块
      if (/^```/.test(line)) {
        var buf = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        html += '<pre><code>' + buf.join('\n') + '</code></pre>';
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
