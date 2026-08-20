/* common.js — RAG 2.0 前端共享逻辑 (index.html / admin.html)
   提供: API 基址, 鉴权头, 统一 fetch, HTML 转义, 主题切换, 会话管理, 图标与 Markdown 命名空间。
   以经典脚本 (defer) 加载; 顶层 const/function 在全局脚本间共享。 */
(function () {
  'use strict';

  // ── 基础 ──────────────────────────────────
  var API = '/api/v1';

  function authHeaders() {
    var k = localStorage.getItem('rag_api_key') || '';
    return k ? { 'X-API-Key': k } : {};
  }

  async function apiFetch(url, opts) {
    opts = opts || {};
    var r = await fetch(url, opts);
    if (!r.ok) {
      var msg = 'HTTP ' + r.status;
      try { var d = await r.json(); if (d.detail) msg = d.detail; } catch (e) {}
      throw new Error(msg);
    }
    return r;
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // ── 主题 (light/dark, 持久化; FOUC 防护由各页 <head> 内联脚本完成) ──
  function toggleTheme() {
    var root = document.documentElement;
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('rag_theme', next); } catch (e) {}
    updateThemeIcon(next);
  }
  function updateThemeIcon(t) {
    var el = document.getElementById('themeToggle');
    if (!el) return;
    el.innerHTML = RAG.icon(t === 'dark' ? 'sun' : 'moon');
  }

  // ── 会话 ID (localStorage 持久化, 跨设备/刷新不丢) ──
  function getSessionId() {
    var sid = localStorage.getItem('rag_session_id');
    if (!sid) {
      sid = (crypto.randomUUID ? crypto.randomUUID() : 's-' + Date.now() + '-' + Math.random().toString(36).slice(2));
      try { localStorage.setItem('rag_session_id', sid); } catch (e) {}
    }
    return sid;
  }
  function resetSessionId() {
    var old = localStorage.getItem('rag_session_id');
    if (old) {
      fetch(API + '/qa/session/' + encodeURIComponent(old), { method: 'DELETE', headers: authHeaders() }).catch(function () {});
    }
    var sid = 's-' + Date.now() + '-' + Math.random().toString(36).slice(2);
    try { localStorage.setItem('rag_session_id', sid); } catch (e) {}
    return sid;
  }

  // ── 共享命名空间 (图标/Markdown 由 icons.js / markdown.js 填充) ──
  var RAG = (window.RAG = window.RAG || {});
  RAG.API = API;
  RAG.authHeaders = authHeaders;
  RAG.apiFetch = apiFetch;
  RAG.escapeHtml = escapeHtml;
  RAG.toggleTheme = toggleTheme;
  RAG.updateThemeIcon = updateThemeIcon;
  RAG.getSessionId = getSessionId;
  RAG.resetSessionId = resetSessionId;
  RAG.icon = RAG.icon || function () { return ''; };
  RAG.renderMarkdown = RAG.renderMarkdown || function (s) { return escapeHtml(s); };

  // DOM 就绪: 替换静态 <i class="fa ..."> 图标, 自动绑定主题按钮
  document.addEventListener('DOMContentLoaded', function () {
    if (typeof RAG.replaceIcons === 'function') RAG.replaceIcons(document);
    var tb = document.getElementById('themeToggle');
    if (tb && !tb.dataset.bound) {
      tb.dataset.bound = '1';
      tb.addEventListener('click', toggleTheme);
      updateThemeIcon(document.documentElement.getAttribute('data-theme'));
    }
  });
})();
