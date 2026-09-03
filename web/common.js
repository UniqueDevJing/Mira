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

  // ── API 错误处理 ──────────────────────────
  function handleApiError(url, err, showToast) {
    var msg = '请求失败，请检查网络后重试';
    if (err && err.name === 'AbortError') return null;
    var statusText = String(err.message || '');
    var m = statusText.match(/HTTP (\d+)/);
    if (m) {
      var code = parseInt(m[1]);
      if (code === 401) msg = 'API Key 无效或已过期，请在顶部设置页重新填入';
      else if (code === 403) msg = '权限不足，请使用管理员密钥';
      else if (code === 429) msg = '请求过于频繁，请稍后再试';
      else if (code >= 500) msg = '服务器内部错误 (' + code + ')，请稍后重试';
      else if (code >= 400) msg = '请求被拒绝 (' + code + ')：' + url;
    } else if (err instanceof TypeError) {
      msg = '网络连接失败，请检查后端是否启动';
    }
    if (showToast !== false) RAG.toast(msg, 'err', 3500);
    return msg;
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

  // ── 主题 (light/dark/system 三态, 持久化)
  //    偏好(localStorage rag_theme): 'light'|'dark'|'system' (默认 system)
  //    data-theme 始终写"生效值"(light/dark), CSS 只需两态, 兼容旧页面;
  //    system 模式由 matchMedia 实时跟随, 无需 CSS 媒体查询嵌套。
  var THEME_ORDER = ['light', 'dark', 'system'];
  var sysMedia = window.matchMedia ? matchMedia('(prefers-color-scheme: dark)') : null;

  function effectiveTheme() {
    var t = document.documentElement.getAttribute('data-theme');
    return t === 'dark' ? 'dark' : 'light';
  }
  function themePreference() {
    try {
      var t = localStorage.getItem('rag_theme');
      if (THEME_ORDER.indexOf(t) >= 0) return t;
    } catch (e) {}
    return 'system';
  }
  function applyTheme(pref) {
    var eff = (pref === 'system')
      ? (sysMedia && sysMedia.matches ? 'dark' : 'light')
      : pref;
    document.documentElement.setAttribute('data-theme', eff);
    try { localStorage.setItem('rag_theme', pref); } catch (e) {}
    updateThemeIcon();
    updateThemeColor();
  }
  // 三态循环: light → dark → system → light
  function cycleTheme() {
    var cur = themePreference();
    var next = THEME_ORDER[(THEME_ORDER.indexOf(cur) + 1) % THEME_ORDER.length];
    applyTheme(next);
  }
  // 兼容旧调用名
  function toggleTheme() { cycleTheme(); }
  function updateThemeIcon() {
    var el = document.getElementById('themeToggle');
    if (!el) return;
    var pref = themePreference();
    var icon = (pref === 'system') ? 'monitor'
      : (effectiveTheme() === 'dark' ? 'sun' : 'moon');
    el.innerHTML = RAG.icon(icon);
    var label = pref === 'system' ? '跟随系统' : (effectiveTheme() === 'dark' ? '暗色' : '亮色');
    el.title = '主题：' + label + ' · 点击切换';
    el.setAttribute('aria-label', el.title);
  }
  function updateThemeColor() {
    var m = document.querySelector('meta[name="theme-color"]');
    if (!m) return;
    var dark = effectiveTheme() === 'dark';
    m.setAttribute('content', dark ? '#0f1320' : '#4f6ef7');
  }
  // system 模式下跟随系统实时切换
  if (sysMedia) {
    sysMedia.addEventListener('change', function () {
      if (themePreference() === 'system') {
        document.documentElement.setAttribute('data-theme', sysMedia.matches ? 'dark' : 'light');
        updateThemeIcon();
        updateThemeColor();
      }
    });
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
  RAG.handleApiError = handleApiError;
  RAG.escapeHtml = escapeHtml;
  RAG.toggleTheme = toggleTheme;
  RAG.updateThemeIcon = updateThemeIcon;
  RAG.getSessionId = getSessionId;
  RAG.resetSessionId = resetSessionId;
  RAG.icon = RAG.icon || function () { return ''; };
  RAG.renderMarkdown = RAG.renderMarkdown || function (s) { return escapeHtml(s); };

  // ── 工具函数 ──────────────────────────────────

  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    var units = ['B', 'KB', 'MB', 'GB'];
    var i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(i ? 1 : 0) + ' ' + units[i];
  }

  /**
   * 相对时间格式化 (e.g. "3 分钟前")
   * @param {number|string} ms - 毫秒时间戳或 ISO 日期字符串
   * @returns {string}
   */
  function timeAgo(ms) {
    var now = Date.now();
    var t = typeof ms === 'string' ? new Date(ms).getTime() : ms;
    var diff = now - t;
    if (diff < 0) return '刚刚';
    var secs = Math.floor(diff / 1000);
    if (secs < 60) return secs + ' 秒前';
    var mins = Math.floor(secs / 60);
    if (mins < 60) return mins + ' 分钟前';
    var hours = Math.floor(mins / 60);
    if (hours < 24) return hours + ' 小时前';
    var days = Math.floor(hours / 24);
    return days + ' 天前';
  }

  /**
   * 防抖工具
   * @param {Function} fn
   * @param {number} delay ms
   * @returns {Function}
   */
  function debounce(fn, delay) {
    var timer;
    return function () {
      var ctx = this, args = arguments;
      clearTimeout(timer);
      timer = setTimeout(function () { fn.apply(ctx, args); }, delay);
    };
  }

  /**
   * 复制文本到剪贴板 (navigator.clipboard 降级 textarea.execCommand)
   * @param {string} text
   * @returns {Promise<boolean>}
   */
  function copyText(text) {
    if (!text) return Promise.resolve(false);
    try {
      if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text).then(function () { return true; });
      }
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand('copy');
      ta.remove();
      return Promise.resolve(ok);
    } catch (e) {
      console.warn('copyText failed:', e.message);
      return Promise.resolve(false);
    }
  }

  /**
   * 显示 toast 提示 (自动管理实例，避免多实例重叠)
   * @param {string} message
   * @param {'ok'|'err'|'warn'} type
   * @param {number} duration ms
   */
  function toast(message, type, duration) {
    type = type || 'ok';
    duration = duration || 2500;
    var wrap = document.querySelector('.toast-wrap');
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.className = 'toast-wrap';
      document.body.appendChild(wrap);
    }
    var t = document.createElement('div');
    t.className = 'toast toast-' + type;
    t.innerHTML = (type === 'ok' ? RAG.icon('check') : RAG.icon('x-circle')) + '<span></span>';
    t.querySelector('span').textContent = message;
    wrap.appendChild(t);
    setTimeout(function () {
      t.classList.add('toast-leave');
      setTimeout(function () { t.remove(); }, 300);
    }, duration);
  }

  /**
   * 在容器内生成骨架屏占位符
   * @param {HTMLElement} container
   * @param {number} count 占位数
   */
  function skeleton(container, count) {
    count = count || 3;
    container.innerHTML = '';
    for (var i = 0; i < count; i++) {
      var row = document.createElement('div');
      row.className = 'skeleton-row';
      row.innerHTML = '<div class="sk sk-w80"></div><div class="sk sk-w60"></div>';
      container.appendChild(row);
    }
  }

  /**
   * 增量解析 SSE 数据块，返回事件数组。
   * 处理多行 data: 字段（SSE spec 允许 \ndata:more 拼接）。
   * Buffer 上限 1MB，超出截断最旧数据。
   * @param {string} chunk - 从 stream 读取的原始文本
   * @returns {Array<Object>} [{type, data}]
   */
  /**
   * 创建独立的 SSE 解析器实例。
   * 每条流一个实例: 内部 buffer 隔离, 并发/快速连续的多条流互不串扰。
   * (旧的全局单例 parseSSEStream 在"停止上一条→立刻发起下一条"时, 残留 buffer
   *  可能把旧流尾部事件混进新流; 闭包实例从根上消除该隐患。)
   * @returns {{push: function(string): Array<Object>}}
   */

  // SSE 累积 buffer 上限 (1MB)。超出时丢弃最旧数据, 防止超长流把内存/单串推爆。
  // 由 createSSEParser 内部 while(buf.length > SSE_MAX) 落实。
  var SSE_MAX = 1024 * 1024;

  function createSSEParser() {
    var buf = '';
    return {
      push: function (chunk) {
        buf += chunk;
        // Buffer cap: 超出则丢弃最旧数据
        while (buf.length > SSE_MAX) {
          var cutoff = buf.indexOf('\n\n');
          buf = cutoff > 0 ? buf.slice(cutoff + 2) : buf.slice(Math.floor(SSE_MAX / 2));
        }
        var events = [];
        var idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          var block = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          var ev = parseSSEEvent(block);
          if (ev) events.push(ev);
        }
        return events;
      },
    };
  }

  // 兼容旧接口: 全局单实例。新代码请用 RAG.createSSEParser()。
  var _legacyParser = createSSEParser();
  function parseSSEStream(chunk) { return _legacyParser.push(chunk); }

  /** 解析单个 SSE data 块 */
  function parseSSEEvent(rawBlock) {
    var line = rawBlock.split('\n').find(function (l) { return l.startsWith('data:'); });
    if (!line) return null;
    try {
      return JSON.parse(line.slice(5).trim());
    } catch (e) {
      console.warn('SSE 块解析失败，已跳过:', line.slice(0, 80));
      return null;
    }
  }

  // 暴露给 RAG namespace
  RAG.formatBytes = formatBytes;
  RAG.timeAgo = timeAgo;
  RAG.debounce = debounce;
  RAG.copyText = copyText;
  RAG.toast = toast;
  RAG.skeleton = skeleton;
  RAG.parseSSEStream = parseSSEStream;

  // DOM 就绪: 替换静态 <i class="fa ..."> 图标, 自动绑定主题按钮
  document.addEventListener('DOMContentLoaded', function () {
    if (typeof RAG.replaceIcons === 'function') RAG.replaceIcons(document);
    var tb = document.getElementById('themeToggle');
    if (tb && !tb.dataset.bound) {
      tb.dataset.bound = '1';
      tb.addEventListener('click', toggleTheme);
      updateThemeIcon();
    }
  });
})();
