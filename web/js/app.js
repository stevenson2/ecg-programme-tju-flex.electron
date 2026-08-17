/* ============================================================
 * ESP32-ECG Web Console · app.js
 * 应用外壳：导航切换、Toast、模块装配
 * ============================================================ */
(function (global) {
  'use strict';

  var ui = {};

  function init() {
    ui.nav = document.getElementById('mainNav');
    ui.toastRoot = document.getElementById('toastRoot');
    ui.heroBpm = document.getElementById('heroBpm');

    ui.nav.addEventListener('click', function (e) {
      var btn = e.target.closest('.nav-btn');
      if (btn) go(btn.dataset.view);
    });

    document.querySelectorAll('[data-go-view]').forEach(function (el) {
      el.addEventListener('click', function () { go(el.dataset.goView); });
    });

    document.querySelectorAll('.brand[data-nav]').forEach(function (el) {
      el.addEventListener('click', function (e) {
        e.preventDefault();
        go(el.dataset.nav);
      });
    });

    global.ECGLive.init();
    global.ECGRecords.init();
    global.ECGPlayback.init();

    /* 演示心率的 Hero 数字跟随实时监测 */
    setInterval(function () {
      var last = global.ECGLive._state && global.ECGLive._state.lastRow;
      if (last && last.bpm > 0) {
        ui.heroBpm.textContent = Math.round(last.bpm) + ' BPM';
      }
    }, 1000);

    var initial = location.hash.replace('#', '');
    var valid = ['overview', 'live', 'records', 'playback'];
    go(valid.indexOf(initial) >= 0 ? initial : 'overview', true);
  }

  function go(view, silent) {
    var views = document.querySelectorAll('.view');
    var found = false;
    views.forEach(function (v) {
      var active = v.dataset.view === view;
      v.classList.toggle('active', active);
      if (active) found = true;
    });
    if (!found) return;

    ui.nav.querySelectorAll('.nav-btn').forEach(function (b) {
      b.classList.toggle('active', b.dataset.view === view);
    });

    if (!silent) {
      try { history.replaceState(null, '', '#' + view); } catch (e) { /* file:// 环境忽略 */ }
    }
    global.scrollTo({ top: 0, behavior: 'auto' });
  }

  function toast(msg, cls) {
    var el = document.createElement('div');
    el.className = 'toast ' + (cls || '');
    el.textContent = msg;
    ui.toastRoot.appendChild(el);
    while (ui.toastRoot.childElementCount > 4) ui.toastRoot.removeChild(ui.toastRoot.firstChild);
    setTimeout(function () {
      el.classList.add('leaving');
      setTimeout(function () { el.remove(); }, 260);
    }, 3600);
  }

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  ready(init);

  global.ECGUI = {
    go: go,
    toast: toast
  };
})(typeof window !== 'undefined' ? window : globalThis);
