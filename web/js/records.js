/* ============================================================
 * ESP32-ECG Web Console · records.js
 * 录制管理：ESP32 SoftAP REST API（GET /api/records ...）
 * ============================================================ */
(function (global) {
  'use strict';

  var Core = global.ECGCore;
  if (!Core) throw new Error('ecg-core.js 必须先加载');

  var ui = {};
  var busy = false;

  function init() {
    ui.base = document.getElementById('apiBase');
    ui.btnFetch = document.getElementById('btnFetchRecords');
    ui.stats = document.getElementById('recordStats');
    ui.statCount = document.getElementById('recStatCount');
    ui.statSize = document.getElementById('recStatSize');
    ui.statAbn = document.getElementById('recStatAbn');
    ui.statDur = document.getElementById('recStatDur');
    ui.corsNote = document.getElementById('corsNote');
    ui.corsDirectLink = document.getElementById('corsDirectLink');
    ui.wrap = document.getElementById('recordsWrap');
    ui.body = document.getElementById('recordsBody');
    ui.empty = document.getElementById('recordsEmpty');

    ui.btnFetch.addEventListener('click', fetchRecords);
    ui.base.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') fetchRecords();
    });
  }

  function baseUrl() {
    var v = (ui.base.value || 'http://192.168.4.1').trim();
    if (!/^https?:\/\//i.test(v)) v = 'http://' + v;
    return v.replace(/\/+$/, '');
  }

  async function fetchRecords() {
    if (busy) return;
    busy = true;
    ui.btnFetch.disabled = true;
    ui.btnFetch.textContent = '获取中…';
    ui.corsNote.hidden = true;
    toast('正在连接 ' + baseUrl() + ' …');

    try {
      var res = await fetchWithTimeout(baseUrl() + '/api/records', { method: 'GET' }, 6000);
      if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + res.statusText);
      var json = await res.json();
      var records = Array.isArray(json) ? json : (json.records || []);
      render(records);
      toast('获取到 ' + records.length + ' 条录制', 'ok');
    } catch (err) {
      ui.wrap.hidden = true;
      ui.stats.hidden = true;
      if (isCorsError(err)) {
        ui.corsDirectLink.href = baseUrl() + '/api/records';
        ui.corsNote.hidden = false;
        toast('请求被浏览器 CORS 拦截，详见提示', 'warn');
      } else {
        toast('获取录制列表失败：' + err.message, 'error');
      }
    } finally {
      busy = false;
      ui.btnFetch.disabled = false;
      ui.btnFetch.textContent = '获取录制列表';
    }
  }

  function render(records) {
    ui.wrap.hidden = false;
    ui.stats.hidden = false;
    ui.body.innerHTML = '';
    ui.empty.hidden = records.length !== 0;

    var totalSize = 0, totalAbn = 0, totalDur = 0;
    records.forEach(function (r) {
      totalSize += Number(r.size) || 0;
      totalAbn += Number(r.abnormal_seconds) || 0;
      totalDur += Number(r.duration) || 0;
    });

    ui.statCount.textContent = records.length;
    ui.statSize.textContent = Core.formatBytes(totalSize);
    ui.statAbn.textContent = Core.formatDuration(totalAbn);
    ui.statDur.textContent = Core.formatDuration(totalDur);

    /* 新 → 旧 */
    records.slice().sort(function (a, b) { return (b.id || 0) - (a.id || 0); }).forEach(function (r) {
      var tr = document.createElement('tr');

      var tdId = document.createElement('td');
      tdId.className = 'cell-id';
      tdId.innerHTML = '<div>#' + String(r.id) + '</div><div style="color:var(--muted);font-size:0.7rem">' +
        (r.start ? Core.formatClock(parseISO(r.start)) : '时间未知') + '</div>';
      tr.appendChild(tdId);

      var tdDur = document.createElement('td');
      tdDur.textContent = Core.formatDuration(r.duration || 0);
      tr.appendChild(tdDur);

      var tdSamples = document.createElement('td');
      tdSamples.textContent = (r.total_samples != null ? r.total_samples : '-').toLocaleString();
      tr.appendChild(tdSamples);

      var tdSize = document.createElement('td');
      tdSize.textContent = Core.formatBytes(r.size || 0);
      tr.appendChild(tdSize);

      var tdAbn = document.createElement('td');
      tdAbn.className = (Number(r.abnormal_seconds) || 0) > 0 ? 'cell-abn' : '';
      tdAbn.textContent = (Number(r.abnormal_seconds) || 0) > 0 ? '⚠ ' + r.abnormal_seconds + 's' : '0s';
      tr.appendChild(tdAbn);

      var tdAct = document.createElement('td');
      tdAct.className = 'cell-actions';
      tdAct.appendChild(actionBtn('回放', 'link-btn', function () { downloadAndPlay(r); }));
      tdAct.appendChild(actionBtn('下载', 'link-btn', function () { downloadFile(r); }));
      tdAct.appendChild(actionBtn('删除', 'btn btn--danger btn-sm', function () { deleteRecord(r); }));
      tr.appendChild(tdAct);

      ui.body.appendChild(tr);
    });
  }

  function actionBtn(text, cls, onClick) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = cls;
    b.textContent = text;
    b.addEventListener('click', onClick);
    return b;
  }

  async function downloadData(id) {
    var url = baseUrl() + '/api/records/' + id + '/data';
    var res = await fetchWithTimeout(url, { method: 'GET' }, 30000);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    return await res.arrayBuffer();
  }

  async function downloadAndPlay(r) {
    if (!global.ECGPlayback) {
      toast('回放模块未加载', 'error');
      return;
    }
    try {
      toast('正在下载录制 #' + r.id + ' …');
      var buf = await downloadData(r.id);
      var record = Core.parseEcgr(buf);
      record.fileName = 'ecg_rec_' + r.id + '.ecgr';
      global.ECGPlayback.loadRecord(record);
      global.ECGUI.go('playback');
      toast('已载入回放：#' + r.id, 'ok');
    } catch (err) {
      if (isCorsError(err)) {
        ui.corsNote.hidden = false;
        ui.corsDirectLink.href = baseUrl() + '/api/records';
        toast('下载被浏览器 CORS 拦截', 'warn');
      } else {
        toast('下载/解析失败：' + err.message, 'error');
      }
    }
  }

  async function downloadFile(r) {
    try {
      var buf = await downloadData(r.id);
      var blob = new Blob([buf], { type: 'application/octet-stream' });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'ecg_rec_' + r.id + '.ecgr';
      a.click();
      setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
      toast('已开始下载 ecg_rec_' + r.id + '.ecgr', 'ok');
    } catch (err) {
      if (isCorsError(err)) {
        ui.corsNote.hidden = false;
        toast('下载被浏览器 CORS 拦截', 'warn');
      } else {
        toast('下载失败：' + err.message, 'error');
      }
    }
  }

  async function deleteRecord(r) {
    if (!window.confirm('确定删除录制 #' + r.id + ' 吗？设备端文件将被永久删除。')) return;
    try {
      var res = await fetchWithTimeout(baseUrl() + '/api/records/' + r.id, { method: 'DELETE' }, 6000);
      if (!res.ok && res.status !== 404) throw new Error('HTTP ' + res.status);
      toast('已删除录制 #' + r.id, 'ok');
      fetchRecords();
    } catch (err) {
      if (isCorsError(err)) {
        ui.corsNote.hidden = false;
        toast('删除请求被浏览器 CORS 拦截', 'warn');
      } else {
        toast('删除失败：' + err.message, 'error');
      }
    }
  }

  function fetchWithTimeout(url, options, ms) {
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, ms);
    return fetch(url, Object.assign({}, options, { signal: ctrl.signal })).finally(function () {
      clearTimeout(timer);
    });
  }

  function isCorsError(err) {
    if (!err) return false;
    var msg = String(err.message || err);
    return /failed to fetch|networkerror|load failed|cors|blocked/i.test(msg);
  }

  /* 固件 ISO8601 "1970-01-01T00:12:34Z" → 本地时间秒 */
  function parseISO(s) {
    if (typeof s !== 'string') return 0;
    var ms = Date.parse(s);
    if (!isFinite(ms)) return 0;
    return Math.floor(ms / 1000);
  }

  function toast(msg, cls) {
    if (global.ECGUI && global.ECGUI.toast) global.ECGUI.toast(msg, cls);
    else console.log('[toast]', msg);
  }

  global.ECGRecords = { init: init, baseUrl: baseUrl, _ui: ui };
})(typeof window !== 'undefined' ? window : globalThis);
