/* ============================================================
 * ESP32-ECG Web Console · playback.js
 * 本地回放：.ecgr 解析、概览、缩放、播放与 CSV 导出
 * ============================================================ */
(function (global) {
  'use strict';

  var Core = global.ECGCore;
  if (!Core) throw new Error('ecg-core.js 必须先加载');

  var state = {
    record: null,
    cursor: 0,          /* 窗口起点（秒） */
    windowSec: 5,
    playing: false,
    speed: 1,
    lastTick: 0,
    rafId: 0,
    overviewDirty: true,
    dragging: null      /* 'overview' | 'main' */
  };

  var ui = {};

  function init() {
    ui.dropzone = document.getElementById('dropzone');
    ui.fileInput = document.getElementById('fileInput');
    ui.playerWrap = document.getElementById('playerWrap');
    ui.pbDur = document.getElementById('pbDur');
    ui.pbRate = document.getElementById('pbRate');
    ui.pbSamples = document.getElementById('pbSamples');
    ui.pbAbn = document.getElementById('pbAbn');
    ui.pbStart = document.getElementById('pbStart');
    ui.pbFile = document.getElementById('pbFile');
    ui.pbTitle = document.getElementById('pbTitle');
    ui.pbWindow = document.getElementById('pbWindow');
    ui.pbCanvas = document.getElementById('pbCanvas');
    ui.pbOverview = document.getElementById('pbOverview');
    ui.pbPlayhead = document.getElementById('pbPlayhead');
    ui.pbMarker = document.getElementById('pbMarker');
    ui.pbTime = document.getElementById('pbTime');
    ui.pbTotal = document.getElementById('pbTotal');
    ui.btnPlay = document.getElementById('btnPbPlay');
    ui.btnSkipBack = document.getElementById('btnPbSkipBack');
    ui.btnSkipFwd = document.getElementById('btnPbSkipFwd');
    ui.pbSpeed = document.getElementById('pbSpeed');
    ui.pbWindowSel = document.getElementById('pbWindowSel');
    ui.btnExport = document.getElementById('btnExportWindowCsv');

    ui.dropzone.addEventListener('click', function () { ui.fileInput.click(); });
    ui.fileInput.addEventListener('change', function () {
      if (ui.fileInput.files && ui.fileInput.files[0]) loadFile(ui.fileInput.files[0]);
    });

    ['dragenter', 'dragover'].forEach(function (ev) {
      ui.dropzone.addEventListener(ev, function (e) {
        e.preventDefault();
        ui.dropzone.classList.add('drag');
      });
    });
    ['dragleave', 'drop'].forEach(function (ev) {
      ui.dropzone.addEventListener(ev, function (e) {
        e.preventDefault();
        ui.dropzone.classList.remove('drag');
      });
    });
    ui.dropzone.addEventListener('drop', function (e) {
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) loadFile(f);
    });

    ui.btnPlay.addEventListener('click', togglePlay);
    ui.btnSkipBack.addEventListener('click', function () { seekBy(-5); });
    ui.btnSkipFwd.addEventListener('click', function () { seekBy(5); });
    ui.pbSpeed.addEventListener('change', function () { state.speed = parseFloat(ui.pbSpeed.value); });
    ui.pbWindowSel.addEventListener('change', function () {
      state.windowSec = parseFloat(ui.pbWindowSel.value);
      ui.pbWindow.textContent = '窗口 ' + state.windowSec + 's';
      clampCursor();
      draw();
    });
    ui.btnExport.addEventListener('click', exportWindowCsv);

    ui.pbOverview.addEventListener('mousedown', function (e) { beginDrag(e, 'overview'); });
    ui.pbCanvas.addEventListener('mousedown', function (e) { beginDrag(e, 'main'); });
    ui.pbCanvas.addEventListener('wheel', onWheel, { passive: false });

    window.addEventListener('mousemove', onDrag);
    window.addEventListener('mouseup', endDrag);
    window.addEventListener('keydown', onKey);
    window.addEventListener('resize', function () {
      state.overviewDirty = true;
      draw();
    });
  }

  /* ================= 载入 ================= */

  function loadFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var record = Core.parseEcgr(reader.result);
        record.fileName = file.name;
        loadRecord(record);
      } catch (err) {
        toast('文件解析失败：' + err.message, 'error');
      }
    };
    reader.onerror = function () {
      toast('文件读取失败', 'error');
    };
    reader.readAsArrayBuffer(file);
  }

  function loadRecord(record) {
    stopPlay();
    state.record = record;
    state.cursor = 0;
    state.windowSec = Math.min(5, record.durationSec || 5);
    ui.pbWindowSel.value = state.windowSec >= 3 ? String(state.windowSec) : '3';
    state.overviewDirty = true;

    ui.dropzone.hidden = true;
    ui.playerWrap.hidden = false;

    var total = record.totalSamples / record.sampleRate;
    ui.pbDur.textContent = Core.formatDuration(record.durationSec || total);
    ui.pbRate.textContent = record.sampleRate + ' Hz';
    ui.pbSamples.textContent = record.totalSamples.toLocaleString();
    ui.pbAbn.textContent = record.abnormalSec > 0 ? record.abnormalSec + ' s' : '无';
    ui.pbStart.textContent = record.startUnix ? Core.formatClock(record.startUnix) : '上电相对时间';
    ui.pbFile.textContent = record.fileName || '录制';
    ui.pbTitle.textContent = record.fileName || 'ECG 录制回放';
    ui.pbTotal.textContent = total.toFixed(1);
    ui.pbWindow.textContent = '窗口 ' + state.windowSec + 's';
    if (record.truncated) {
      toast('警告：头部计数大于实际数据，已按可用数据截断', 'warn');
    }
    clampCursor();
    draw();
  }

  /* ================= 播放控制 ================= */

  function totalSeconds() {
    return state.record ? state.record.totalSamples / state.record.sampleRate : 0;
  }

  function clampCursor() {
    var total = totalSeconds();
    state.cursor = Core.clamp(state.cursor, 0, Math.max(0, total - state.windowSec));
  }

  function togglePlay() {
    if (!state.record) return;
    state.playing = !state.playing;
    ui.btnPlay.textContent = state.playing ? '⏸' : '▶';
    ui.btnPlay.classList.toggle('icon-btn--primary', state.playing);
    if (state.playing) {
      state.lastTick = performance.now();
      state.rafId = requestAnimationFrame(playTick);
    } else if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = 0;
    }
  }

  function stopPlay() {
    state.playing = false;
    ui.btnPlay.textContent = '▶';
    ui.btnPlay.classList.remove('icon-btn--primary');
    if (state.rafId) cancelAnimationFrame(state.rafId);
    state.rafId = 0;
  }

  function playTick(now) {
    if (!state.playing) return;
    var dt = Math.min(0.2, (now - state.lastTick) / 1000);
    state.lastTick = now;
    state.cursor += dt * state.speed;
    var total = totalSeconds();
    if (state.cursor >= total - state.windowSec) {
      state.cursor = Math.max(0, total - state.windowSec);
      stopPlay();
    }
    draw();
    if (state.playing) state.rafId = requestAnimationFrame(playTick);
  }

  function seekBy(delta) {
    if (!state.record) return;
    state.cursor += delta;
    clampCursor();
    draw();
  }

  /* ================= 绘制 ================= */

  function draw() {
    drawMain();
    drawOverview();
    ui.pbTime.textContent = state.cursor.toFixed(1);
    var cw = ui.pbOverview.clientWidth || 1;
    ui.pbPlayhead.style.left = ((state.cursor / Math.max(0.1, totalSeconds())) * 100).toFixed(3) + '%';
    ui.pbMarker.style.left = '0px';
  }

  function drawMain() {
    var rec = state.record;
    var canvas = ui.pbCanvas;
    if (!rec || !canvas || !canvas.isConnected) return;
    var dpr = global.devicePixelRatio || 1;
    var cw = canvas.clientWidth;
    var ch = canvas.clientHeight;
    if (!cw || !ch) return;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    var fs = rec.sampleRate;
    var startIdx = Math.floor(state.cursor * fs);
    var n = Math.min(rec.totalSamples - startIdx, Math.ceil(state.windowSec * fs));
    if (n <= 1) return;

    var padX = 58;
    var padTop = 14;
    var padBottom = 24;
    var innerW = cw - padX - 14;
    var innerH = ch - padTop - padBottom;

    /* 电压范围：自动 + 最小跨度 0.4V */
    var mn = Infinity, mx = -Infinity;
    for (var i = 0; i < n; i++) {
      var v = rec.voltAt(startIdx + i);
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (!isFinite(mn) || !isFinite(mx)) { mn = -1; mx = 1; }
    var center = (mn + mx) / 2;
    var half = Math.max((mx - mn) / 2, 0.2) * 1.12;
    mn = center - half;
    mx = center + half;

    /* 网格 + 电压标签 */
    var step = pickVoltStep(2 * half);
    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    var first = Math.ceil(mn / step) * step;
    for (var gv = first; gv <= mx + 1e-9; gv += step) {
      var gy = padTop + innerH * (1 - (gv - mn) / (mx - mn));
      ctx.strokeStyle = 'rgba(148,178,224,0.08)';
      ctx.beginPath();
      ctx.moveTo(padX, gy);
      ctx.lineTo(cw - 14, gy);
      ctx.stroke();
      ctx.fillStyle = 'rgba(148,163,184,0.7)';
      ctx.fillText(gv.toFixed(2) + 'V', padX - 8, gy);
    }

    var timeStep = state.windowSec <= 3 ? 0.2 : (state.windowSec <= 5 ? 0.5 : (state.windowSec <= 10 ? 1 : (state.windowSec <= 30 ? 2 : 5)));
    ctx.textAlign = 'center';
    for (var gt = 0; gt <= state.windowSec + 1e-9; gt += timeStep) {
      var gx = padX + innerW * (gt / state.windowSec);
      ctx.strokeStyle = 'rgba(148,178,224,0.08)';
      ctx.beginPath();
      ctx.moveTo(gx, padTop);
      ctx.lineTo(gx, ch - padBottom);
      ctx.stroke();
      ctx.fillStyle = 'rgba(148,163,184,0.7)';
      ctx.fillText((gt).toFixed(1) + 's', gx, ch - padBottom + 12);
    }

    /* 异常位图覆盖 */
    if (rec.bitmap) {
      var winStart = state.cursor;
      var winEnd = state.cursor + state.windowSec;
      ctx.fillStyle = 'rgba(244,63,94,0.14)';
      for (var s = Math.floor(winStart); s <= Math.ceil(winEnd) && s < rec.bitmap.length; s++) {
        if (s >= 0 && rec.bitmap[s]) {
          var x0 = padX + innerW * ((s - winStart) / state.windowSec);
          var x1 = padX + innerW * ((s + 1 - winStart) / state.windowSec);
          ctx.fillRect(x0, padTop, Math.max(1, x1 - x0), innerH);
        }
      }
    }

    /* 波形 */
    var yOf = function (val) { return padTop + innerH * (1 - (val - mn) / (mx - mn)); };
    var xOf = function (idx) { return padX + innerW * ((idx - startIdx) / (n - 1)); };
    ctx.beginPath();
    for (var j = 0; j < n; j++) {
      var px = xOf(startIdx + j);
      var py = yOf(rec.voltAt(startIdx + j));
      if (j === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = '#2dd4bf';
    ctx.lineWidth = 1.7;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.shadowColor = 'rgba(45,212,191,0.8)';
    ctx.shadowBlur = 9;
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = 'rgba(45,212,191,0.9)';
    ctx.font = 'bold 10px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'left';
    ctx.fillText('clean · ' + fs + ' Hz · 8000 LSB/V', cw - 14 - 170, padTop + 4);
  }

  function drawOverview() {
    var rec = state.record;
    var canvas = ui.pbOverview;
    if (!rec || !canvas || !canvas.isConnected) return;
    var dpr = global.devicePixelRatio || 1;
    var cw = canvas.clientWidth;
    var ch = canvas.clientHeight;
    if (!cw || !ch) return;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
      state.overviewDirty = true;
    }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    if (state.overviewDirty) {
      state.overviewDirty = false;
      ctx.clearRect(0, 0, cw, ch);
      var cols = Math.min(cw, rec.totalSamples);
      var dm = Core.decimateMinMax(rec.samples, cols);
      var mn = Infinity, mx = -Infinity;
      for (var c = 0; c < dm.columns; c++) {
        if (dm.mins[c] < mn) mn = dm.mins[c];
        if (dm.maxs[c] > mx) mx = dm.maxs[c];
      }
      if (!isFinite(mn) || !isFinite(mx) || mx - mn < 1) { mn = -1000; mx = 1000; }
      var span = (mx - mn) * 1.06;
      var mid = (mn + mx) / 2;
      ctx.fillStyle = 'rgba(45,212,191,0.75)';
      for (var x = 0; x < dm.columns; x++) {
        var y0 = ch / 2 - ((dm.maxs[x] - mid) / span) * ch;
        var y1 = ch / 2 - ((dm.mins[x] - mid) / span) * ch;
        ctx.fillRect(x, y0, 1, Math.max(1, y1 - y0));
      }
      if (rec.bitmap) {
        ctx.fillStyle = 'rgba(244,63,94,0.5)';
        var total = totalSeconds();
        for (var s = 0; s < rec.bitmap.length; s++) {
          if (rec.bitmap[s]) {
            var bx = (s / total) * cw;
            var bw = Math.max(1, cw / total);
            ctx.fillRect(bx, 0, bw, ch);
          }
        }
      }
    }
  }

  function pickVoltStep(span) {
    var candidates = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2];
    for (var i = 0; i < candidates.length; i++) {
      if (span / candidates[i] <= 10) return candidates[i];
    }
    return 5;
  }

  /* ================= 交互 ================= */

  function beginDrag(e, zone) {
    if (!state.record) return;
    state.dragging = zone;
    seekFromEvent(e, zone);
  }

  function onDrag(e) {
    if (!state.dragging || !state.record) return;
    seekFromEvent(e, state.dragging);
  }

  function endDrag() { state.dragging = null; }

  function seekFromEvent(e, zone) {
    var el = zone === 'overview' ? ui.pbOverview : ui.pbCanvas;
    var rect = el.getBoundingClientRect();
    var frac = Core.clamp((e.clientX - rect.left) / rect.width, 0, 1);
    var total = totalSeconds();
    if (zone === 'overview') {
      state.cursor = frac * total;
    } else {
      state.cursor = frac * total - state.windowSec / 2;
    }
    clampCursor();
    draw();
  }

  function onWheel(e) {
    if (!state.record) return;
    e.preventDefault();
    var before = state.windowSec;
    var center = state.cursor + state.windowSec / 2;
    var factor = Math.exp(e.deltaY * 0.0012);
    state.windowSec = Core.clamp(state.windowSec * factor, 1, Math.min(60, totalSeconds()));
    var shift = center - (state.cursor + before / 2);
    state.cursor += shift;
    clampCursor();
    ui.pbWindowSel.value = state.windowSec;
    ui.pbWindow.textContent = '窗口 ' + state.windowSec.toFixed(1) + 's';
    draw();
  }

  function onKey(e) {
    if (!state.record) return;
    var view = document.getElementById('view-playback');
    if (!view || !view.classList.contains('active')) return;
    if (e.target && /^(input|select|textarea)$/i.test(e.target.tagName)) return;
    if (e.code === 'Space') {
      e.preventDefault();
      togglePlay();
    } else if (e.code === 'ArrowLeft') {
      e.preventDefault();
      seekBy(-0.1);
    } else if (e.code === 'ArrowRight') {
      e.preventDefault();
      seekBy(0.1);
    }
  }

  /* ================= CSV 导出 ================= */

  function exportWindowCsv() {
    var rec = state.record;
    if (!rec) return;
    var startIdx = Math.floor(state.cursor * rec.sampleRate);
    var n = Math.min(rec.totalSamples - startIdx, Math.ceil(state.windowSec * rec.sampleRate));
    if (n <= 0) return;

    var lines = [Core.CSV_COLUMNS.join(',')];
    for (var i = 0; i < n; i++) {
      var idx = startIdx + i;
      var clean = rec.voltAt(idx).toFixed(4);
      var abn = rec.bitmap ? (rec.bitmap[Math.min(rec.bitmap.length - 1, Math.floor(idx / rec.sampleRate))] ? 1 : 0) : 0;
      /* .ecgr 仅存 clean 通道：其余列为占位，便于沿用 9 列工具链 */
      lines.push([clean, clean, clean, 0, 0, '1.000', 0, abn, abn ? '0.990' : '0.000'].join(','));
    }
    var blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = (rec.fileName || 'recording').replace(/\.ecgr$/i, '') + '_window_' + state.cursor.toFixed(1) + 's.csv';
    a.click();
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
    toast('已导出当前窗口 ' + n + ' 行 CSV（仅 clean 通道有效，其余为占位列）', 'ok');
  }

  function toast(msg, cls) {
    if (global.ECGUI && global.ECGUI.toast) global.ECGUI.toast(msg, cls);
    else console.log('[toast]', msg);
  }

  global.ECGPlayback = {
    init: init,
    loadRecord: loadRecord,
    loadFile: loadFile,
    _state: state
  };
})(typeof window !== 'undefined' ? window : globalThis);
