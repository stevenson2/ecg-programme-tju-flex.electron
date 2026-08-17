/* ============================================================
 * ESP32-ECG Web Console · live.js
 * 实时监测：演示合成 / Web Serial / Web Bluetooth + Canvas 波形渲染
 * ============================================================ */
(function (global) {
  'use strict';

  var Core = global.ECGCore;
  if (!Core) throw new Error('ecg-core.js 必须先加载');

  var CHANNELS = [
    { key: 'clean', label: 'Clean', color: '#2dd4bf' },
    { key: 'noisy', label: 'Noisy', color: '#f59e0b' },
    { key: 'filtered', label: 'Filtered', color: '#34d399' }
  ];

  var BUFFER_SECONDS = 120;   /* 环形缓冲 120 秒 */
  var MAX_CSV_ROWS = 120000;  /* CSV 会话上限（100Hz 约 20 分钟） */

  var state = {
    source: 'demo',
    running: true,
    fs: 100,
    capacity: 0,
    count: 0,
    buffers: {},
    visible: { clean: true, noisy: true, filtered: true },
    gain: 1,
    windowSec: 5,
    lastRow: null,
    qrs: null,
    demo: null,
    demoLast: 0,
    demoAccum: 0,
    rafId: 0,
    serialPort: null,
    serialReader: null,
    serialDone: false,
    bleDevice: null,
    bleServer: null,
    bleTx: null,
    bleRx: null,
    bleFrameBuf: '',
    csvRecording: false,
    csvRows: [],
    initialized: false
  };

  var ui = {};

  /* ================= UI 初始化 ================= */

  function init() {
    if (state.initialized) return;
    state.initialized = true;

    ui.canvas = document.getElementById('liveCanvas');
    ui.liveStatusDot = document.getElementById('liveStatusDot');
    ui.liveSourceName = document.getElementById('liveSourceName');
    ui.liveSampleRate = document.getElementById('liveSampleRate');
    ui.chartHint = document.getElementById('chartHint');
    ui.alarmOverlay = document.getElementById('alarmOverlay');
    ui.vitalBpm = document.getElementById('vitalBpm');
    ui.vitalHrState = document.getElementById('vitalHrState');
    ui.vitalTrueBpm = document.getElementById('vitalTrueBpm');
    ui.heartIcon = document.getElementById('heartIcon');
    ui.vitalAi = document.getElementById('vitalAi');
    ui.vitalAiBadge = document.getElementById('vitalAiBadge');
    ui.vitalAiConf = document.getElementById('vitalAiConf');
    ui.vitalAiBar = document.getElementById('vitalAiBar');
    ui.vitalSqi = document.getElementById('vitalSqi');
    ui.vitalSqiBar = document.getElementById('vitalSqiBar');
    ui.vitalMotion = document.getElementById('vitalMotion');
    ui.vitalMotionBadge = document.getElementById('vitalMotionBadge');
    ui.eventLog = document.getElementById('eventLog');
    ui.csvStats = document.getElementById('csvStats');
    ui.navLiveDot = document.getElementById('navLiveDot');
    ui.globalStatusText = document.getElementById('globalStatusText');
    ui.globalStatus = document.getElementById('globalStatus');
    ui.sourceSeg = document.getElementById('sourceSeg');
    ui.gainRange = document.getElementById('gainRange');
    ui.gainOut = document.getElementById('gainOut');
    ui.windowSel = document.getElementById('windowSel');
    ui.channelChips = document.getElementById('channelChips');
    ui.serialPanel = document.getElementById('serialPanel');
    ui.blePanel = document.getElementById('blePanel');
    ui.serialBaud = document.getElementById('serialBaud');
    ui.btnSerialConnect = document.getElementById('btnSerialConnect');
    ui.btnBleConnect = document.getElementById('btnBleConnect');
    ui.btnBleRecStart = document.getElementById('btnBleRecStart');
    ui.btnBleRecStop = document.getElementById('btnBleRecStop');
    ui.btnBleRecStatus = document.getElementById('btnBleRecStatus');
    ui.btnLivePlay = document.getElementById('btnLivePlay');
    ui.btnSnapshot = document.getElementById('btnSnapshot');
    ui.btnCsvRecord = document.getElementById('btnCsvRecord');
    ui.recDot = document.getElementById('recDot');
    ui.btnCsvRecordText = document.getElementById('btnCsvRecordText');
    ui.btnClearLog = document.getElementById('btnClearLog');

    bindUi();
    setSource('demo', true);
    startDemo();
    requestAnimationFrame(loop);
  }

  function bindUi() {
    ui.sourceSeg.addEventListener('click', function (e) {
      var btn = e.target.closest('.source-seg__btn');
      if (!btn) return;
      setSource(btn.dataset.source);
    });

    ui.gainRange.addEventListener('input', function () {
      state.gain = parseFloat(ui.gainRange.value);
      ui.gainOut.textContent = state.gain.toFixed(2) + '×';
    });

    ui.windowSel.addEventListener('change', function () {
      state.windowSec = parseFloat(ui.windowSel.value);
    });

    ui.channelChips.addEventListener('click', function (e) {
      var chip = e.target.closest('.chip');
      if (!chip) return;
      var key = chip.dataset.chan;
      state.visible[key] = !state.visible[key];
      chip.classList.toggle('active', state.visible[key]);
    });

    ui.btnLivePlay.addEventListener('click', function () {
      state.running = !state.running;
      ui.btnLivePlay.textContent = state.running ? '暂停' : '继续';
      log(state.running ? '已继续刷新' : '显示已暂停（数据停止追加）', state.running ? 'ok' : 'warn');
    });

    ui.btnSnapshot.addEventListener('click', exportSnapshot);

    ui.btnCsvRecord.addEventListener('click', function () {
      if (!state.csvRecording) startCsvRecording();
      else stopCsvRecording();
    });

    ui.btnClearLog.addEventListener('click', function () {
      ui.eventLog.innerHTML = '<p class="muted">等待事件…</p>';
    });

    ui.btnSerialConnect.addEventListener('click', toggleSerial);
    ui.btnBleConnect.addEventListener('click', toggleBle);
    ui.btnBleRecStart.addEventListener('click', function () { sendBleCommand('REC_START'); });
    ui.btnBleRecStop.addEventListener('click', function () { sendBleCommand('REC_STOP'); });
    ui.btnBleRecStatus.addEventListener('click', function () { sendBleCommand('REC_STATUS'); });
  }

  /* ================= 数据源切换 ================= */

  function setSource(source, force) {
    if (!force && source === state.source) return;

    /* 停掉旧源 */
    if (state.source === 'demo') stopDemo();
    if (state.source === 'serial' && state.serialPort) disconnectSerial('已切换数据源');
    if (state.source === 'ble' && state.bleDevice) disconnectBle();

    state.source = source;
    state.fs = source === 'ble' ? 125 : 100;
    resetBuffers();
    setLastRow(null);
    updateGlobalStatus();

    ui.sourceSeg.querySelectorAll('.source-seg__btn').forEach(function (b) {
      b.classList.toggle('active', b.dataset.source === source);
    });
    ui.serialPanel.hidden = source !== 'serial';
    ui.blePanel.hidden = source !== 'ble';
    ui.liveSourceName.textContent = source === 'demo' ? '演示信号' : (source === 'serial' ? '串口设备' : 'BLE NUS 设备');
    ui.liveSampleRate.textContent = state.fs + ' Hz · 缓冲 0s';
    ui.chartHint.hidden = false;

    if (source === 'demo') {
      startDemo();
      log('演示模式已启动：浏览器内合成 PQRST 心电 @100Hz', 'ok');
    } else if (source === 'serial') {
      log('已选择串口模式，请点击「连接串口」并选择 ESP32 端口', 'warn');
    } else {
      log('已选择 BLE 模式，请点击「连接设备」（需 Chrome/Edge）', 'warn');
    }
  }

  function resetBuffers() {
    state.capacity = Math.floor(BUFFER_SECONDS * state.fs);
    state.count = 0;
    state.buffers = {};
    CHANNELS.forEach(function (c) {
      state.buffers[c.key] = new Float32Array(state.capacity);
    });
    state.qrs = new Core.QrsDetector(state.fs);
  }

  /* ================= 数据入列 ================= */

  function pushRow(row) {
    var cap = state.capacity;
    var idx = state.count % cap;
    CHANNELS.forEach(function (c) {
      state.buffers[c.key][idx] = row[c.key];
    });
    state.count++;

    setLastRow(row);
    var beat = state.qrs.update(row.filtered);
    if (beat && row.bpm > 0) pulseHeart();

    if (state.csvRecording) {
      if (state.csvRows.length >= MAX_CSV_ROWS) {
        stopCsvRecording(true);
      } else {
        state.csvRows.push(row);
        ui.csvStats.textContent = state.csvRows.length.toLocaleString() + ' 行';
      }
    }
  }

  function setLastRow(row) {
    state.lastRow = row;
    if (!row) {
      ui.vitalBpm.textContent = '--';
      ui.vitalAiConf.textContent = '--';
      ui.vitalSqi.textContent = '--';
      return;
    }
    var bpm = Math.round(row.bpm);
    ui.vitalBpm.textContent = bpm > 0 ? bpm : '--';
    ui.vitalHrState.textContent = row.motion ? '运动中（冻结）' : (bpm > 0 ? '实时检测' : '学习中');
    ui.vitalTrueBpm.textContent = row.trueBpm > 0 ? row.trueBpm + ' BPM' : '--';
    ui.vitalSqi.textContent = row.sqi.toFixed(2);
    ui.vitalSqiBar.style.width = Core.clamp(row.sqi * 100, 0, 100).toFixed(1) + '%';
    ui.vitalAiConf.textContent = row.confidence.toFixed(2);
    ui.vitalAiBar.style.width = Core.clamp(row.confidence * 100, 0, 100).toFixed(1) + '%';

    if (row.abnormal) {
      ui.vitalAi.classList.add('is-alarm');
      ui.vitalAiBadge.className = 'badge badge--alarm';
      ui.vitalAiBadge.textContent = '⚠ 异常';
      ui.alarmOverlay.hidden = false;
    } else {
      ui.vitalAi.classList.remove('is-alarm');
      ui.vitalAiBadge.className = 'badge badge--ok';
      ui.vitalAiBadge.textContent = '正常';
      ui.alarmOverlay.hidden = true;
    }

    if (row.motion) {
      ui.vitalMotion.classList.add('is-alarm');
      ui.vitalMotionBadge.className = 'badge badge--alarm';
      ui.vitalMotionBadge.textContent = '⚠ 运动中';
    } else {
      ui.vitalMotion.classList.remove('is-alarm');
      ui.vitalMotionBadge.className = 'badge badge--muted';
      ui.vitalMotionBadge.textContent = '无运动';
    }

    var dot = ui.navLiveDot;
    var glob = ui.globalStatus.querySelector('.status-dot');
    if (row.abnormal) {
      dot.classList.add('on');
      glob.className = 'status-dot alarm';
    } else {
      dot.classList.remove('on');
    }
  }

  function pulseHeart() {
    ui.heartIcon.style.animation = 'none';
    void ui.heartIcon.offsetWidth;
    var bpm = state.lastRow && state.lastRow.bpm > 0 ? state.lastRow.bpm : 75;
    ui.heartIcon.style.animation = 'heartbeat ' + Core.clamp(60 / bpm, 0.35, 1.6).toFixed(2) + 's ease-in-out infinite';
  }

  /* ================= 演示引擎 ================= */

  function startDemo() {
    stopDemo();
    state.demo = new Core.EcgSimulator(state.fs);
    state.demoLast = performance.now();
    state.demoAccum = 0;
    state.rafId = requestAnimationFrame(demoTick);
  }

  function stopDemo() {
    if (state.rafId) cancelAnimationFrame(state.rafId);
    state.rafId = 0;
    state.demo = null;
  }

  function demoTick(now) {
    if (state.source !== 'demo' || !state.demo) return;
    state.rafId = requestAnimationFrame(demoTick);
    if (!state.running) {
      state.demoLast = now;
      return;
    }
    var elapsed = (now - state.demoLast) / 1000;
    state.demoLast = now;
    state.demoAccum += elapsed * state.fs;
    var need = Math.floor(state.demoAccum);
    if (need > 60) need = 60; /* 标签页休眠恢复时防一次补太多 */
    state.demoAccum -= need;
    for (var i = 0; i < need; i++) pushRow(state.demo.next());
  }

  /* ================= Web Serial ================= */

  function serialSupported() {
    return !!(global.navigator && global.navigator.serial);
  }

  async function connectSerial() {
    if (!serialSupported()) {
      log('当前浏览器不支持 Web Serial（请使用 Chrome / Edge 桌面版）', 'error');
      toast('Web Serial 需要 Chrome / Edge 桌面版', 'warn');
      return;
    }
    try {
      var port = await navigator.serial.requestPort();
      var baud = parseInt(ui.serialBaud.value, 10) || 460800;
      await port.open({ baudRate: baud });
      state.serialPort = port;
      state.serialDone = false;
      ui.btnSerialConnect.textContent = '断开串口';
      ui.btnSerialConnect.classList.remove('btn--primary');
      ui.btnSerialConnect.classList.add('btn--danger');
      log('串口已连接 @' + baud + '，等待 9 列 CSV 数据…', 'ok');
      updateGlobalStatus();
      readSerialLoop(port);
    } catch (err) {
      if (err && err.name !== 'NotFoundError') {
        log('串口连接失败：' + (err.message || err), 'error');
        toast('串口连接失败：' + (err.message || err), 'error');
      }
    }
  }

  async function readSerialLoop(port) {
    var decoder = new TextDecoder();
    var tail = '';
    while (port.readable && !state.serialDone) {
      var reader;
      try {
        reader = port.readable.getReader();
        state.serialReader = reader;
      } catch (e) {
        break;
      }
      try {
        while (true) {
          var r = await reader.read();
          if (r.done) break;
          var text = tail + decoder.decode(r.value, { stream: true });
          tail = '';
          var lines = text.split(/\r?\n/);
          if (lines.length > 1) {
            tail = lines.pop();
            feedLines(lines);
          } else {
            tail = text;
          }
        }
      } catch (err) {
        if (!state.serialDone) {
          log('串口读取中断：' + (err.message || err), 'error');
        }
        break;
      } finally {
        try { reader.releaseLock(); } catch (e) { /* noop */ }
      }
    }
    if (state.serialPort === port && !state.serialDone) {
      afterSerialClosed(port);
    }
  }

  function feedLines(lines) {
    var accepted = 0;
    for (var i = 0; i < lines.length; i++) {
      var row = Core.parseCsvLine(lines[i]);
      if (!row) continue;
      if (state.source !== 'serial' && state.source !== 'ble') continue;
      if (state.running) pushRow(row);
      accepted++;
    }
    if (accepted === 0 && lines.length > 1) return;
  }

  async function disconnectSerial(reason) {
    state.serialDone = true;
    var port = state.serialPort;
    state.serialPort = null;
    if (state.serialReader) {
      try { await state.serialReader.cancel(); } catch (e) { /* noop */ }
      state.serialReader = null;
    }
    if (port) {
      try { await port.close(); } catch (e) { /* noop */ }
    }
    afterSerialClosed(port);
    if (reason) log(reason, 'warn');
  }

  function afterSerialClosed() {
    ui.btnSerialConnect.textContent = '连接串口';
    ui.btnSerialConnect.classList.add('btn--primary');
    ui.btnSerialConnect.classList.remove('btn--danger');
    ui.liveSampleRate.textContent = state.fs + ' Hz · 缓冲 ' + bufferSeconds().toFixed(0) + 's';
    updateGlobalStatus();
  }

  async function toggleSerial() {
    if (state.serialPort) {
      await disconnectSerial('串口已断开');
    } else {
      await connectSerial();
    }
  }

  /* ================= Web Bluetooth (NUS) ================= */

  function bleSupported() {
    return !!(global.navigator && navigator.bluetooth);
  }

  async function connectBle() {
    if (!bleSupported()) {
      log('当前浏览器不支持 Web Bluetooth（请使用 Chrome / Edge）', 'error');
      toast('Web Bluetooth 需要 Chrome / Edge', 'warn');
      return;
    }
    try {
      var device = await navigator.bluetooth.requestDevice({
        filters: [{ services: [Core.NUS_SERVICE_UUID] }]
      });
      await openBleDevice(device);
    } catch (err) {
      if (err && err.name !== 'NotFoundError') {
        log('BLE 连接失败：' + (err.message || err), 'error');
        toast('BLE 连接失败：' + (err.message || err), 'error');
      }
    }
  }

  async function openBleDevice(device) {
    var server = await device.gatt.connect();
    var service = await server.getPrimaryService(Core.NUS_SERVICE_UUID);
    var tx = await service.getCharacteristic(Core.NUS_TX_UUID);
    var rx = await service.getCharacteristic(Core.NUS_RX_UUID);

    state.bleDevice = device;
    state.bleServer = server;
    state.bleTx = tx;
    state.bleRx = rx;
    state.bleFrameBuf = '';

    device.addEventListener('gattserverdisconnected', onBleDisconnected);
    tx.addEventListener('characteristicvaluechanged', onBleValue);
    await tx.startNotifications();

    ui.btnBleConnect.textContent = '断开设备';
    ui.btnBleConnect.classList.remove('btn--primary');
    ui.btnBleConnect.classList.add('btn--danger');
    setBleButtonsEnabled(true);
    log('BLE 已连接：' + (device.name || 'ESP32-ECG') + '，接收 NUS TX @125Hz…', 'ok');
    updateGlobalStatus();
  }

  function onBleValue(event) {
    var value = event.target.value;
    if (!value) return;
    var text = '';
    for (var i = 0; i < value.byteLength; i++) text += String.fromCharCode(value.getUint8(i));
    state.bleFrameBuf += text;

    /* 固件每帧以 ';' 结尾，一个 notify 可能含多帧或半帧 */
    var parts = state.bleFrameBuf.split(/[;\r\n]+/);
    state.bleFrameBuf = parts.pop() || '';
    feedLines(parts);
  }

  async function sendBleCommand(cmd) {
    if (!state.bleRx) return;
    var data = new TextEncoder().encode(cmd + '\n');
    try {
      if (state.bleRx.properties.writeWithoutResponse) {
        await state.bleRx.writeValueWithoutResponse(data);
      } else {
        await state.bleRx.writeValue(data);
      }
      log('已发送 BLE 命令：' + cmd, 'ok');
    } catch (err) {
      log('BLE 命令发送失败：' + (err.message || err), 'error');
    }
  }

  async function disconnectBle() {
    var device = state.bleDevice;
    state.bleDevice = null;
    state.bleServer = null;
    state.bleTx = null;
    state.bleRx = null;
    setBleButtonsEnabled(false);
    ui.btnBleConnect.textContent = '连接设备';
    ui.btnBleConnect.classList.add('btn--primary');
    ui.btnBleConnect.classList.remove('btn--danger');
    if (device && device.gatt && device.gatt.connected) {
      try { await device.gatt.disconnect(); } catch (e) { /* noop */ }
    }
    log('BLE 已断开', 'warn');
    updateGlobalStatus();
  }

  function onBleDisconnected() {
    log('BLE 连接被设备断开', 'warn');
    disconnectBle();
  }

  async function toggleBle() {
    if (state.bleDevice) await disconnectBle();
    else await connectBle();
  }

  function setBleButtonsEnabled(on) {
    ui.btnBleRecStart.disabled = !on;
    ui.btnBleRecStop.disabled = !on;
    ui.btnBleRecStatus.disabled = !on;
  }

  /* ================= 状态显示 ================= */

  function bufferSeconds() {
    return Math.min(state.count, state.capacity) / state.fs;
  }

  function updateGlobalStatus() {
    var dot = ui.globalStatus.querySelector('.status-dot');
    var connected = state.source === 'serial' ? !!state.serialPort :
      (state.source === 'ble' ? !!state.bleDevice : false);
    if (connected) {
      dot.className = 'status-dot on';
      ui.globalStatusText.textContent = state.source === 'serial' ? '串口已连接' : 'BLE 已连接';
    } else if (state.source === 'demo') {
      dot.className = 'status-dot on';
      ui.globalStatusText.textContent = '演示模式';
    } else {
      dot.className = 'status-dot';
      ui.globalStatusText.textContent = state.source === 'serial' ? '串口未连接' : 'BLE 未连接';
    }
    ui.liveSampleRate.textContent = state.fs + ' Hz · 缓冲 ' + bufferSeconds().toFixed(1) + 's';
  }

  /* ================= 主渲染循环 ================= */

  function loop() {
    requestAnimationFrame(loop);
    draw();
    if (state.count > 0) ui.chartHint.hidden = true;
    if (state.count % Math.floor(state.fs / 2) === 0) {
      ui.liveSampleRate.textContent = state.fs + ' Hz · 缓冲 ' + bufferSeconds().toFixed(1) + 's';
    }
  }

  function draw() {
    var canvas = ui.canvas;
    if (!canvas || !canvas.isConnected) return;
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

    if (state.count < 2) return;

    var fs = state.fs;
    var windowSec = state.windowSec;
    var visCount = CHANNELS.filter(function (c) { return state.visible[c.key]; }).length;
    if (visCount === 0) return;

    var n = Math.min(state.count, Math.round(windowSec * fs));
    var start = state.count - n;
    var padX = 54;
    var padTop = 16;
    var padBottom = 22;
    var laneH = (ch - padTop - padBottom) / visCount;

    /* 网格 */
    ctx.strokeStyle = 'rgba(148,178,224,0.10)';
    ctx.lineWidth = 1;
    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    ctx.fillStyle = 'rgba(148,163,184,0.75)';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    var gridStep = windowSec <= 5 ? 0.5 : (windowSec <= 10 ? 1 : (windowSec <= 20 ? 2 : 5));
    for (var tSec = 0; tSec <= windowSec + 1e-9; tSec += gridStep) {
      var x = padX + (cw - padX - 12) * (tSec / windowSec);
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, ch - padBottom);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText((-windowSec + tSec).toFixed(0) + 's', x, ch - padBottom + 12);
      ctx.textAlign = 'right';
    }

    /* 通道曲线 */
    var lane = 0;
    for (var ci = 0; ci < CHANNELS.length; ci++) {
      var meta = CHANNELS[ci];
      if (!state.visible[meta.key]) continue;

      var buf = state.buffers[meta.key];
      var top = padTop + lane * laneH;
      var innerTop = top + 14;
      var innerBottom = top + laneH - 8;

      var mn = Infinity, mx = -Infinity;
      for (var i = start; i < state.count; i++) {
        var v = buf[i % state.capacity];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      if (!isFinite(mn) || !isFinite(mx) || mx - mn < 1e-9) {
        mn = -0.5; mx = 0.5;
      }
      var span = (mx - mn) * 0.16;
      mn -= span;
      mx += span;

      /* 车道水平网格 */
      ctx.strokeStyle = 'rgba(148,178,224,0.07)';
      for (var g = 0; g <= 2; g++) {
        var gy = innerTop + (innerBottom - innerTop) * g / 2;
        ctx.beginPath();
        ctx.moveTo(padX, gy);
        ctx.lineTo(cw - 12, gy);
        ctx.stroke();
      }

      ctx.fillStyle = meta.color;
      ctx.textAlign = 'right';
      ctx.fillText(fmtVolt(mx), padX - 8, innerTop + 3);
      ctx.fillText(fmtVolt(mn), padX - 8, innerBottom - 3);

      /* 波形 */
      var yOf = function (val) {
        var k = (val - mn) / (mx - mn);
        return innerBottom - k * (innerBottom - innerTop);
      };
      ctx.beginPath();
      var xOf = function (idx) {
        return padX + (cw - padX - 12) * ((idx - start) / (n - 1));
      };
      for (var j = start; j < state.count; j++) {
        var px = xOf(j);
        var py = yOf(buf[j % state.capacity]);
        if (j === start) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = meta.color;
      ctx.lineWidth = 1.7;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.shadowColor = meta.color;
      ctx.shadowBlur = 9;
      ctx.globalAlpha = 0.95;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;

      /* 车道名 */
      ctx.fillStyle = meta.color;
      ctx.font = 'bold 10px ' + getComputedStyle(document.body).fontFamily;
      ctx.textAlign = 'left';
      ctx.fillText(meta.label, cw - 14 - ctx.measureText(meta.label).width, innerTop + 3);

      lane++;
    }
  }

  function fmtVolt(v) {
    var a = Math.abs(v);
    if (a > 0 && a < 0.01) return (v * 1000).toFixed(1) + ' mV';
    return v.toFixed(2) + ' V';
  }

  /* ================= 截图 / CSV ================= */

  function exportSnapshot() {
    var canvas = ui.canvas;
    try {
      canvas.toBlob(function (blob) {
        if (!blob) return;
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'esp32-ecg_' + Date.now() + '.png';
        a.click();
        setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
        toast('截图已导出', 'ok');
      }, 'image/png');
    } catch (err) {
      toast('截图导出失败：' + err.message, 'error');
    }
  }

  function startCsvRecording() {
    state.csvRecording = true;
    state.csvRows = [];
    ui.csvStats.textContent = '0 行';
    ui.recDot.classList.add('on');
    ui.btnCsvRecordText.textContent = '停止并导出 CSV';
    log('CSV 会话开始记录（' + state.fs + ' Hz 源数据）', 'ok');
  }

  function stopCsvRecording(autoLimit) {
    state.csvRecording = false;
    ui.recDot.classList.remove('on');
    ui.btnCsvRecordText.textContent = '开始记录 CSV';
    var rows = state.csvRows;
    var text = Core.CSV_COLUMNS.join(',') + '\n';
    for (var i = 0; i < rows.length; i++) text += Core.rowToCsv(rows[i]) + '\n';
    var blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'esp32-ecg_session_' + Date.now() + '.csv';
    a.click();
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
    ui.csvStats.textContent = rows.length.toLocaleString() + ' 行（已导出）';
    log((autoLimit ? '达到上限，' : '') + 'CSV 已导出：' + rows.length.toLocaleString() + ' 行', 'ok');
    toast('CSV 已导出：' + rows.length.toLocaleString() + ' 行', 'ok');
  }

  /* ================= 日志 / Toast ================= */

  function ts() {
    var d = new Date();
    function p(n) { return n < 10 ? '0' + n : '' + n; }
    return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
  }

  function log(msg, cls) {
    var el = ui.eventLog;
    if (el.querySelector('.muted') && el.childElementCount === 1 && el.firstElementChild.textContent === '等待事件…') {
      el.innerHTML = '';
    }
    var line = document.createElement('p');
    line.className = 'log-line ' + (cls || '');
    line.innerHTML = '<span class="t">' + ts() + '</span>' + escapeHtml(msg);
    el.appendChild(line);
    while (el.childElementCount > 300) el.removeChild(el.firstChild);
    el.scrollTop = el.scrollHeight;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function toast(msg, cls) {
    if (global.ECGUI && global.ECGUI.toast) global.ECGUI.toast(msg, cls);
    else console.log('[toast]', msg);
  }

  /* 页面隐藏时，演示引擎回帧自然被 cap；切回继续 */

  global.ECGLive = {
    init: init,
    setSource: setSource,
    log: log,
    toast: toast,
    _state: state
  };
})(typeof window !== 'undefined' ? window : globalThis);
