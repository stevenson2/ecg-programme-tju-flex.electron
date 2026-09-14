/* eslint-disable no-var */
/* ============================================================
 * ESP32-ECG Web Console · ecg-core.js
 * 纯逻辑模块（不依赖 DOM），可在 Node 中单独测试。
 *
 * 内容：
 *  - 串口/BLE 9 列 CSV 解析
 *  - .ecgr 录制文件解析（32 字节头 + int16 LE 样本 + 异常位图）
 *  - 浏览器内合成 ECG 信号发生器（演示模式）
 *  - 轻量 QRS 检测器（心跳动画 / 节拍标记）
 *  - 概览图 min/max 抽取与格式化工具
 * ============================================================ */
(function (global) {
  'use strict';

  /* ---------------- 常量 ---------------- */

  /* 协议契约（P0-1）：由 scripts/gen_protocol_constants.py 从 protocol/ecg_proto.json 生成。 */
  /* 契约常量来源：浏览器由 index.html 先加载 protocol_generated.js；
   * Node 测试由 CommonJS require 加载。 */
  var PROTO = (typeof ECGProtocol !== 'undefined') ? ECGProtocol : require('./protocol_generated.js');
  var CSV_COLUMNS = ['clean', 'noisy', 'filtered', 'bpm', 'trueBpm', 'sqi', 'motion', 'abnormal', 'confidence', 'asrc'];
  /* v2 追加第 10 列 asrc；旧 9 列帧仍可解析（asrc 缺省 0）。 */
  var CSV_LINE_RE_V2 = /^\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*;?\s*$/;
  var CSV_LINE_RE = /^\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*;?\s*$/;

  var NUS_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
  var NUS_TX_UUID = '6e400002-b5a3-f393-e0a9-e50e24dcca9e';
  var NUS_RX_UUID = '6e400003-b5a3-f393-e0a9-e50e24dcca9e';

  var ECGR_MAGIC = 'ECGR';
  var ECGR_VERSION = PROTO.ECGR_VERSION_CURRENT;
  var ECGR_VERSION_1 = 1;
  var ECGR_VERSION_2 = 2;
  var ECGR_HEADER_SIZE = 32;
  var ECGR_FLAG_HAS_ABNORMAL_BITMAP = 0x01;
  var ECGR_SCALE_TO_VOLTS = PROTO.ECGR_SCALE_TO_VOLTS;
  var ECGR_RESERVED0_ABNORMAL_IS_ASRC = PROTO.ECGR_RESERVED0_ABNORMAL_IS_ASRC;

  /* ---------------- 工具 ---------------- */

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function pad2(n) { return n < 10 ? '0' + n : '' + n; }

  function formatDuration(totalSeconds) {
    var s = Math.max(0, Math.round(totalSeconds));
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var sec = s % 60;
    if (h > 0) return h + ':' + pad2(m) + ':' + pad2(sec);
    return m + ':' + pad2(sec);
  }

  function formatBytes(bytes) {
    if (!isFinite(bytes) || bytes <= 0) return '0 B';
    var units = ['B', 'KB', 'MB', 'GB'];
    var i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
    var v = bytes / Math.pow(1024, i);
    return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + ' ' + units[i];
  }

  function formatClock(unixSec) {
    var d = new Date(unixSec * 1000);
    return d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()) +
      ' ' + pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
  }

  /* ---------------- 9 列 CSV 解析 ---------------- */

  /**
   * 解析一行 9 列 CSV（串口 / BLE 帧，可带末尾 ';'）。
   * 失败返回 null。诊断日志行（"[心率] ..." 等）自然被拒。
   */
  function parseCsvLine(line) {
    if (typeof line !== 'string') return null;
    /* v2 优先（10 列，含 asrc）；不匹配退回 v1 的 9 列行为。 */
    var m2 = CSV_LINE_RE_V2.exec(line);
    if (m2) {
      return {
        clean: parseFloat(m2[1]),
        noisy: parseFloat(m2[2]),
        filtered: parseFloat(m2[3]),
        bpm: parseFloat(m2[4]),
        trueBpm: parseFloat(m2[5]),
        sqi: parseFloat(m2[6]),
        motion: parseInt(m2[7], 10) === 1,
        abnormal: parseInt(m2[8], 10) !== 0,
        confidence: parseFloat(m2[9]),
        asrc: parseInt(m2[10], 10) || 0,
        protoVer: PROTO.PROTO_VERSION
      };
    }
    var m = CSV_LINE_RE.exec(line);
    if (!m) return null;
    return {
      clean: parseFloat(m[1]),
      noisy: parseFloat(m[2]),
      filtered: parseFloat(m[3]),
      bpm: parseFloat(m[4]),
      trueBpm: parseFloat(m[5]),
      sqi: parseFloat(m[6]),
      motion: parseInt(m[7], 10) === 1,
      abnormal: parseInt(m[8], 10) !== 0,
      confidence: parseFloat(m[9]),
      asrc: parseInt(m[8], 10) !== 0 ? 1 : 0,
      protoVer: 1
    };
  }

  function rowToCsv(row) {
    var asrc = (row.asrc === undefined || row.asrc === null) ? (row.abnormal ? 1 : 0) : row.asrc;
    return [
      row.clean.toFixed(4),
      row.noisy.toFixed(4),
      row.filtered.toFixed(4),
      Math.round(row.bpm),
      Math.round(row.trueBpm),
      row.sqi.toFixed(3),
      row.motion ? 1 : 0,
      row.abnormal ? 1 : 0,
      row.confidence.toFixed(3),
      (asrc >>> 0)
    ].join(',');
  }

  /* ---------------- .ecgr 解析 ---------------- */

  function readU32LE(buf, off) {
    return (buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16) | (buf[off + 3] << 24)) >>> 0;
  }

  /**
   * 解析 ECGR 二进制（ArrayBuffer / Uint8Array）。
   * 成功返回记录对象，失败抛出 Error（含中文提示）。
   */
  function parseEcgr(input) {
    var bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
    if (!bytes || bytes.length < ECGR_HEADER_SIZE) {
      throw new Error('文件太小，不是有效的 .ecgr 录制（至少需要 32 字节头部）');
    }
    if (String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== ECGR_MAGIC) {
      throw new Error('魔数不匹配：不是 ESP32-ECG 的 .ecgr 录制文件');
    }
    var version = bytes[4];
    if (version !== ECGR_VERSION_1 && version !== ECGR_VERSION_2) {
      throw new Error('不支持的 ECGR 版本：' + version);
    }

    var flags = bytes[5];
    var reserved0 = bytes[26];
    var bitmapIsAsrc = version === ECGR_VERSION_2 &&
        (reserved0 & ECGR_RESERVED0_ABNORMAL_IS_ASRC) !== 0;
    var sampleRate = readU32LE(bytes, 6);
    var startUnix = readU32LE(bytes, 10);
    var durationSec = readU32LE(bytes, 14);
    var headerSamples = readU32LE(bytes, 18);
    var abnormalSec = readU32LE(bytes, 22);

    if (sampleRate < 50 || sampleRate > 4000) {
      throw new Error('采样率字段异常：' + sampleRate + ' Hz');
    }

    var hasBitmap = (flags & ECGR_FLAG_HAS_ABNORMAL_BITMAP) !== 0;
    var payload = bytes.length - ECGR_HEADER_SIZE;
    /* P0-3 统一截断容忍（正向解码）：
     * - 样本流按头部 totalSamples 读取，文件不足则读到字节上限；
     * - 位图从样本流之后读取，文件不足则尾部按 0 补齐；
     * - 任一处不足即 truncated=true，绝不抛错拒收。 */
    var availableSampleBytes = payload;
    var totalSamples;
    if (headerSamples > 0) {
      totalSamples = Math.floor(Math.min(headerSamples * 2, availableSampleBytes) / 2);
    } else if (hasBitmap) {
      totalSamples = Math.floor(Math.max(0, payload - durationSec) / 2);
    } else {
      totalSamples = Math.floor(payload / 2);
    }
    var truncated = headerSamples > 0 && totalSamples < headerSamples;
    if (headerSamples === 0 && hasBitmap && payload < durationSec + totalSamples * 2) {
      truncated = true; /* 头部未声明样本数时只能估算，标记不完整 */
    }

    var samples = new Int16Array(bytes.buffer, bytes.byteOffset + ECGR_HEADER_SIZE, totalSamples);

    var bitmap = null;
    if (hasBitmap) {
      var bmpStart = ECGR_HEADER_SIZE + totalSamples * 2;
      var got = Math.min(durationSec, Math.max(0, bytes.length - bmpStart));
      bitmap = new Uint8Array(durationSec);
      for (var bi = 0; bi < got; bi++) bitmap[bi] = bytes[bmpStart + bi];
      if (got < durationSec) truncated = true;
    }

    return {
      bytes: bytes,
      fileName: null,
      version: version,
      flags: flags,
      bitmapIsAsrc: bitmapIsAsrc,
      sampleRate: sampleRate,
      startUnix: startUnix,
      durationSec: durationSec,
      totalSamples: totalSamples,
      abnormalSec: abnormalSec,
      hasBitmap: hasBitmap,
      bitmap: bitmap,
      samples: samples,
      truncated: truncated,
      /* 样本 i → 电压 (V)：固件写入时 cleanSample(V) * 8000 */
      voltAt: function (i) { return samples[i] / ECGR_SCALE_TO_VOLTS; }
    };
  }

  /* ---------------- 轻量 QRS 检测器 ---------------- */

  /**
   * 差分 → 平方 → 滑动窗积分 → 自适应阈值的简化 QRS 检测。
   * 用于网页端心跳动画与节拍标记，不作为临床算法。
   */
  function QrsDetector(sampleRate) {
    this.fs = sampleRate || 250;
    this.win = Math.max(4, Math.round(this.fs * 0.04));       /* 40ms 积分窗 */
    this.refractory = Math.round(this.fs * 0.28);             /* 280ms 不应期 */
    this.buf = new Float32Array(6);                           /* 差分用 */
    this.ring = new Float32Array(this.win);
    this.ringPos = 0;
    this.ringSum = 0;
    this.ringFilled = 0;
    this.lastBeat = -1e9;
    this.spki = 0;      /* 信号峰值估计 */
    this.npki = 1;      /* 噪声峰值估计 */
    this.idx = 0;
  }

  QrsDetector.prototype.update = function (v) {
    var i = this.idx;
    var prev = this.buf[0];
    this.buf[0] = v;
    var diff = v - prev;
    var energy = diff * diff;

    if (this.ringFilled < this.win) this.ringFilled++;
    this.ringSum -= this.ring[this.ringPos];
    this.ring[this.ringPos] = energy;
    this.ringSum += energy;
    this.ringPos = (this.ringPos + 1) % this.win;
    this.idx++;

    var integ = this.ringSum / this.win;
    var threshold = this.npki + 0.22 * (this.spki - this.npki);

    if (integ > threshold && (i - this.lastBeat) > this.refractory) {
      this.lastBeat = i;
      this.spki = 0.875 * this.spki + 0.125 * integ;
      return true;
    }
    if (integ < this.npki) {
      this.npki = 0.94 * this.npki + 0.06 * integ;
    } else if (integ < threshold) {
      this.npki = 0.97 * this.npki + 0.03 * integ;
    }
    return false;
  };

  /* ---------------- 合成 ECG 信号发生器 ---------------- */

  function gauss(x, mu, sigma) {
    var d = (x - mu) / sigma;
    return Math.exp(-0.5 * d * d);
  }

  /** 心拍形态（相对 R 峰的时间，单位秒）：P-Q-R-S-T 高斯混合 */
  function ecgMorphology(tRel) {
    return 0.12 * gauss(tRel, -0.22, 0.042)      /* P 波 */
      - 0.16 * gauss(tRel, -0.032, 0.010)       /* Q 波 */
      + 1.18 * gauss(tRel, 0.0, 0.011)          /* R 波 */
      - 0.24 * gauss(tRel, 0.040, 0.013)        /* S 波 */
      + 0.32 * gauss(tRel, 0.270, 0.075);       /* T 波 */
  }

  /**
   * @param {number} [sampleRate=100] 输出采样率（串口 100 / BLE 125）
   * @param {number} [baseBpm] 起始心率，缺省随机 62~88
   */
  function EcgSimulator(sampleRate, baseBpm) {
    this.fs = sampleRate || 100;
    this.dt = 1 / this.fs;
    this.t = 0;
    this.baseBpm = baseBpm || Math.round(62 + Math.random() * 26);
    this.trueBpm = this.baseBpm;
    this.bpm = 0;
    this.beatCount = 0;
    this.smoothBpm = 0;
    this.phaseInBeat = 0;
    this.rr = 60 / this.trueBpm;
    this.motionUntil = -1;
    this.nextMotion = 18 + Math.random() * 8;
    this.abnormalUntil = -1;
    this.nextAbnormal = 70 + Math.random() * 40;
    this.abnormalConf = 0;
    this.sqi = 0.9;
    this.confidence = 0.12;
    this.noiseState = 0;
    this.filteredPrev = 0;
    this.baselineState = 0;
  }

  EcgSimulator.prototype._randn = function () {
    /* Box-Muller（缓存一对） */
    if (this._spare !== undefined) {
      var s = this._spare;
      this._spare = undefined;
      return s;
    }
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    var mag = Math.sqrt(-2.0 * Math.log(u));
    this._spare = mag * Math.sin(2.0 * Math.PI * v);
    return mag * Math.cos(2.0 * Math.PI * v);
  };

  EcgSimulator.prototype.next = function () {
    var t = this.t;
    this.t += this.dt;

    /* 心率漂移：30s 周期 ±4 BPM，RR 逐拍 ±3% 抖动 */
    this.trueBpm = clamp(this.baseBpm + 4 * Math.sin(t * 2 * Math.PI / 30), 42, 130);
    this.rr = (60 / this.trueBpm) * (1 + 0.03 * Math.sin(t * 0.7 + this.baseBpm));

    this.phaseInBeat += this.dt;
    if (this.phaseInBeat >= this.rr) {
      this.phaseInBeat -= this.rr;
      this.beatCount++;
      var inst = 60 / this.rr;
      this.smoothBpm = this.smoothBpm === 0 ? inst : 0.82 * this.smoothBpm + 0.18 * inst;
      this.bpm = Math.round(clamp(this.smoothBpm, 30, 220));
    }

    /* 运动片段：6~9s，噪声放大 + SQI 降低 + 心率冻结 */
    var inMotion = t < this.motionUntil;
    if (!inMotion && t >= this.nextMotion) {
      this.motionUntil = t + 6 + Math.random() * 3;
      this.nextMotion = t + 22 + Math.random() * 10;
      inMotion = true;
    }

    /* 演示性 AI 异常片段：锁存 5s */
    var abnormal = t < this.abnormalUntil;
    if (!abnormal && t >= this.nextAbnormal) {
      this.abnormalUntil = t + 5;
      this.nextAbnormal = t + 75 + Math.random() * 40;
      abnormal = true;
      this.abnormalConf = 0.78 + Math.random() * 0.18;
    }

    var tRel = this.phaseInBeat - 0.0;
    var clean = ecgMorphology(tRel - 0.0);
    /* P 波位于心拍前段，重采样一份 tRel-1.0 以补上前一拍末尾的 P/T */
    clean += 0.10 * ecgMorphology(tRel - this.rr);

    var baseline = 0.05 * Math.sin(2 * Math.PI * 0.28 * t) + 0.02 * Math.sin(2 * Math.PI * 0.09 * t);
    clean += baseline + 0.004 * this._randn();

    var powerline = 0.06 * Math.sin(2 * Math.PI * 50 * t + 0.7);
    this.noiseState = 0.92 * this.noiseState + 0.08 * this._randn();
    var muscle = inMotion ? 0.22 * this.noiseState * 3 : 0.05 * this.noiseState;
    var noisy = clean + powerline + muscle;

    /* “filtered”：去基线 + 一阶低通（近似固件显示链，演示用途） */
    var fb = clean - baseline;
    this.filteredPrev += (fb - this.filteredPrev) * 0.12;
    var filtered = this.filteredPrev;

    var sqi = clamp(inMotion ? 0.30 + 0.2 * Math.random() : 0.86 + 0.1 * Math.random(), 0, 1);
    var confidence = abnormal ? this.abnormalConf : clamp(0.05 + 0.08 * Math.abs(this.noiseState) + (inMotion ? 0.12 : 0), 0.02, 0.35);

    return {
      clean: clean,
      noisy: noisy,
      filtered: filtered,
      bpm: inMotion ? this.bpm : this.bpm,
      trueBpm: Math.round(this.trueBpm),
      sqi: sqi,
      motion: inMotion,
      abnormal: abnormal,
      confidence: confidence,
      t: t
    };
  };

  /* ---------------- 概览 min/max 抽取 ---------------- */

  /**
   * 把长数组按列宽抽取 min/max，供概览图绘制。
   * @returns {{mins:Float32Array, maxs:Float32Array, columns:number}}
   */
  function decimateMinMax(data, columns) {
    columns = Math.max(1, Math.floor(columns) || 1);
    var n = data.length;
    if (n === 0) return { mins: new Float32Array(0), maxs: new Float32Array(0), columns: 0 };
    var outN = Math.min(columns, n);
    var mins = new Float32Array(outN);
    var maxs = new Float32Array(outN);
    for (var c = 0; c < outN; c++) {
      var start = Math.floor(c * n / outN);
      var end = Math.max(start + 1, Math.floor((c + 1) * n / outN));
      var mn = Infinity, mx = -Infinity;
      for (var i = start; i < end; i++) {
        var v = data[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      mins[c] = mn === Infinity ? 0 : mn;
      maxs[c] = mx === -Infinity ? 0 : mx;
    }
    return { mins: mins, maxs: maxs, columns: outN };
  }

  /* ---------------- 导出 ---------------- */

  var api = {
    CSV_COLUMNS: CSV_COLUMNS,
    NUS_SERVICE_UUID: NUS_SERVICE_UUID,
    NUS_TX_UUID: NUS_TX_UUID,
    NUS_RX_UUID: NUS_RX_UUID,
    ECGR_SCALE_TO_VOLTS: ECGR_SCALE_TO_VOLTS,
    clamp: clamp,
    formatDuration: formatDuration,
    formatBytes: formatBytes,
    formatClock: formatClock,
    parseCsvLine: parseCsvLine,
    rowToCsv: rowToCsv,
    parseEcgr: parseEcgr,
    primaryAsrc: function (asrc) { return PROTO.primaryAsrc(asrc || 0); },
    ASRC: PROTO.ASRC,
    QrsDetector: QrsDetector,
    EcgSimulator: EcgSimulator,
    decimateMinMax: decimateMinMax
  };

  global.ECGCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
