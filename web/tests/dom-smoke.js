/* DOM 冒烟测试：jsdom 加载完整页面并触发模块初始化
 * 用法: node tests/dom-smoke.js <web目录绝对路径> <jsdom模块目录>
 */
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

const webDir = process.argv[2];
const jsdomDir = process.argv[3];
const { JSDOM } = require(path.join(jsdomDir, 'node_modules', 'jsdom'));

const fileUrl = p => pathToFileURL(p).href;
const html = fs.readFileSync(path.join(webDir, 'index.html'), 'utf8');
/* 把相对脚本/样式路径替换为绝对 file:// URL，便于 jsdom 加载 */
const htmlAbs = html
  .replace(/css\/styles\.css/g, fileUrl(path.join(webDir, 'css', 'styles.css')))
  .replace(/js\/ecg-core\.js/g, fileUrl(path.join(webDir, 'js', 'ecg-core.js')))
  .replace(/js\/live\.js/g, fileUrl(path.join(webDir, 'js', 'live.js')))
  .replace(/js\/records\.js/g, fileUrl(path.join(webDir, 'js', 'records.js')))
  .replace(/js\/playback\.js/g, fileUrl(path.join(webDir, 'js', 'playback.js')))
  .replace(/js\/app\.js/g, fileUrl(path.join(webDir, 'js', 'app.js')));

const dom = new JSDOM(htmlAbs, {
  url: fileUrl(path.join(webDir, 'index.html')),
  runScripts: 'dangerously',
  resources: 'usable',
  pretendToBeVisual: true,
  beforeParse(window) {
    /* Canvas 2D stub（jsdom 无实现） */
    const noop = () => {};
    const ctxStub = new Proxy({}, {
      get(target, prop) {
        if (prop === 'measureText') return () => ({ width: 0 });
        if (typeof prop === 'string' && !(prop in target)) target[prop] = noop;
        return target[prop];
      },
      set(target, prop, value) { target[prop] = value; return true; }
    });
    window.HTMLCanvasElement.prototype.getContext = function () { return ctxStub; };
    window.requestAnimationFrame = () => 0;
    window.cancelAnimationFrame = noop;
    window.URL.createObjectURL = () => 'blob:smoke';
    window.URL.revokeObjectURL = noop;
    window.scrollTo = noop;
  }
});

const assert = require('assert');
const { window } = dom;
const { document } = window;

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

(async () => {
  await new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('DOMContentLoaded 超时')), 8000);
    window.addEventListener('DOMContentLoaded', () => { clearTimeout(t); resolve(); });
    if (document.readyState === 'complete') { clearTimeout(t); resolve(); }
  });

  assert.ok(window.ECGCore, 'ECGCore 应已加载');
  assert.ok(window.ECGLive, 'ECGLive 应已加载');
  assert.ok(window.ECGRecords, 'ECGRecords 应已加载');
  assert.ok(window.ECGPlayback, 'ECGPlayback 应已加载');
  assert.ok(window.ECGUI, 'ECGUI 应已加载');

  /* 视图切换 */
  window.ECGUI.go('live');
  assert.ok(document.getElementById('view-live').classList.contains('active'), '应切到实时监测');
  window.ECGUI.go('records');
  window.ECGUI.go('playback');
  assert.ok(document.getElementById('view-playback').classList.contains('active'), '应切到本地回放');
  window.ECGUI.go('overview');

  /* 事件日志接口 */
  window.ECGLive.log('dom smoke test', 'ok');
  assert.ok(document.getElementById('eventLog').textContent.includes('dom smoke test'));

  /* 生命体征面板在无数据时应安全渲染 */
  assert.ok(document.getElementById('vitalBpm').textContent === '--', '初始 BPM 应显示 --');

  /* 回放模块：构造合法 ECGR 并载入 */
  const total = 500, fs = 250, dur = 2;
  const buf = new ArrayBuffer(32 + total * 2 + dur);
  const u8 = new Uint8Array(buf);
  u8[0] = 69; u8[1] = 67; u8[2] = 71; u8[3] = 82; u8[4] = 1; u8[5] = 1;
  const w32 = (off, v) => { u8[off] = v & 255; u8[off+1] = (v >>> 8) & 255; u8[off+2] = (v >>> 16) & 255; u8[off+3] = (v >>> 24) & 255; };
  w32(6, fs); w32(10, 1000); w32(14, dur); w32(18, total); w32(22, 1);
  const dv = new DataView(buf);
  for (let i = 0; i < total; i++) dv.setInt16(32 + i * 2, Math.round(Math.sin(i / 12) * 4000), true);
  const rec = window.ECGCore.parseEcgr(buf);
  rec.fileName = 'smoke.ecgr';
  window.ECGPlayback.loadRecord(rec);
  assert.ok(!document.getElementById('playerWrap').hidden, '回放面板应显示');
  assert.strictEqual(document.getElementById('pbSamples').textContent, '500');
  assert.strictEqual(document.getElementById('pbDur').textContent, '0:02');

  console.log('✓ DOM 冒烟测试通过');
  process.exit(0);
})().catch(err => {
  console.error('✗ DOM 冒烟测试失败:', err && err.stack || err);
  process.exit(1);
});
