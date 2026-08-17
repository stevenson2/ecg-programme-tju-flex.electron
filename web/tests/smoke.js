/* Node 冒烟测试：验证 ecg-core.js 纯逻辑
 * 运行：node tests/smoke.js
 */
'use strict';
const assert = require('assert');
const Core = require('../js/ecg-core.js');

/* 1. CSV 解析 */
const row = Core.parseCsvLine('1.234,-0.056,0.789,75,72,0.912,0,0,0.123');
assert.ok(row, '9 列 CSV 应解析成功');
assert.strictEqual(row.clean, 1.234);
assert.strictEqual(row.motion, false);
assert.strictEqual(Core.parseCsvLine('1.234,-0.056,0.789,75,72,0.912,0,0,0.123;').bpm, 75, '尾部分号应被接受');
assert.strictEqual(Core.parseCsvLine('[心率] 检测 75 BPM'), null, '诊断行应返回 null');
assert.strictEqual(Core.parseCsvLine('1,2,3'), null, '列数不足应返回 null');

/* 2. 模拟器：500 秒输出基本物理量合理 */
const sim = new Core.EcgSimulator(100, 75);
let beatSeen = 0, abnormalSeen = 0, minV = Infinity, maxV = -Infinity;
for (let i = 0; i < 100 * 500; i++) {
  const r = sim.next();
  if (r.clean < minV) minV = r.clean;
  if (r.clean > maxV) maxV = r.clean;
  if (r.abnormal) abnormalSeen++;
  if (r.bpm > 0 && r.bpm < 200) beatSeen++;
}
assert.ok(beatSeen > 0, '应检测到心率');
assert.ok(maxV - minV > 0.3 && maxV - minV < 5, '电压范围应合理: ' + (maxV - minV));
assert.ok(abnormalSeen > 0, '演示应出现异常片段');

/* 3. .ecgr 解析：构造 32 字节头 + 400 int16 样本 + 2 字节位图 */
const fs = 250, duration = 2, total = 400;
const buf = new ArrayBuffer(32 + total * 2 + duration);
const u8 = new Uint8Array(buf);
u8[0] = 69; u8[1] = 67; u8[2] = 71; u8[3] = 82; /* ECGR */
u8[4] = 1; u8[5] = 1; /* flags: has bitmap */
const write32 = (off, v) => { u8[off] = v & 255; u8[off + 1] = (v >>> 8) & 255; u8[off + 2] = (v >>> 16) & 255; u8[off + 3] = (v >>> 24) & 255; };
write32(6, fs);
write32(10, 1000);
write32(14, duration);
write32(18, total);
write32(22, 1);
const dv = new DataView(buf);
for (let i = 0; i < total; i++) dv.setInt16(32 + i * 2, Math.round(Math.sin(i / 10) * 5000), true);
u8[32 + total * 2] = 1; /* 第 0 秒异常 */
const rec = Core.parseEcgr(buf);
assert.strictEqual(rec.totalSamples, total);
assert.strictEqual(rec.sampleRate, fs);
assert.ok(rec.bitmap && rec.bitmap[0] === 1);
assert.ok(Math.abs(rec.voltAt(10) - Math.sin(1) * 5000 / 8000) < 1e-4); /* int16 量化误差 */

/* 4. min/max 抽取 */
const data = new Float32Array([0, 1, -2, 3, -4, 5]);
const dm = Core.decimateMinMax(data, 3);
assert.strictEqual(dm.columns, 3);
assert.strictEqual(dm.mins[1], -2);
assert.strictEqual(dm.maxs[2], 5);

/* 5. QRS 检测：合成脉冲序列 */
const det = new Core.QrsDetector(100);
let beats = 0;
for (let i = 0; i < 100 * 10; i++) {
  const t = i / 100;
  const pulse = Math.exp(-0.5 * Math.pow((t % 1 - 0.2) / 0.02, 2)) * 2;
  if (det.update(pulse)) beats++;
}
assert.ok(beats >= 8 && beats <= 12, 'QRS 检测应在 10 秒内找到约 10 拍: ' + beats);

console.log('✓ 全部冒烟测试通过');
