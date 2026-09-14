/* 金样测试（P0-1）：固定 BLE/CSV 帧 + ECGR 字节 -> 三端同断言。
 * 运行: node web/tests/protocol-golden.js
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const Core = require('../js/ecg-core.js');

const ROOT = path.resolve(__dirname, '..', '..');
const GOLDEN = path.join(ROOT, 'protocol', 'golden');
const framesDoc = JSON.parse(fs.readFileSync(path.join(GOLDEN, 'ble_frames.json'), 'utf8'));
const expectedDoc = JSON.parse(fs.readFileSync(path.join(GOLDEN, 'ecgr_expected.json'), 'utf8'));

let n = 0;
function ok(cond, msg) { assert.ok(cond, msg); n++; }

/* 1. BLE/CSV 金样帧 */
for (const [name, frame] of Object.entries(framesDoc.frames)) {
  const row = Core.parseCsvLine(frame);
  const exp = framesDoc.expected[name];
  if (!exp) continue; // HELLO / 诊断行单独断言
  ok(row, `${name}: 应解析成功`);
  for (const [k, v] of Object.entries(exp)) {
    const actual = (k === 'abnormal') ? (row.abnormal ? 1 : 0) : row[k];
    ok(actual === v, `${name}.${k}: 期望 ${v} 实际 ${actual}`);
  }
}
ok(Core.parseCsvLine(framesDoc.frames.diagnostic_line) === null, '诊断行应拒绝');
ok(Core.parseCsvLine(framesDoc.frames.hello) === null, 'HELLO 行不应作为 CSV 数据解析');

/* 2. ECGR v1/v2 金样 */
for (const [file, exp] of Object.entries(expectedDoc)) {
  const bytes = fs.readFileSync(path.join(GOLDEN, file));
  const rec = Core.parseEcgr(bytes);
  ok(rec.version === exp.version, `${file}: version`);
  ok(rec.sampleRate === exp.sampleRate, `${file}: sampleRate`);
  ok(rec.durationSec === exp.durationSec, `${file}: durationSec`);
  ok(rec.totalSamples === exp.totalSamples, `${file}: totalSamples`);
  ok(rec.abnormalSec === exp.abnormalSeconds, `${file}: abnormalSeconds`);
  ok(rec.hasBitmap === exp.hasBitmap, `${file}: hasBitmap`);
  ok(rec.bitmapIsAsrc === exp.bitmapIsAsrc, `${file}: bitmapIsAsrc`);
  ok(Array.from(rec.bitmap).join(',') === exp.bitmap.join(','), `${file}: bitmap`);
  ok(rec.truncated === exp.truncated, `${file}: truncated`);
  ok(Math.abs(rec.voltAt(1) - 1000 / 8000) < 1e-9, `${file}: voltAt`);
}

/* 3. asrc 分级（生成常量） */

/* 4. asrc 分级文案（共享生成常量） */
ok(Core.primaryAsrc(0x10).name === 'LEADOFF', 'LEADOFF tier');
ok(Core.primaryAsrc(0x01).name === 'AI', 'AI tier');
ok(Core.primaryAsrc(0x1c).name === 'LEADOFF', 'priority: LEADOFF > AI/FLAT/VF');
ok(Core.primaryAsrc(0x00) === null, 'normal -> null');

/* 5. rowToCsv 必须输出 v2 第 10 列（web 导入/导出闭环） */
const row10 = Core.parseCsvLine('0.1,0.2,0.3,70,0,0.9,0,1,0.5,16');
ok(row10.asrc === 16, 'v2 frame asrc value');
const csv10 = Core.rowToCsv(row10);
ok(csv10.split(',').length === 10, 'rowToCsv must emit 10 columns');
ok(csv10.endsWith(',16'), 'rowToCsv keeps asrc bit 0x10');
/* v1 行重放应降级 asrc 0x01（非零即报警） */
const rowV1 = Core.parseCsvLine('0.1,0.2,0.3,70,0,0.9,0,1,0.5');
ok(rowV1.asrc === 1, 'v1 abnormal -> asrc 0x01 fallback');

console.log(`✓ protocol golden (web) 通过: ${n} 断言`);