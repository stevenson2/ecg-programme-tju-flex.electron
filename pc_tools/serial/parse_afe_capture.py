#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 esp_timer_check_afe.txt 关键行与 CSV 统计"""
import re
import statistics

p = r'esp_timer_check_afe.txt'
text = open(p, encoding='utf-8', errors='replace').read()
lines = text.splitlines()
print('总行数', len(lines))
for pat in ['真实AFE', '当前输入模式', '[AFE]', '[WiFi]', 'AP started', 'SSID']:
    hits = [l for l in lines if pat in l]
    print(f'-- {pat}: {len(hits)} 行')
    for h in hits[:5]:
        print('   ', h.strip()[:160])

drops = [l for l in lines if '[SAMPLE]' in l]
total = sum(int(re.search(r'dropped=(\d+)', l).group(1)) for l in drops)
print('SAMPLE 行数', len(drops), '总 dropped =', total)

ble = [l for l in lines if '[BLE]' in l]
print('BLE 行数', len(ble))
for b in ble[:10]:
    print('   ', b.strip()[:160])

hr = [l for l in lines if '[心率]' in l]
print('心率行数', len(hr))
for h in hr[:10]:
    print('   ', h.strip()[:120])
print('   ...末尾5条:')
for h in hr[-5:]:
    print('   ', h.strip()[:120])

csv_rows = []
for l in lines:
    if re.match(r'^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},', l):
        parts = l.split(',')
        if len(parts) >= 9:
            csv_rows.append(parts)
print('CSV 行数', len(csv_rows))
if csv_rows:
    bpm = [int(x[3]) for x in csv_rows if x[3].isdigit()]
    sqi = [float(x[5]) for x in csv_rows]
    abn = sum(1 for x in csv_rows if x[7] == '1')
    confs = [float(x[8]) for x in csv_rows if re.match(r'^[-0-9.]+$', x[8])]
    print('BPM 非零数', len(bpm), '中位', statistics.median(bpm) if bpm else 0,
          '唯一值', sorted(set(bpm))[:30])
    print('SQI min/med/max', round(min(sqi), 3), round(statistics.median(sqi), 3), round(max(sqi), 3))
    print('abnormal=1 行数', abn, '/', len(csv_rows))
    if confs:
        print('conf max', max(confs), '非零行数', sum(1 for c in confs if c > 0))
