#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 OVS4/OVS1 两轮捕获: 模式/BLE/VF/abnormal/心率/SQI"""
import re
import statistics
import sys

for path in sys.argv[1:]:
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    tags = {}
    for t in ('[BLE]', '[VF]', '[AF]', '[SAFETY]', '[ALARM]', '[温度]', '真实AFE', '回放'):
        tags[t] = [l for l in lines if t in l]
    rows = []
    for l in lines:
        if re.match(r'^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},', l):
            p = l.split(',')
            if len(p) >= 9:
                rows.append(p)
    print('=' * 70)
    print(path)
    print('总行数', len(lines), ' CSV', len(rows))
    for t in ('真实AFE', '回放', '[BLE]', '[VF]', '[AF]', '[SAFETY]', '[ALARM]'):
        print(f'  {t}: {len(tags[t])}')
        for l in tags[t][:6]:
            print('     ', l.strip()[:140])
    if rows:
        bpm = [int(x[3]) for x in rows if x[3].isdigit()]
        sqi = [float(x[5]) for x in rows]
        abn = sum(1 for x in rows if x[7] == '1')
        confs = [float(x[8]) for x in rows if re.match(r'^[-0-9.]+$', x[8])]
        print('  BPM: n=%d 中位=%s 唯一值=%s' % (len(bpm), statistics.median(bpm) if bpm else 0,
              sorted(set(bpm))[:25]))
        print('  SQI: min/med/max = %.3f / %.3f / %.3f' % (min(sqi), statistics.median(sqi), max(sqi)))
        print('  abnormal=1: %d/%d; conf 非零=%d, max=%.3f' % (
            abn, len(rows), sum(1 for c in confs if c > 0), max(confs) if confs else 0.0))
