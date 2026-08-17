#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 esp_timer_check_afe.txt 中检索各报警/诊断标签行"""
import re

p = r'esp_timer_check_afe.txt'
lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
tags = ['[ALARM]', '[AF]', '[VF]', '[SAFETY]', '[运动]', '[温度]', '[REC]',
        '[SCHED]', '[系统]', '[警告]', '[AI]', 'flatline', '[SAMPLE]', '[BLE]']
for t in tags:
    hits = [l for l in lines if t in l]
    print(f'--- {t}: {len(hits)}')
    for h in hits[:12]:
        print('   ', h.strip()[:180])

# 前 40 个非 CSV 行 (boot/模式切换过程)
print('\n--- 前 40 个非 CSV 行:')
n = 0
for l in lines:
    if not re.match(r'^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},', l):
        print('   ', l.strip()[:180])
        n += 1
        if n >= 40:
            break
