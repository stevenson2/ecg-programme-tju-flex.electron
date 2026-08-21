#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复 TUNING_HISTORY.md 第五十五章乱码块 (UTF-8 被 GBK 误读, 无法无损还原)"""
import io
import time

PATH = '/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/TUNING_HISTORY.md'

replacement = """## 第五十五章 心率检测重构（能量包络）与 5 个 bug 修复（编码修复注记）

> **历史注记（2026-08-16 会话）**：本节原文在历史上被 GBK 误读后写回 UTF-8，
> 形成不可无损还原的乱码；其内容与紧随其后的第五十六章（完整、可读）重合
> （五个根因：RR 单位 ms/秒混用、导数平方放大尖锐伪影 8×、5-15Hz 带通砍窄 R、
> 形态学宽度旧标定误杀、beatCount 显示 bug）。为保持权威文档可读，本会话以
> 本注记替换乱码块，细节一律以第五十六章为准，不重建原文。

"""

with io.open(PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1-based 行号 3975..3997 (乱码块正文; 398 空行保留)
start, end = 3975, 3997
assert 0 < start <= end <= len(lines), (len(lines), start, end)
# 双重校验: 首行必须以乱码特征开头
assert lines[start - 1].startswith('## 绗'), repr(lines[start - 1][:20])
assert '閬楃暀' in lines[end - 1], repr(lines[end - 1])

new_lines = lines[: start - 1] + [replacement] + lines[end:]
for attempt in range(5):
    try:
        with io.open(PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print('OK: replaced lines', start, '-', end, '; total lines now', len(new_lines))
        break
    except OSError as e:
        print('write retry', attempt, repr(e))
        time.sleep(2)
else:
    raise SystemExit('write failed after retries')
