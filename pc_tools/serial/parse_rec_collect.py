#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 rec_collect.py 输出, 校验 totalSamples/duration ≈ 250"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'rec_afe_60s.txt'
text = open(path, encoding='utf-8', errors='replace').read()
print(text.strip()[:3000])
newrecs = []
for l in text.splitlines():
    m = re.search(r'NEWREC.*id=(\d+).*dur=(\d+)s samples=(\d+) abnSec=(\d+) size=(\d+)', l)
    if m:
        newrecs.append(tuple(int(x) for x in m.groups()))
    m2 = re.match(r'^(\d+),(\d+),(\d+),(\d+),(\d+)$', l.strip())
    if m2:
        pass
print('\n新增记录:')
for r in newrecs:
    rid, dur, samples, abn, size = r
    print(f'  id={rid} dur={dur}s samples={samples} abnSec={abn} size={size} '
          f'ratio={samples/dur if dur else 0:.2f} (期望≈250)')
