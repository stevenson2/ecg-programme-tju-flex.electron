#!/usr/bin/env python3
"""MIT-BIH 全量数据集下载器 (48条记录)
直接运行: python download_full.py
断点续传: 已下载的文件自动跳过
"""

import urllib.request, time, sys
from pathlib import Path

RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

BASE_URL = 'https://physionet.org/files/mitdb/1.0.0'
OUTDIR = Path(__file__).resolve().parent / 'data' / 'raw' / 'mit-bih-arrhythmia-database'
OUTDIR.mkdir(parents=True, exist_ok=True)

total = len(RECORDS) * 3
ok = skip = fail = 0

print(f'Downloading {len(RECORDS)} records ({total} files) to:')
print(f'  {OUTDIR}')
print()

for rec in RECORDS:
    for ext in ['hea', 'dat', 'atr']:
        fname = f'{rec}.{ext}'
        fpath = OUTDIR / fname
        if fpath.exists():
            skip += 1
            continue
        try:
            url = f'{BASE_URL}/{fname}'
            data = urllib.request.urlopen(url, timeout=30).read()
            fpath.write_bytes(data)
            ok += 1
            size_kb = len(data) / 1024
            sys.stdout.write(f'\r  [{ok+skip}/{total}] {fname} ({size_kb:.0f} KB) OK    ')
            sys.stdout.flush()
        except Exception as e:
            fail += 1
            sys.stdout.write(f'\r  [{ok+skip}/{total}] {fname} FAIL: {e}\n')
            sys.stdout.flush()
        time.sleep(0.1)

print(f'\n\nDone! {ok} OK, {skip} skipped, {fail} FAIL')
