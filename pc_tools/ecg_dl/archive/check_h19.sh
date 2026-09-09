#!/bin/bash
cd '/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master' || exit 1
python3 - <<'PY'
import json, re, sys
root = '.'
jr = json.load(open('pc_tools/ecg_dl/models/ludb_hr_v6_eval.json', encoding='utf-8'))
s = jr['summary']
vals = {
  'se_pct': f"{s['se']*100:.2f}%",
  'ppv_pct': f"{s['ppv']*100:.2f}%",
  'f1': f"{s['f1']:.3f}",
  'mae': f"{s['bpm_mae']:.2f}",
  'median': f"{s['bpm_median_err']:.2f}",
}
print('JSON v6:', vals, 'TP/FP/FN =', s['tp'], s['fp'], s['fn'])
manu = open('docs/manuscript_sections_1_4.md', encoding='utf-8').read()
fin = open('docs/FINAL_RESULTS.md', encoding='utf-8').read()
need = ['96.40%', '78.87%', '0.868', '4.16', '1.46']
for name, text in [('manuscript', manu), ('FINAL_RESULTS', fin)]:
    miss = [v for v in need if v not in text]
    print(name, 'missing:', miss if miss else 'NONE (all v6 numbers present)')
# T12 deployed row check
t12 = re.search(r'\*\*v6 \(energy envelope \+ recalibrated gates, deployed\)\*\*.*', manu)
print('T12 v6 row:', t12.group(0) if t12 else 'MISSING')
t9 = re.search(r'\*\*v6 \(能量包络\+重标定门限, 部署\)\*\*.*', fin)
print('表9附 v6 row:', t9.group(0) if t9 else 'MISSING')
# audit sums
assert s['tp'] + s['fn'] == s['gold_beats']
assert s['tp'] + s['fp'] == s['det_beats']
print('AUDIT: TP+FN==gold, TP+FP==det OK; warnings =', s['audit']['warnings'])
PY