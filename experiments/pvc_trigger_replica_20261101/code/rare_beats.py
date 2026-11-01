"""(1) Per-beat context of the V beats of chosen patients. (2) Crude QRS-width proxy by class and by model score."""
import sys
import numpy as np
from common import load, zwin, LABELS
from dissect import context

S, M = load('test')
lab, g = M['t_label'], M['t_g']
C, F = context(M, S)
runs = {'ref_s42': 'runs/ref_s42/test_trigger_margins_ep08.npy', 'trig_s42': 'runs/trig_s42/test_trigger_margins_ep08.npy',
        'trig_s43': 'runs/trig_s43/test_trigger_margins_ep08.npy'}
TM = {k: np.load(v).astype(np.float64) for k, v in runs.items()}


def width_ms(idx):
    """Samples within +-120 ms of the anchor whose |z| exceeds 30% of the local peak, in ms."""
    out = np.empty(len(idx))
    for s in range(0, len(idx), 32768):
        x = np.abs(zwin(S, g[idx[s:s + 32768]])[..., 0][:, 95:155])
        out[s:s + 32768] = (x > 0.3 * x.max(1, keepdims=True)).sum(1) * 4
    return out


non = np.sort(g[np.isin(lab, [1, 3, 4])])
print('## V beats of selected patients')
print('pt seg   anchor | preRR postRR post/pre amp | width_ms | S/Q/unm within 2s | margin ref_s42 trig_s42 trig_s43 | prev,next trigger')
for p in [int(x) for x in sys.argv[1].split(',')]:
    idx = np.flatnonzero((M['t_patient'] == p) & (lab == 2))
    w = width_ms(idx)
    for n, i in enumerate(idx):
        j = np.searchsorted(non, g[i])
        d = min(abs(g[i] - non[max(j - 1, 0)]), abs(non[min(j, len(non) - 1)] - g[i]))
        prev = LABELS[int(lab[i - 1])] if i > 0 and M['t_seg'][i - 1] == M['t_seg'][i] else '-'
        nxt = LABELS[int(lab[i + 1])] if i + 1 < len(lab) and M['t_seg'][i + 1] == M['t_seg'][i] else '-'
        print(f"{p:2d} {M['t_seg'][i]:3d} {M['t_anchor'][i]:8d} | {C['pre'][i]:5.2f} {C['post'][i]:6.2f} {C['pp'][i]:8.2f} {C['amp'][i]:5.2f} | {w[n]:5.0f} | {str(d <= 500):5s} | "
              + ' '.join(f'{TM[k][i]:+7.2f}' for k in runs) + f' | {prev},{nxt}')

print('\n## QRS-width proxy (ms), median [p10, p90]')
rng = np.random.default_rng(0)
iN = rng.choice(np.flatnonzero(lab == 0), 20000, replace=False)
iS, iV = np.flatnonzero(lab == 1), np.flatnonzero(lab == 2)
iQ = rng.choice(np.flatnonzero(lab == 3), 20000, replace=False)
wN, wS, wV, wQ = width_ms(iN), width_ms(iS), width_ms(iV), width_ms(iQ)
fmt = lambda w: f'{np.median(w):.0f} [{np.percentile(w, 10):.0f}, {np.percentile(w, 90):.0f}] n={len(w)}'
print('N', fmt(wN)); print('S', fmt(wS)); print('V', fmt(wV)); print('Q', fmt(wQ))
print('AUC of width alone, V vs N:', end=' ')
from scipy.stats import rankdata
r = rankdata(np.r_[wV, wN]); print(round(float((r[:len(wV)].sum() - len(wV) * (len(wV) + 1) / 2) / (len(wV) * len(wN))), 3))
m = TM['trig_s42'][iV]
for name, k in [('V margin<0 (scored as N)', m < 0), ('V margin 0-5', (m >= 0) & (m < 5)), ('V margin>=5', m >= 5)]:
    print(f'{name:26s} width {fmt(wV[k])} | med amp {np.nanmedian(C["amp"][iV][k]):.2f} | med preRR {np.nanmedian(C["pre"][iV][k]):.2f} | frac preRR>=0.85 {np.nanmean(C["pre"][iV][k] >= .85):.2f}')
for name, k in [('N amp>2 (top noise bin)', C['amp'][iN] > 2), ('N premature <0.85', C['pre'][iN] < .85), ('N typical', (C['amp'][iN] <= 2) & (C['pre'][iN] >= .85))]:
    print(f'{name:26s} width {fmt(wN[k])} | med margin trig_s42 {np.median(TM["trig_s42"][iN][k]):+.2f}')
ms = TM['trig_s42'][iS]
print(f'S premature <0.85          width {fmt(wS[C["pre"][iS] < .85])} | med margin {np.median(ms[C["pre"][iS] < .85]):+.2f} | p99 {np.percentile(ms[C["pre"][iS] < .85], 99):+.2f}')
# does width explain the S confusion? AUC of width V vs S
r = rankdata(np.r_[wV, wS]); print('AUC of width alone, V vs S:', round(float((r[:len(wV)].sum() - len(wV) * (len(wV) + 1) / 2) / (len(wV) * len(wS))), 3))
print('AUC of trig_s42 margin, V vs S:', end=' ')
r = rankdata(np.r_[TM['trig_s42'][iV], ms]); print(round(float((r[:len(iV)].sum() - len(iV) * (len(iV) + 1) / 2) / (len(iV) * len(iS))), 3))
