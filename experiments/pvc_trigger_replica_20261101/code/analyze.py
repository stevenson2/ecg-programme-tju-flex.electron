"""Continuous full-background metrics and oracle decompositions from saved test trigger margins.

Variants (pool and random baseline unchanged):
  base     deployed rule: grid score = max margin over all triggers in the grid
  NVonly   oracle: only triggers matched to a reference N or V count; other triggers -inf
  noQgrid  oracle: grids containing a reference Q but no reference V get -inf
"""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from common import load, LABELS

KS = (3, 10, 30)


def grids(M, tm):
    seg_len = M['seg_len']
    ng = seg_len // 1000
    off = np.r_[0, np.cumsum(ng)]
    gi = M['t_anchor'] // 1000
    ok = gi < ng[M['t_seg']]
    tk = np.flatnonzero(ok)
    gg = off[M['t_seg'][tk]] + gi[tk]
    order = np.lexsort((-tm[tk], gg))
    tk, gg = tk[order], gg[order]
    first = np.r_[True, gg[1:] != gg[:-1]]
    G = gg[first]
    win = tk[first]
    score = tm[win].astype(np.float64)
    cnt = np.zeros((off[-1], 4), np.int64)
    rgi = M['r_sample'] // 1000
    rok = rgi < ng[M['r_seg']]
    rg = off[M['r_seg'][rok]] + rgi[rok]
    lab = M['r_label'][rok]
    for c in range(4):
        np.add.at(cnt[:, c], rg[lab == c], 1)
    seg_of = np.repeat(np.arange(len(ng)), ng)
    pat = M['seg_patient'][seg_of[G]]
    return dict(G=G, score=score, win=win, cnt=cnt[G], pat=pat)


def per_patient(T, M):
    out = []
    for p in np.unique(T['pat']):
        k = np.flatnonzero(T['pat'] == p)
        s, G, c, w = T['score'][k], T['G'][k], T['cnt'][k], T['win'][k]
        order = np.lexsort((G, -s))
        isV = c[:, 2] > 0
        n, g = len(k), int(isV.sum())
        availV = int(c[:, 2].sum())
        nsv = int(c[:, :3].sum())
        rec = dict(patient=int(p), grids=n, V_grids=g, available_V=availV,
                   V_fraction=availV / nsv if nsv else 0.0)
        for kk in KS:
            rec[f'D{kk}'] = bool(isV[order[:kk]].any())
            rec[f'rand{kk}'] = 1 - math.comb(n - g, kk) / math.comb(n, kk) if n >= kk else float(g > 0)
        top = order[:3]
        rec['cats'] = ['V_supported' if c[j, 2] else 'Q_undetermined' if c[j, 3] else 'S_present_noV' if c[j, 1]
                       else 'N_only_noV' if c[j, 0] else 'no_reference_beat' for j in top]
        rec['winners'] = [LABELS[int(M['t_label'][w[j]])] for j in top]
        if g:
            ranks = np.empty(n, np.int64)
            ranks[order] = np.arange(1, n + 1)
            rec['best_V_rank'] = int(ranks[isV].min())
            if g < n:
                r = rankdata(s)
                rec['auc'] = float((r[isV].sum() - g * (g + 1) / 2) / (g * (n - g)))
                nv = np.sort(s[~isV])[::-1]
                rec['gap'] = float(s[isV].max() - nv[min(2, len(nv) - 1)])
        out.append(rec)
    return out


def summarize(per):
    pos = [r for r in per if r['available_V'] > 0]
    groups = dict(all=pos, lt1=[r for r in pos if r['V_fraction'] < 0.01], rare=[r for r in pos if r['available_V'] <= 5])
    out = {}
    for name, grp in groups.items():
        d = dict(n=len(grp))
        for kk in KS:
            d[f'D{kk}'] = sum(r[f'D{kk}'] for r in grp)
            d[f'rand{kk}'] = round(sum(r[f'rand{kk}'] for r in grp), 2)
        d['med_rank'] = float(np.median([r['best_V_rank'] for r in grp]))
        d['auc'] = round(float(np.nanmean([r.get('auc', np.nan) for r in grp])), 4)
        d['med_gap'] = round(float(np.nanmedian([r.get('gap', np.nan) for r in grp])), 2)
        out[name] = d
    cats = [c for r in per for c in r['cats']]
    wins = [w for r in per for w in r['winners']]
    out['cats'] = {c: cats.count(c) for c in ('V_supported', 'Q_undetermined', 'S_present_noV', 'N_only_noV', 'no_reference_beat')}
    out['winners'] = {l: wins.count(l) for l in ('V', 'N', 'S', 'Q', 'unmatched')}
    return out


def quantiles(M, tm):
    q = {}
    for c, name in LABELS.items():
        m = tm[M['t_label'] == c]
        if len(m):
            q[name] = [round(float(x), 2) for x in np.quantile(m, [.5, .9, .99, .999])]
    return q


def analyze(run, ep, M):
    tm = np.load(Path(run) / f'test_trigger_margins_ep{ep:02d}.npy').astype(np.float64)
    res = {'quantiles_p50_p90_p99_p999': quantiles(M, tm)}
    nv = tm.copy()
    nv[~np.isin(M['t_label'], [0, 2])] = -np.inf
    for name, t in (('base', tm), ('NVonly', nv)):
        res[name] = summarize(per_patient(grids(M, t), M))
    T = grids(M, tm)
    T['score'] = np.where((T['cnt'][:, 3] > 0) & (T['cnt'][:, 2] == 0), -np.inf, T['score'])
    res['noQgrid'] = summarize(per_patient(T, M))
    return res


if __name__ == '__main__':
    S, M = load('test')
    results = {}
    for spec in sys.argv[1:]:
        run, ep = spec.rsplit(':', 1)
        results[f'{Path(run).name}:ep{int(ep)}'] = r = analyze(run, int(ep), M)
        print('#####', spec)
        for k in ('base', 'NVonly', 'noQgrid'):
            print(f'  [{k}]', json.dumps(r[k]))
        print('  [quantiles]', json.dumps(r['quantiles_p50_p90_p99_p999']))
    Path('/tmp/pvcexp/analysis.json').write_text(json.dumps(results, indent=1))
