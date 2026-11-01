"""Failure-mode dissection from saved test-trigger margins. No training.

Per run:epoch
  A  pairwise AUCs and extreme-value arithmetic (which N quantile a V must beat)
  B  prematurity: margin vs pre-RR ratio per class (shortcut-learning test)
  C  noise: margin vs window-amplitude ratio per class
  D  what the top 0.05% of N triggers look like
  E  captured vs missed V relative to each patient's 3rd-best non-V grid
  F  per-patient signal-vs-background table
Global
  G  seed ensembles of saved ep8 margins
  H  (--fuse run:ep) logistic fusion of CNN margin with RR/amplitude context, fit on dev, applied on test
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr
from common import load, OFF, LABELS
from analyze import grids, per_patient, summarize

warnings.filterwarnings('ignore')
J = lambda o: json.dumps(o, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
PRE_EDGES = [0, .7, .85, 1.15, np.inf]
AMP_EDGES = [0, .5, 2, 4, np.inf]
_DEV = {}


def prev_median(x, w, seg):
    """Median of the previous w values inside the same segment; NaN when fewer than 3 exist."""
    out = np.full(len(x), np.nan)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    for b, e in zip(starts, np.r_[starts[1:], len(x)]):
        n = e - b
        if n < 4:
            continue
        p = np.r_[np.full(w, np.nan), x[b:e - 1]]
        W = swv(p, w)[:n]
        med = np.nanmedian(W, axis=1)
        med[np.isfinite(W).sum(1) < 3] = np.nan
        out[b:e] = med
    return out


def context(M, S):
    a, seg = M['t_anchor'].astype(np.float64), M['t_seg']
    n = len(a)
    same = seg[1:] == seg[:-1]
    d = np.diff(a)
    pre, post = np.full(n, np.nan), np.full(n, np.nan)
    pre[1:][same] = d[same]
    post[:-1][same] = d[same]
    ref = prev_median(pre, 8, seg)
    sd = np.empty(n, np.float64)
    g = M['t_g']
    for s in range(0, n, 65536):
        sd[s:s + 65536] = np.asarray(S[g[s:s + 65536, None] + OFF], np.float32).std(1)
    sref = prev_median(sd, 16, seg)
    with np.errstate(all='ignore'):
        C = dict(pre=pre / ref, post=post / ref, pp=post / pre, amp=sd / sref)
        F = np.stack([np.log(C[k]) for k in ('pre', 'post', 'pp', 'amp')], 1)
    F[~np.isfinite(F)] = 0
    return C, np.clip(F, -3, 3)


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = rankdata(np.r_[pos, neg])
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def frac(v, cond):
    f = np.isfinite(v)
    return round(float(cond[f].mean()), 3) if f.any() else None


def bins(tm, mask, ratio, edges, T):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = mask & (ratio >= lo) & (ratio < hi)
        rows.append(dict(bin=f'{lo}-{hi}', n=int(k.sum()), med=round(float(np.median(tm[k])), 2) if k.any() else None,
                         p99=round(float(np.percentile(tm[k], 99)), 2) if k.sum() >= 100 else None,
                         frac_ge_Vmed=round(float(np.mean(tm[k] >= T)), 4) if k.any() else None))
    k = mask & ~np.isfinite(ratio)
    rows.append(dict(bin='undef', n=int(k.sum()), med=round(float(np.median(tm[k])), 2) if k.any() else None, p99=None,
                     frac_ge_Vmed=round(float(np.mean(tm[k] >= T)), 4) if k.any() else None))
    return rows


def section_A(tm, lab, F):
    V, N = tm[lab == 2], tm[lab == 0]
    isV, isN = lab == 2, lab == 0
    out = {f'AUC_V_vs_{LABELS[c]}': auc(V, tm[lab == c]) for c in (0, 1, 3, 4)}
    out['single_feature_AUC_V_vs_N'] = dict(prematurity=auc(-F[isV, 0], -F[isN, 0]), pause=auc(F[isV, 2], F[isN, 2]),
                                             amplitude=auc(F[isV, 3], F[isN, 3]))
    qs = [.99, .999, .9999, .99995]
    Nq = np.quantile(N, qs)
    out['N_quantiles'] = {str(q): round(float(v), 2) for q, v in zip(qs, Nq)}
    out['frac_V_above_N_quantile'] = {str(q): round(float(np.mean(V > v)), 3) for q, v in zip(qs, Nq)}
    out['V_p10_p50_p90'] = [round(float(x), 2) for x in np.quantile(V, [.1, .5, .9])]
    return out


def section_D(tm, lab, C, M, frac_top=5e-4):
    g = M['t_g']
    idxN = np.flatnonzero(lab == 0)
    k = max(int(len(idxN) * frac_top), 50)
    top = idxN[np.argsort(-tm[idxN])[:k]]
    non = np.sort(g[lab != 0])

    def near(ix, r=500):
        pos = g[ix]
        j = np.searchsorted(non, pos)
        left = np.where(j > 0, pos - non[np.maximum(j - 1, 0)], np.inf)
        right = np.where(j < len(non), non[np.minimum(j, len(non) - 1)] - pos, np.inf)
        return np.minimum(left, right) <= r

    def rates(ix):
        return dict(n=int(len(ix)), premature=frac(C['pre'][ix], C['pre'][ix] < .85), pause_after=frac(C['pp'][ix], C['pp'][ix] > 1.2),
                    noisy=frac(C['amp'][ix], (C['amp'][ix] > 2) | (C['amp'][ix] < .5)), near_nonN_2s=round(float(near(ix).mean()), 3),
                    margin_min=round(float(tm[ix].min()), 2))
    return dict(top_N=rates(top), all_N=rates(idxN))


def third_bg(T):
    thr = {}
    for p in np.unique(T['pat']):
        s = np.sort(T['score'][(T['pat'] == p) & (T['cnt'][:, 2] == 0)])[::-1]
        thr[int(p)] = float(s[2]) if len(s) >= 3 else -np.inf
    return thr


def section_E(tm, lab, C, M, T):
    thr = third_bg(T)
    iV = np.flatnonzero(lab == 2)
    t3 = np.array([thr[int(p)] for p in M['t_patient'][iV]])
    cap = tm[iV] > t3
    inrun = ((np.r_[-1, lab[:-1]] == 2) | (np.r_[lab[1:], -1] == 2))[iV]

    def desc(k):
        if not k.any():
            return dict(n=0)
        ix = iV[k]
        return dict(n=int(k.sum()), med_margin=round(float(np.median(tm[ix])), 2), med_margin_minus_bg3=round(float(np.median(tm[ix] - t3[k])), 2),
                    med_pre=round(float(np.nanmedian(C['pre'][ix])), 3), frac_premature=frac(C['pre'][ix], C['pre'][ix] < .85),
                    med_pp=round(float(np.nanmedian(C['pp'][ix])), 3), med_amp=round(float(np.nanmedian(C['amp'][ix])), 3),
                    frac_in_run=round(float(inrun[k].mean()), 3), frac_pre_undef=round(float((~np.isfinite(C['pre'][ix])).mean()), 3))
    return dict(captured=desc(cap), missed=desc(~cap))


def section_F(tm, lab, M, T):
    rows = []
    for p in np.unique(T['pat']):
        k = np.flatnonzero(T['pat'] == p)
        c, s, w = T['cnt'][k], T['score'][k], T['win'][k]
        if c[:, 2].sum() == 0:
            continue
        nv = np.flatnonzero(c[:, 2] == 0)
        o = nv[np.argsort(-s[nv])]
        kt = M['t_patient'] == p
        mN = tm[kt & (lab == 0)]
        j = o[0]
        rows.append(dict(pt=int(p), V=int(c[:, 2].sum()), Vg=int((c[:, 2] > 0).sum()), grids=len(k), bestV=round(float(s[c[:, 2] > 0].max()), 2),
                         bg3=round(float(s[o[2]]), 2), bg1_win=LABELS[int(lab[w[j]])],
                         bg1_cat='Q' if c[j, 3] else 'S' if c[j, 1] else 'N' if c[j, 0] else '-',
                         Np999=round(float(np.percentile(mN, 99.9)), 2), Nmax=round(float(mN.max()), 2),
                         fQ=round(float(np.mean(lab[kt] == 3)), 3), fU=round(float(np.mean(lab[kt] == 4)), 3), trig=int(kt.sum())))
    return rows


def brief(s):
    out = {g: f"D3 {s[g]['D3']}/{s[g]['n']} D10 {s[g]['D10']} D30 {s[g]['D30']} rank {s[g]['med_rank']} gap {s[g]['med_gap']:+.2f}"
           for g in ('all', 'lt1', 'rare')}
    out.update(Vitems=s['cats']['V_supported'], Qitems=s['cats']['Q_undetermined'], winQU=s['winners']['Q'] + s['winners']['unmatched'])
    return out


def section_G(M):
    ens = {'ref': ['ref_s42', 'ref_s43'], 'trig': ['trig_s42', 'trig_s43'], 'trigNV': ['trigNV_s42', 'trigNV_s43']}
    out, allm = {}, []
    for arm, runs in ens.items():
        ms = [np.load(f'runs/{r}/test_trigger_margins_ep08.npy').astype(np.float64) for r in runs]
        allm += ms
        out[arm] = dict(single=[brief(summarize(per_patient(grids(M, m), M))) for m in ms],
                        mean2=brief(summarize(per_patient(grids(M, np.mean(ms, 0)), M))))
    out['all6_mean'] = brief(summarize(per_patient(grids(M, np.mean(allm, 0)), M)))
    return out


def fit_lr(X, y, l2=1e-4):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))
    w = w / w.sum()

    def f(b):
        z = Z @ b
        loss = np.sum(w * (np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))) + l2 * np.sum(b[:-1] ** 2)
        p = 1 / (1 + np.exp(-z))
        return loss, Z.T @ (w * (p - y)) + 2 * l2 * np.r_[b[:-1], 0]
    b = minimize(f, np.zeros(Z.shape[1]), jac=True, method='L-BFGS-B', options=dict(maxiter=500)).x
    return (lambda Xn: np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ b), b


def design(m, F, kind):
    cols = {'margin': [m],
            'margin+RR': [m, F[:, 0], F[:, 1], F[:, 2]],
            'margin+amp': [m, F[:, 3], F[:, 3] ** 2, m * F[:, 3]],
            'margin+all': [m, F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 3] ** 2, m * F[:, 3]],
            'context_only': [F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 3] ** 2]}[kind]
    return np.stack(cols, 1).astype(np.float64)


def section_H(run, ep, M, F, tm_test, lab):
    import tensorflow as tf
    from models import build
    from evaluate import logits_fn, margins
    tf.config.threading.set_intra_op_parallelism_threads(3)
    seed = int(run.rsplit('_s', 1)[1])
    if 'F' not in _DEV:
        Sd, Md = load('dev')
        _DEV['S'], _DEV['M'] = Sd, Md
        _DEV['F'] = context(Md, Sd)[1]
    Sd, Md, Fd = _DEV['S'], _DEV['M'], _DEV['F']
    cache = Path(run) / f'dev_trigger_margins_ep{ep:02d}.npy'
    if cache.exists():
        md = np.load(cache)
    else:
        model = build('S', seed)
        model.load_weights(f'{run}/ep{ep:02d}.weights.h5')
        md = margins(logits_fn(model), Sd, Md['t_g'])
        np.save(cache, md)
    md = md.astype(np.float64)
    yd = (Md['t_label'] == 2).astype(np.float64)
    out = {}
    for kind in ('margin', 'margin+RR', 'margin+amp', 'margin+all', 'context_only'):
        f, b = fit_lr(design(md, Fd, kind), yd)
        s = f(design(tm_test, F, kind))
        out[kind] = dict(coef=[round(float(x), 3) for x in b], test_AUC_V_vs_rest=auc(s[lab == 2], s[lab != 2]),
                         test_AUC_V_vs_N=auc(s[lab == 2], s[lab == 0]), **brief(summarize(per_patient(grids(M, s), M))))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('specs', nargs='*')
    ap.add_argument('--fuse', action='append', default=[])
    ap.add_argument('--no-ensemble', action='store_true')
    a = ap.parse_args()
    S, M = load('test')
    lab = M['t_label']
    C, F = context(M, S)
    OUT = {'context': dict(triggers=int(len(lab)), pre_defined=round(float(np.isfinite(C['pre']).mean()), 3))}
    print('[context]', J(OUT['context']), flush=True)
    for spec in a.specs:
        run, ep = spec.rsplit(':', 1)
        ep = int(ep)
        name = f'{Path(run).name}:ep{ep}'
        tm = np.load(f'{run}/test_trigger_margins_ep{ep:02d}.npy').astype(np.float64)
        T = grids(M, tm)
        Tv = float(np.median(tm[lab == 2]))
        fin = np.isfinite(C['pre']) & (lab == 2)
        fa = fin & np.isfinite(C['amp'])
        R = dict(A=section_A(tm, lab, F),
                 B=dict(V_median=round(Tv, 2), N=bins(tm, lab == 0, C['pre'], PRE_EDGES, Tv), S=bins(tm, lab == 1, C['pre'], PRE_EDGES, Tv),
                        V=bins(tm, lab == 2, C['pre'], PRE_EDGES, Tv),
                        spearman_V_margin_vs_preRR=round(float(spearmanr(tm[fin], C['pre'][fin])[0]), 3),
                        spearman_V_margin_vs_amp=round(float(spearmanr(tm[fa], C['amp'][fa])[0]), 3)),
                 C={LABELS[c]: bins(tm, lab == c, C['amp'], AMP_EDGES, Tv) for c in (0, 2, 3, 4)},
                 D=section_D(tm, lab, C, M), E=section_E(tm, lab, C, M, T), F=section_F(tm, lab, M, T))
        OUT[name] = R
        print(f'##### {name}')
        for k, v in R.items():
            print(f'[{k}]', J(v), flush=True)
    if not a.no_ensemble:
        OUT['G_ensembles'] = section_G(M)
        print('[G]', J(OUT['G_ensembles']), flush=True)
    for spec in a.fuse:
        run, ep = spec.rsplit(':', 1)
        ep = int(ep)
        tm = np.load(f'{run}/test_trigger_margins_ep{ep:02d}.npy').astype(np.float64)
        OUT[f'H_{Path(run).name}:ep{ep}'] = r = section_H(run, ep, M, F, tm, lab)
        print(f'[H {Path(run).name}:ep{ep}]', J(r), flush=True)
    Path('/tmp/pvcexp/dissect.json').write_text(json.dumps(OUT, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
