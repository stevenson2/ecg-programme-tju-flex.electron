"""Score windows and compute (1) the clean-beat metric used so far and (2) the deployed
full-background top-3 evidence metric (same definitions as full_background/evaluate.py)."""
import math
import numpy as np
from common import zwin, LABELS

CAT_ORDER = ('V_supported', 'Q_undetermined', 'S_present_noV', 'N_only_noV', 'no_reference_beat')


def logits_fn(model):
    import tensorflow as tf
    ins = model.inputs if len(model.inputs) > 1 else model.inputs[0]
    shared = tf.keras.Model(ins, model.get_layer('do').output)
    out = model.get_layer('out')

    @tf.function(reduce_retracing=True)
    def f(x):
        feat = shared(x, training=False)
        lg = tf.linalg.matmul(feat, out.kernel) + out.bias
        return lg[:, 1] - lg[:, 0]
    return f


def margins(f, S, g, ctx=None, batch=2048):
    out = np.empty(len(g), np.float32)
    for s in range(0, len(g), batch):
        x = zwin(S, g[s:s + batch])
        out[s:s + batch] = (f([x, ctx[s:s + batch]]) if ctx is not None else f(x)).numpy()
    return out


def macro_rates(m, y, pat, t):
    fpr, rec = [], []
    for p in np.unique(pat):
        k = pat == p
        n, v = k & (y == 0), k & (y == 1)
        if n.any(): fpr.append(np.mean(m[n] >= t))
        if v.any(): rec.append(np.mean(m[v] >= t))
    return float(np.mean(fpr)), float(np.mean(rec)) if rec else float('nan')


def cutoff_for_fpr(m, y, pat, budget=0.02):
    cand = np.unique(m[y == 0])
    lo, hi = 0, len(cand) - 1
    while lo < hi:  # smallest threshold whose macro N FPR <= budget
        mid = (lo + hi) // 2
        if macro_rates(m, y, pat, cand[mid])[0] <= budget: hi = mid
        else: lo = mid + 1
    return float(np.nextafter(cand[lo], np.inf)) if macro_rates(m, y, pat, cand[lo])[0] > budget else float(cand[lo])


def clean_metric(m, y, pat, cutoff=None):
    c = cutoff_for_fpr(m, y, pat) if cutoff is None else cutoff
    fpr, rec = macro_rates(m, y, pat, c)
    return dict(cutoff=c, macro_N_FPR=fpr, macro_V_recall=rec)


def background(M, tm, k=3):
    """Per patient top-k complete 4 s grids by max trigger margin. tm: margin per trigger."""
    seg_len = M['seg_len']
    rows, per = [], []
    for p in np.unique(M['seg_patient']):
        segs = np.flatnonzero(M['seg_patient'] == p)
        base = []
        for s in segs:
            ng = seg_len[s] // 1000
            tk = np.flatnonzero(M['t_seg'] == s)
            gi = M['t_anchor'][tk] // 1000
            ok = gi < ng
            tk, gi = tk[ok], gi[ok]
            rk = np.flatnonzero(M['r_seg'] == s)
            rg = M['r_sample'][rk] // 1000
            rok = rg < ng
            cnt = np.zeros((ng, 4), np.int64)
            lab = M['r_label'][rk][rok]
            for c in range(4):
                np.add.at(cnt[:, c], rg[rok][lab == c], 1)
            if not len(tk):
                continue
            order = np.argsort(gi, kind='stable')
            tk, gi = tk[order], gi[order]
            cuts = np.r_[0, np.flatnonzero(np.diff(gi)) + 1, len(gi)]
            for a, b in zip(cuts[:-1], cuts[1:]):
                j = a + int(np.argmax(tm[tk[a:b]]))
                g = int(gi[a])
                base.append((float(tm[tk[j]]), int(s), g, *map(int, cnt[g]), int(M['t_label'][tk[j]])))
        base.sort(key=lambda r: (-r[0], r[1], r[2]))
        n = len(base)
        gV = sum(r[5] > 0 for r in base)
        availV = sum(r[5] for r in base)
        NSV = sum(r[3] + r[4] + r[5] for r in base)
        top = base[:k]
        cats = []
        for r in top:
            N, Sx, V, Q = r[3:7]
            cats.append('V_supported' if V else 'Q_undetermined' if Q else 'S_present_noV' if Sx else 'N_only_noV' if N else 'no_reference_beat')
        vranks = [i + 1 for i, r in enumerate(base) if r[5] > 0]
        per.append(dict(patient=int(p), grids=int(n), V_grids=int(gV), available_V=int(availV),
                        V_fraction=float(availV / NSV) if NSV else 0.0, hit=any(c == 'V_supported' for c in cats),
                        V_items=sum(c == 'V_supported' for c in cats),
                        random_hit=1 - math.comb(n - gV, k) / math.comb(n, k) if n >= k else float(gV > 0),
                        best_V_rank=min(vranks) if vranks else None,
                        cats=cats, winners=[LABELS[r[7]] for r in top]))
    return per


def summarize_background(per):
    pos = [r for r in per if r['available_V'] > 0]
    low = [r for r in pos if r['V_fraction'] < 0.01]
    rare = [r for r in pos if r['available_V'] <= 5]
    out = {}
    for name, grp in [('all_V_positive', pos), ('lt1pct', low), ('rare_1to5', rare)]:
        out[name] = dict(patients=len(grp), D3=sum(r['hit'] for r in grp), random_expected=round(sum(r['random_hit'] for r in grp), 3),
                         median_best_V_rank=float(np.median([r['best_V_rank'] for r in grp])) if grp else None)
    items = [c for r in per for c in r['cats']]
    win = [w for r in per for w in r['winners']]
    out['items'] = len(items)
    out['item_categories'] = {c: items.count(c) for c in CAT_ORDER}
    out['winner_trigger_labels'] = {l: win.count(l) for l in ('V', 'N', 'S', 'Q', 'unmatched', 'other')}
    return out
