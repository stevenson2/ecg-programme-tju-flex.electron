"""Evaluate a train2.py run. Dev per epoch with the tail selection metric (mean log2 best-V-grid rank over V-positive
patients; ties -> larger D10), then test at the final and the dev-selected epoch. For the 4-class head the primary
score is l_V - logsumexp(l_N, l_S, l_X); l_V - l_N is reported as secondary. Test margins are saved in the same
file names as run_eval.py so analyze.py works unchanged."""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from common import load, zwin, rwin
from features import trigger_features, rel_scale
from variants import build_variant
from analyze import grids, per_patient, summarize

p = argparse.ArgumentParser()
p.add_argument('run')
p.add_argument('--threads', type=int, default=4)
a = p.parse_args()
tf.config.threading.set_intra_op_parallelism_threads(a.threads)
tf.config.threading.set_inter_op_parallelism_threads(1)
run = Path(a.run)
cfg = json.loads((run / 'train_log.json').read_text())
arm, seed, K, ctx_cols = cfg['args']['arm'], cfg['args']['seed'], cfg['K'], cfg['ctx_cols']
aux4, relamp = cfg.get('aux4', False), cfg.get('relamp', arm == 'trig_relamp')
model = build_variant(seed, n_out=K, ctx_dim=len(ctx_cols) if ctx_cols else 0, aux4=aux4)
shared = tf.keras.Model(model.inputs, model.get_layer('do').output)
heads = [model.get_layer('out')] + ([model.get_layer('out4')] if aux4 else [])
NL = K + (4 if aux4 else 0)


@tf.function(reduce_retracing=True)
def logits(x, c=None):
    f = shared([x, c] if c is not None else x, training=False)
    return tf.concat([tf.matmul(f, h.kernel) + h.bias for h in heads], axis=1)


SPLIT = {}


def prepare(split):
    S, M = load(split)
    feat = trigger_features(split, M, S)
    ctx = feat['F'][:, ctx_cols] if ctx_cols else None
    scale = rel_scale(feat) if relamp else None
    SPLIT[split] = (S, M, ctx, scale)


def score(split):
    """Returns M and a dict of score reductions; the first entry is the primary (deployable) score."""
    S, M, ctx, scale = SPLIT[split]
    g = M['t_g']
    LG = np.empty((len(g), NL), np.float32)
    for s in range(0, len(g), 2048):
        ix = np.arange(s, min(len(g), s + 2048))
        x = tf.constant(rwin(S, g[ix], scale[ix]) if scale is not None else zwin(S, g[ix]))
        LG[s:s + 2048] = (logits(x, tf.constant(ctx[ix])) if ctx is not None else logits(x)).numpy()
    if K == 2 and not aux4:
        return M, {'primary': LG[:, 1] - LG[:, 0]}
    if K == 4:
        return M, {'primary': LG[:, 2] - np.logaddexp.reduce(LG[:, [0, 1, 3]], axis=1), 'vn': LG[:, 2] - LG[:, 0]}
    L4 = LG[:, 2:]  # binary head first, then the auxiliary N/S/V/X head
    return M, {'primary': LG[:, 1] - LG[:, 0], 'vn': L4[:, 2] - L4[:, 0], 'vr': L4[:, 2] - np.logaddexp.reduce(L4[:, [0, 1, 3]], axis=1)}


def tail(M, tm):
    per = per_patient(grids(M, tm), M)
    s = summarize(per)
    pos = [r for r in per if r['available_V'] > 0]
    s['mean_log2_rank'] = round(float(np.mean([np.log2(r['best_V_rank']) for r in pos])), 3)
    return s


def brief(s):
    return dict(mean_log2_rank=s['mean_log2_rank'],
                **{g: f"D3 {s[g]['D3']}/{s[g]['n']} D10 {s[g]['D10']} D30 {s[g]['D30']} rank {s[g]['med_rank']} gap {s[g]['med_gap']:+.2f} auc {s[g]['auc']}"
                   for g in ('all', 'lt1', 'rare')},
                Vitems=s['cats']['V_supported'], Qitems=s['cats']['Q_undetermined'], Nitems=s['cats']['N_only_noV'],
                winQU=s['winners']['Q'] + s['winners']['unmatched'])


prepare('dev')
dev = []
for w in sorted(run.glob('ep*.weights.h5')):
    model.load_weights(w)
    M, sc = score('dev')
    s = tail(M, sc['primary'])
    dev.append(dict(epoch=int(w.name[2:4]), **brief(s), D10_all=s['all']['D10']))
    print('DEV', json.dumps(dev[-1]), flush=True)
best = min(dev, key=lambda r: (r['mean_log2_rank'], -r['D10_all']))['epoch']
prepare('test')
test = {}
for e in sorted({dev[-1]['epoch'], best}):
    model.load_weights(run / f'ep{e:02d}.weights.h5')
    M, sc = score('test')
    test[e] = {}
    for name, tm in sc.items():
        suffix = '' if name == 'primary' else f'_{name}'
        np.save(run / f'test_trigger_margins{suffix}_ep{e:02d}.npy', tm.astype(np.float32))
        test[e][name] = brief(tail(M, tm))
    print('TEST', e, 'selected' if e == best else 'final', json.dumps(test[e]), flush=True)
(run / 'eval.json').write_text(json.dumps(dict(arm=arm, seed=seed, K=K, ctx_cols=ctx_cols, aux4=aux4, dev=dev, selected_epoch=best,
                                               final_epoch=dev[-1]['epoch'], test=test), indent=1))
