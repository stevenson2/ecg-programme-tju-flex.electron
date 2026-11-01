"""Dev metrics for every epoch (selection signals), then test for the final and dev-selected epochs."""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from common import load, pools
from models import build
from evaluate import logits_fn, margins, clean_metric, background, summarize_background

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
p = argparse.ArgumentParser()
p.add_argument('run')
args = p.parse_args()
run = Path(args.run)
arm, seed = run.name.rsplit('_s', 1)
seed = int(seed)
model = build('S', seed)
f = logits_fn(model)


def balanced_ce(m, y, pat):
    nll = np.where(y == 1, np.logaddexp(0, -m), np.logaddexp(0, m))
    means = []
    for c in (0, 1):
        means.append(np.mean([nll[(pat == q) & (y == c)].mean() for q in np.unique(pat) if ((pat == q) & (y == c)).any()]))
    return float(0.5 * sum(means))


def score(split):
    S, M = load(split)
    tm = margins(f, S, M['t_g'])
    gr, yr, pr = pools(M, 'ref')
    return M, tm, (margins(f, S, gr), yr, pr)


def native(M, tm, ref):
    if arm == 'ref':
        return balanced_ce(*ref)
    keep = {'trig': np.ones(len(tm), bool), 'trig_noQ': M['t_label'] != 3,
            'trigNV': np.isin(M['t_label'], [0, 2])}[arm]
    return balanced_ce(tm[keep], (M['t_label'][keep] == 2).astype(int), M['t_patient'][keep])


epochs = sorted(run.glob('ep*.weights.h5'))
dev = []
for w in epochs:
    model.load_weights(w)
    M, tm, ref = score('dev')
    bg = summarize_background(background(M, tm))
    dev.append(dict(epoch=int(w.name[2:4]), native_ce=native(M, tm, ref), clean=clean_metric(*ref),
                    D3=bg['all_V_positive']['D3'], V_items=bg['item_categories']['V_supported']))
    print('DEV', json.dumps(dev[-1]), flush=True)
best = min(dev, key=lambda r: r['native_ce'])['epoch']
chosen = sorted({dev[-1]['epoch'], best})
test = {}
for e in chosen:
    model.load_weights(run / f'ep{e:02d}.weights.h5')
    M, tm, ref = score('test')
    per = background(M, tm)
    d = next(r for r in dev if r['epoch'] == e)
    test[e] = dict(clean_self=clean_metric(*ref), clean_dev_locked=clean_metric(*ref, cutoff=d['clean']['cutoff']),
                   background=summarize_background(per), per_patient=per)
    np.save(run / f'test_trigger_margins_ep{e:02d}.npy', tm)
    print('TEST', e, json.dumps({k: v for k, v in test[e].items() if k != 'per_patient'}), flush=True)
(run / 'eval.json').write_text(json.dumps(dict(arm=arm, seed=seed, dev=dev, native_best_epoch=best,
                                               final_epoch=dev[-1]['epoch'], test=test), indent=1))
