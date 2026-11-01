"""(1) Signed trigger-anchor minus matched-annotation offset on the test split.
(2) Re-score a trained run on test triggers with every deployed window moved by d samples."""
import json, sys
import numpy as np
import tensorflow as tf
from common import load
from models import build
from evaluate import logits_fn, margins
from analyze import grids, per_patient, summarize

tf.config.threading.set_intra_op_parallelism_threads(2)
S, M = load('test')
r_start = np.r_[0, np.cumsum(np.bincount(M['r_seg'], minlength=len(M['seg_len'])))][:-1]
m = M['t_ref'] >= 0
off = M['t_anchor'][m] - M['r_sample'][r_start[M['t_seg'][m]] + M['t_ref'][m]]
lab = M['t_label'][m]
for c, name in ((0, 'N'), (2, 'V')):
    o = off[lab == c]
    print(f'offset {name}: n={len(o)} median={np.median(o):+.0f} p10={np.percentile(o, 10):+.0f} p90={np.percentile(o, 90):+.0f} '
          f'mean={o.mean():+.2f}', flush=True)

run, seed = sys.argv[1], int(sys.argv[2])
model = build('S', seed)
f = logits_fn(model)
res = {}
for ep in map(int, sys.argv[3].split(',')):
    model.load_weights(f'{run}/ep{ep:02d}.weights.h5')
    for d in map(int, sys.argv[4].split(',')):
        g = np.where(M['t_anchor'] + d - 125 >= 0, M['t_g'] + d, M['t_g'])
        s = summarize(per_patient(grids(M, margins(f, S, g)), M))
        res[f'ep{ep}_d{d:+d}'] = s
        print(f'ep{ep} d={d:+d}', ' | '.join(f"{k}: D3 {s[k]['D3']}/{s[k]['n']} rank {s[k]['med_rank']} gap {s[k]['med_gap']:+.2f}"
                                           for k in ('all', 'lt1', 'rare')), '| Vitems', s['cats']['V_supported'],
              'Qitems', s['cats']['Q_undetermined'], flush=True)
json.dump(res, open(f'{run}/shift_test.json', 'w'), indent=1)
