"""Smoke-test arms on the trigger stream. Shared with train.py: architecture seed, Adam + cosine 3e-4 (alpha .01),
batch 256, class-weighted sparse CE, all V + fresh 250K negatives per epoch (sampling RNG identical to train.py).

arms
  trig            parity control for train.py 'trig'
  trigctx         + 4 context scalars (preRR, postRR, post/pre, amplitude ratio)
  trigctx_causal  + 2 causal scalars (preRR, amplitude ratio)
  trig4c          4-class head N/S/V/X (X = Q or unmatched)
  trig_hnm        from epoch 2 on, half of the negatives are the hardest of a random 500K subset
  trig_relamp     input window normalised by patient-adaptive scale (rwin) instead of per-window z-score
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from common import load, zwin, rwin
from features import trigger_features, rel_scale
from variants import build_variant

ARMS = {'trig': {}, 'trigctx': dict(ctx=[0, 1, 2, 3]), 'trigctx_causal': dict(ctx=[0, 3]), 'trig4c': dict(K=4),
        'trig_hnm': dict(hnm=True), 'trig_relamp': dict(relamp=True),
        'trigctx4c': dict(ctx=[0, 1, 2, 3], aux4=True),  # context + binary main head + N/S/V/X auxiliary head
        'trig4c2h': dict(aux4=True)}
p = argparse.ArgumentParser()
p.add_argument('arm', choices=sorted(ARMS))
p.add_argument('--epochs', type=int, default=8)
p.add_argument('--neg-per-epoch', type=int, default=250000)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--out', default='/tmp/pvcexp/runs')
args = p.parse_args()
tf.config.threading.set_intra_op_parallelism_threads(args.threads)
tf.config.threading.set_inter_op_parallelism_threads(1)

out = Path(args.out) / f'{args.arm}_s{args.seed}'
out.mkdir(parents=True, exist_ok=True)
S, M = load('train')
feat = trigger_features('train', M, S)
lab, g = M['t_label'], M['t_g']
cfg = ARMS[args.arm]
K, ctx_cols, aux4 = cfg.get('K', 2), cfg.get('ctx'), cfg.get('aux4', False)
y4 = np.select([lab == 0, lab == 1, lab == 2], [0, 1, 2], 3).astype(np.int32)
y = y4 if K == 4 else (lab == 2).astype(np.int32)
ctx = feat['F'][:, ctx_cols] if ctx_cols else None
scale = rel_scale(feat) if cfg.get('relamp') else None
win = (lambda ix: rwin(S, g[ix], scale[ix])) if scale is not None else (lambda ix: zwin(S, g[ix]))

model = build_variant(args.seed, n_out=K, ctx_dim=len(ctx_cols) if ctx_cols else 0, aux4=aux4)
pos, neg = np.flatnonzero(lab == 2), np.flatnonzero(lab != 2)
n_neg = min(args.neg_per_epoch, len(neg))
steps = (len(pos) + n_neg + 255) // 256
sched = tf.keras.optimizers.schedules.CosineDecay(3e-4, steps * args.epochs, alpha=.01)
opt = tf.keras.optimizers.Adam(sched)
print(json.dumps(dict(arm=args.arm, K=K, aux4=aux4, pool=len(y), pos=len(pos), neg=len(neg), neg_per_epoch=n_neg, steps=steps,
                      params=model.count_params(), ctx_cols=ctx_cols)), flush=True)
scce = tf.keras.losses.sparse_categorical_crossentropy


def total_loss(pr, yy, yy4, cw, cw4):
    if aux4:
        return tf.reduce_mean(scce(yy, pr[0]) * tf.gather(cw, yy)) + tf.reduce_mean(scce(yy4, pr[1]) * tf.gather(cw4, yy4))
    return tf.reduce_mean(scce(yy, pr) * tf.gather(cw, yy))


YSIG = [tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32), tf.TensorSpec([K]), tf.TensorSpec([4])]
if ctx is None:
    @tf.function(input_signature=[tf.TensorSpec([None, 250, 1])] + YSIG)
    def step(x, yy, yy4, cw, cw4):
        with tf.GradientTape() as tape:
            loss = total_loss(model(x, training=True), yy, yy4, cw, cw4)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss
else:
    @tf.function(input_signature=[tf.TensorSpec([None, 250, 1]), tf.TensorSpec([None, len(ctx_cols)])] + YSIG)
    def step(x, c, yy, yy4, cw, cw4):
        with tf.GradientTape() as tape:
            loss = total_loss(model([x, c], training=True), yy, yy4, cw, cw4)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

shared = tf.keras.Model(model.inputs, model.get_layer('do').output)
outl = model.get_layer('out')


def binary_margin(ix):
    """l_V - l_N with the current weights, inference mode; used only for hard-negative mining."""
    x = tf.constant(win(ix))
    f = shared([x, tf.constant(ctx[ix])] if ctx is not None else x, training=False)
    lg = tf.matmul(f, outl.kernel) + outl.bias
    return (lg[:, 1] - lg[:, 0]).numpy()


rng = np.random.default_rng(10000 + args.seed)
log = []
for ep in range(args.epochs):
    t0 = time.time()
    if cfg.get('hnm') and ep >= 1:
        cand = np.sort(rng.choice(neg, min(500000, len(neg)), replace=False))
        sc = np.concatenate([binary_margin(cand[s:s + 4096]) for s in range(0, len(cand), 4096)])
        hard = cand[np.argsort(-sc)[:n_neg // 2]]
        rnd = rng.choice(np.setdiff1d(neg, hard, assume_unique=True), n_neg - len(hard), replace=False)
        negsel = np.concatenate([hard, rnd])
        mined = dict(mining_seconds=round(time.time() - t0, 1), hard_margin_min=float(sc[np.argsort(-sc)[n_neg // 2 - 1]]),
                     hard_label_counts={int(k): int(v) for k, v in zip(*np.unique(lab[hard], return_counts=True))})
    else:
        negsel = rng.choice(neg, n_neg, replace=False)
        mined = {}
    idx = np.concatenate([pos, negsel])
    rng.shuffle(idx)
    n = len(idx)
    cnt = np.bincount(y[idx], minlength=K).astype(np.float64)
    cw = tf.constant(np.where(cnt > 0, n / (K * np.maximum(cnt, 1)), 0.0), tf.float32)
    cnt4 = np.bincount(y4[idx], minlength=4).astype(np.float64)
    cw4 = tf.constant(np.where(cnt4 > 0, n / (4 * np.maximum(cnt4, 1)), 0.0), tf.float32)
    tot = 0.0
    for b in range(0, n, 256):
        ix = np.sort(idx[b:b + 256])
        x = tf.constant(win(ix))
        ys = (tf.constant(y[ix]), tf.constant(y4[ix]), cw, cw4)
        loss = step(x, *ys) if ctx is None else step(x, tf.constant(ctx[ix]), *ys)
        tot += float(loss) * len(ix)
        if b // 256 % 500 == 0:
            print(f'ep{ep + 1} batch {b // 256}/{steps} loss {tot / (b + len(ix)):.4f} {time.time() - t0:.0f}s', flush=True)
    model.save_weights(out / f'ep{ep + 1:02d}.weights.h5')
    log.append(dict(epoch=ep + 1, loss=tot / n, seconds=time.time() - t0, class_counts=cnt.astype(int).tolist(), **mined))
    print(json.dumps(log[-1]), flush=True)
(out / 'train_log.json').write_text(json.dumps(dict(args=vars(args), K=K, ctx_cols=ctx_cols, aux4=aux4, relamp=bool(cfg.get('relamp')), log=log), indent=1))
