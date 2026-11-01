"""Train the deployed 62,834-parameter network (models.build('S')) under one data recipe.

Identical across arms: architecture, init seed, Adam + cosine 3e-4 (alpha .01), batch 256,
class-weighted sparse CE with weights n/(2*n_c) recomputed on each epoch's pool,
fresh random subsample of 250K negatives each epoch. Arms differ only in which windows are trained on."""
import argparse, json, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from common import load, zwin, pools, ctx_features
from models import build

p = argparse.ArgumentParser()
p.add_argument('arm', choices=['ref', 'trig', 'trig_noQ', 'trigNV', 'trigctx'])
p.add_argument('--epochs', type=int, default=8)
p.add_argument('--neg-per-epoch', type=int, default=250000)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--out', default='/tmp/pvcexp/runs')
args = p.parse_args()
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

out = Path(args.out) / f'{args.arm}_s{args.seed}'
out.mkdir(parents=True, exist_ok=True)
S, M = load('train')
g, y, pat = pools(M, 'trig' if args.arm == 'trigctx' else args.arm)
ctx = ctx_features(M, S) if args.arm == 'trigctx' else None

if args.arm == 'trigctx':
    from ctxmodel import build_ctx
    model = build_ctx(args.seed)
else:
    model = build('S', args.seed)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
n_neg = min(args.neg_per_epoch, len(neg))
steps = (len(pos) + n_neg + 255) // 256
sched = tf.keras.optimizers.schedules.CosineDecay(3e-4, steps * args.epochs, alpha=.01)
opt = tf.keras.optimizers.Adam(sched)
print(json.dumps(dict(arm=args.arm, pool=len(y), pos=len(pos), neg=len(neg), neg_per_epoch=n_neg,
                      steps=steps, params=model.count_params())), flush=True)

if ctx is None:
    @tf.function(input_signature=[tf.TensorSpec([None, 250, 1]), tf.TensorSpec([None], tf.int32), tf.TensorSpec([2])])
    def step(x, yy, cw):
        with tf.GradientTape() as tape:
            pr = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(yy, pr) * tf.gather(cw, yy))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss
else:
    @tf.function(input_signature=[tf.TensorSpec([None, 250, 1]), tf.TensorSpec([None, 4]), tf.TensorSpec([None], tf.int32), tf.TensorSpec([2])])
    def step(x, c, yy, cw):
        with tf.GradientTape() as tape:
            pr = model([x, c], training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(yy, pr) * tf.gather(cw, yy))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

rng = np.random.default_rng(10000 + args.seed)
log = []
for ep in range(args.epochs):
    t0 = time.time()
    idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(idx)
    n = len(idx)
    cw = tf.constant([n / (2 * n_neg), n / (2 * len(pos))], tf.float32)
    tot = 0.0
    for b in range(0, n, 256):
        ix = np.sort(idx[b:b + 256])
        x = tf.constant(zwin(S, g[ix]))
        if ctx is None:
            loss = step(x, tf.constant(y[ix]), cw)
        else:
            loss = step(x, tf.constant(ctx[ix]), tf.constant(y[ix]), cw)
        tot += float(loss) * len(ix)
        if b // 256 % 500 == 0:
            print(f'ep{ep + 1} batch {b // 256}/{steps} loss {tot / (b + len(ix)):.4f} {time.time() - t0:.0f}s', flush=True)
    model.save_weights(out / f'ep{ep + 1:02d}.weights.h5')
    log.append(dict(epoch=ep + 1, loss=tot / n, seconds=time.time() - t0))
    print(json.dumps(log[-1]), flush=True)
(out / 'train_log.json').write_text(json.dumps(dict(args=vars(args), log=log), indent=1))
