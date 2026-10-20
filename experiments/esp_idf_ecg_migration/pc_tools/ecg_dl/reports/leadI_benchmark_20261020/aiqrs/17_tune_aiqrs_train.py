#!/usr/bin/env python3
"""Tune ai_qrs params on INCART TRAIN records only (frozen test never touched).
Metric per config: pooled Se_all, Se_V, PPV (match tol 150 ms @ AI 250 Hz domain after
the ecg_ai 200 ms anti-retrigger), clean + NSTDB em @6 dB; plus median R offset."""
import os, sys, json, subprocess, itertools, numpy as np, wfdb
os.environ.setdefault("OMP_NUM_THREADS", "2")
B = "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/benchmark_leadI_20261019"
sys.path.insert(0, B + "/scripts")
from common_preprocess import resample_ecg, AAMI_NSVQ
HR = B + "/tools/hr_host/hr_host"
INC = "/mnt/c/ecg_data/incart/files"
split = json.load(open(B + "/splits/incart_leadI_split.json"))
train = split["train_rids"]
assert not set(train) & set(split["test_rids"])
# pick train records: 10 with most V + 3 with fewest
nv = {}
for rid in train:
    a = wfdb.rdann(f"{INC}/I{rid:02d}", "atr")
    nv[rid] = sum(1 for s in a.symbol if AAMI_NSVQ.get(s) == 2)
order = sorted(train, key=lambda r: -nv[r])
pick = order[:10] + order[-3:]
print("train picks", pick, [nv[r] for r in pick], flush=True)
nst = wfdb.rdrecord("/mnt/d/ecg_data_dl/nstdb/em"); em = resample_ecg(nst.p_signal[:, 0], nst.fs, 500)
data = []
for rid in pick:
    r = wfdb.rdrecord(f"{INC}/I{rid:02d}"); a = wfdb.rdann(f"{INC}/I{rid:02d}", "atr")
    x = resample_ecg(r.p_signal[:, 0], r.fs, 500)
    gp, gc = [], []
    for s, y in zip(a.sample, a.symbol):
        if y in AAMI_NSVQ: gp.append(int(round(s * 250 / r.fs))); gc.append(AAMI_NSVQ[y])
    rng = np.random.default_rng(rid)
    off = int(rng.integers(0, len(em) - len(x)))
    nz = em[off:off + len(x)] - em[off:off + len(x)].mean()
    xn = x + nz * np.sqrt(np.var(x) / (np.var(nz) * 10 ** 0.6))
    data.append((rid, x.astype(np.float32), xn.astype(np.float32), np.array(gp), np.array(gc)))

def run(x, mode, args):
    out = subprocess.run([HR, str(mode)] + [str(a) for a in args], input=x.tobytes(), capture_output=True, check=True).stdout.split()
    pk = np.asarray([int(v) for v in out], np.int64) // 2
    keep, last = [], -10**9
    min_rr = 150 if mode == 1 else 50
    for p in pk:
        if p - last >= min_rr: keep.append(p); last = p
    return np.asarray(keep, np.int64)

def match(det, gp, tol=37):
    if len(det) == 0: return np.zeros(len(gp), bool), np.full(len(gp), 999)
    j = np.clip(np.searchsorted(det, gp), 1, len(det) - 1)
    c = np.where(np.abs(det[j - 1] - gp) <= np.abs(det[j] - gp), j - 1, j)
    d = det[c] - gp
    return np.abs(d) <= tol, d

def score(mode, args):
    res = {}
    for k, idx in (("clean", 1), ("em6", 2)):
        tp = n = nV = tpV = nd = 0; offs = []
        for rec in data:
            det = run(rec[idx], mode, args); gp, gc = rec[3], rec[4]
            h, d = match(det, gp)
            tp += h.sum(); n += len(gp); nV += (gc == 2).sum(); tpV += h[gc == 2].sum(); nd += len(det)
            offs += list(d[h])
        res[k] = dict(Se=tp / n, SeV=tpV / max(1, nV), PPV=min(1, tp / max(1, nd)), off_med=float(np.median(offs)) if offs else None)
    return res

print("LEGACY", json.dumps(score(1, [])), flush=True)
grid = list(itertools.product([100, 125], [0.05, 0.08, 0.12, 0.2], [0.08], [0, 180]))
rows = []
for refr, ratio, floor, tg in grid:
    s = score(2, [refr, ratio, floor, tg, 0.5, 16])
    rows.append(((refr, ratio, floor, tg), s))
    print((refr, ratio, floor, tg), json.dumps(s), flush=True)
def obj(s):  # prioritise V detection, keep PPV, average clean & noisy
    return np.mean([s[k]["SeV"] + 0.5 * s[k]["Se"] + 1.0 * s[k]["PPV"] for k in ("clean", "em6")])
best = max(rows, key=lambda r: obj(r[1]))
print("BEST", best[0], json.dumps(best[1]), flush=True)
json.dump({"picks": pick, "rows": [[list(a), b] for a, b in rows], "best": list(best[0])}, open(B + "/results/aiqrs_tune_train.json", "w"), indent=1, default=float)
