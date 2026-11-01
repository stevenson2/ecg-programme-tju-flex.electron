"""Run the firmware AI chain over whole Icentia11k segments; label every trigger."""
import ctypes as C, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import wfdb
from scipy.signal import resample_poly

ROOT = Path('/tmp/icentia')
OUT = Path('/tmp/pvcexp/cache')
LIB = '/tmp/pvcexp/libhostchain.so'
CODES = {'N': 0, 'S': 1, 'V': 2, 'Q': 3}
UNMATCHED, OTHER = 4, 5
MATCH = 37  # 148 ms at 250 Hz


def chain(x500, min_rr=50):
    lib = C.CDLL(LIB)
    FP, IP = C.POINTER(C.c_float), C.POINTER(C.c_int32)
    lib.run_chain.argtypes = [FP, C.c_int, FP, IP, IP, C.c_int, C.c_int]
    x500 = np.ascontiguousarray(x500, np.float32)
    n = len(x500)
    stream = np.zeros((n + 1) // 2, np.float32)
    cap = n // 50 + 16
    fire, rsc = np.zeros(cap, np.int32), np.zeros(cap, np.int32)
    nf = lib.run_chain(x500.ctypes.data_as(FP), n, stream.ctypes.data_as(FP),
                       fire.ctypes.data_as(IP), rsc.ctypes.data_as(IP), cap, min_rr)
    return stream, rsc[:nf].astype(np.int64)


def one(job):
    pdir, seg = job
    base = ROOT / pdir / seg
    rec = wfdb.rdrecord(str(base), physical=True, channels=[0])
    raw = rec.p_signal[:, 0].astype(np.float64)
    raw = np.nan_to_num(raw - raw[0])
    stream, anchor = chain(resample_poly(raw, 2, 1))
    assert len(stream) == len(raw)
    ann = wfdb.rdann(str(base), 'atr')
    sym = np.asarray(ann.symbol)
    smp = np.asarray(ann.sample, np.int64)
    beat = np.isin(sym, list(CODES) + ['a', 'J', 'A', 'F', 'E', 'j', 'e', 'L', 'R', '/', 'f'])
    rs, rsym = smp[beat], sym[beat]
    rlab = np.array([CODES.get(s, OTHER) for s in rsym], np.int8)
    n = len(stream)
    ok = (anchor - 125 >= 0) & (anchor + 125 <= n)
    anchor = anchor[ok]
    if len(rs):
        j = np.clip(np.searchsorted(rs, anchor), 1, len(rs) - 1) if len(rs) > 1 else np.zeros(len(anchor), np.int64)
        if len(rs) > 1:
            left, right = rs[j - 1], rs[j]
            pick = np.where(np.abs(anchor - left) <= np.abs(right - anchor), j - 1, j)
        else:
            pick = j
        dist = np.abs(anchor - rs[pick])
        tl = np.where(dist <= MATCH, rlab[pick], UNMATCHED).astype(np.int8)
        tref = np.where(dist <= MATCH, pick, -1)
    else:
        dist = np.full(len(anchor), 10 ** 6)
        tl = np.full(len(anchor), UNMATCHED, np.int8)
        tref = np.full(len(anchor), -1)
    rok = (rs - 125 >= 0) & (rs + 125 <= n)
    return dict(pdir=pdir, seg=seg, stream=stream, anchor=anchor, tlabel=tl, tdist=dist.astype(np.int32),
                tref=tref.astype(np.int32), rsample=rs, rlabel=rlab, rok=rok,
                symbols=dict(zip(*np.unique(sym, return_counts=True))), n=n)


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    split = json.loads(Path('/tmp/pvcexp/split.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    pdirs = {p['patient']: p for p in plan['patients']}
    for name, patients in split.items():
        jobs = [(pdirs[p]['dir'], s) for p in patients for s in pdirs[p]['segments']]
        with Pool(4) as pool:
            res = pool.map(one, jobs, chunksize=1)
        offs = np.cumsum([0] + [r['n'] for r in res])
        pid = {p: i for i, p in enumerate(patients)}
        seg_patient = np.array([pid[Path(r['pdir']).name] for r in res], np.int32)
        cat = lambda k: np.concatenate([r[k] for r in res])
        tseg = np.concatenate([np.full(len(r['anchor']), i, np.int32) for i, r in enumerate(res)])
        rseg = np.concatenate([np.full(len(r['rsample']), i, np.int32) for i, r in enumerate(res)])
        np.save(OUT / f'{name}_stream.npy', np.concatenate([r['stream'] for r in res]).astype(np.float32))
        np.savez(OUT / f'{name}_meta.npz', seg_offset=offs, seg_patient=seg_patient, seg_len=np.array([r['n'] for r in res]),
                 t_seg=tseg, t_anchor=cat('anchor'), t_label=cat('tlabel'), t_dist=cat('tdist'), t_ref=cat('tref'),
                 r_seg=rseg, r_sample=cat('rsample'), r_label=cat('rlabel'), r_ok=cat('rok'),
                 patients=np.array(patients), segments=np.array([r['seg'] for r in res]))
        sym = {}
        for r in res:
            for k, v in r['symbols'].items():
                sym[k] = sym.get(k, 0) + int(v)
        tl = cat('tlabel')
        print(name, 'segments', len(res), 'hours', round(offs[-1] / 250 / 3600, 1), 'triggers', len(tl),
              'trigger labels', {k: int((tl == c).sum()) for k, c in [('N', 0), ('S', 1), ('V', 2), ('Q', 3), ('unm', 4), ('oth', 5)]},
              'ann symbols', sym, flush=True)


if __name__ == '__main__':
    main()
