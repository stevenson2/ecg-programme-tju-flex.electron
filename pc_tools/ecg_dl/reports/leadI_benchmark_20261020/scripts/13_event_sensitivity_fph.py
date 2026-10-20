#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""13 · Dual primary metric: event-level sensitivity (FULL alarm chain) + FP/h.

Chain (per record, everything causal-ish & identical for all models):
  raw -> resample 250 Hz -> causal HP0.5/LP40/notch50 -> OWN R-peak detector
  (simple_rpeaks, same as scripts 04/07/08, but threshold computed per 60 s block)
  -> 250-sample R-centred z-scored beat -> model score -> 1 Hz flag (any beat>=thr)
  -> alarm state machine (dens 8/10, hold 12 s, release <=1/5, cooldown 25 s).
No GT R-peaks are used for scoring (GT only to define events / label seconds).

Events (from reference annotations, V = AAMI V class: V,E,F,!):
  * spec event   : episode where an ORACLE (1 Hz flag = 'second contains a true V beat')
                   would enter alarm under the same rules (cooldown 0). = what the spec
                   promises to catch. Detected if model alarm is active at any second in
                   [onset, end + GRACE_S]. Latency = first active second - onset
                   (onset = first V-second in the oracle density window at oracle entry).
  * VT run       : >=3 consecutive V beats in the annotation sequence (reported separately;
                   runs not overlapping a spec event are 'spec-undetectable by design').
  * isolated PVC : V beats outside any spec event (burden per hour only).
FP entry: model alarm entry whose 10 s density window contains no true V second.
Conditions: INCART frozen test Lead I clean; + NSTDB em / ma at SNR 12/6/0 dB (INCART);
MIT-BIH frozen test records (MLII — NOT Lead I, cross-lead, honest flag); PTB Lead I
controls; 4 real-device sessions (~100 Hz serial, no annotations).
Resource: CPU, 2 threads, single process; run under nice -n 19.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import json, sys, time, csv
from pathlib import Path
import numpy as np
import wfdb
from scipy import signal as sp

BENCH = Path("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/benchmark_leadI_20261019")
WS = BENCH.parent
REPO = WS / "ecg-programme-tju-flex.electron-master"
DATA, RES = BENCH / "data", BENCH / "results"
FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)
OUTD = RES / "event_sens_fph"; OUTD.mkdir(parents=True, exist_ok=True)
CACHE = OUTD / "score_cache"; CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BENCH / "scripts"))
from common_preprocess import TARGET_FS, WIN, AAMI_NSVQ, resample_ecg, causal_filter_chain

INCART_DIR = Path("/mnt/c/ecg_data/incart/files")
MIT_DIR = Path("/mnt/d/ecg_data_dl/mitdb")
NST_DIR = Path("/mnt/d/ecg_data_dl/nstdb")
PTB_DIR = Path("/mnt/c/ecg_data/ecg_database")
INBOX = WS / "real_captures" / "inbox"
SPLIT = json.loads((BENCH / "splits" / "incart_leadI_split.json").read_text(encoding="utf-8"))
TEST_RIDS = list(SPLIT["test_rids"])
MIT_SPLIT = json.loads((BENCH / "splits" / "mit_mlii_split.json").read_text(encoding="utf-8"))
MIT_TEST = []
for p in MIT_SPLIT["test_patients"]:
    MIT_TEST += p.replace("mit_", "").split("_")  # mit_201_202 -> 201, 202

THR = 0.50
RULE = dict(win=10, th=8, hold_s=12, rel_win=5, rel_th=1, cooldown_s=25)
GRACE_S = 15
THR_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
SNRS = [12, 6, 0]
NOISES = ["em", "ma"]
NBOOT = 2000
RNG = np.random.default_rng(20261020)

MODELS = {
    "student_hard": ("keras", BENCH / "models/student_leadI_resnet_large.h5"),
    "distill_v1": ("keras", BENCH / "models/student_leadI_distill_fm_center.h5"),
    "distill_v2": ("keras", BENCH / "models/student_leadI_distill_fm_v2.h5"),
    "v3a_h5": ("keras", REPO / "pc_tools/ecg_dl/models/best_resnet_large_clean_baseline_v3.h5"),
    "exp0_h5": ("keras", REPO / "pc_tools/ecg_dl/models/best_resnet_large_exp0_baseline.h5"),
    "exp5_int8": ("tflite", REPO / "pc_tools/ecg_dl/models/deploy_match/exp5_clean_int8.tflite"),
}
LEAK = {"exp5_int8": "historical split may overlap INCART test & MIT-BIH (leakage NOT excluded)",
        "v3a_h5": "trained on MIT-BIH/INCART historical split (MIT leakage likely)",
        "exp0_h5": "trained on MIT-BIH/INCART historical split (MIT leakage likely)"}

# ---------------------------------------------------------------- signal chain
def simple_rpeaks(sig250, fs=250, block_s=60):
    b, a = sp.butter(2, [5 / (fs / 2), 20 / (fs / 2)], btype="band")
    y = sp.filtfilt(b, a, sig250)
    env = np.abs(sp.hilbert(y))
    out = []
    B = block_s * fs
    for s0 in range(0, len(env), B):
        e = env[s0:s0 + B]
        if len(e) < fs:
            continue
        thr = np.percentile(e, 90)
        pk, _ = sp.find_peaks(e, height=thr, distance=int(0.3 * fs))
        out.extend((pk + s0).tolist())
    pk = np.asarray(sorted(set(out)), np.int64)
    if len(pk) > 1:  # enforce refractory across block borders
        keep = [pk[0]]
        for p in pk[1:]:
            if p - keep[-1] >= int(0.3 * fs):
                keep.append(p)
        pk = np.asarray(keep)
    return pk


HR_HOST = BENCH / "tools/hr_host/hr_host"
AI_MIN_RR = 150  # ecg_ai_on_rpeak anti-retrigger, samples @250 Hz (600 ms)


def fw_rpeaks(raw, fs):
    """Firmware-faithful: resample 500 Hz (mV) -> applyFilter + hrProcess (heartrate.cpp v6,
    compiled on host, tools/hr_host) -> beatDetected indices -> /2 to 250 Hz ->
    ecg_ai_on_rpeak 600 ms anti-retrigger. Window centred at detection sample (as on board)."""
    import subprocess
    x = resample_ecg(raw, fs, 500).astype(np.float32)
    out = subprocess.run([str(HR_HOST), "1"], input=x.tobytes(), capture_output=True, check=True).stdout.split()
    pk = np.asarray([int(v) for v in out], np.int64) // 2
    keep, last = [], -10**9
    for p in pk:
        if p - last >= AI_MIN_RR:
            keep.append(p); last = p
    return np.asarray(keep, np.int64)


def beats_at(sig250, peaks):
    half = WIN // 2
    beats, used = [], []
    for p in peaks:
        lo, hi = p - half, p - half + WIN
        if lo < 0 or hi > len(sig250):
            continue
        seg = sig250[lo:hi]
        s = seg.std()
        if s < 1e-8:
            continue
        beats.append(((seg - seg.mean()) / s).astype(np.float32))
        used.append(int(p))
    if not beats:
        return np.zeros((0, WIN), np.float32), np.zeros(0, np.int64)
    return np.stack(beats), np.asarray(used, np.int64)


def alarm_sim(flags, win, th, hold_s, rel_win, rel_th, cooldown_s=0):
    n = len(flags); flags = np.asarray(flags, bool)
    cs = np.concatenate([[0], np.cumsum(flags)])
    alarm = np.zeros(n, bool); active = False; last = -10**9; cd_until = -1; entries = []
    for i in range(n):
        dens = cs[i + 1] - cs[max(0, i - win + 1)]
        if dens >= th:
            if not active:
                if i >= cd_until:
                    active = True; entries.append(i); last = i
            else:
                last = i
        elif active:
            rel = cs[i + 1] - cs[max(0, i - rel_win + 1)]
            if (i - last) >= hold_s and rel <= rel_th:
                active = False; cd_until = i + 1 + cooldown_s
        alarm[i] = active
    return entries, alarm


def gt_beats(ann_samp, ann_sym, fs_orig):
    r = TARGET_FS / float(fs_orig)
    pk, cls = [], []
    for s, y in zip(ann_samp, ann_sym):
        if y in AAMI_NSVQ:
            pk.append(int(round(s * r))); cls.append(AAMI_NSVQ[y])
    return np.asarray(pk, np.int64), np.asarray(cls, np.int32)


def define_events(gt_pk, gt_cls, n_sec):
    isv = gt_cls == 2
    vsec = np.zeros(n_sec, bool)
    for p in gt_pk[isv]:
        if 0 <= p // 250 < n_sec:
            vsec[p // 250] = True
    o_entries, o_alarm = alarm_sim(vsec, **{**RULE, "cooldown_s": 0})
    events = []
    # episodes = contiguous oracle-active spans
    i = 0
    while i < n_sec:
        if o_alarm[i]:
            j = i
            while j + 1 < n_sec and o_alarm[j + 1]:
                j += 1
            lo = max(0, i - RULE["win"] + 1)
            onset = lo + int(np.argmax(vsec[lo:i + 1]))
            events.append({"onset": int(onset), "oracle_entry": int(i), "end": int(j)})
            i = j + 1
        else:
            i += 1
    # VT runs: >=3 consecutive V in annotation sequence (only AAMI beats)
    runs = []
    k = 0
    while k < len(gt_cls):
        if gt_cls[k] == 2:
            m = k
            while m + 1 < len(gt_cls) and gt_cls[m + 1] == 2:
                m += 1
            if m - k + 1 >= 3:
                runs.append({"start": int(gt_pk[k] // 250), "end": int(gt_pk[m] // 250), "n_beats": int(m - k + 1)})
            k = m + 1
        else:
            k += 1
    for r in runs:
        r["in_spec_event"] = any(e["onset"] - 10 <= r["end"] and r["start"] <= e["end"] for e in events)
    in_ev = np.zeros(n_sec, bool)
    for e in events:
        in_ev[e["onset"]:e["end"] + 1] = True
    v_secs = (gt_pk[isv] // 250).clip(0, n_sec - 1)
    iso = int((~in_ev[v_secs]).sum()) if len(v_secs) else 0
    return events, runs, vsec, iso, int(isv.sum())


def evaluate(flags, vsec, events, runs, n_sec):
    entries, alarm = alarm_sim(flags, **RULE)
    cs = np.concatenate([[0], np.cumsum(vsec)])
    fp = [t for t in entries if cs[t + 1] - cs[max(0, t - RULE["win"] + 1)] == 0]
    det, lat = 0, []
    for e in events:
        a, b = e["onset"], min(n_sec - 1, e["end"] + GRACE_S)
        seg = alarm[a:b + 1]
        if seg.any():
            det += 1; lat.append(int(np.argmax(seg)))
    rdet = 0
    for r in runs:
        a, b = r["start"], min(n_sec - 1, r["end"] + GRACE_S)
        rdet += int(alarm[a:b + 1].any())
    return {"n_entries": len(entries), "n_fp": len(fp), "n_events": len(events), "n_det": det,
            "lat": lat, "n_runs": len(runs), "n_runs_det": rdet,
            "n_runs_undet_spec": sum(1 for r in runs if not r["in_spec_event"]),
            "n_runs_undet_spec_det": sum(int(alarm[r["start"]:min(n_sec - 1, r["end"] + GRACE_S) + 1].any())
                                         for r in runs if not r["in_spec_event"]),
            "hours": n_sec / 3600.0}


def flags_from(peaks, scores, n_sec, thr):
    f = np.zeros(n_sec, bool)
    sec = peaks // 250
    m = (scores >= thr) & (sec < n_sec)
    f[sec[m]] = True
    return f

# ---------------------------------------------------------------- models
class Scorer:
    def __init__(self):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        self.tf = tf; self.m = {}
        for k, (kind, p) in MODELS.items():
            if not Path(p).exists():
                print("MISSING model", k, p, flush=True); continue
            if kind == "keras":
                self.m[k] = ("keras", tf.keras.models.load_model(str(p), compile=False))
            else:
                it = tf.lite.Interpreter(model_path=str(p), num_threads=2); it.allocate_tensors()
                self.m[k] = ("tflite", (it, it.get_input_details()[0], it.get_output_details()[0]))

    def score(self, k, beats):
        kind, m = self.m[k]
        if len(beats) == 0:
            return np.zeros(0, np.float32)
        if kind == "keras":
            out = np.asarray(m.predict(beats[..., None], batch_size=512, verbose=0)).reshape(len(beats), -1)
            return (out[:, -1] if out.shape[1] > 1 else out[:, 0]).astype(np.float32)
        it, inp, outd = m
        sc = []
        scale, zp = inp["quantization"] if inp["dtype"] == np.int8 else (None, None)
        for b in beats:
            if inp["dtype"] == np.int8:
                it.set_tensor(inp["index"], np.clip(np.round(b / scale + zp), -128, 127).astype(np.int8).reshape(inp["shape"]))
            else:
                it.set_tensor(inp["index"], b.reshape(inp["shape"]).astype(inp["dtype"]))
            it.invoke()
            o = it.get_tensor(outd["index"]).reshape(-1)
            if outd["dtype"] == np.int8:
                os_, oz = outd["quantization"]; o = (o.astype(np.float32) - oz) * os_
            sc.append(float(o[-1] if len(o) > 1 else o[0]))
        return np.asarray(sc, np.float32)

# ---------------------------------------------------------------- data
def load_nstdb(name):
    r = wfdb.rdrecord(str(NST_DIR / name))
    return r.p_signal[:, 0].astype(np.float64), float(r.fs)


def add_noise(sig, fs, noise, nfs, snr_db, seed):
    n = resample_ecg(noise, nfs, fs) if abs(nfs - fs) > 1e-6 else noise.copy()
    rng = np.random.default_rng(seed)
    if len(n) < len(sig):
        n = np.tile(n, int(np.ceil(len(sig) / len(n))) + 1)
    off = int(rng.integers(0, len(n) - len(sig) + 1))
    n = n[off:off + len(sig)]
    n = n - n.mean()
    ps = np.var(sig - np.mean(sig)); pn = np.var(n)
    return sig + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))


def process(rec_id, cond, raw, fs, gt_pk, gt_cls, scorer, det="fw"):
    cpath = CACHE / f"{rec_id}__{cond}__{det}.npz"
    if cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        return {"peaks": z["peaks"], "n_sec": int(z["n_sec"]),
                "scores": {k: z["s_" + k] for k in scorer_keys() if "s_" + k in z.files}}
    sig250 = causal_filter_chain(resample_ecg(raw, fs))
    n_sec = len(sig250) // 250
    pk = fw_rpeaks(raw, fs) if det == "fw" else simple_rpeaks(sig250)
    beats, used = beats_at(sig250, pk)
    sc = {k: scorer.score(k, beats) for k in scorer.m}
    np.savez_compressed(cpath, peaks=used, n_sec=n_sec, **{"s_" + k: v for k, v in sc.items()})
    return {"peaks": used, "n_sec": n_sec, "scores": sc}


_KEYS = list(MODELS.keys())
def scorer_keys():
    return _KEYS


def detector_stats(det_pk, gt_pk, gt_cls, tol=37):  # 150 ms @250
    if len(gt_pk) == 0:
        return {}
    j = np.searchsorted(det_pk, gt_pk)
    d = np.full(len(gt_pk), 10**9)
    for o in (0, -1):
        jj = np.clip(j + o, 0, max(0, len(det_pk) - 1))
        if len(det_pk):
            d = np.minimum(d, np.abs(det_pk[jj] - gt_pk))
    hit = d <= tol
    return {"n_gt": int(len(gt_pk)), "n_det": int(len(det_pk)), "Se_all": float(hit.mean()),
            "Se_V": float(hit[gt_cls == 2].mean()) if (gt_cls == 2).any() else float("nan"),
            "PPV_approx": float(min(1.0, hit.sum() / max(1, len(det_pk))))}

# ---------------------------------------------------------------- FM approx
def fm_probe_scores_incart():
    """ECG-FM center_step LogReg probe; cached embeddings exist only at GT-annotated peaks
    (script 10). Detected peaks are mapped to nearest GT peak (<=37 samples); unmatched -> 0.
    => APPROXIMATION, optimistic for FM (uses GT-aligned window), clean INCART only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    z = np.load(DATA / "ecg_fm_incart_emb.npz", allow_pickle=True)
    m = z["y_train"] >= 0
    scl = StandardScaler().fit(z["center_train"][m])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(scl.transform(z["center_train"][m]), z["y_train"][m])
    E = np.load(DATA / "ecg_fm_incart_test_replay_emb.npz", allow_pickle=True)
    return clf, scl, E

# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    scorer = Scorer()
    global _KEYS
    _KEYS = list(scorer.m.keys())
    recs = []   # list of dict(rec, db, cond, gt..., out)
    det_rows = []

    noise = {}
    for nm in NOISES:
        try:
            noise[nm] = load_nstdb(nm)
        except Exception as e:
            print("NSTDB missing", nm, e, flush=True)

    # INCART test
    for rid in TEST_RIDS:
        name = f"I{rid:02d}"
        r = wfdb.rdrecord(str(INCART_DIR / name)); a = wfdb.rdann(str(INCART_DIR / name), "atr")
        raw = r.p_signal[:, 0].astype(np.float64); fs = float(r.fs)
        gt_pk, gt_cls = gt_beats(a.sample, a.symbol, fs)
        conds = [("clean", raw)]
        for nm, (nsig, nfs) in noise.items():
            for snr in SNRS:
                conds.append((f"{nm}{snr}dB", add_noise(raw, fs, nsig, nfs, snr, seed=rid * 100 + snr + (0 if nm == "em" else 7))))
        conds.insert(1, ("clean_proxyDet", raw))
        for cond, x in conds:
            print(f"[{time.time()-t0:7.0f}s] INCART {name} {cond}", flush=True)
            out = process(name, cond, x, fs, gt_pk, gt_cls, scorer, det="proxy" if cond.endswith("proxyDet") else "fw")
            recs.append({"db": "INCART_LeadI", "rec": name, "cond": cond, "gt_pk": gt_pk, "gt_cls": gt_cls, **out})
            det_rows.append({"db": "INCART_LeadI", "rec": name, "cond": cond, **detector_stats(out["peaks"], gt_pk, gt_cls)})
    # MIT-BIH test (MLII, cross-lead)
    for nm in MIT_TEST:
        p = MIT_DIR / nm
        dat = MIT_DIR / f"{nm}.dat"
        if not p.with_suffix(".hea").exists() or not dat.exists() or dat.stat().st_size < 1950000:
            print("MIT missing", nm, flush=True); continue
        r = wfdb.rdrecord(str(p)); a = wfdb.rdann(str(p), "atr")
        ch = [s.upper() for s in r.sig_name].index("MLII") if "MLII" in [s.upper() for s in r.sig_name] else 0
        raw = r.p_signal[:, ch].astype(np.float64); fs = float(r.fs)
        gt_pk, gt_cls = gt_beats(a.sample, a.symbol, fs)
        print(f"[{time.time()-t0:7.0f}s] MIT {nm} ch={r.sig_name[ch]}", flush=True)
        for cond in ["clean", "clean_proxyDet"]:
            out = process(f"mit{nm}", cond, raw, fs, gt_pk, gt_cls, scorer, det="proxy" if cond.endswith("proxyDet") else "fw")
            recs.append({"db": f"MITBIH_{r.sig_name[ch]}_crosslead", "rec": f"mit{nm}", "cond": cond, "gt_pk": gt_pk, "gt_cls": gt_cls, **out})
            det_rows.append({"db": "MITBIH", "rec": nm, "cond": cond, **detector_stats(out["peaks"], gt_pk, gt_cls)})
    # PTB Lead I controls (normal-only, full length)
    ctrl = PTB_DIR / "CONTROLS"
    pats = []
    if ctrl.exists():
        pats = [l.strip() for l in ctrl.read_text(errors="ignore").splitlines() if l.strip().startswith("patient")]
    for line in pats:
        stem = PTB_DIR / line if "/" in line else None
        if stem is None:
            heas = sorted((PTB_DIR / line).glob("*.hea"))
            if not heas: continue
            stem = heas[0].with_suffix("")
        try:
            r = wfdb.rdrecord(str(stem))
        except Exception:
            continue
        names = [s.upper() for s in r.sig_name]
        if "I" not in names: continue
        raw = r.p_signal[:, names.index("I")].astype(np.float64)
        key = "ptb_" + str(stem.relative_to(PTB_DIR)).replace("/", "_")
        out = process(key, "clean", raw, float(r.fs), np.zeros(0, np.int64), np.zeros(0, np.int32), scorer)
        recs.append({"db": "PTB_LeadI_control", "rec": key, "cond": "clean", "gt_pk": np.zeros(0, np.int64), "gt_cls": np.zeros(0, np.int32), **out})
    print(f"PTB controls: {sum(1 for r in recs if r['db'].startswith('PTB'))}", flush=True)
    # Real device sessions (~100 Hz serial; no annotations)
    import pandas as pd
    for fn in ["serial_20261019_145244_leadI_standard_AD8232.csv", "serial_20261019_155455_leadI_standard_AD8232.csv",
               "serial_20261019_143011_wearable_Lchest_horiz_AD8232.csv", "serial_20261019_143825_wearable_Lchest_horiz_AD8232.csv"]:
        path = INBOX / fn
        if not path.exists():
            print("real missing", fn, flush=True); continue
        df = pd.read_csv(path)
        if "filtered" in df.columns:
            x = df["filtered"].to_numpy(np.float64)
        else:
            x = df["raw_line"].astype(str).str.split(",", expand=True)[2].astype(float).to_numpy(np.float64)
        fs = 100.0
        mp = path.with_name(path.stem + "_meta.json")
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8")); dur = meta.get("duration_s_actual") or meta.get("duration_s")
            if dur: fs = len(x) / float(dur)
        key = "real_" + fn.split("_")[2]
        for cond in ["clean", "clean_proxyDet"]:
            out = process(key, cond, x, fs, np.zeros(0, np.int64), np.zeros(0, np.int32), scorer, det="proxy" if cond.endswith("proxyDet") else "fw")
            recs.append({"db": "REAL_" + ("leadI" if "leadI" in fn else "wearable"), "rec": key, "cond": cond,
                         "gt_pk": np.zeros(0, np.int64), "gt_cls": np.zeros(0, np.int32), **out})

    # FM approx scores on INCART clean
    fm_ok = False
    try:
        clf, scl, E = fm_probe_scores_incart()
        for r in recs:
            if r["db"] != "INCART_LeadI" or r["cond"] not in ("clean", "clean_proxyDet"): continue
            key = f"{r['rec']}_ctr_r"
            if key not in E.files: continue
            # reproduce script-10 GT window order: AAMI beats with full window in bounds and std>0
            emb = E[key]
            gtp = r["gt_pk"]
            gtp_ok = gtp[(gtp - 125 >= 0)]
            if len(gtp_ok) != len(emb):
                print(f"  FM align {r['rec']}: gt={len(gtp_ok)} emb={len(emb)} (trim tail)", flush=True)
                if len(gtp_ok) - len(emb) > 3 or len(gtp_ok) < len(emb):
                    print("  FM align mismatch too large, skip", r['rec'], flush=True); continue
                gtp_ok = gtp_ok[:len(emb)]
            s_fm_gt = clf.predict_proba(scl.transform(emb[:len(gtp_ok)]))[:, 1].astype(np.float32)
            dp = r["peaks"]
            j = np.clip(np.searchsorted(gtp_ok, dp), 0, len(gtp_ok) - 1)
            jm = np.clip(j - 1, 0, len(gtp_ok) - 1)
            best = np.where(np.abs(gtp_ok[jm] - dp) < np.abs(gtp_ok[j] - dp), jm, j)
            s = np.where(np.abs(gtp_ok[best] - dp) <= 37, s_fm_gt[best], 0.0).astype(np.float32)
            r["scores"]["ecgfm_probe_approx"] = s
        fm_ok = True
    except Exception as e:
        print("FM approx skipped:", e, flush=True)

    # ------------------------------------------------ evaluation
    for r in recs:
        ev, runs, vsec, iso, nV = define_events(r["gt_pk"], r["gt_cls"], r["n_sec"])
        r.update(events=ev, runs=runs, vsec=vsec, iso=iso, nV=nV)
    models = list(scorer.m.keys()) + (["ecgfm_probe_approx"] if fm_ok else [])

    rows = []  # per model, db, cond, rec, thr
    for r in recs:
        for mdl in models:
            if mdl not in r["scores"]: continue
            for thr in THR_GRID:
                f = flags_from(r["peaks"], r["scores"][mdl], r["n_sec"], thr)
                e = evaluate(f, r["vsec"], r["events"], r["runs"], r["n_sec"])
                rows.append({"model": mdl, "db": r["db"], "cond": r["cond"], "rec": r["rec"], "thr": thr, **e})

    def agg(sel, boot=True):
        if not sel: return None
        recs_ = sorted(set(x["rec"] for x in sel))
        by = {k: [x for x in sel if x["rec"] == k] for k in recs_}
        def stat(keys):
            ne = sum(x["n_events"] for k in keys for x in by[k]); nd = sum(x["n_det"] for k in keys for x in by[k])
            h = sum(x["hours"] for k in keys for x in by[k]); fp = sum(x["n_fp"] for k in keys for x in by[k])
            return (nd / ne if ne else np.nan), (fp / h if h else np.nan)
        se, fph = stat(recs_)
        lat = [l for x in sel for l in x["lat"]]
        out = {"n_rec": len(recs_), "hours": sum(x["hours"] for x in sel), "n_events": sum(x["n_events"] for x in sel),
               "n_det": sum(x["n_det"] for x in sel), "sens": se, "n_fp": sum(x["n_fp"] for x in sel), "fph": fph,
               "lat_med": float(np.median(lat)) if lat else np.nan, "lat_p95": float(np.percentile(lat, 95)) if lat else np.nan,
               "n_runs": sum(x["n_runs"] for x in sel), "n_runs_det": sum(x["n_runs_det"] for x in sel),
               "n_runs_undet_spec": sum(x["n_runs_undet_spec"] for x in sel),
               "n_runs_undet_spec_det": sum(x["n_runs_undet_spec_det"] for x in sel)}
        if boot and len(recs_) > 1:
            bs, bf = [], []
            for _ in range(NBOOT):
                ks = list(RNG.choice(recs_, len(recs_), replace=True))
                s_, f_ = stat(ks); bs.append(s_); bf.append(f_)
            bs = np.asarray(bs, float); bf = np.asarray(bf, float)
            out["sens_ci"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))] if np.isfinite(bs).any() else [np.nan, np.nan]
            out["fph_ci"] = [float(np.nanpercentile(bf, 2.5)), float(np.nanpercentile(bf, 97.5))]
        else:
            out["sens_ci"] = [np.nan, np.nan]; out["fph_ci"] = [np.nan, np.nan]
        return out

    summary = []
    groups = sorted(set((x["db"], x["cond"]) for x in rows))
    for mdl in models:
        for db, cond in groups:
            for thr in THR_GRID:
                sel = [x for x in rows if x["model"] == mdl and x["db"] == db and x["cond"] == cond and x["thr"] == thr]
                a = agg(sel, boot=(thr == THR))
                if a: summary.append({"model": mdl, "db": db, "cond": cond, "thr": thr, **a})

    # event inventory
    inv = []
    for r in recs:
        if r["cond"] != "clean": continue
        h = r["n_sec"] / 3600
        inv.append({"db": r["db"], "rec": r["rec"], "hours": round(h, 3), "n_V": r["nV"], "spec_events": len(r["events"]),
                    "vt_runs": len(r["runs"]), "vt_runs_outside_spec": sum(1 for x in r["runs"] if not x["in_spec_event"]),
                    "isolated_V": r["iso"], "isolated_V_per_h": round(r["iso"] / h, 1) if h else None})

    # write CSVs
    def wcsv(p, rows_):
        if not rows_: return
        keys = list(rows_[0].keys())
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for x in rows_:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in x.items()})
    wcsv(OUTD / "summary.csv", summary)
    wcsv(OUTD / "per_record.csv", [{k: v for k, v in x.items() if k != "lat"} | {"lat": json.dumps(x["lat"])} for x in rows])
    wcsv(OUTD / "event_inventory.csv", inv)
    wcsv(OUTD / "detector_stats.csv", det_rows)

    # ranking (gate on INCART clean FP/h <=1 at thr=0.5; rank by INCART clean sensitivity)
    def get(mdl, db, cond, thr=THR):
        for s in summary:
            if s["model"] == mdl and s["db"] == db and s["cond"] == cond and s["thr"] == thr: return s
    rank = []
    mitdb = next((g[0] for g in groups if g[0].startswith("MITBIH")), None)
    for mdl in models:
        ic = get(mdl, "INCART_LeadI", "clean")
        if not ic: continue
        noise_fph = {c: (get(mdl, "INCART_LeadI", c) or {}).get("fph") for c in [f"{n}{s}dB" for n in NOISES for s in SNRS]}
        noise_sens = {c: (get(mdl, "INCART_LeadI", c) or {}).get("sens") for c in [f"{n}{s}dB" for n in NOISES for s in SNRS]}
        mit = get(mdl, mitdb, "clean") if mitdb else None
        incP = get(mdl, "INCART_LeadI", "clean_proxyDet")
        mitP = get(mdl, mitdb, "clean_proxyDet") if mitdb else None
        realLP = get(mdl, "REAL_leadI", "clean_proxyDet"); realWP = get(mdl, "REAL_wearable", "clean_proxyDet")
        ptb = get(mdl, "PTB_LeadI_control", "clean")
        realL = get(mdl, "REAL_leadI", "clean"); realW = get(mdl, "REAL_wearable", "clean")
        # best threshold satisfying FP/h<=1 on INCART clean
        best = None
        for thr in THR_GRID:
            s = get(mdl, "INCART_LeadI", "clean", thr)
            if s and s["fph"] <= 1.0 and (best is None or s["sens"] > best[1]): best = (thr, s["sens"], s["fph"])
        rank.append({"model": mdl, "leak_flag": LEAK.get(mdl, "") + (" | APPROX: GT-aligned FM emb, clean INCART only" if mdl.startswith("ecgfm") else ""),
                     "incart": ic, "mit": mit, "incart_proxyDet": incP, "mit_proxyDet": mitP, "realL_proxyDet": realLP, "realW_proxyDet": realWP, "ptb": ptb, "realL": realL, "realW": realW,
                     "noise_fph": noise_fph, "noise_sens": noise_sens,
                     "gate_pass_thr050": bool(ic["fph"] <= 1.0), "best_thr_under_gate": best})
    rank.sort(key=lambda x: (not x["gate_pass_thr050"], -(x["incart"]["sens"] if np.isfinite(x["incart"]["sens"]) else -1)))
    (OUTD / "ranking.json").write_text(json.dumps({"generated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rule": RULE, "thr": THR,
        "grace_s": GRACE_S, "rank": rank, "inventory": inv, "elapsed_s": round(time.time() - t0, 1)}, indent=1, default=float, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------ figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = plt.cm.tab10(np.arange(len(models)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (db, cond, ttl) in zip(axes, [("INCART_LeadI", "clean", "INCART Lead I test (clean)"), ("INCART_LeadI", "em6dB", "INCART + NSTDB em @6 dB")]):
        for c, mdl in zip(cols, models):
            pts = [(s["fph"], s["sens"], s["thr"]) for s in summary if s["model"] == mdl and s["db"] == db and s["cond"] == cond]
            if not pts: continue
            pts.sort(key=lambda p: p[2])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=c, ms=3, label=mdl)
            p5 = [p for p in pts if p[2] == THR]
            if p5: ax.plot(p5[0][0], p5[0][1], "*", color=c, ms=13, mec="k")
        ax.axvline(1.0, color="r", ls="--", lw=1); ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xlabel("FP alarm entries / hour (symlog)"); ax.set_ylabel("Event sensitivity (spec events)")
        ax.set_title(ttl + "\nthreshold sweep 0.3–0.98, ★ = 0.50; red = FP/h gate 1"); ax.grid(alpha=.3); ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "event_sens_vs_fph_tradeoff.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    conds = ["clean"] + [f"{n}{s}dB" for n in NOISES for s in SNRS]
    x = np.arange(len(conds)); w = 0.8 / len(models)
    for i, (c, mdl) in enumerate(zip(cols, models)):
        f = [(get(mdl, "INCART_LeadI", cd) or {}).get("fph", np.nan) for cd in conds]
        s = [(get(mdl, "INCART_LeadI", cd) or {}).get("sens", np.nan) for cd in conds]
        axes[0].bar(x + i * w, f, w, color=c, label=mdl); axes[1].bar(x + i * w, s, w, color=c, label=mdl)
    for ax, yl in zip(axes, ["FP/h @0.50", "Event sensitivity @0.50"]):
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(conds, rotation=20); ax.set_ylabel(yl); ax.grid(axis="y", alpha=.3)
    axes[0].axhline(1, color="r", ls="--"); axes[0].set_yscale("symlog", linthresh=1); axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7); fig.suptitle("INCART Lead I test · full chain (own R-peak) · noise stress (NSTDB)")
    fig.tight_layout(); fig.savefig(FIG / "event_sens_fph_noise_stress.png", dpi=140); plt.close(fig)
    print("DONE elapsed", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
