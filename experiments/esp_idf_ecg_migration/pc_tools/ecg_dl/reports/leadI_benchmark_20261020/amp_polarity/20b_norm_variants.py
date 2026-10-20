"""Part B: amplitude-normalisation variants in PC replay (aiqrs peaks), models student_hard/distill_v2/exp5_int8.
Variants are fixed a priori (constants from the TRAINING set only; nothing tuned on INCART test):
  z       : firmware path (per-window z-score)                       [baseline]
  gain0.5 : signal x0.5 before z-score (sanity: gain is cancelled by z-score)
  realign : re-centre window on max|x| within +-60 ms, then z-score
  ctx4s   : (w - mean_ctx)/std_ctx, ctx = 4 s ending at window end (causal-implementable)
  rscale  : (w - median(w)) / |R| * 5.48  (5.48 = median |R| of training N windows in z units)
Also simulates the 100 Hz serial capture path on INCART (500 Hz -> every 5th sample -> replay resample)
to measure how much the capture/replay itself changes the model input.
Outputs results/amp_analysis/variants_*.csv, gallery data npz."""
import os, json, time, importlib.util
os.environ["DET"] = "aiqrs"; os.environ.setdefault("MODELS_SUBSET", "student_hard,distill_v2,exp5_int8")
from pathlib import Path
import numpy as np, pandas as pd, wfdb
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s13", HERE / "13_event_sensitivity_fph.py"); s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
B = s13.BENCH; OUT = B / "results/amp_analysis"; OUT.mkdir(exist_ok=True, parents=True)
R_TRAIN = 5.48
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

def windows(sig, pk, var):
    half, W = 125, 250; out, used, oi = [], [], []
    x = sig * (0.5 if var == "gain0.5" else 1.0)
    for k0, p in enumerate(pk):
        p = int(p)
        if var == "realign":
            lo_, hi_ = max(0, p - 15), min(len(x), p + 16); p = lo_ + int(np.argmax(np.abs(x[lo_:hi_])))
        lo, hi = p - half, p - half + W
        if lo < 0 or hi > len(x): continue
        seg = x[lo:hi].astype(np.float64)
        if var == "ctx4s":
            ctx = x[max(0, hi - 1000):hi]; s = ctx.std(); m = ctx.mean()
        elif var == "rscale":
            m = np.median(seg); s = np.abs(seg[120:131] - m).max() / R_TRAIN
        else:
            m = seg.mean(); s = seg.std()
        if s < 1e-8: continue
        out.append(((seg - m) / s).astype(np.float32)); used.append(p); oi.append(k0)
    windows.oi = np.asarray(oi, int)
    return (np.stack(out) if out else np.zeros((0, W), np.float32)), np.asarray(used, np.int64)

VARS = ["z", "gain0.5", "realign", "ctx4s", "rscale"]
scorer = s13.Scorer(); MODELS = list(scorer.m.keys()); log("models", MODELS)
def score_all(Wn): return {m: scorer.score(m, Wn) for m in MODELS}

# ---------- real sessions
FILES = {"real_145244": "serial_20261019_145244_leadI_standard_AD8232.csv", "real_155455": "serial_20261019_155455_leadI_standard_AD8232.csv",
         "real_143011": "serial_20261019_143011_wearable_Lchest_horiz_AD8232.csv", "real_143825": "serial_20261019_143825_wearable_Lchest_horiz_AD8232.csv"}
lab = json.loads(sorted((B / "results/labels").glob("ecg_labels_*.json"))[-1].read_text(encoding="utf-8"))["sessions"]
rows, gal = [], {}
for key, fn in FILES.items():
    df = pd.read_csv(s13.INBOX / fn); disp = df["raw_line"].astype(str).str.split(",", expand=True)[2].astype(float).to_numpy()
    meta = json.loads((s13.INBOX / fn.replace(".csv", "_meta.json")).read_text(encoding="utf-8")); fs = len(disp) / float(meta["duration_s_actual"])
    sig = s13.causal_filter_chain(s13.resample_ecg(disp, fs)); pk = np.load(s13.CACHE / f"{key}__clean__aiqrs.npz")["peaks"]
    L = lab.get(key, {}).get("beats", {})
    for v in VARS:
        Wn, used = windows(sig, pk, v); sc = score_all(Wn)
        lb = np.array([L.get(str(int(pk[k])), "") for k in windows.oi])
        for m in MODELS:
            isN = lb == "N"
            rows.append({"set": key, "variant": v, "model": m, "n_beats": len(Wn), "flag_rate_all": float((sc[m] >= .5).mean()),
                         "n_labN": int(isN.sum()), "Vcall_rate_labN": float((sc[m][isN] >= .5).mean()) if isN.any() else np.nan})
        if key == "real_145244" and v == "z":
            gal["real_W"] = Wn; gal["real_lab"] = lb; gal["real_t"] = used / 250
            for m in MODELS: gal["real_s_" + m] = sc[m]
    log(key, "done")
pd.DataFrame(rows).to_csv(OUT / "variants_real.csv", index=False)

# ---------- INCART frozen test: events & FP/h with aiqrs peaks; beat-level N V-call / V Se at GT peaks
erows, brows, inc_gal = [], [], {"W": [], "s": {m: [] for m in MODELS}, "rec": []}
for rid in s13.TEST_RIDS:
    nm = f"I{rid:02d}"; r = wfdb.rdrecord(str(s13.INCART_DIR / nm)); a = wfdb.rdann(str(s13.INCART_DIR / nm), "atr")
    x = r.p_signal[:, 0].astype(np.float64); fs = float(r.fs); gp, gc = s13.gt_beats(a.sample, a.symbol, fs)
    sig = s13.causal_filter_chain(s13.resample_ecg(x, fs)); n_sec = len(sig) // 250
    pk = np.load(s13.CACHE / f"{nm}__clean__aiqrs.npz")["peaks"]
    events, runs, vsec, _, _ = s13.define_events(gp, gc, n_sec)
    # 100 Hz serial-path simulation (beat level, GT peaks)
    x500 = s13.resample_ecg(x, fs, 500); x100 = x500[::5]; sig100 = s13.causal_filter_chain(s13.resample_ecg(x100, 100.0))
    for v in VARS:
        Wn, used = windows(sig, pk, v); sc = score_all(Wn)
        for m in MODELS:
            ev = s13.evaluate(s13.flags_from(used, sc[m], n_sec, .5), vsec, events, runs, n_sec)
            erows.append({"rec": nm, "variant": v, "model": m, **{k: ev[k] for k in ["n_entries", "n_fp", "n_events", "n_det", "hours"]}})
    for path, sg in [("direct250", sig), ("serial100", sig100)]:
        for v in (["z", "realign", "ctx4s", "rscale"] if path == "serial100" else VARS):
            Wn, used = windows(sg, gp, v); cl = gc[windows.oi]; sc = score_all(Wn)
            for m in MODELS:
                brows.append({"rec": nm, "path": path, "variant": v, "model": m, "nN": int((cl == 0).sum()), "nV": int((cl == 2).sum()),
                              "Vcall_N": float((sc[m][cl == 0] >= .5).mean()), "Se_V": float((sc[m][cl == 2] >= .5).mean()) if (cl == 2).any() else np.nan})
            if path == "direct250" and v == "z":
                hv = np.where((cl == 2) & np.all([sc[m] >= .9 for m in MODELS], 0))[0][:3]
                inc_gal["W"] += list(Wn[hv]); inc_gal["rec"] += [nm] * len(hv)
                for m in MODELS: inc_gal["s"][m] += list(sc[m][hv])
    log(nm, "done")
pd.DataFrame(erows).to_csv(OUT / "variants_incart_events.csv", index=False); pd.DataFrame(brows).to_csv(OUT / "variants_incart_beats.csv", index=False)

# ---------- PTB controls FP/h
prow = []
ctrl = s13.PTB_DIR / "CONTROLS"; pats = [l.strip() for l in ctrl.read_text(errors="ignore").splitlines() if l.strip().startswith("patient")]
for line in pats:
    heas = sorted((s13.PTB_DIR / line).glob("*.hea")) if "/" not in line else [s13.PTB_DIR / (line + ".hea")]
    if not heas: continue
    stem = heas[0].with_suffix("")
    try: r = wfdb.rdrecord(str(stem))
    except Exception: continue
    names = [s.upper() for s in r.sig_name]
    if "I" not in names: continue
    key = "ptb_" + str(stem.relative_to(s13.PTB_DIR)).replace("/", "_"); cp = s13.CACHE / f"{key}__clean__aiqrs.npz"
    if not cp.exists(): continue
    sig = s13.causal_filter_chain(s13.resample_ecg(r.p_signal[:, names.index("I")].astype(np.float64), float(r.fs))); n_sec = len(sig) // 250
    pk = np.load(cp)["peaks"]
    for v in VARS:
        Wn, used = windows(sig, pk, v); sc = score_all(Wn)
        for m in MODELS:
            ent, _ = s13.alarm_sim(s13.flags_from(used, sc[m], n_sec, .5), **s13.RULE)
            prow.append({"rec": key, "variant": v, "model": m, "n_entries": len(ent), "hours": n_sec / 3600, "beat_flag": float((sc[m] >= .5).mean())})
log("PTB", len(prow) // (len(VARS) * len(MODELS)), "records")
pd.DataFrame(prow).to_csv(OUT / "variants_ptb.csv", index=False)
tz = np.load(B / "data/student_train_leadI.npz", allow_pickle=True)
np.savez_compressed(OUT / "_gallery_tmp.npz", inc_W=np.array(inc_gal["W"]), inc_rec=np.array(inc_gal["rec"]),
                    **{f"inc_s_{m}": np.array(v) for m, v in inc_gal["s"].items()}, **gal,
                    trainN=np.median(tz["beats"][tz["labels_nsvq"] == 0][::3], 0), trainV=np.median(tz["beats"][tz["labels_nsvq"] == 2], 0))
# ---------- summary
E = pd.DataFrame(erows).groupby(["variant", "model"]).sum(numeric_only=True); E["sens"] = E.n_det / E.n_events; E["FPh"] = E.n_fp / E.hours
P = pd.DataFrame(prow).groupby(["variant", "model"]).sum(numeric_only=True); P["PTB_FPh"] = P.n_entries / P.hours
R = pd.DataFrame(rows); R1 = R[R.set == "real_145244"].set_index(["variant", "model"])["Vcall_rate_labN"]
Rall = R.groupby(["variant", "model"]).flag_rate_all.mean()
Bt = pd.DataFrame(brows); Bt["nVc"] = Bt.Vcall_N * Bt.nN; Bt["nVs"] = Bt.Se_V.fillna(0) * Bt.nV
Bg = Bt.groupby(["path", "variant", "model"])[["nVc", "nN", "nVs", "nV"]].sum(); Bg["Vcall_N"] = Bg.nVc / Bg.nN; Bg["SeV"] = Bg.nVs / Bg.nV
S = pd.concat([R1.rename("real145244_Ncall"), Rall.rename("real4_flag_mean"), E[["sens", "FPh"]].rename(columns={"sens": "INCART_event_sens", "FPh": "INCART_FPh"}), P["PTB_FPh"],
               Bg.loc["direct250"][["Vcall_N", "SeV"]].rename(columns={"Vcall_N": "INCART_beat_Ncall", "SeV": "INCART_beat_SeV"})], axis=1)
S.to_csv(OUT / "variants_summary.csv"); print(S.round(3).to_string())
print(Bg[["Vcall_N", "SeV"]].round(3).to_string())
log("DONE")
