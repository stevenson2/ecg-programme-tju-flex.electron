"""Part A: amplitude / clipping / baseline / spectrum stats, real sessions vs training & INCART (no models)."""
import os, json, importlib.util
os.environ["DET"] = "aiqrs"
from pathlib import Path
import numpy as np, pandas as pd, wfdb
from scipy import signal as sp
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s13", HERE / "13_event_sensitivity_fph.py"); s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
B = s13.BENCH; OUT = B / "results/amp_analysis"; OUT.mkdir(exist_ok=True, parents=True)
FILES = {"real_145244": "serial_20261019_145244_leadI_standard_AD8232.csv", "real_155455": "serial_20261019_155455_leadI_standard_AD8232.csv",
         "real_143011": "serial_20261019_143011_wearable_Lchest_horiz_AD8232.csv", "real_143825": "serial_20261019_143825_wearable_Lchest_horiz_AD8232.csv"}
z = np.load(B / "data/student_train_leadI.npz", allow_pickle=True)
print("train notes:", str(z["notes"])[:600])
li = np.load(B / "data/leadI_incart_beats.npz", allow_pickle=True); print("incart beats filter:", li["filter"], "fs", li["fs"], "window", li["window"])
tb, tl = z["beats"], z["labels_nsvq"]
def wstats(W):
    c = W[:, 120:131]; r = c[np.arange(len(W)), np.abs(c).argmax(1)]
    return {"R_center_z_med": float(np.median(np.abs(r))), "R_center_z_p10": float(np.percentile(np.abs(r), 10)), "R_center_z_p90": float(np.percentile(np.abs(r), 90)),
            "maxabs_z_med": float(np.median(np.abs(W).max(1))), "R_pos_frac": float((r > 0).mean()),
            "edge_trend_med": float(np.median(np.abs(W[:, -10:].mean(1) - W[:, :10].mean(1))))}
rows = []
for c, nm in [(0, "train_N"), (2, "train_V")]:
    rows.append({"set": nm, "n": int((tl == c).sum()), **wstats(tb[tl == c][::5])})
real = {}
for key, fn in FILES.items():
    df = pd.read_csv(s13.INBOX / fn); cols = df["raw_line"].astype(str).str.split(",", expand=True).iloc[:, :3].astype(float).to_numpy()
    meta = json.loads((s13.INBOX / fn.replace(".csv", "_meta.json")).read_text(encoding="utf-8")); fs = len(cols) / float(meta["duration_s_actual"])
    adc = cols[:, 1] + 1.65          # noisyNoDC + DC_OFFSET_REMOVE = ADC volts (cal.)
    disp = cols[:, 2]
    lo, hi = adc <= 0.05, adc >= 3.25
    sat = lo | hi
    segs = []; i = 0
    while i < len(sat):
        if sat[i]:
            j = i
            while j + 1 < len(sat) and sat[j + 1]: j += 1
            segs.append([round(i / fs, 2), round((j + 1) / fs, 2), "low" if lo[i] else "high"]); i = j + 1
        else: i += 1
    sig = s13.causal_filter_chain(s13.resample_ecg(disp, fs))
    c = np.load(s13.CACHE / f"{key}__clean__aiqrs.npz"); pk = c["peaks"]
    W, _ = s13.beats_at(sig, pk)
    # raw R amplitude (V at ADC) from display col @fs: peak-to-peak in +-80ms around peak
    pr = (pk / 250 * fs).astype(int); h = int(0.08 * fs)
    rpp = np.array([np.ptp(disp[max(0, p - h):p + h]) for p in pr])
    bw = sp.filtfilt(*sp.butter(2, 0.5 / (fs / 2)), disp)  # baseline wander (<0.5 Hz)
    real[key] = dict(fs=fs, sig=sig, disp=disp, adc=adc, pk=pk, W=W, segs=segs)
    rows.append({"set": key, "n": len(W), "fs_capture": round(fs, 2), **wstats(W),
                 "adc_min_V": float(adc.min()), "adc_max_V": float(adc.max()), "adc_p1_V": float(np.percentile(adc, 1)), "adc_p99_V": float(np.percentile(adc, 99)),
                 "sat_frac": float(sat.mean()), "sat_low_frac": float(lo.mean()), "sat_high_frac": float(hi.mean()), "n_sat_segments": len(segs),
                 "sat_s_total": round(float(sat.sum() / fs), 2), "R_pp_V_med": float(np.median(rpp)), "R_pp_mV_input_if_gain100": float(np.median(rpp) * 10),
                 "baseline_std_V": float(bw.std()), "baseline_std_over_Rpp": float(bw.std() / np.median(rpp)), "baseline_range_V": float(np.ptp(bw))})
    print(key, "fs", round(fs, 2), "sat segments", segs[:12])
# INCART test raw (Lead I)
inc = {}
for rid in s13.TEST_RIDS:
    nm = f"I{rid:02d}"; r = wfdb.rdrecord(str(s13.INCART_DIR / nm)); a = wfdb.rdann(str(s13.INCART_DIR / nm), "atr")
    x = r.p_signal[:, 0]; fs = r.fs; gp, gc = s13.gt_beats(a.sample, a.symbol, fs)
    sig = s13.causal_filter_chain(s13.resample_ecg(x, fs)); W, used = s13.beats_at(sig, gp)
    cl = gc[np.isin(gp, used)]
    pr = (a.sample[np.isin(a.symbol, ["N"])]).astype(int); h = int(.08 * fs)
    rpp = np.array([np.ptp(x[max(0, p - h):p + h]) for p in pr[:2000]])
    bw = sp.filtfilt(*sp.butter(2, 0.5 / (fs / 2)), x)
    inc[nm] = dict(sig=sig, W=W, cl=cl, gp=used)
    rows.append({"set": "INCART_" + nm, "n": len(W), **wstats(W[cl == 0]), "R_pp_mV_input_if_gain100": float(np.median(rpp)),
                 "baseline_std_over_Rpp": float(bw.std() / np.median(rpp))})
df = pd.DataFrame(rows); df.to_csv(OUT / "amp_stats.csv", index=False); print(df.to_string())
# spectra (model-chain signal, 250 Hz), normalised power
spec_rows = {}
for k, v in list(real.items()) + [(k, v) for k, v in inc.items()]:
    f, P = sp.welch(v["sig"], 250, nperseg=1024); spec_rows[k] = P / P[(f > 1) & (f < 40)].sum()
pd.DataFrame(spec_rows, index=f).to_csv(OUT / "spectra_norm.csv")
np.savez_compressed(OUT / "_tmp_parta.npz", **{f"{k}_W": v["W"] for k, v in real.items()},
                    **{f"{k}_W": v["W"][:3000] for k, v in inc.items()}, **{f"{k}_cl": v["cl"][:3000] for k, v in inc.items()})
# high-frequency content fraction
for k in spec_rows: print(k, "frac 15-40Hz %.3f  >40Hz %.4f" % (spec_rows[k][(f >= 15) & (f < 40)].sum(), spec_rows[k][f >= 40].sum()))
