"""Alignment audit: where does the QRS sit inside the 250-sample windows used for training/test?
- offset of max|x| (QRS energy peak) from window centre (sample 125), by class, for training npz, INCART val/test npz,
  and for windows re-cut from raw INCART test with the causal chain (as script 13 does)
- group delay of causal HP0.5/LP40/notch50 chain at QRS frequencies
- INCART annotation position vs raw-signal local max|x| (annotation offset)"""
import os, importlib.util
os.environ["DET"] = "aiqrs"
from pathlib import Path
import numpy as np, pandas as pd, wfdb
from scipy import signal as sp
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s13", HERE / "13_event_sensitivity_fph.py"); s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
B = s13.BENCH; OUT = B / "results/alignment"; OUT.mkdir(parents=True, exist_ok=True)
def qrs_off(W, lo=-40, hi=41):
    """offset of max |band-passed| energy within +-160 ms of centre, and of max|x| over whole window"""
    b, a = sp.butter(2, [8 / 125, 30 / 125], "band"); E = np.abs(sp.filtfilt(b, a, W, axis=1))
    c = E[:, 125 + lo:125 + hi]; o1 = c.argmax(1) + lo
    o2 = np.abs(W).argmax(1) - 125; o3 = E.argmax(1) - 125
    return o1, o2, o3
rows = []
def add(name, W, cls):
    for c, cn in [(0, "N"), (2, "V")]:
        m = cls == c
        if m.sum() < 10: continue
        o1, o2, o3 = qrs_off(W[m][:20000])
        rows.append({"set": name, "cls": cn, "n": int(m.sum()), "qrsE_off_med": float(np.median(o1)), "qrsE_off_IQR": f"{np.percentile(o1,25):.0f}..{np.percentile(o1,75):.0f}",
                     "qrsE_within_pm10": float((np.abs(o1) <= 10).mean()), "maxabs_off_med": float(np.median(o2)), "maxabs_within_pm10": float((np.abs(o2) <= 10).mean()),
                     "globalE_within_pm20": float((np.abs(o3) <= 20).mean())})
z = np.load(B / "data/student_train_leadI.npz", allow_pickle=True); add("student_train_leadI", z["beats"], z["labels_nsvq"])
li = np.load(B / "data/leadI_incart_beats.npz", allow_pickle=True)
for sp_ in ["val", "test"]:
    m = li["split"] == sp_; add(f"leadI_incart_beats[{sp_}]", li["beats"][m], li["labels_nsvq"][m])
# re-cut from raw (script-13 path, GT peaks) + annotation offset vs raw local max
ann_rows = []
Wall, Call = [], []
for rid in s13.TEST_RIDS:
    nm = f"I{rid:02d}"; r = wfdb.rdrecord(str(s13.INCART_DIR / nm)); a = wfdb.rdann(str(s13.INCART_DIR / nm), "atr")
    x = r.p_signal[:, 0].astype(np.float64); fs = float(r.fs); gp, gc = s13.gt_beats(a.sample, a.symbol, fs)
    sig = s13.causal_filter_chain(s13.resample_ecg(x, fs)); W, used = s13.beats_at(sig, gp); Wall.append(W); Call.append(gc[np.isin(gp, used)])
    # raw (zero-phase band) local max around annotation
    b, a_ = sp.butter(2, [8 / (fs / 2), 30 / (fs / 2)], "band"); e = np.abs(sp.filtfilt(b, a_, x))
    for c, cn in [(0, "N"), (2, "V")]:
        idx = [s for s, y in zip(a.sample, a.symbol) if s13.AAMI_NSVQ.get(y) == c and 40 < s < len(x) - 40]
        if len(idx) < 5: continue
        off = np.array([np.argmax(e[s - 40:s + 41]) - 40 for s in idx])
        ann_rows.append({"rec": nm, "cls": cn, "n": len(idx), "ann_vs_rawQRS_off_ms_med": float(np.median(off) / fs * 1000)})
add("INCART_test_recut_causal(script13)", np.concatenate(Wall), np.concatenate(Call))
R = pd.DataFrame(rows); R.to_csv(OUT / "window_alignment.csv", index=False); print(R.to_string())
A = pd.DataFrame(ann_rows); A.to_csv(OUT / "annotation_offset.csv", index=False); print(A.groupby("cls").ann_vs_rawQRS_off_ms_med.describe())
# group delay of causal chain
fs = 250
bh, ah = sp.butter(2, 0.5 / 125, "high"); bl, al = sp.butter(2, 40 / 125); bn, an = sp.iirnotch(50, 20, fs)
b = np.convolve(np.convolve(bh, bl), bn); a = np.convolve(np.convolve(ah, al), an)
w, gd = sp.group_delay((b, a), w=[5, 10, 15, 20, 25], fs=fs)
print("causal chain group delay (samples @250) at 5/10/15/20/25 Hz:", np.round(gd, 2))
# per-record N alignment in test windows
pr = []
for W, C, rid in zip(Wall, Call, s13.TEST_RIDS):
    for c, cn in [(0, "N"), (2, "V")]:
        m = C == c
        if m.sum() < 5: continue
        o1, o2, _ = qrs_off(W[m]); pr.append({"rec": f"I{rid:02d}", "cls": cn, "n": int(m.sum()), "qrsE_off_med": float(np.median(o1)), "maxabs_off_med": float(np.median(o2)), "maxabs_within_pm10": float((np.abs(o2) <= 10).mean())})
P = pd.DataFrame(pr); P.to_csv(OUT / "per_record_alignment.csv", index=False); print(P.to_string())
np.savez_compressed(OUT / "_tmp_testV.npz", W=np.concatenate(Wall)[np.concatenate(Call) == 2][::7], N=np.concatenate(Wall)[np.concatenate(Call) == 0][::50])
