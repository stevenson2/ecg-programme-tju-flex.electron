"""Diagnostic: score real sessions with inverted polarity (x -> -x) and report N V-call rate; also R/S ratio stats.
Not a proposed fix by itself; tests whether RS morphology/polarity drives the misjudgment."""
import os, json, importlib.util
os.environ["DET"] = "aiqrs"; os.environ.setdefault("MODELS_SUBSET", "student_hard,distill_v2,exp5_int8")
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s13", HERE / "13_event_sensitivity_fph.py"); s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
B = s13.BENCH; OUT = B / "results/amp_analysis"
FILES = {"real_145244": "serial_20261019_145244_leadI_standard_AD8232.csv", "real_155455": "serial_20261019_155455_leadI_standard_AD8232.csv",
         "real_143011": "serial_20261019_143011_wearable_Lchest_horiz_AD8232.csv", "real_143825": "serial_20261019_143825_wearable_Lchest_horiz_AD8232.csv"}
lab = json.loads(sorted((B / "results/labels").glob("ecg_labels_*.json"))[-1].read_text(encoding="utf-8"))["sessions"]
sc_ = s13.Scorer(); rows = []
def rs(W):  # R (max) and S (min) within +-60 ms of centre, z units
    c = W[:, 110:141]; return c.max(1), -c.min(1)
tz = np.load(B / "data/student_train_leadI.npz", allow_pickle=True); tb, tl = tz["beats"][::4], tz["labels_nsvq"][::4]
for nm, W in [("train_N", tb[tl == 0]), ("train_V", tb[tl == 2])]:
    r, s = rs(W); rows.append({"set": nm, "pol": "+", "R_med": np.median(r), "S_med": np.median(s), "S_gt_R_frac": float((s > r).mean())})
for key, fn in FILES.items():
    df = pd.read_csv(s13.INBOX / fn); d = df["raw_line"].astype(str).str.split(",", expand=True)[2].astype(float).to_numpy()
    meta = json.loads((s13.INBOX / fn.replace(".csv", "_meta.json")).read_text(encoding="utf-8")); fs = len(d) / float(meta["duration_s_actual"])
    pk = np.load(s13.CACHE / f"{key}__clean__aiqrs.npz")["peaks"]; L = lab.get(key, {}).get("beats", {})
    for pol in [1, -1]:
        sig = s13.causal_filter_chain(s13.resample_ecg(pol * d, fs))
        # recentre on max|x| (+-60ms) so inverted beats stay centred
        pk2 = np.array([max(0, p - 15) + int(np.argmax(np.abs(sig[max(0, p - 15):p + 16]))) for p in pk])
        W, used = s13.beats_at(sig, pk2); keep = [i for i, p in enumerate(pk2) if 125 <= p <= len(sig) - 125]
        lb = np.array([L.get(str(int(pk[i])), "") for i in keep])[:len(W)]
        r, s = rs(W); row = {"set": key, "pol": "+" if pol > 0 else "-", "R_med": np.median(r), "S_med": np.median(s), "S_gt_R_frac": float((s > r).mean()), "n": len(W)}
        for m in sc_.m:
            v = sc_.score(m, W) >= .5; row["flag_" + m] = float(v.mean())
            if (lb == "N").any(): row["Ncall_" + m] = float(v[lb == "N"].mean())
        rows.append(row)
# INCART test N beats R/S for reference
inc = np.load(B / "data/leadI_incart_beats.npz", allow_pickle=True); m_ = (inc["split"] == "test") & (inc["labels_nsvq"] == 0)
r, s = rs(inc["beats"][m_][::5]); rows.append({"set": "INCART_test_N", "pol": "+", "R_med": np.median(r), "S_med": np.median(s), "S_gt_R_frac": float((s > r).mean())})
P = pd.DataFrame(rows); P.to_csv(OUT / "polarity_rs_test.csv", index=False); print(P.round(3).to_string())
