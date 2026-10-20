"""Candidate fix: automatic polarity at session start (fixed rule, no tuning):
   during the first 30 s, take median R-centred z-window; if its min within +-60 ms is deeper than its max
   (S deeper than R), invert the signal for the rest of the session. Evaluate on real sessions, INCART frozen test, PTB."""
import os, json, importlib.util
os.environ["DET"] = "aiqrs"; os.environ.setdefault("MODELS_SUBSET", "student_hard,distill_v2,exp5_int8")
from pathlib import Path
import numpy as np, pandas as pd, wfdb
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s13", HERE / "13_event_sensitivity_fph.py"); s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
B = s13.BENCH; OUT = B / "results/amp_analysis"
sc_ = s13.Scorer(); M = list(sc_.m)
def decide(sig, pk):
    W, used = s13.beats_at(sig, pk[pk < 30 * 250])
    if len(W) < 5: return 1
    med = np.median(W, 0)[110:141]; return -1 if -med.min() > med.max() else 1
def run(sig, pk, pol):
    if pol < 0:
        sig = -sig; pk = np.array([max(0, p - 15) + int(np.argmax(np.abs(sig[max(0, p - 15):p + 16]))) for p in pk])
    W, used = s13.beats_at(sig, pk); return W, used, {m: sc_.score(m, W) for m in M}
rows = []
lab = json.loads(sorted((B / "results/labels").glob("ecg_labels_*.json"))[-1].read_text(encoding="utf-8"))["sessions"]
FILES = {"real_145244": "serial_20261019_145244_leadI_standard_AD8232.csv", "real_155455": "serial_20261019_155455_leadI_standard_AD8232.csv",
         "real_143011": "serial_20261019_143011_wearable_Lchest_horiz_AD8232.csv", "real_143825": "serial_20261019_143825_wearable_Lchest_horiz_AD8232.csv"}
for key, fn in FILES.items():
    df = pd.read_csv(s13.INBOX / fn); d = df["raw_line"].astype(str).str.split(",", expand=True)[2].astype(float).to_numpy()
    meta = json.loads((s13.INBOX / fn.replace(".csv", "_meta.json")).read_text(encoding="utf-8")); fs = len(d) / float(meta["duration_s_actual"])
    sig = s13.causal_filter_chain(s13.resample_ecg(d, fs)); pk = np.load(s13.CACHE / f"{key}__clean__aiqrs.npz")["peaks"]; n_sec = len(sig) // 250
    pol = decide(sig, pk); W, used, sc = run(sig, pk, pol); L = lab.get(key, {}).get("beats", {})
    # label lookup: nearest original peak
    lb = np.array([L.get(str(int(pk[np.argmin(np.abs(pk - u))])), "") for u in used])
    for m in M:
        ent, _ = s13.alarm_sim(s13.flags_from(used, sc[m], n_sec, .5), **s13.RULE)
        rows.append({"db": "REAL", "rec": key, "pol": pol, "model": m, "flag": float((sc[m] >= .5).mean()), "n_entries": len(ent), "hours": n_sec / 3600,
                     "Ncall": float((sc[m][lb == "N"] >= .5).mean()) if (lb == "N").any() else np.nan})
for rid in s13.TEST_RIDS:
    nm = f"I{rid:02d}"; r = wfdb.rdrecord(str(s13.INCART_DIR / nm)); a = wfdb.rdann(str(s13.INCART_DIR / nm), "atr")
    x = r.p_signal[:, 0].astype(np.float64); fs = float(r.fs); gp, gc = s13.gt_beats(a.sample, a.symbol, fs)
    sig = s13.causal_filter_chain(s13.resample_ecg(x, fs)); n_sec = len(sig) // 250; pk = np.load(s13.CACHE / f"{nm}__clean__aiqrs.npz")["peaks"]
    ev_, runs, vsec, _, _ = s13.define_events(gp, gc, n_sec); pol = decide(sig, pk); W, used, sc = run(sig, pk, pol)
    for m in M:
        e = s13.evaluate(s13.flags_from(used, sc[m], n_sec, .5), vsec, ev_, runs, n_sec)
        rows.append({"db": "INCART", "rec": nm, "pol": pol, "model": m, "n_events": e["n_events"], "n_det": e["n_det"], "n_fp": e["n_fp"], "hours": e["hours"]})
    print(nm, "pol", pol, flush=True)
ctrl = s13.PTB_DIR / "CONTROLS"; pats = [l.strip() for l in ctrl.read_text(errors="ignore").splitlines() if l.strip().startswith("patient")]
nflip = 0
for line in pats:
    if "/" in line: stem = s13.PTB_DIR / line
    else:
        heas = sorted((s13.PTB_DIR / line).glob("*.hea"))
        if not heas: continue
        stem = heas[0].with_suffix("")
    key = "ptb_" + str(stem.relative_to(s13.PTB_DIR)).replace("/", "_"); cp = s13.CACHE / f"{key}__clean__aiqrs.npz"
    if not cp.exists(): continue
    try: r = wfdb.rdrecord(str(stem))
    except Exception: continue
    if "I" not in [s.upper() for s in r.sig_name]: continue
    r = wfdb.rdrecord(str(stem)); names = [s.upper() for s in r.sig_name]
    sig = s13.causal_filter_chain(s13.resample_ecg(r.p_signal[:, names.index("I")].astype(np.float64), float(r.fs))); n_sec = len(sig) // 250
    pk = np.load(cp)["peaks"]; pol = decide(sig, pk); nflip += pol < 0; W, used, sc = run(sig, pk, pol)
    for m in M:
        ent, _ = s13.alarm_sim(s13.flags_from(used, sc[m], n_sec, .5), **s13.RULE)
        rows.append({"db": "PTB", "rec": key, "pol": pol, "model": m, "n_entries": len(ent), "hours": n_sec / 3600})
D = pd.DataFrame(rows); D.to_csv(OUT / "autopolarity_detail.csv", index=False)
print("PTB flipped records:", nflip)
I = D[D.db == "INCART"].groupby("model").sum(numeric_only=True); P = D[D.db == "PTB"].groupby("model").sum(numeric_only=True)
S = pd.DataFrame({"INCART_event_sens": I.n_det / I.n_events, "INCART_FPh": I.n_fp / I.hours, "PTB_FPh": P.n_entries / P.hours})
S.to_csv(OUT / "autopolarity_summary.csv"); print(S.round(3)); print(D[D.db == "REAL"].round(3).to_string())
print("INCART flipped:", D[(D.db == "INCART") & (D.pol < 0)].rec.unique())
