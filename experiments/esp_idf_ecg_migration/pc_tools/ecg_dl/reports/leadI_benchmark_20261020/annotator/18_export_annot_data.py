"""Export the 4 real-device sessions for the offline annotation tool (tools/annotator).
Reads the script-13 score cache (DET=aiqrs: new AI QRS trigger via tools/hr_host, scores of
student_hard / distill_v2 / exp5_int8), recomputes the 250 Hz display signal with the same
causal chain, and simulates alarm entries (thr 0.50, same RULE). No model inference here.
Run (WSL): nice -n 19 python3 scripts/18_export_annot_data.py
Output: tools/annotator/ecg_annot_data.js  (window.ECG_DATA = {...})"""
import os, json, importlib.util
os.environ["DET"] = "aiqrs"
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s13", HERE / "13_event_sensitivity_fph.py")
s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
MODELS = ["student_hard", "distill_v2", "exp5_int8"]
THR = s13.THR
FILES = ["serial_20261019_145244_leadI_standard_AD8232.csv", "serial_20261019_155455_leadI_standard_AD8232.csv",
         "serial_20261019_143011_wearable_Lchest_horiz_AD8232.csv", "serial_20261019_143825_wearable_Lchest_horiz_AD8232.csv"]
out = {"generated_by": "scripts/18_export_annot_data.py", "detector": "aiqrs (ECG_AI_PERMISSIVE_QRS=1, tools/hr_host mode 2)",
       "fs": 250, "thr": THR, "rule": s13.RULE, "models": MODELS, "sessions": []}
for fn in FILES:
    path = s13.INBOX / fn
    if not path.exists():
        print("missing", fn); continue
    df = pd.read_csv(path)
    x = df["filtered"].to_numpy(np.float64) if "filtered" in df.columns else \
        df["raw_line"].astype(str).str.split(",", expand=True)[2].astype(float).to_numpy(np.float64)
    fs = 100.0
    mp = path.with_name(path.stem + "_meta.json")
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8")); dur = meta.get("duration_s_actual") or meta.get("duration_s")
        if dur: fs = len(x) / float(dur)
    key = "real_" + fn.split("_")[2]
    z = np.load(s13.CACHE / f"{key}__clean__aiqrs.npz", allow_pickle=True)
    peaks, n_sec = z["peaks"].astype(int), int(z["n_sec"])
    sig = s13.causal_filter_chain(s13.resample_ecg(x, fs))[:n_sec * 250]
    sig = (sig - np.median(sig)) / (np.percentile(np.abs(sig - np.median(sig)), 99) + 1e-9)
    scores, alarms = {}, {}
    for m in MODELS:
        sc = z["s_" + m].astype(float); scores[m] = [round(v, 3) for v in sc]
        ent, act = s13.alarm_sim(s13.flags_from(peaks, sc, n_sec, THR), **s13.RULE)
        spans, i = [], 0
        while i < n_sec:
            if act[i]:
                j = i
                while j + 1 < n_sec and act[j + 1]: j += 1
                spans.append([i, j + 1]); i = j + 1
            else: i += 1
        alarms[m] = {"entries": [int(e) for e in ent], "spans": spans}
    out["sessions"].append({"key": key, "file": fn, "kind": "leadI" if "leadI" in fn else "wearable",
                            "fs_orig": round(fs, 3), "n_sec": n_sec, "sig": [round(float(v), 3) for v in sig],
                            "peaks": peaks.tolist(), "scores": scores, "alarms": alarms})
    print(key, "n_sec", n_sec, "beats", len(peaks), {m: len(alarms[m]["entries"]) for m in MODELS})
dst = HERE.parent / "tools/annotator/ecg_annot_data.js"; dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("window.ECG_DATA = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
print("wrote", dst, dst.stat().st_size, "bytes")
