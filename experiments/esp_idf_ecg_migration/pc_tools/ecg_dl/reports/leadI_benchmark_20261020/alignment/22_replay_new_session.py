"""Replay a new real_captures session through the AI QRS trigger (tools/hr_host mode 2) + student_hard/distill_v2/exp5,
writing the script-13 score cache (results/event_sens_fph/score_cache/<key>__clean__aiqrs.npz). Also integrity check.
  python3 scripts/22_replay_new_session.py serial_20261020_160942_wearable_single_lead_AD8232.csv"""
import os, sys, json, importlib.util
os.environ["DET"] = "aiqrs"; os.environ.setdefault("MODELS_SUBSET", "student_hard,distill_v2,exp5_int8")
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s13", HERE / "13_event_sensitivity_fph.py"); s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
fn = sys.argv[1]; path = s13.INBOX / fn
df = pd.read_csv(path); parts = df["raw_line"].astype(str).str.split(",", expand=True)
ncol = parts.notna().sum(1); ok = ncol >= 10
num = parts.iloc[:, :3].apply(pd.to_numeric, errors="coerce"); ok &= num.notna().all(1)
meta = json.loads(path.with_name(path.stem + "_meta.json").read_text(encoding="utf-8"))
dur = float(meta["duration_s_actual"]); x = num[ok].iloc[:, 2].to_numpy(np.float64); fs = len(x) / dur
ts = df["host_ts_s"].to_numpy(); gaps = np.diff(ts)
adc = num[ok].iloc[:, 1].to_numpy() + 1.65
tb = parts[4][ok].astype(str).value_counts().to_dict()
print(json.dumps({"rows": len(df), "rows_valid": int(ok.sum()), "rows_bad": int((~ok).sum()), "duration_s": dur, "fs_est": round(fs, 3),
                  "host_ts_first_last": [float(ts[0]), float(ts[-1])], "max_host_gap_s": float(gaps.max()), "n_gaps_gt_0.5s": int((gaps > 0.5).sum()),
                  "true_bpm_hist": tb, "adc_floor_frac": float((adc <= 0.05).mean()), "adc_top_frac": float((adc >= 3.25).mean())}, ensure_ascii=False))
key = "real_" + fn.split("_")[2]
scorer = s13.Scorer(); s13._KEYS = list(scorer.m.keys())
out = s13.process(key, "clean", x, fs, np.zeros(0, np.int64), np.zeros(0, np.int32), scorer, det="aiqrs")
n_sec = out["n_sec"]
for m, sc in out["scores"].items():
    ent, _ = s13.alarm_sim(s13.flags_from(out["peaks"], sc, n_sec, .5), **s13.RULE)
    print(f"{m}: beats={len(sc)} flagged={float((sc>=.5).mean()):.3f} alarm_entries={len(ent)} ({len(ent)/(n_sec/3600):.1f}/h)")
print("HR ~ %.1f bpm from aiqrs peaks" % (60 / np.median(np.diff(out["peaks"]) / 250)))
