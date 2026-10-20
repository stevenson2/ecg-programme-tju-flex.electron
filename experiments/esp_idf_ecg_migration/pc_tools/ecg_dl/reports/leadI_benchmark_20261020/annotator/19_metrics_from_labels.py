"""Recompute per-model real-device metrics from labels exported by tools/annotator/ecg_annotator.html.
Pure python + stdlib (runs on Windows `python` or WSL). Low resource.

  python scripts/19_metrics_from_labels.py results/labels/ecg_labels_xxx.json [more.json ...]
        [--unlabeled-unflagged N|exclude]  (default N: un-reviewed beats NOT flagged by any model are assumed normal)

Beat level (detected beats = new AI QRS trigger, scores at thr from the data file, default 0.50):
  truth V   = detected beats labelled V  + manually added (missed) beats labelled V
  predicted = detected beat with model score >= thr
  N / 误判(FD) -> non-V; 不确定(UNK) -> excluded; 伪影(ART) -> non-V in "all", excluded in "excl_art"
  flagged-but-unlabelled beats -> excluded (counted in n_flag_unlab)
  Se = TP/(TP+FN)  (FN includes missed beats added by hand), PPV = TP/(TP+FP)
Alarm level (entries pre-computed at thr with the firmware rule):
  label V -> true alarm; N/FD/ART -> false alarm; UNK -> excluded;
  unlabelled -> false if no truth-V beat in the 10 s density window before the entry, else true (counted n_auto)
  FP/h = false alarms / hours.  "excl_art": entries inside artifact segments dropped, artifact time removed from hours.
Outputs: results/real_label_metrics.csv and printed table."""
import sys, json, csv, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("labels", nargs="+")
ap.add_argument("--data", default=str(HERE.parent / "tools/annotator/ecg_annot_data.js"))
ap.add_argument("--unlabeled-unflagged", default="N", choices=["N", "exclude"])
ap.add_argument("--out", default=str(HERE.parent / "results/real_label_metrics.csv"))
a = ap.parse_args()
txt = Path(a.data).read_text(encoding="utf-8")
D = json.loads(txt[txt.index("=") + 1:].strip().rstrip(";"))
FS, THR, WIN = D["fs"], D["thr"], D["rule"]["win"]
LS = {}
for f in a.labels:  # later files override earlier per session
    LS.update(json.loads(Path(f).read_text(encoding="utf-8"))["sessions"])

def in_art(t, arts): return any(s <= t <= e for s, e in arts)

def session_metrics(S, L, m, mode):
    arts = L.get("artifacts", []) if mode == "excl_art" else []
    beats, manual, alarms = L.get("beats", {}), L.get("manual", {}), L.get("alarms", {})
    tp = fp = fn = tn = nfl = nex = 0
    vt = []
    for j, p in enumerate(S["peaks"]):
        t = p / FS
        if in_art(t, arts): nex += 1; continue
        lb = beats.get(str(p))
        flagged = any(S["scores"][mm][j] >= THR for mm in D["models"])
        if lb is None:
            if flagged: nfl += 1; continue
            if a.unlabeled_unflagged == "exclude": continue
            lb = "N"
        if lb == "UNK" or (lb == "ART" and mode == "excl_art"): nex += 1; continue
        isv, pred = lb == "V", S["scores"][m][j] >= THR
        if isv: vt.append(t)
        tp += isv and pred; fn += isv and not pred; fp += (not isv) and pred; tn += (not isv) and not pred
    nmiss = 0
    for p, lb in manual.items():
        t = int(p) / FS
        if lb == "V" and not in_art(t, arts):
            fn += 1; nmiss += 1; vt.append(t)
    art_s = sum(min(e, S["n_sec"]) - max(0, s) for s, e in arts)
    hours = max(1e-9, (S["n_sec"] - art_s) / 3600)
    fa = ta = nauto = nexa = 0
    for e in S["alarms"][m]["entries"]:
        if in_art(e, arts) or in_art(e + 0.999, arts): nexa += 1; continue
        lb = alarms.get(f"{m}@{e}")
        if lb == "UNK": nexa += 1; continue
        if lb is None:
            nauto += 1
            true = any(e - WIN + 1 <= t < e + 1 for t in vt)
        else:
            true = lb == "V"
        ta += true; fa += not true
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n_missed_manual_V=nmiss, n_flag_unlab=nfl, n_beat_excl=nex,
                n_alarm=ta + fa, n_alarm_true=ta, n_alarm_false=fa, n_alarm_auto=nauto, n_alarm_excl=nexa,
                hours=hours, artifact_s=art_s)

def fin(r):
    r["Se"] = r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else float("nan")
    r["PPV"] = r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else float("nan")
    r["FP_per_h"] = r["n_alarm_false"] / r["hours"]
    return r

rows = []
for S in D["sessions"]:
    L = LS.get(S["key"])
    if L is None:
        print("no labels for", S["key"]); continue
    for mode in ["all", "excl_art"]:
        for m in D["models"]:
            rows.append(fin({"group": S["key"], "kind": S["kind"], "mode": mode, "model": m, **session_metrics(S, L, m, mode)}))
num = ["tp", "fp", "fn", "tn", "n_missed_manual_V", "n_flag_unlab", "n_beat_excl", "n_alarm", "n_alarm_true",
       "n_alarm_false", "n_alarm_auto", "n_alarm_excl", "hours", "artifact_s"]
for grp in ["leadI", "wearable", "ALL"]:
    for mode in ["all", "excl_art"]:
        for m in D["models"]:
            sub = [r for r in rows if r["mode"] == mode and r["model"] == m and r["group"].startswith("real_") and (grp == "ALL" or r["kind"] == grp)]
            if sub:
                rows.append(fin({"group": "SUM_" + grp, "kind": grp, "mode": mode, "model": m, **{k: sum(r[k] for r in sub) for k in num}}))
cols = ["group", "kind", "mode", "model", "Se", "PPV", "FP_per_h"] + num
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in cols})
f3 = lambda v: "  nan" if v != v else f"{v:5.2f}"
print(f"thr={THR}  unlabelled-unflagged={a.unlabeled_unflagged}   ->  {a.out}")
print(f"{'group':14s} {'mode':8s} {'model':13s}   Se   PPV  FP/h  TP FP FN  alarms(F/T/auto/excl)  flagUnlab  hours")
for r in rows:
    print(f"{r['group']:14s} {r['mode']:8s} {r['model']:13s} {f3(r['Se'])} {f3(r['PPV'])} {r['FP_per_h']:5.1f} {r['tp']:3d}{r['fp']:3d}{r['fn']:3d}"
          f"   {r['n_alarm_false']}/{r['n_alarm_true']}/{r['n_alarm_auto']}/{r['n_alarm_excl']}            {r['n_flag_unlab']:4d}  {r['hours']:.3f}")
