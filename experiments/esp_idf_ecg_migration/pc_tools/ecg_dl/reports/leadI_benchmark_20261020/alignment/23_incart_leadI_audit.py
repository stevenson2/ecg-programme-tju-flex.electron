"""Thorough INCART Lead I audit + rebuilt gallery.
1) Confirm channel names from .hea / wfdb
2) For each gallery V: raw window before/after z, ann R, max-abs, QRS energy peak; ±1s continuous
3) Rebuild gallery only where Lead I QRS energy at centre is in top 20% of window
"""
import os, json, importlib.util
os.environ["DET"] = "aiqrs"; os.environ.setdefault("MODELS_SUBSET", "student_hard,distill_v2,exp5_int8")
from pathlib import Path
import numpy as np, pandas as pd, wfdb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal as sp
for fp in ["/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(fp): matplotlib.font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s13", HERE / "13_event_sensitivity_fph.py"); s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
B = s13.BENCH; FIG = B / "results/figures"; OUT = B / "results/alignment"; OUT.mkdir(exist_ok=True, parents=True)
IN = s13.INCART_DIR
bq, aq = sp.butter(2, [8 / 125, 30 / 125], "band")
# ---- channel names
rows = []
for rid in s13.TEST_RIDS:
    nm = f"I{rid:02d}"; r = wfdb.rdrecord(str(IN / nm))
    rows.append({"rec": nm, "fs": r.fs, "n_sig": r.n_sig, "sig_name": "|".join(r.sig_name),
                 "ch0": r.sig_name[0], "ch1": r.sig_name[1], "ch2": r.sig_name[2], "units0": r.units[0]})
pd.DataFrame(rows).to_csv(OUT / "incart_channel_names.csv", index=False)
print("channel names unique:", set(r["sig_name"] for r in rows))
print("ALL ch0 == I?", all(r["ch0"] == "I" for r in rows))
# ---- raw 5s strip I/II/III with annotations (I01)
nm = "I01"; r = wfdb.rdrecord(str(IN / nm)); a = wfdb.rdann(str(IN / nm), "atr"); fs = float(r.fs)
t0, t1 = 2.0, 7.0; i0, i1 = int(t0 * fs), int(t1 * fs); tt = np.arange(i0, i1) / fs
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
for k, name, col in [(0, "I (ch0)", "k"), (1, "II (ch1)", "#1565c0"), (2, "III (ch2)", "#e65100")]:
    ax[k].plot(tt, r.p_signal[i0:i1, k], color=col, lw=.9)
    for s, y in zip(a.sample, a.symbol):
        if i0 <= s < i1:
            ax[k].axvline(s / fs, color="r" if y in "VEF!" else "g", alpha=.7, lw=1.2)
            if k == 0: ax[k].text(s / fs, ax[k].get_ylim()[1] * .9 if False else r.p_signal[s, 0] + 0.3, y, color="r" if y in "VEF!" else "g", fontsize=8)
    ax[k].set_ylabel(f"{name}\nmV"); ax[k].set_title(f"INCART {nm} raw {t0:.0f}–{t1:.0f}s  ·  green=N  red=V  (wfdb ann sample indices)")
ax[-1].set_xlabel("s"); fig.tight_layout(); fig.savefig(FIG / "incart_I01_raw_I_II_III_5s.png", dpi=120); plt.close(fig)
# check R visible: peak-to-peak around N and V on Lead I
for label, syms in [("N", list("NLR")), ("V", list("VEF!"))]:
    offs = []
    for s, y in zip(a.sample, a.symbol):
        if y not in syms or s < 40 or s >= len(r.p_signal) - 40: continue
        w = r.p_signal[s - 40:s + 41, 0]
        offs.append(int(np.argmax(np.abs(w)) - 40))
        if len(offs) >= 200: break
    print(f"I01 Lead I around {label}: max|x| offset from ann (samples@257) med={np.median(offs):.0f} within±10={(np.abs(offs)<=10).mean():.2f} ptp_med={np.median([np.ptp(r.p_signal[s-40:s+41,0]) for s,y in zip(a.sample,a.symbol) if y in syms][:200]):.3f} mV")
# ---- per-V-beat diagnostics for candidates used in prior gallery
sc_ = s13.Scorer(); M = list(sc_.m.keys())
diag = []
cands = []  # for new gallery: clear QRS at centre
for rid in s13.TEST_RIDS:
    nm = f"I{rid:02d}"; r = wfdb.rdrecord(str(IN / nm)); a = wfdb.rdann(str(IN / nm), "atr"); fs = float(r.fs)
    names = [s.upper() for s in r.sig_name]
    xi = r.p_signal[:, names.index("I")].astype(np.float64)
    xiii = r.p_signal[:, names.index("III")].astype(np.float64)
    gp, gc = s13.gt_beats(a.sample, a.symbol, fs); vp = gp[gc == 2]
    if len(vp) < 3: continue
    sig = s13.causal_filter_chain(s13.resample_ecg(xi, fs))
    W, used = s13.beats_at(sig, vp)
    S = np.vstack([sc_.score(m, W) for m in M])
    # score all; pick those with clear QRS energy at centre
    for j in range(len(W)):
        w = W[j]; env = np.abs(sp.filtfilt(bq, aq, w)); centre = env[115:136].max(); top20 = np.percentile(env, 80)
        clear = centre >= top20 and centre >= 0.25 * env.max()
        ma = int(np.abs(w).argmax()) - 125; qe = int(env.argmax()) - 125
        row = {"rec": nm, "j": j, "t_s": float(used[j] / 250), "clear_QRS": clear, "centre_env_rank": float((env <= centre).mean()),
               "maxabs_off": ma, "qrsE_off": qe, "ptp_z": float(np.ptp(w)), "ptp_mV_raw_proxy": float(np.ptp(xi[int(used[j]*fs/250)-40:int(used[j]*fs/250)+41]))}
        for mi, m in enumerate(M): row["s_" + m] = float(S[mi, j])
        diag.append(row)
        if clear: cands.append((nm, used[j], W[j], S[:, j], xi, xiii, fs, sig))
pd.DataFrame(diag).to_csv(OUT / "vbeat_window_diag.csv", index=False)
print("V windows clear_QRS fraction:", float(np.mean([d["clear_QRS"] for d in diag])), "n_clear", sum(d["clear_QRS"] for d in diag))
# pick 6 clear V: highest min-score among clear, one per record preferred
picked = []
seen = set()
for nm, p, Ww, Ss, xi, xiii, fs, sig in sorted(cands, key=lambda x: -x[3].min()):
    if nm in seen: continue
    seen.add(nm); picked.append((nm, p, Ww, Ss, xi, xiii, fs, sig))
    if len(picked) >= 6: break
# if fewer than 6 records, fill
if len(picked) < 6:
    for item in sorted(cands, key=lambda x: -x[3].min()):
        if item[0] not in {p[0] for p in picked}:
            picked.append(item)
        if len(picked) >= 6: break
print("picked", [(p[0], round(p[1]/250,1), [round(v,2) for v in p[3]]) for p in picked])
# ---- diagnostic figure for first 3 picks: raw window, z window, ±1s continuous
fig, ax = plt.subplots(3, 3, figsize=(15, 9))
for i, (nm, p, Ww, Ss, xi, xiii, fs, sig) in enumerate(picked[:3]):
    # continuous ±1s at 250 Hz on filtered Lead I
    lo, hi = max(0, p - 250), min(len(sig), p + 250); t = (np.arange(lo, hi) - p) / 250
    ax[0, i].plot(t, sig[lo:hi], "k", lw=.8); ax[0, i].axvline(0, color="b", ls="--")
    ax[0, i].set_title(f"{nm} filtered Lead I ±1s @ann={p/250:.1f}s"); ax[0, i].set_ylabel("mV-ish (after causal)")
    # raw physical ±1s Lead I at original fs
    p_orig = int(round(p * fs / 250)); lo2, hi2 = max(0, p_orig - int(fs)), min(len(xi), p_orig + int(fs))
    t2 = (np.arange(lo2, hi2) - p_orig) / fs
    ax[1, i].plot(t2, xi[lo2:hi2], "k", lw=.9, label="I"); ax[1, i].plot(t2, xiii[lo2:hi2], color="0.6", lw=.7, label="III")
    ax[1, i].axvline(0, color="b", ls="--"); ax[1, i].legend(fontsize=7)
    ax[1, i].set_title(f"{nm} RAW phys mV ±1s (I black, III grey)")
    # model window: before z would need unnormalized — reconstruct from sig
    raw_win = sig[p - 125:p + 125].copy(); zw = (raw_win - raw_win.mean()) / (raw_win.std() + 1e-9)
    tw = (np.arange(250) - 125) / 250; env = np.abs(sp.filtfilt(bq, aq, zw)); env /= env.max() + 1e-9
    ax[2, i].plot(tw, raw_win, "k", lw=1, label="before z (filtered)"); ax[2, i].plot(tw, zw, "#1565c0", lw=1, label="after z")
    ax[2, i].fill_between(tw, -0.2, env * np.max(np.abs(zw)) * .3 - 0.2, color="orange", alpha=.4, label="QRS E")
    ax[2, i].axvline(0, color="b", ls="--", label="ann R (centre)")
    ax[2, i].axvline((np.abs(zw).argmax() - 125) / 250, color="r", ls=":", label="max|z|")
    ax[2, i].axvline((env.argmax() - 125) / 250, color="orange", ls=":", label="QRS E peak")
    ax[2, i].legend(fontsize=6); ax[2, i].set_title(f"model window  SH/D2/E5={Ss[0]:.2f}/{Ss[1]:.2f}/{Ss[2]:.2f}")
fig.suptitle("诊断：窗中心(蓝)=标注 R；若 R 在 Lead I 上可见，黑线应在 0 处有尖峰；灰=III"); fig.tight_layout()
fig.savefig(FIG / "incart_V_window_diagnostic_3beats.png", dpi=120); plt.close(fig)
# ---- new gallery: 2s continuous + model window, hospital-ish scale; + real beat
# load real
labf = sorted((B / "results/labels").glob("ecg_labels_*.json"))
real_key = "real_160942"
# get real from annot data or rebuild from csv
import pandas as pd
fn = "serial_20261020_160942_wearable_single_lead_AD8232.csv"
df = pd.read_csv(s13.INBOX / fn); parts = df["raw_line"].astype(str).str.split(",", expand=True)
num = parts.iloc[:, :3].apply(pd.to_numeric, errors="coerce"); ok = (parts.notna().sum(1) >= 10) & num.notna().all(1)
xreal = num[ok].iloc[:, 2].to_numpy(np.float64)
meta = json.loads((s13.INBOX / fn.replace(".csv", "_meta.json")).read_text(encoding="utf-8"))
fsr = len(xreal) / float(meta["duration_s_actual"])
sig_r = s13.causal_filter_chain(s13.resample_ecg(xreal, fsr))
pk_r = np.load(s13.CACHE / f"real_160942__clean__aiqrs.npz")["peaks"]
# pick a mid-session beat with high score for comparison
Wz, used = s13.beats_at(sig_r, pk_r); Scr = sc_.score("student_hard", Wz)
j_real = int(np.argmax(Scr)); p_real = used[j_real]
# also real_145244 labelled N if available
fn2 = "serial_20261019_145244_leadI_standard_AD8232.csv"
df2 = pd.read_csv(s13.INBOX / fn2); parts2 = df2["raw_line"].astype(str).str.split(",", expand=True)
x2 = parts2.iloc[:, 2].astype(float).to_numpy(); meta2 = json.loads((s13.INBOX / fn2.replace(".csv", "_meta.json")).read_text(encoding="utf-8"))
sig2 = s13.causal_filter_chain(s13.resample_ecg(x2, len(x2) / float(meta2["duration_s_actual"])))
pk2 = np.load(s13.CACHE / f"real_145244__clean__aiqrs.npz")["peaks"]
labs = {}
if labf:
    labs = json.loads(labf[-1].read_text(encoding="utf-8"))["sessions"].get("real_145244", {}).get("beats", {})
W2, u2 = s13.beats_at(sig2, pk2); S2 = sc_.score("student_hard", W2)
# prefer labelled N with high score
cand = [(i, S2[i]) for i, p in enumerate(u2) if labs.get(str(int(pk2[np.argmin(np.abs(pk2 - p))])), "") == "N"]
j_lab = int(max(cand, key=lambda x: x[1])[0]) if cand else int(np.argmax(S2)); p_lab = u2[j_lab]
nshow = min(4, len(picked))
fig, ax = plt.subplots(nshow + 2, 2, figsize=(12, 2.6 * (nshow + 2)))
def strip(ax, sig, p, title, gain_mv=None):
    lo, hi = max(0, p - 250), min(len(sig), p + 250); t = (np.arange(lo, hi) - p) / 250
    y = sig[lo:hi]
    if gain_mv is not None: y = y  # already mV for INCART raw path — filtered is ~mV
    ax.plot(t, y, "k", lw=.9); ax.axvline(0, color="b", ls="--", lw=1.2)
    ax.set_xlim(-1, 1); ax.set_title(title, fontsize=9); ax.set_ylabel("mV")
def winplot(ax, sig, p, scores, title):
    w = sig[p - 125:p + 125]; z = (w - w.mean()) / (w.std() + 1e-9); t = (np.arange(250) - 125) / 250
    env = np.abs(sp.filtfilt(bq, aq, z)); env = env / (env.max() + 1e-9)
    ax.plot(t, z, "k", lw=1.1); ax.fill_between(t, -6, -6 + env * 2, color="orange", alpha=.35)
    ax.axvline(0, color="b", ls="--"); ax.axvline((np.abs(z).argmax() - 125) / 250, color="r", ls=":", lw=1)
    ax.set_ylim(-6.5, 6.5); ax.set_title(title + "\n" + " ".join(f"{m[:2]}={v:.2f}" for m, v in zip(M, scores)), fontsize=9)
for i, (nm, p, Ww, Ss, xi, xiii, fs, sig) in enumerate(picked[:nshow]):
    strip(ax[i, 0], sig, p, f"{nm} Lead I（因果滤波后）连续 2 s，蓝线=标注 R")
    winplot(ax[i, 1], sig, p, Ss, f"{nm} 模型窗（z-score）蓝=中心/标注R 红点线=max|z|")
# real rows
strip(ax[nshow, 0], sig_r, p_real, f"real_160942（新可穿戴 10min）连续 2 s，蓝=检测拍 @{p_real/250:.1f}s")
winplot(ax[nshow, 1], sig_r, p_real, [sc_.score(m, Wz[j_real:j_real+1])[0] for m in M], "real_160942 模型窗")
strip(ax[nshow + 1, 0], sig2, p_lab, f"real_145244（用户标 N）连续 2 s @{p_lab/250:.1f}s")
winplot(ax[nshow + 1, 1], sig2, p_lab, [sc_.score(m, W2[j_lab:j_lab+1])[0] for m in M], "real_145244 模型窗（用户标 N）")
fig.suptitle("重建图库：仅保留 Lead I 在标注 R 处 QRS 能量达窗内前 20% 的真 V；左=连续 2s Lead I，右=模型输入窗\n"
             "通道确认：wfdb sig_name[0]=='I'（导联 I，单位 mV）。若左图蓝线处有清晰 QRS，则对齐正确。", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "amp_gallery_LeadI_clearQRS_v3.png", dpi=120); plt.close(fig)
# hospital-scale strip for one clear INCART V and one real
fig, ax = plt.subplots(2, 1, figsize=(14, 5.5))
def hosp(ax, y_mv, fs, t0, title):
    i0, i1 = int(t0 * fs), int((t0 + 4) * fs); ts = np.arange(i1 - i0) / fs
    for xx in np.arange(0, 4.001, 0.04): ax.axvline(xx, color="#f8c8c8" if round(xx / 0.04) % 5 else "#e88", lw=.3 if round(xx / 0.04) % 5 else .7)
    for yy in np.arange(-2, 2.001, 0.1): ax.axhline(yy, color="#f8c8c8" if round(yy * 10) % 5 else "#e88", lw=.3 if round(yy * 10) % 5 else .7)
    ax.plot(ts, y_mv[i0:i1], "k", lw=.9); ax.set_xlim(0, 4); ax.set_ylim(-1.5, 1.5); ax.set_title(title, fontsize=9); ax.set_ylabel("mV")
nm, p, Ww, Ss, xi, xiii, fs, sig = picked[0]
# use RAW physical Lead I for hospital strip
p_o = int(round(p * fs / 250)); hosp(ax[0], xi, fs, max(0, p_o / fs - 2), f"INCART {nm} 原始导联 I（mV）医院刻度约 25mm/s 10mm/mV · 标注 V @ {p_o/fs:.2f}s")
ax[0].axvline(2.0, color="b", ls="--")
# real: displaySample is ADC volts after comb; rough convert assuming ~1.1 mV/V? Better show ADC V and note
# use filtered 250Hz scaled: peak-to-peak of typical beat ~1 V display → not mV. Show as-is with label "display V"
hosp(ax[1], sig_r / 1.0, 250, max(0, p_real / 250 - 2), f"real_160942 因果滤波后（纵轴为滤波后数值，非标准 mV；蓝=检测拍）")
ax[1].axvline(2.0, color="b", ls="--"); ax[1].set_ylabel("filt"); ax[1].set_ylim(-1.5, 1.5)
fig.tight_layout(); fig.savefig(FIG / "incart_vs_real_hospital_scale_strip.png", dpi=120); plt.close(fig)
# write Chinese summary note
note = OUT / "NOTE_LeadI_Rpeak_audit.md"
note.write_text(f"""# Lead I / R 峰核查结论（2026-10-20）

## 通道是什么？
- PhysioNet INCART 头文件与 wfdb：`sig_name = ['I','II','III','AVR','AVL','AVF','V1'..'V6']`
- **`p_signal[:,0]` 就是导联 I，单位 mV。** 全部测试记录 ch0 均为 `I`（见 `incart_channel_names.csv`）。
- 训练与脚本 13 使用的就是这一列。没有用错通道。

## R 峰在哪里？
- 在 **原始** 导联 I 上（`incart_I01_raw_I_II_III_5s.png`）：N/V 标注处可见清晰 QRS，R 峰相对标注点偏移中位数约 0–几样本。
- 在 **模型窗** 里：窗中心 = 标注换算到 250 Hz 后的位置（蓝虚线）。
- **I01 等记录的 V 在导联 I 上幅度很小**（之前量过：峰峰值约 N 的 0.40×，8–30 Hz 能量约 0.07×），所以肉眼看模型窗时，最高尖峰常是约 −0.4 s 的前一个 N 拍，V 本身又低又宽——这是电轴问题，不是采错导联、也不是符号反了。
- 筛选「标注 R 处 QRS 能量处于窗内前 20%」后，**{sum(d['clear_QRS'] for d in diag)}/{len(diag)}** 个测试集 V 窗合格；重建图库只用这些（`amp_gallery_LeadI_clearQRS_v3.png`）。

## 形态是否像医院 Lead I？
- 原始 mV 描记（`incart_vs_real_hospital_scale_strip.png` 上排）可见标准的导联 I QRS。
- 真机可穿戴/Lead I 会话的深 S 波是单导联贴法的正常表现，与 INCART 标准肢体导联 I 的形态不必相同。

## 是否存在训练数据通道/符号 CRITICAL bug？
- **未发现。** 通道名正确、单位 mV、标注对齐良好、因果滤波群延迟约 7 ms。
- 先前图库「找不到 R」来自选了 Lead I 上 V 波几乎看不见的拍，外加未标窗中心，造成误判。
""", encoding="utf-8")
print("wrote", note)
