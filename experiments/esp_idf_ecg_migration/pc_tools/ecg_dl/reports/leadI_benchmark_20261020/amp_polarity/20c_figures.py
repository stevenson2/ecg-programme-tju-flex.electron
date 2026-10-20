"""Figures for amplitude/clipping study (reads results/amp_analysis/*, raw CSV locally; PNG only)."""
import os, json, importlib.util
os.environ["DET"] = "aiqrs"
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal as sp
for fp in ["/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(fp): matplotlib.font_manager.fontManager.addfont(fp)
for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist): plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]; break
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s13", HERE / "13_event_sensitivity_fph.py"); s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
B = s13.BENCH; A = B / "results/amp_analysis"; FIG = B / "results/figures"
M = ["student_hard", "distill_v2", "exp5_int8"]; SH = {"student_hard": "SH", "distill_v2": "D2", "exp5_int8": "E5"}
FILES = {"real_145244": "serial_20261019_145244_leadI_standard_AD8232.csv", "real_155455": "serial_20261019_155455_leadI_standard_AD8232.csv",
         "real_143011": "serial_20261019_143011_wearable_Lchest_horiz_AD8232.csv", "real_143825": "serial_20261019_143825_wearable_Lchest_horiz_AD8232.csv"}
def load(key):
    fn = FILES[key]; df = pd.read_csv(s13.INBOX / fn); c = df["raw_line"].astype(str).str.split(",", expand=True).iloc[:, :3].astype(float).to_numpy()
    meta = json.loads((s13.INBOX / fn.replace(".csv", "_meta.json")).read_text(encoding="utf-8")); return c, len(c) / float(meta["duration_s_actual"])
pa = np.load(A / "_tmp_parta.npz"); tz = np.load(B / "data/student_train_leadI.npz", allow_pickle=True)
def rz(W): c = W[:, 120:131]; return np.abs(c[np.arange(len(W)), np.abs(c).argmax(1)])
def pkpos(W): return np.abs(W).argmax(1) - 125
# ---- fig 1: input-domain stats
fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
tb, tl = tz["beats"][::4], tz["labels_nsvq"][::4]
bins = np.linspace(0, 8, 41)
ax[0, 0].hist(rz(tb[tl == 0]), bins, density=True, alpha=.45, color="g", label="训练 N")
ax[0, 0].hist(rz(tb[tl == 2]), bins, density=True, alpha=.45, color="r", label="训练 V")
incN = np.concatenate([pa[k][pa[k.replace("_W", "_cl")] == 0] for k in pa.files if k.startswith("I") and k.endswith("_W")])
ax[0, 0].hist(rz(incN), bins, density=True, histtype="step", lw=2, color="k", label="INCART 测试 N")
for k, c in zip(FILES, ["#1565c0", "#00838f", "#ef6c00", "#8e24aa"]): ax[0, 0].hist(rz(pa[k + "_W"]), bins, density=True, histtype="step", lw=2, color=c, label=k)
ax[0, 0].set_title("模型输入（每窗 z-score 后）中心 R 高度 |z|"); ax[0, 0].set_xlabel("|z| at R (±20 ms)"); ax[0, 0].legend(fontsize=8)
b2 = np.arange(-125, 126, 5)
ax[0, 1].hist(pkpos(tb[tl == 0]), b2, density=True, alpha=.45, color="g", label="训练 N"); ax[0, 1].hist(pkpos(incN), b2, density=True, histtype="step", lw=2, color="k", label="INCART 测试 N")
for k, c in zip(FILES, ["#1565c0", "#00838f", "#ef6c00", "#8e24aa"]): ax[0, 1].hist(pkpos(pa[k + "_W"]), b2, density=True, histtype="step", lw=2, color=c, label=k)
ax[0, 1].set_title("窗内 max|z| 相对窗中心的位置（样本 @250Hz）"); ax[0, 1].set_yscale("log"); ax[0, 1].legend(fontsize=8)
S = pd.read_csv(A / "spectra_norm.csv", index_col=0)
for k, c in zip(FILES, ["#1565c0", "#00838f", "#ef6c00", "#8e24aa"]): ax[1, 0].semilogy(S.index, S[k], color=c, lw=1.5, label=k)
for k in [c for c in S.columns if c.startswith("I")]: ax[1, 0].semilogy(S.index, S[k], color="0.6", lw=.7)
ax[1, 0].axvline(50, color="r", ls=":"); ax[1, 0].text(50.5, S.values.max() / 3, "100 Hz 采集 Nyquist", color="r", fontsize=8)
ax[1, 0].set_xlim(0, 80); ax[1, 0].set_title("模型链信号功率谱（1–40 Hz 归一；灰=INCART 测试）"); ax[1, 0].set_xlabel("Hz"); ax[1, 0].legend(fontsize=8)
for k, c in zip(FILES, ["#1565c0", "#00838f", "#ef6c00", "#8e24aa"]):
    cc, fs = load(k); adc = cc[:, 1] + 1.65; n = int(fs)
    sec = np.array([(adc[i * n:(i + 1) * n] <= .05).mean() for i in range(len(adc) // n)])
    ax[1, 1].plot(sec * 100, color=c, label=f"{k}（全程 {100*(adc<=.05).mean():.2f}%）")
ax[1, 1].set_title("ADC 触底（≤0.05 V）样本占比，逐秒"); ax[1, 1].set_xlabel("s"); ax[1, 1].set_ylabel("%"); ax[1, 1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "amp_input_domain_stats.png", dpi=120); plt.close(fig)
# ---- fig 2: gallery
G = np.load(A / "_gallery_tmp.npz", allow_pickle=True); t = (np.arange(250) - 125) / 250
nI = min(6, len(G["inc_W"])); lab = G["real_lab"]
idxN = np.where(lab == "N")[0]; sm = np.min([G["real_s_" + m][idxN] for m in M], 0); idxN = idxN[np.argsort(-sm)][:6]
fig, ax = plt.subplots(2, 6, figsize=(18, 6.5), sharey=True)
for i in range(6):
    a = ax[0, i]
    if i < nI:
        a.plot(t, G["trainV"], color="r", lw=2.5, alpha=.35, label="训练 V 中位模板"); a.plot(t, G["trainN"], color="g", lw=2.5, alpha=.35, label="训练 N 中位模板")
        a.plot(t, G["inc_W"][i], "k", lw=1.2); a.set_title(f"INCART {G['inc_rec'][i]} 真 V\n" + " ".join(f"{SH[m]}={G['inc_s_'+m][i]:.2f}" for m in M), fontsize=9)
    a = ax[1, i]; j = idxN[i]
    a.plot(t, G["trainV"], color="r", lw=2.5, alpha=.35); a.plot(t, G["trainN"], color="g", lw=2.5, alpha=.35)
    a.plot(t, G["real_W"][j], color="#1565c0", lw=1.2); a.set_title(f"real_145244 @{G['real_t'][j]:.1f}s 用户标 N\n" + " ".join(f"{SH[m]}={G['real_s_'+m][j]:.2f}" for m in M), fontsize=9)
ax[0, 0].legend(fontsize=7, loc="lower left"); ax[0, 0].set_ylabel("模型输入 (z)"); ax[1, 0].set_ylabel("模型输入 (z)")
for a in ax[1]: a.set_xlabel("s（R=0）")
fig.suptitle("(上) 公开库真 V 被正确高分；(下) 真机正常拍被误判高分 —— 模型输入视图（每窗 z-score，250 点）", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "amp_gallery_correctV_vs_falseV.png", dpi=120); plt.close(fig)
# ---- fig 3: strips hospital scale vs current
GAIN = 1100.0   # AD8232 module total gain (assumption: SparkFun-type 100x11); ADC V -> input mV = V/1.1
for key, t0 in [("real_145244", 40.0), ("real_155455", 20.0), ("real_143011", 30.0)]:
    cc, fs = load(key); d = cc[:, 2]
    b, a_ = sp.butter(2, [0.5 / (fs / 2), 40 / (fs / 2)], "band"); y = sp.lfilter(b, a_, d)
    i0, i1 = int(t0 * fs), int((t0 + 10) * fs); ts = np.arange(i1 - i0) / fs; mv = y[i0:i1] / GAIN * 1000
    fig = plt.figure(figsize=(25 / 2.54 * 1.0 + 1, 6 / 2.54 + 1 + 3.2))
    ax1 = fig.add_axes([0.04, 0.42, 0.94, 0.5]); ax2 = fig.add_axes([0.04, 0.07, 0.94, 0.26])
    for xx in np.arange(0, 10.001, 0.04): ax1.axvline(xx, color="#f8c8c8" if round(xx / 0.04) % 5 else "#e88", lw=.4 if round(xx / 0.04) % 5 else .8)
    for yy in np.arange(-3, 3.001, 0.1): ax1.axhline(yy, color="#f8c8c8" if round(yy * 10) % 5 else "#e88", lw=.4 if round(yy * 10) % 5 else .8)
    ax1.plot(ts, mv, "k", lw=.9); ax1.set_xlim(0, 10); ax1.set_ylim(-1.5, 1.5); ax1.set_aspect(0.4 / 1.0 * 1)  # 25 mm/s : 10 mm/mV -> 1 s = 2.5 mV visually
    ax1.set_title(f"{key} {t0:.0f}–{t0+10:.0f}s  医院标准刻度 25 mm/s、10 mm/mV（假设 AD8232 总增益 1100；0.5–40 Hz）", fontsize=9); ax1.set_ylabel("mV（输入折算）")
    ax2.plot(ts, d[i0:i1], color="#1565c0", lw=.8); ax2.set_xlim(0, 10); ax2.set_title("当前：串口 displaySample（ADC 伏，自动纵轴）", fontsize=9); ax2.set_ylabel("V")
    fig.savefig(FIG / f"amp_strip_{key}_hospital_vs_current.png", dpi=130); plt.close(fig)
print("figs ok")
