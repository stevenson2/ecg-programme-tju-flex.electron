"""Fixed gallery: INCART test true-V beats (1 per record, highest min-score across models) shown in model-input view
(Lead I, z) WITH centre marker + 8-30 Hz QRS energy envelope, plus the same 1 s of lead III (where V is large) to prove
the window is centred on the annotated V. Bottom: real_145244 user-N beats scored high. + alignment histogram figure."""
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
B = s13.BENCH; FIG = B / "results/figures"; M = ["student_hard", "distill_v2", "exp5_int8"]; SH = {"student_hard": "SH", "distill_v2": "D2", "exp5_int8": "E5"}
sc_ = s13.Scorer(); bq, aq = sp.butter(2, [8 / 125, 30 / 125], "band")
tz = np.load(B / "data/student_train_leadI.npz", allow_pickle=True); trN = np.median(tz["beats"][tz["labels_nsvq"] == 0][::3], 0); trV = np.median(tz["beats"][tz["labels_nsvq"] == 2], 0)
ex = []
for rid in s13.TEST_RIDS:
    nm = f"I{rid:02d}"; r = wfdb.rdrecord(str(s13.INCART_DIR / nm)); a = wfdb.rdann(str(s13.INCART_DIR / nm), "atr"); fs = float(r.fs)
    gp, gc = s13.gt_beats(a.sample, a.symbol, fs); vp = gp[gc == 2]
    if len(vp) < 5: continue
    sig = s13.causal_filter_chain(s13.resample_ecg(r.p_signal[:, 0], fs)); W, used = s13.beats_at(sig, vp)
    sig3 = s13.causal_filter_chain(s13.resample_ecg(r.p_signal[:, [s.upper() for s in r.sig_name].index("III")], fs))
    S = np.array([sc_.score(m, W) for m in M]); k = int(np.argmax(S.min(0)))
    p = used[k]; w3 = sig3[p - 125:p + 125]; w3 = (w3 - w3.mean()) / (w3.std() + 1e-9)
    ex.append(dict(rec=nm, W=W[k], W3=w3, s=S[:, k], t=p / 250))
lab = json.loads(sorted((B / "results/labels").glob("ecg_labels_*.json"))[-1].read_text(encoding="utf-8"))["sessions"]["real_145244"]["beats"]
G = np.load(B / "results/amp_analysis/_gallery_tmp.npz", allow_pickle=True)
idxN = np.where(G["real_lab"] == "N")[0]; idxN = idxN[np.argsort(-np.min([G["real_s_" + m][idxN] for m in M], 0))][:4]
t = (np.arange(250) - 125) / 250; n = len(ex)
fig, ax = plt.subplots(2, max(n, 4), figsize=(3.0 * max(n, 4), 7), sharey=True)
for i, e in enumerate(ex):
    a = ax[0, i]; env = np.abs(sp.filtfilt(bq, aq, e["W"])); env = env / env.max() * 3
    a.plot(t, trV, "r", lw=2.2, alpha=.3); a.plot(t, e["W3"] - 0, color="0.65", lw=1, label="同拍 III 导联 (z)")
    a.fill_between(t, -6, -6 + env, color="orange", alpha=.4, label="Lead I 8–30Hz 能量")
    a.plot(t, e["W"], "k", lw=1.3, label="模型输入 Lead I (z)"); a.axvline(0, color="b", ls=":", lw=1)
    a.set_title(f"INCART {e['rec']} 真 V @{e['t']:.0f}s\n" + " ".join(f"{SH[m]}={v:.2f}" for m, v in zip(M, e["s"])), fontsize=9)
for i in range(n, ax.shape[1]): ax[0, i].axis("off")
for i, j in enumerate(idxN):
    a = ax[1, i]; a.plot(t, trN, "g", lw=2.2, alpha=.35, label="训练 N 中位模板"); a.plot(t, trV, "r", lw=2.2, alpha=.3, label="训练 V 中位模板")
    a.plot(t, G["real_W"][j], color="#1565c0", lw=1.3); a.axvline(0, color="b", ls=":", lw=1)
    a.set_title(f"real_145244 @{G['real_t'][j]:.1f}s 用户标 N\n" + " ".join(f"{SH[m]}={G['real_s_'+m][j]:.2f}" for m in M), fontsize=9)
for i in range(4, ax.shape[1]): ax[1, i].axis("off")
ax[0, 0].legend(fontsize=7, loc="upper left"); ax[1, 0].legend(fontsize=7, loc="upper left"); ax[0, 0].set_ylim(-6.5, 7)
fig.suptitle("修正版图库：窗中心(蓝虚线)=标注 R；上排 INCART 测试真 V（每条记录取三模型最低分最高的一拍），灰线=同一时刻 III 导联，橙=Lead I QRS 能量包络\n"
             "注：I01/I65 等 V 在 Lead I 上幅度很小，窗内最高尖峰是前一个正常拍（约 −0.4 s），并非窗口错位", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "amp_gallery_correctV_vs_falseV_v2.png", dpi=115); plt.close(fig)
# alignment histogram
R = pd.read_csv(B / "results/alignment/window_alignment.csv")
z_ = np.load(B / "data/student_train_leadI.npz", allow_pickle=True); li = np.load(B / "data/leadI_incart_beats.npz", allow_pickle=True)
def off(W):
    E = np.abs(sp.filtfilt(bq, aq, W, axis=1)); return E[:, 85:166].argmax(1) - 40, np.abs(W).argmax(1) - 125
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for nm, W, C in [("训练集", z_["beats"], z_["labels_nsvq"]), ("INCART 测试", li["beats"][li["split"] == "test"], li["labels_nsvq"][li["split"] == "test"])]:
    for c, cn, col in [(0, "N", "g"), (2, "V", "r")]:
        o1, o2 = off(W[C == c][:20000])
        ax[0].hist(o1 * 4, np.arange(-160, 165, 8), density=True, histtype="step", lw=2, ls="-" if nm == "训练集" else "--", color=col, label=f"{nm} {cn}")
        ax[1].hist(o2 * 4, np.arange(-500, 505, 20), density=True, histtype="step", lw=2, ls="-" if nm == "训练集" else "--", color=col, label=f"{nm} {cn}")
ax[0].set_title("QRS 能量峰（8–30 Hz，±160 ms 内）相对窗中心 (ms)"); ax[1].set_title("整窗 max|x| 位置相对窗中心 (ms) —— V 拍常落在相邻正常拍上")
for a in ax: a.legend(fontsize=8); a.set_xlabel("ms"); a.set_yscale("log")
fig.tight_layout(); fig.savefig(FIG / "alignment_qrs_offset_hist.png", dpi=120)
print("ok", [e["rec"] for e in ex])
