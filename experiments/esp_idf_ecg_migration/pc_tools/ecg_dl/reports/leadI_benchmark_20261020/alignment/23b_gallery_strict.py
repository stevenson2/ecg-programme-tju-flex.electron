"""Rebuild gallery with STRICT centre-QRS rule: 8-30Hz energy peak within ±40 ms of centre,
AND centre energy >= 50% of window max. Also show INCART Lead I N reference + real beats."""
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
B = s13.BENCH; FIG = B / "results/figures"; OUT = B / "results/alignment"
bq, aq = sp.butter(2, [8 / 125, 30 / 125], "band")
sc_ = s13.Scorer(); M = list(sc_.m.keys())
def is_strict(w):
    env = np.abs(sp.filtfilt(bq, aq, w)); qe = int(env.argmax()) - 125
    return abs(qe) <= 10 and env[125] >= 0.5 * env.max()
# collect strict V and N
strictV, strictN = [], []
for rid in s13.TEST_RIDS:
    nm = f"I{rid:02d}"; r = wfdb.rdrecord(str(s13.INCART_DIR / nm)); a = wfdb.rdann(str(s13.INCART_DIR / nm), "atr"); fs = float(r.fs)
    names = [s.upper() for s in r.sig_name]
    xi = r.p_signal[:, names.index("I")].astype(np.float64)
    xiii = r.p_signal[:, names.index("III")].astype(np.float64)
    gp, gc = s13.gt_beats(a.sample, a.symbol, fs)
    sig = s13.causal_filter_chain(s13.resample_ecg(xi, fs))
    for cls, bag in [(2, strictV), (0, strictN)]:
        pk = gp[gc == cls]
        if len(pk) < 2: continue
        W, used = s13.beats_at(sig, pk)
        S = np.vstack([sc_.score(m, W) for m in M])
        for j in range(len(W)):
            if not is_strict(W[j]): continue
            bag.append(dict(rec=nm, p=int(used[j]), W=W[j], S=S[:, j], xi=xi, xiii=xiii, fs=fs, sig=sig))
print(f"strict V={len(strictV)} / strict N={len(strictN)}")
# pick: 1 per record preferred, high min-score for V; mid for N
def pick(bag, n=4, by_score=True):
    bag = sorted(bag, key=lambda d: -d["S"].min() if by_score else abs(d["S"].mean() - 0.1))
    out, seen = [], set()
    for d in bag:
        if d["rec"] in seen: continue
        seen.add(d["rec"]); out.append(d)
        if len(out) >= n: break
    return out
Vs = pick(strictV, 4); Ns = pick(strictN, 2, by_score=False)
print("V picks", [(d["rec"], round(d["p"]/250,1), [round(float(x),2) for x in d["S"]]) for d in Vs])
print("N picks", [(d["rec"], round(d["p"]/250,1)) for d in Ns])
# real sessions
def load_real(fn, key):
    df = pd.read_csv(s13.INBOX / fn); parts = df["raw_line"].astype(str).str.split(",", expand=True)
    num = parts.iloc[:, :3].apply(pd.to_numeric, errors="coerce"); ok = (parts.notna().sum(1) >= 10) & num.notna().all(1)
    x = num[ok].iloc[:, 2].to_numpy(np.float64)
    meta = json.loads((s13.INBOX / fn.replace(".csv", "_meta.json")).read_text(encoding="utf-8"))
    fs = len(x) / float(meta["duration_s_actual"]); sig = s13.causal_filter_chain(s13.resample_ecg(x, fs))
    pk = np.load(s13.CACHE / f"{key}__clean__aiqrs.npz")["peaks"]; W, used = s13.beats_at(sig, pk)
    return sig, W, used, {m: sc_.score(m, W) for m in M}
sig160, W160, u160, S160 = load_real("serial_20261020_160942_wearable_single_lead_AD8232.csv", "real_160942")
sig145, W145, u145, S145 = load_real("serial_20261019_145244_leadI_standard_AD8232.csv", "real_145244")
labs = {}
for f in sorted((B / "results/labels").glob("ecg_labels_*.json")):
    labs = json.loads(f.read_text(encoding="utf-8"))["sessions"].get("real_145244", {}).get("beats", {})
# real_145244: labelled N with highest SH score
idxN = [i for i, p in enumerate(u145) if labs.get(str(int(p)), "") == "N"]
j145 = max(idxN, key=lambda i: S145["student_hard"][i]) if idxN else int(np.argmax(S145["student_hard"]))
j160 = int(np.argmax(S160["student_hard"]))
# figure
nrows = len(Vs) + len(Ns) + 2
fig, ax = plt.subplots(nrows, 3, figsize=(15, 2.35 * nrows))
def row(axr, sig, p, scores, title, xi=None, fs=None, xiii=None):
    # col0: continuous 2s filtered Lead I
    lo, hi = max(0, p - 250), min(len(sig), p + 250); t = (np.arange(lo, hi) - p) / 250
    axr[0].plot(t, sig[lo:hi], "k", lw=.85); axr[0].axvline(0, color="b", ls="--", lw=1.3)
    axr[0].set_xlim(-1, 1); axr[0].set_title(title + " · 连续2s Lead I（滤波）", fontsize=8); axr[0].set_ylabel("filt")
    # col1: RAW physical if available
    if xi is not None:
        po = int(round(p * fs / 250)); lo2, hi2 = max(0, po - int(fs)), min(len(xi), po + int(fs))
        t2 = (np.arange(lo2, hi2) - po) / fs
        axr[1].plot(t2, xi[lo2:hi2], "k", lw=.9, label="I")
        if xiii is not None: axr[1].plot(t2, xiii[lo2:hi2], color="0.55", lw=.7, label="III")
        axr[1].axvline(0, color="b", ls="--"); axr[1].legend(fontsize=6, loc="upper right")
        axr[1].set_xlim(-1, 1); axr[1].set_title("原始 mV（黑=I 灰=III）", fontsize=8)
    else:
        axr[1].plot(t, sig[lo:hi], "#1565c0", lw=.85); axr[1].axvline(0, color="b", ls="--")
        axr[1].set_xlim(-1, 1); axr[1].set_title("真机（无原始 mV；显示滤波值）", fontsize=8)
    # col2: model window z
    w = sig[p - 125:p + 125]; z = (w - w.mean()) / (w.std() + 1e-9); tw = (np.arange(250) - 125) / 250
    env = np.abs(sp.filtfilt(bq, aq, z)); env = env / (env.max() + 1e-9)
    axr[2].plot(tw, z, "k", lw=1.05); axr[2].fill_between(tw, -6, -6 + 2 * env, color="orange", alpha=.35)
    axr[2].axvline(0, color="b", ls="--"); axr[2].axvline((np.abs(z).argmax() - 125) / 250, color="r", ls=":", lw=1)
    axr[2].set_ylim(-6.5, 6.5)
    axr[2].set_title("模型窗 z  " + " ".join(f"{m[:2]}={float(v):.2f}" for m, v in zip(M, scores)), fontsize=8)
r = 0
for d in Vs:
    row(ax[r], d["sig"], d["p"], d["S"], f"INCART {d['rec']} 真V @{d['p']/250:.1f}s（严格：QRS能量峰在中心±40ms）", d["xi"], d["fs"], d["xiii"]); r += 1
for d in Ns:
    row(ax[r], d["sig"], d["p"], d["S"], f"INCART {d['rec']} 真N @{d['p']/250:.1f}s（对照：导联I直立R）", d["xi"], d["fs"], d["xiii"]); r += 1
row(ax[r], sig160, int(u160[j160]), [S160[m][j160] for m in M], f"real_160942 可穿戴 @{u160[j160]/250:.1f}s"); r += 1
row(ax[r], sig145, int(u145[j145]), [S145[m][j145] for m in M], f"real_145244 用户标N @{u145[j145]/250:.1f}s")
fig.suptitle("严格筛选图库：QRS 能量峰必须落在标注 R ±40 ms，且中心能量 ≥ 窗内最大的 50%\n"
             "通道：wfdb ch0=='I'（导联 I，mV）。左=滤波连续2s，中=原始mV（I/III），右=模型输入。蓝虚线=标注/检测中心。", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "amp_gallery_LeadI_strictCentreQRS_v4.png", dpi=115); plt.close(fig)
# also one hospital-scale raw Lead I with clear N and clear V from same record if possible
# prefer I05 or I35 that has both
dN = Ns[0]; dV = next((d for d in Vs if d["rec"] == dN["rec"]), Vs[0])
fig, ax = plt.subplots(2, 1, figsize=(14, 6))
def hosp_raw(ax, xi, fs, centre_s, title):
    t0 = max(0, centre_s - 2); i0, i1 = int(t0 * fs), int((t0 + 4) * fs); ts = np.arange(i1 - i0) / fs
    for xx in np.arange(0, 4.001, 0.04): ax.axvline(xx, color="#f8c8c8" if round(xx / 0.04) % 5 else "#e88", lw=.3 if round(xx / 0.04) % 5 else .7)
    for yy in np.arange(-1, 3.001, 0.1): ax.axhline(yy, color="#f8c8c8" if round(yy * 10) % 5 else "#e88", lw=.3 if round(yy * 10) % 5 else .7)
    y = xi[i0:i1]; ax.plot(ts, y - np.median(y), "k", lw=.9)  # remove DC for display
    ax.axvline(centre_s - t0, color="b", ls="--", lw=1.5)
    ax.set_xlim(0, 4); ax.set_ylim(-1.0, 2.0); ax.set_title(title, fontsize=9); ax.set_ylabel("mV")
hosp_raw(ax[0], dN["xi"], dN["fs"], dN["p"] / 250, f"INCART {dN['rec']} 原始导联 I · 真 N @ {dN['p']/250:.2f}s（蓝线）· 约 25mm/s 10mm/mV")
hosp_raw(ax[1], dV["xi"], dV["fs"], dV["p"] / 250, f"INCART {dV['rec']} 原始导联 I · 真 V @ {dV['p']/250:.2f}s（蓝线，严格中心 QRS）")
fig.tight_layout(); fig.savefig(FIG / "incart_LeadI_hospital_N_and_strictV.png", dpi=120); plt.close(fig)
# update note
(OUT / "NOTE_LeadI_Rpeak_audit.md").write_text(f"""# Lead I / R 峰核查结论（2026-10-20，修订）

## 1. 画的是哪一导？
- PhysioNet / wfdb：`sig_name = ['I','II','III','AVR','AVL','AVF','V1'..'V6']`
- **`p_signal[:,0]` = 导联 I，单位 mV。** 全部测试记录如此（`incart_channel_names.csv`）。
- 证据图：`incart_I01_raw_I_II_III_5s.png` — 导联 I 上 N 拍为清晰**直立 R**，标注竖线正好落在 R 上；II/III 形态符合肢体导联关系。
- **未发现通道错误或符号统一取反。** 不是 CRITICAL 训练数据 bug。

## 2. R 峰在哪里？
- **原始导联 I**：在标注样本处，N 拍 max|x| 偏移中位 1 样本（@257 Hz），100% 在 ±10 样本内；峰峰值中位约 1.3 mV。
- **模型 1 s 窗**：中心 = 标注换算到 250 Hz。因果滤波群延迟约 7 ms。
- **为何旧图库“看不到 R”：** 很多 INCART 的 V 在导联 I 上幅度很小（电轴接近垂直于 I）。此时窗内最高尖峰往往是约 −0.4 s 的前一个 N 拍，V 本身又低又宽。这是 Lead I 单导联物理限制，不是绘图把 R 画丢了，也不是采了 II/III。
- 诊断图 `incart_V_window_diagnostic_3beats.png`：同一时刻 III（灰）在 0 s 有大 QRS，而 I（黑）接近平坦 — 正是“V 在 I 上几乎看不见”的样子。

## 3. 严格筛选后的图库
规则：8–30 Hz QRS 能量峰必须在窗中心 **±40 ms**，且中心能量 ≥ 窗内最大的 50%。
- 测试集 V 窗：严格合格 **{len(strictV)}** 个（此前宽松“前 20%”会误放行邻居 N 峰主导的窗）。
- 新图：`amp_gallery_LeadI_strictCentreQRS_v4.png`（真 V + 真 N 对照 + real_160942 + real_145244）
- 医院刻度：`incart_LeadI_hospital_N_and_strictV.png`

## 4. 和真机形态
- INCART 标准肢体导联 I：多数 N 为直立 R。
- 真机可穿戴 / 单导 Lead I：深 S 常见，与公开库肢体 I 不必同形；不建议固件自动取反，应用负向 QRS 的 N 拍做训练增强。

## 5. CRITICAL？
- **否（通道/符号/对齐）。** 训练与测试用的就是导联 I，R 在 N 拍上对齐良好。
- **是风险（模型行为）：** 高分 V 窗里仍有相当比例在导联 I 上看不见中心 QRS，模型可能在利用早搏时序 / 邻拍，而非 V 形态。评估与增强时应按“中心是否有清晰 QRS”分层。
""", encoding="utf-8")
print("done")
