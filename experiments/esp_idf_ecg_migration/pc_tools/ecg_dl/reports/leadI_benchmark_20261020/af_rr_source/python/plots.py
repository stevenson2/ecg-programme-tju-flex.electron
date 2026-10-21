"""Figures for the AF RR-source report. No raw WFDB files are written."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _fp in (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
):
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from constants import MIN_RR_SEC
from io_wfdb import load_signal
from sources import SOURCE_NAMES


SRC_LABEL = {
    "ann": "标注峰",
    "ann_manual": "人工修正峰",
    "fw_strict": "固件严格",
    "fw_aiqrs": "固件宽松",
    "xqrs": "XQRS",
}


def _bar_with_ci(ax, labels, vals, lo, hi, ylabel, title):
    x = np.arange(len(labels))
    vals = np.asarray(vals, dtype=float)
    yerr = np.vstack([
        np.clip(vals - np.asarray(lo, dtype=float), 0, None),
        np.clip(np.asarray(hi, dtype=float) - vals, 0, None),
    ])
    ax.bar(x, vals, yerr=yerr, capsize=3, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def _grab(summary, split, setting, scheme, src, key):
    blob = (
        summary.get("by_split", {})
        .get(split, {})
        .get(setting, {})
        .get(scheme, {})
        .get(src)
    )
    if not blob:
        return None, None, None
    v = blob["pooled"].get(key)
    ci = blob.get("ci95", {}).get(key, [None, None])
    return v, (ci[0] if ci else None), (ci[1] if ci else None)


def make_all_plots(root: Path, per_record, summary, rec_payloads, example_candidates, data_dir: Path):
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    split = summary.get("hypothesis_verdict", {}).get("split_used") or "test_ltaf"
    if split not in summary.get("by_split", {}) or not summary["by_split"][split].get("default", {}).get("af_only"):
        split = "test_mit" if "test_mit" in summary.get("by_split", {}) else "dev"

    srcs = [s for s in SOURCE_NAMES if _grab(summary, split, "default", "af_only", s, "event_se")[0] is not None]
    labels = [SRC_LABEL.get(s, s) for s in srcs]

    # 1. event Se default vs calibrated
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, setting, title in zip(axes, ("default", "calibrated"), ("固件默认阈值", "开发集校准阈值")):
        vals, lo, hi = [], [], []
        for s in srcs:
            v, a, b = _grab(summary, split, setting, "af_only", s, "event_se")
            vals.append(v if v is not None else np.nan)
            lo.append(a if a is not None else np.nan)
            hi.append(b if b is not None else np.nan)
        _bar_with_ci(ax, labels, vals, lo, hi, "事件灵敏度", f"{title} · {split} · 仅AF")
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_event_se.png", dpi=140)
    plt.close(fig)

    # 2. FP/h
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, setting, title in zip(axes, ("default", "calibrated"), ("固件默认阈值", "开发集校准阈值")):
        vals, lo, hi = [], [], []
        for s in srcs:
            v, a, b = _grab(summary, split, setting, "af_only", s, "fp_per_hour_total")
            vals.append(v if v is not None else np.nan)
            lo.append(a if a is not None else np.nan)
            hi.append(b if b is not None else np.nan)
        _bar_with_ci(ax, labels, vals, lo, hi, "误报事件 / h（总时长分母）", f"{title} · {split}")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_fph_total.png", dpi=140)
    plt.close(fig)

    # 3. analyzable + delay
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vals = [(_grab(summary, split, "default", "af_only", s, "analyzable_frac")[0] or np.nan) for s in srcs]
    axes[0].bar(labels, vals, color="#54A24B")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("可分析时间占比（拒判单独计入，不从分母剔除）")
    axes[0].tick_params(axis="x", rotation=20)
    vals = [(_grab(summary, split, "default", "af_only", s, "mean_delay_s")[0] or np.nan) for s in srcs]
    axes[1].bar(labels, vals, color="#F58518")
    axes[1].set_title("检出延迟均值 (s)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_analyzable_delay.png", dpi=140)
    plt.close(fig)

    # 4. mechanism QRS se AF vs non-AF
    mech = (
        summary.get("mechanism_pooled", {}).get("gold_peaks")
        or summary.get("mechanism_pooled", {}).get(split)
        or {}
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(srcs))
    w = 0.35
    afs = [mech.get(s, {}).get("qrs_se_af", np.nan) for s in srcs]
    nas = [mech.get(s, {}).get("qrs_se_non_af", np.nan) for s in srcs]
    ax.bar(x - w / 2, afs, w, label="AF 段")
    ax.bar(x + w / 2, nas, w, label="非 AF 段")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("QRS 灵敏度 vs 参考峰")
    ax.set_title("检测器在 AF / 非 AF 段的漏检是否不对称")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_qrs_se_af_vs_naf.png", dpi=140)
    plt.close(fig)

    # 5. short RR fraction
    fig, ax = plt.subplots(figsize=(7, 4))
    ref_s = [mech.get(s, {}).get("short_rr_frac_ref_af", np.nan) for s in srcs]
    det_s = [mech.get(s, {}).get("short_rr_frac_det_af", np.nan) for s in srcs]
    ax.bar(x - w / 2, ref_s, w, label="参考 RR < 0.48 s")
    ax.bar(x + w / 2, det_s, w, label="该路径 RR < 0.48 s")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("AF 段短 RR 占比")
    ax.set_title(f"固件 MIN_RR={MIN_RR_SEC:.3f}s 是否截断 AF 短 RR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_short_rr_af.png", dpi=140)
    plt.close(fig)

    # 6. RR histograms from payloads (fw_strict vs reviewed peaks on AF)
    _rr_hist(fig_dir, rec_payloads)

    # 7. miss reason stacked
    _miss_reason(fig_dir, per_record, split)

    # 8. example waveforms
    _example_waveforms(fig_dir, data_dir, rec_payloads, example_candidates)

    print(f"[plots] wrote {list(fig_dir.glob('*.png'))}", flush=True)


def _rr_hist(fig_dir: Path, rec_payloads):
    bins = np.linspace(0.2, 1.6, 29)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, src, title in zip(
        axes,
        ("ann_manual", "fw_strict"),
        ("人工修正 / 参考 RR", "固件严格 RR"),
    ):
        af, na = [], []
        for p in rec_payloads:
            t = p["peaks"].get(src, np.zeros(0))
            if t.size < 3:
                continue
            rr = np.diff(t)
            end = t[1:]
            y = p["y_af"]
            idx = np.floor(end).astype(int)
            ok = (idx >= 0) & (idx < len(y))
            in_af = np.zeros(rr.size, dtype=bool)
            in_af[ok] = y[idx[ok]]
            af.append(rr[in_af])
            na.append(rr[~in_af])
        af = np.concatenate(af) if af else np.zeros(0)
        na = np.concatenate(na) if na else np.zeros(0)
        if af.size:
            ax.hist(af, bins=bins, density=True, alpha=0.6, label=f"AF n={af.size}")
        if na.size:
            ax.hist(na, bins=bins, density=True, alpha=0.5, label=f"非AF n={na.size}")
        ax.axvline(MIN_RR_SEC, color="r", ls="--", label="MIN_RR 0.48s")
        ax.set_title(title)
        ax.set_xlabel("RR (s)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_rr_hist_af_vs_naf.png", dpi=140)
    plt.close(fig)


def _miss_reason(fig_dir: Path, per_record, split):
    rows = [r for r in per_record if r["split"] == split and r["setting"] == "default" and r["scheme"] == "af_only"]
    srcs = [s for s in SOURCE_NAMES if any(r["source"] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(srcs))
    hit = []
    nrm = []
    und = []
    for s in srcs:
        rs = [r for r in rows if r["source"] == s]
        tp = sum(r.get("event_tp") or 0 for r in rs)
        fn0 = sum(r.get("event_fn_called_normal") or 0 for r in rs)
        fn2 = sum(r.get("event_fn_undetermined") or 0 for r in rs)
        tot = tp + fn0 + fn2
        if tot == 0:
            hit.append(0); nrm.append(0); und.append(0)
        else:
            hit.append(tp / tot); nrm.append(fn0 / tot); und.append(fn2 / tot)
    ax.bar(x, hit, label="检出")
    ax.bar(x, nrm, bottom=hit, label="漏检·判为正常")
    ax.bar(x, und, bottom=np.asarray(hit) + np.asarray(nrm), label="漏检·无法判定")
    ax.set_xticks(x)
    ax.set_xticklabels([SRC_LABEL.get(s, s) for s in srcs], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("AF 事件去向（拒判不从分母剔除）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_event_disposition.png", dpi=140)
    plt.close(fig)


def _example_waveforms(fig_dir: Path, data_dir: Path, rec_payloads, example_candidates):
    by_key = {(p["db"], p["rec"]): p for p in rec_payloads}

    def _rank(e):
        db = e.get("db")
        if db == "ltafdb":
            return 0
        if db == "mitdb":
            return 1
        return 2

    misses = sorted(
        [e for e in example_candidates if e.get("kind") == "miss"],
        key=_rank,
    )
    fps = sorted(
        [e for e in example_candidates if e.get("kind") == "fp"],
        key=_rank,
    )
    picks = []
    for pool, limit in ((misses, 2), (fps, 2)):
        n_from = 0
        seen_rec = set()
        for e in pool:
            key = (e["db"], e["rec"])
            if key not in by_key or key in seen_rec:
                continue
            if float(e.get("start_s") or 0) < 60:
                continue
            picks.append(e)
            seen_rec.add(key)
            n_from += 1
            if n_from >= limit:
                break
    if not picks:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "无可用波形示例（可能该次运行没有漏检/误报事件）", ha="center")
        ax.axis("off")
        fig.savefig(fig_dir / "fig_examples.png", dpi=140)
        plt.close(fig)
        return
    n = min(4, len(picks))
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), squeeze=False)
    for ax, e in zip(axes[:, 0], picks[:n]):
        db, rec = e["db"], e["rec"]
        try:
            sig = load_signal(data_dir / db, rec)
        except Exception as ex:
            ax.text(0.5, 0.5, f"load fail {db}/{rec}: {ex}", ha="center")
            continue
        fs = sig["fs"]
        t0 = max(0.0, float(e["start_s"]) - 2.0)
        t1 = min(sig["duration_s"], float(e["end_s"]) + 2.0)
        if t1 - t0 > 25:
            t1 = t0 + 20
        i0, i1 = int(t0 * fs), int(t1 * fs)
        x = np.arange(i0, i1) / fs
        y = sig["sig"][i0:i1]
        ax.plot(x, y, color="k", lw=0.6)
        payload = by_key[(db, rec)]
        for src, col, ms in (("ann", "g", 8), ("fw_strict", "r", 6), ("fw_aiqrs", "b", 5)):
            t = payload["peaks"].get(src, np.zeros(0))
            tt = t[(t >= t0) & (t <= t1)]
            if tt.size:
                yi = np.interp(tt, x, y) if y.size else np.zeros_like(tt)
                ax.scatter(tt, yi, s=ms, c=col, label=SRC_LABEL.get(src, src), zorder=3)
        ax.axvspan(e["start_s"], min(e["end_s"], t1), color="orange", alpha=0.15)
        kind = "漏检 AF 事件" if e["kind"] == "miss" else "误报事件"
        extra = e.get("miss_reason", "")
        ax.set_title(f"{kind} {db}/{rec} {e['start_s']:.0f}–{e['end_s']:.0f}s {extra}")
        ax.set_xlim(t0, t1)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_examples.png", dpi=140)
    plt.close(fig)
