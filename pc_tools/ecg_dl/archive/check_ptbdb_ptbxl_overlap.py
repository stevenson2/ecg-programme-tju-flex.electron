#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_ptbdb_ptbxl_overlap.py — ptbdb(KD 训练源) ↔ PTB-XL(MI 评估集) 患者重叠核查

背景（innovation_and_rigor_audit.md 漏洞#2）：
  KD 心梗专家在 PTB-XL 上报告记录级 MI AUC 0.777。若 ptbdb（KD 训练数据源）
  与 PTB-XL（评估集）存在同一患者，该 AUC 为泄漏值，双专家心梗腿结论作废。
  本脚本将 2026-08-20 交互式会话中执行的三层核查协议固化为可复现产物。

三层协议：
  L1 粗指纹扫描：ptbdb 全部患者各段记录取 10s 探针（lead II，1000Hz→125Hz 粗化，
     z-score）对全部 PTB-XL 记录（500Hz→125Hz 粗化，z-score）算 Pearson 相关，
     取 top-K 可疑对。
  L2 多导联全分辨率复核：top-K 对逐对做 500Hz 全分辨率滑移对齐 + 12 导联相关。
     同源判据：12 导联全部 ≥ 0.95（同源信号在所有导联都应高相关；
     仅下壁导联高而 I/V 导联崩溃 = 形态模板化相似，非同源）。
  L3 元数据交叉验证：比对可疑对的年龄/性别（ptbdb 头注释 vs PTB-XL database.csv）。
     同源判据：性别一致且年龄差 ≤ 5 岁。

判定：任一对同时通过 L2+L3 → 泄漏成立，M0 结论作废；否则无重叠证据。

依赖：仅 numpy + 标准库（不依赖 wfdb/scipy/tensorflow）。
用法（项目根目录）：
  python pc_tools/ecg_dl/check_ptbdb_ptbxl_overlap.py                 # 全量
  python pc_tools/ecg_dl/check_ptbdb_ptbxl_overlap.py --limit-xl 300  # 冒烟测试
输出：pc_tools/ecg_dl/models/ptbdb_ptbxl_overlap_check.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent          # pc_tools/ecg_dl
ROOT = REPO.parent.parent                        # 项目根
PTBDB_DIR = ROOT / "ECG-Database"
PTBXL_DIR = ROOT / "PTB-XL_ECG"
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"
OUT_JSON = REPO / "models" / "ptbdb_ptbxl_overlap_check.json"
XL_CACHE = REPO / "models" / "cache_ptbxl_lead2_125hz.npy"

PROBE_FS = 125          # 粗化后采样率
PROBE_LEN = PROBE_FS * 10   # 10s 探针 = 1250 点
FULL_FS = 500           # L2 全分辨率采样率（ptbdb 1000→500 对齐 PTB-XL 500）
SHIFT_MAX = 250         # L2 滑移搜索范围 ±250 样本 @500Hz = ±0.5s
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_II_IDX = 1


# ---------------- WFDB 手工解析（format 16 = interleaved int16） ----------------

def _num(field):
    """'6.2(mV)' -> 6.2；'-32768(0)' -> -32768。"""
    return float(str(field).split("(")[0])


def parse_wfdb_header(header_path: Path):
    """解析 WFDB 头文件。

    返回 (fs, n_samples, n_ch, gains, baselines, comments, n_dat_ch)。
    n_dat_ch = .dat 文件内的实际通道数（ptbdb 的 12 导联在 .dat，
    3 个 Frank 导联在独立 .xyz 文件——reshape 必须用各自文件的通道数）。
    """
    gains, baselines = [], []
    fs, n_samples, n_ch = None, None, None
    n_dat_ch = 0
    comments = []
    with open(header_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                comments.append(line.lstrip("# ").strip())
                continue
            parts = line.split()
            if not parts:
                continue
            if ".dat" in parts[0] and len(parts) >= 4:
                # 通道行（.dat 文件内）
                try:
                    gains.append(_num(parts[2]))
                    baselines.append(_num(parts[3]))
                    n_dat_ch += 1
                except ValueError:
                    pass
            elif fs is None and len(parts) >= 4 and parts[1].isdigit():
                # 记录行（仅首个；防止 .xyz 行覆盖 fs/n_samples）
                try:
                    n_ch = int(parts[1])
                    fs = float(parts[2])
                    n_samples = int(parts[3])
                except ValueError:
                    pass
    return fs, n_samples, n_ch, gains, baselines, comments, n_dat_ch


def read_wfdb_channel(dat_path: Path, ch_idx: int, n_ch: int, gain: float,
                      baseline: float, max_samples: int = 0):
    """读取单通道 int16 数据并转物理单位（mV）。"""
    raw = np.fromfile(dat_path, dtype="<i2")
    if n_ch > 1:
        raw = raw.reshape(-1, n_ch)[:, ch_idx]
    sig = (raw.astype(np.float64) - baseline) / max(gain, 1e-9)
    return sig[:max_samples] if max_samples > 0 else sig


def parse_comment(comments, key):
    """从头注释取字段值（如 'age: 60' -> '60'）。"""
    for c in comments:
        if c.lower().startswith(key.lower()):
            return c.split(":", 1)[1].strip()
    return ""


# ---------------- 探针提取与相关计算 ----------------

def coarsen_z(sig, src_fs):
    """整数倍抽取粗化到 PROBE_FS 后 z-score（与原会话'粗化'语义一致）。"""
    step = int(round(src_fs / PROBE_FS))
    x = sig[::step][:PROBE_LEN]
    if len(x) < PROBE_LEN:
        return None
    x = x - x.mean()
    sd = x.std()
    return x / sd if sd > 1e-9 else None


def pearson(a, b):
    b = b - b.mean()
    sa = b.std()
    if sa < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (len(a) * sa))


def collect_ptbdb_probes():
    """遍历 ptbdb 全部记录，每段取 1-3 个 10s 探针（短记录 1 个，长记录均匀多点）。"""
    probes, meta = [], []
    rec_lines = [l.strip() for l in open(PTBDB_DIR / "RECORDS") if l.strip()
                 and not l.startswith("#")]
    for rel in rec_lines:
        base = PTBDB_DIR / rel
        hdr = base.with_suffix(".hea")
        if not hdr.exists():
            continue
        fs, n_samples, n_ch, gains, baselines, _, n_dat_ch = parse_wfdb_header(hdr)
        if not fs or not n_samples or len(gains) <= LEAD_II_IDX:
            continue
        dur_s = n_samples / fs
        if dur_s < 12:
            continue
        # 探针起点：<60s 取 1 个；60-120s 取 2 个；更长取 3 个（均匀）
        n_probe = 1 if dur_s < 60 else (2 if dur_s < 120 else 3)
        starts = [int(dur_s * i / (n_probe + 1)) for i in range(1, n_probe + 1)]
        dat = base.with_suffix(".dat")
        need = int(max(starts) * fs + PROBE_FS * 10 * (fs / PROBE_FS)) + 1
        sig = read_wfdb_channel(dat, LEAD_II_IDX, n_dat_ch, gains[LEAD_II_IDX],
                                baselines[LEAD_II_IDX], max_samples=min(n_samples, need))
        for s_start in starts:
            seg = sig[int(s_start * fs):]
            p = coarsen_z(seg, fs)
            if p is not None:
                probes.append(p)
                meta.append(rel)
    return np.asarray(probes, dtype=np.float32), meta


def build_xl_matrix(limit=0):
    """全部 PTB-XL 记录 lead II 粗化探针矩阵（带磁盘缓存）。"""
    rows = list(csv.DictReader(open(PTBXL_CSV, encoding="utf-8")))
    names = [r["filename_hr"] for r in rows]
    if limit:
        rows = rows[:limit]
        names = names[:limit]
    elif XL_CACHE.exists():
        return np.load(XL_CACHE), names
    mat = np.zeros((len(rows), PROBE_LEN), dtype=np.float32)
    names = []
    for i, r in enumerate(rows):
        dat = PTBXL_DIR / (r["filename_hr"] + ".dat")
        try:
            fs, n_samples, n_ch, gains, baselines, _, n_dat_ch = parse_wfdb_header(
                dat.with_suffix(".hea"))
            sig = read_wfdb_channel(dat, LEAD_II_IDX, n_dat_ch, gains[LEAD_II_IDX],
                                    baselines[LEAD_II_IDX], max_samples=n_samples)
        except Exception:
            sig = np.zeros(0)
            fs = 500
        p = coarsen_z(sig, fs if fs else 500)
        mat[i] = p if p is not None else 0.0
        names.append(r["filename_hr"])
        if (i + 1) % 2000 == 0:
            print(f"  PTB-XL {i+1}/{len(rows)}")
    if limit == 0:
        np.save(XL_CACHE, mat)
    return mat, names


# ---------------- L2 多导联全分辨率复核 ----------------

def load_pair_full(ptbdb_rel, xl_hr):
    """读一对记录的 12 导联 @500Hz（ptbdb 1000Hz 整数倍抽取）。"""
    def ptb_leads(rel):
        hdr = (PTBDB_DIR / rel).with_suffix(".hea")
        fs, n_samples, n_ch, gains, baselines, comments, n_dat_ch = parse_wfdb_header(hdr)
        dat = (PTBDB_DIR / rel).with_suffix(".dat")
        raw = np.fromfile(dat, dtype="<i2").reshape(-1, n_dat_ch).astype(np.float64)
        step = int(round(fs / FULL_FS))
        raw = raw[::step][:FULL_FS * 10]
        leads = [(raw[:, c] - baselines[c]) / max(gains[c], 1e-9) for c in range(min(12, n_dat_ch))]
        return leads, fs, comments

    def xl_leads(hr):
        dat = PTBXL_DIR / (hr + ".dat")
        hdr = dat.with_suffix(".hea")
        fs, n_samples, n_ch, gains, baselines, _, n_dat_ch = parse_wfdb_header(hdr)
        raw = np.fromfile(dat, dtype="<i2").reshape(-1, n_dat_ch).astype(np.float64)
        raw = raw[:FULL_FS * 10]
        leads = [(raw[:, c] - baselines[c]) / max(gains[c], 1e-9) for c in range(min(12, n_dat_ch))]
        return leads

    pl, _, comments = ptb_leads(ptbdb_rel)
    xl = xl_leads(xl_hr)
    n = min(len(pl[0]), len(xl[0]))
    pl = [x[:n] for x in pl]
    xl = [x[:n] for x in xl]
    return pl, xl, comments


def aligned_corr(pl, xl):
    """先用 lead II 搜索最优滑移（±SHIFT_MAX），再按该偏移算 12 导联相关。"""
    ref_p = pl[LEAD_II_IDX] - pl[LEAD_II_IDX].mean()
    ref_x = xl[LEAD_II_IDX] - xl[LEAD_II_IDX].mean()
    best_shift, best_c = 0, -2.0
    for sh in range(-SHIFT_MAX, SHIFT_MAX + 1, 5):
        if sh >= 0:
            a, b = ref_p[sh:], ref_x[:len(ref_x) - sh]
        else:
            a, b = ref_p[:sh], ref_x[-sh:]
        m = min(len(a), len(b))
        if m < PROBE_LEN // 2:
            continue
        c = pearson(a[:m] / (a[:m].std() + 1e-12), b[:m])
        if c > best_c:
            best_c, best_shift = c, sh
    sh = best_shift
    out = {}
    for li, name in enumerate(LEADS):
        a = pl[li][sh:] if sh >= 0 else pl[li][:sh]
        b = xl[li][:len(xl[li]) - sh] if sh >= 0 else xl[li][-sh:]
        m = min(len(a), len(b))
        a = a[:m] - np.mean(a[:m])
        b = b[:m] - np.mean(b[:m])
        denom = a.std() * b.std() * m
        out[name] = float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0
    return best_shift, out


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description="ptbdb↔PTB-XL 患者重叠核查（三层协议）")
    ap.add_argument("--top-k", type=int, default=5, help="L1 取前 K 可疑对进入 L2/L3")
    ap.add_argument("--limit-xl", type=int, default=0, help="限制 PTB-XL 记录数（冒烟测试）")
    ap.add_argument("--age-tol", type=float, default=5.0, help="L3 年龄差容差（岁）")
    args = ap.parse_args()

    t0 = time.time()
    print("[L1] 收集 ptbdb 探针 ...")
    probes, probe_names = collect_ptbdb_probes()
    print(f"  ptbdb 探针: {len(probes)} 个（来自 {len(set(n.split('/')[0] for n in probe_names))} 患者）")

    print("[L1] 构建 PTB-XL 探针矩阵 ...")
    xl_mat, xl_names = build_xl_matrix(args.limit_xl)
    n_xl = xl_mat.shape[0]
    print(f"  PTB-XL 记录: {n_xl}")

    # 批量 Pearson：两侧均已 z-score，corr = 点积 / n
    corr = (probes @ xl_mat.T) / PROBE_LEN
    flat_idx = np.argsort(corr.ravel())[::-1][:args.top_k * 10]
    seen, candidates = set(), []
    for fi in flat_idx:
        pi, xi = divmod(int(fi), n_xl)
        key = (pi, xi)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"ptbdb_record": probe_names[pi], "xl_record": (
            xl_names[xi] if xl_names else f"row{xi}"), "coarse_corr": round(float(corr[pi, xi]), 4)})
        if len(candidates) >= args.top_k:
            break
    print(f"[L1] top-{len(candidates)} 粗相关: "
          + ", ".join(f"{c['coarse_corr']:.4f}" for c in candidates))

    # L2 + L3
    xl_rows = {r["filename_hr"]: r for r in csv.DictReader(open(PTBXL_CSV, encoding="utf-8"))}
    results = []
    for cand in candidates:
        entry = dict(cand)
        try:
            pl, xl, comments = load_pair_full(cand["ptbdb_record"], cand["xl_record"])
            shift, lead_corr = aligned_corr(pl, xl)
            entry["shift_samples"] = shift
            entry["lead_corr"] = {k: round(v, 4) for k, v in lead_corr.items()}
            entry["lead_corr_mean"] = round(float(np.mean(list(lead_corr.values()))), 4)
            entry["l2_same_source"] = bool(min(lead_corr.values()) >= 0.95)
        except Exception as e:
            entry["l2_error"] = str(e)
            entry["l2_same_source"] = None
            results.append(entry)
            continue
        # L3 元数据
        xl_row = xl_rows.get(cand["xl_record"], {})
        p_age = parse_comment(comments, "age")
        p_sex = parse_comment(comments, "sex")
        x_age, x_sex = xl_row.get("age", ""), xl_row.get("sex", "")
        entry["metadata"] = {
            "ptbdb": {"age": p_age, "sex": p_sex},
            "ptbxl": {"patient_id": xl_row.get("patient_id", ""),
                      "age": x_age, "sex": x_sex},
        }
        try:
            age_diff = abs(float(p_age) - float(x_age)) if p_age and x_age else None
        except ValueError:
            age_diff = None
        sex_ok = bool(p_sex and x_sex and p_sex.lower()[0] == x_sex.lower()[0])
        age_ok = bool(age_diff is not None and age_diff <= args.age_tol)
        entry["l3"] = {"age_diff": age_diff, "sex_match": sex_ok,
                       "consistent": bool(sex_ok and age_ok)}
        entry["leak"] = bool(entry.get("l2_same_source") and entry["l3"]["consistent"])
        results.append(entry)
        print(f"[L2/L3] {cand['ptbdb_record']} vs {cand['xl_record']}: "
              f"12导联均值 {entry.get('lead_corr_mean')} 同源判据={entry.get('l2_same_source')} "
              f"年龄差={entry['l3']['age_diff']} 性别一致={entry['l3']['sex_match']}")

    leak_confirmed = any(r.get("leak") for r in results)
    report = {
        "protocol": {
            "l1": "ptbdb lead II 10s 探针(1000→125Hz 粗化,z-score) × PTB-XL 全量(500→125Hz) Pearson",
            "l2": f"top-K 对 500Hz 全分辨率滑移对齐(±{SHIFT_MAX}) 12 导联相关；同源判据=12 导联全部≥0.95",
            "l3": f"年龄差≤{args.age_tol} 且性别一致；ptbdb 头注释 vs ptbxl_database.csv",
        },
        "n_probes_ptbdb": len(probes),
        "n_records_ptbxl": n_xl,
        "candidates": results,
        "verdict": "LEAK CONFIRMED" if leak_confirmed else "NO OVERLAP EVIDENCE",
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n===== 判定: {report['verdict']} =====")
    print(f"报告已保存: {OUT_JSON}")
    return 0 if not leak_confirmed else 2


if __name__ == "__main__":
    sys.exit(main())
