#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecgfounder_embed_1lead.py — ECGFounder 1-lead 离线特征提取 / 硬负样本准备
================================================================================
用途：
  使用 PKUDigitalHealth/ECGFounder 的 1-lead 预训练模型，对
  1. PTB-XL Lead II 10s 记录
  2. 真实 AFE 单导联 10s 连续段
  提取 1024 维 deep features，供离线硬负样本挖掘/域距离分析。

输入：
  - PTB-XL 数据库（项目根 PTB-XL_ECG）
  - ECGFounder checkpoint：../ECGFounder/checkpoint/1_lead_ECGFounder.pth
  - 真实 AFE ECGR：pc_tools/ecg_dl/data/real/*.ecgr

输出：
  - pc_tools/ecg_dl/models/ecgfounder/ptbxl_1lead_features.npy
  - pc_tools/ecg_dl/models/ecgfounder/ptbxl_1lead_meta.json
  - pc_tools/ecg_dl/models/ecgfounder/real_afe_1lead_features.npy
  - pc_tools/ecg_dl/models/ecgfounder/real_afe_1lead_meta.json

用法：
  python3 pc_tools/ecg_dl/ecgfounder_embed_1lead.py --ptbxl --real
  python3 pc_tools/ecg_dl/ecgfounder_embed_1lead.py --real
"""
import sys, json, time, argparse
from pathlib import Path

import numpy as np
import torch

# 定位 ECGFounder 仓库（位于项目根上一级同名目录之外？实际为 项目根/../ECGFounder）
REPO = Path(__file__).resolve().parents[2]          # pc_tools/ecg_dl -> pc_tools -> 项目根
PROJECT_ROOT = REPO
ECGFOUNDER_DIR = PROJECT_ROOT.parent / "ECGFounder"
if not ECGFOUNDER_DIR.exists():
    # 兼容放在项目根内的情况
    ECGFOUNDER_DIR = PROJECT_ROOT / "ECGFounder"
sys.path.insert(0, str(ECGFOUNDER_DIR))

from net1d import Net1D
from util import filter_bandpass

MODEL_DIR = Path(__file__).resolve().parent / "models" / "ecgfounder"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CKPT_1LEAD = ECGFOUNDER_DIR / "checkpoint" / "1_lead_ECGFounder.pth"
PTBXL_DIR = PROJECT_ROOT / "PTB-XL_ECG"
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"

FS = 500
SEG_LEN = 5000    # 10s * 500Hz
TASKS = [l.strip() for l in open(ECGFOUNDER_DIR / "tasks.txt", encoding="utf-8") if l.strip()]

# 全局节律异常（先用于硬负样本挖掘候选；不代表可直接用于拍级训练）
GLOBAL_ABN = {
    "AFIB", "AFLT", "1AVB", "2AVB", "3AVB", "PACE",
    "SVARR", "SVTAC", "PSVT",
}
NORMAL_CODES = {"NORM", "SR", "SBRAD", "STACH", "SARRH"}
# 形态学/逐拍形态/结构标签，筛选时排除
STRUCT_FORM = set(
    "IMI ASMI AMI ALMI ILMI LMI IPLMI PMI LVH RVH LAO/LAE RAO/RAE SEHYP VCLVH "
    "ISC_ ISCAL ISCIN ISCIL ISCAS ISCLA INJAS INJAL INJIN INJIL NST_ DIG LOWT "
    "NT_ INVT TAB_ STE_ STD_ ABQRS QWAVE LVOLT HVOLT NDT ANEUR EL DTI CRBBB "
    "CLBBB IRBBB IVCD LAFB LPFB LPR WPW PVC PAC PRC(S) BIGU TRIGU".split()
)


def build_1lead_model():
    """加载 ECGFounder 1-lead 模型，返回 eval 模式 model。"""
    if not CKPT_1LEAD.exists():
        raise FileNotFoundError(f"缺少 checkpoint: {CKPT_1LEAD}")
    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
        return_features=True,
    )
    ck = torch.load(str(CKPT_1LEAD), map_location="cpu")
    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()
    return model


def ecgfounder_preprocess(sig_1d_500):
    """严格按 ECGFounder dataset.py 的单导联预处理：filter_bandpass + z-score。"""
    sig = np.asarray(sig_1d_500, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    filtered = filter_bandpass(sig, FS)
    # z-score per lead
    mu = filtered.mean(axis=-1, keepdims=True)
    sd = filtered.std(axis=-1, keepdims=True) + 1e-8
    z = (filtered - mu) / sd
    return z.astype(np.float32)


def embed_batch(model, inputs):
    """inputs: list[np.ndarray (1,5000)] -> (n,1024) features + (n,150) logits"""
    x = np.stack(inputs, axis=0).astype(np.float32)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        logits, feats = model(xt)
    return feats.numpy(), logits.numpy()


# ---------------- PTB-XL ----------------
def select_ptbxl_records(max_records=None):
    """返回候选记录行列表：全局节律异常 + 正常对照，排除结构/逐拍形态标签。"""
    import pandas as pd, ast
    if not PTBXL_CSV.exists():
        raise FileNotFoundError(PTBXL_CSV)
    db = pd.read_csv(PTBXL_CSV)
    db = db[db["validated_by_human"] == True].copy()
    rows = []
    for _, r in db.iterrows():
        codes = set(ast.literal_eval(r["scp_codes"]).keys())
        if codes & STRUCT_FORM:
            continue
        if codes & GLOBAL_ABN and (codes <= (GLOBAL_ABN | NORMAL_CODES)):
            rows.append((r, "abnormal"))
        elif codes <= NORMAL_CODES:
            rows.append((r, "normal"))
    # 控制平衡：异常全选，正常最多与异常等比或指定上限
    abn = [x for x in rows if x[1] == "abnormal"]
    norm = [x for x in rows if x[1] == "normal"]
    # 正常对照最多取与异常数量相同，并随机抽样固定种子
    if len(norm) > len(abn):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(norm), size=len(abn), replace=False)
        norm = [norm[i] for i in idx]
    selected = abn + norm
    if max_records:
        selected = selected[:max_records]
    return selected


def process_ptbxl(model, max_records=None):
    import pandas as pd
    rows = select_ptbxl_records(max_records=max_records)
    print(f"[PTBXL] selected {len(rows)} records", flush=True)
    feats_list, logits_list, meta = [], [], []
    t0 = time.time()
    batch = []
    batch_meta = []

    def flush():
        if not batch:
            return
        feats, logits = embed_batch(model, batch)
        feats_list.append(feats)
        logits_list.append(logits)
        meta.extend(batch_meta)
        batch.clear()
        batch_meta.clear()

    for i, (r, label) in enumerate(rows):
        try:
            rec = __import__("wfdb").rdrecord(str(PTBXL_DIR / r["filename_hr"]))
            lead2 = rec.p_signal[:, 1].astype(np.float64)
            if len(lead2) != SEG_LEN:
                # 严格对齐 10s
                n = int(len(lead2) * SEG_LEN / (rec.fs * 10))
                from scipy.signal import resample
                lead2 = resample(lead2, SEG_LEN)
            prep = ecgfounder_preprocess(lead2)
            batch.append(prep)
            import ast as _ast
            codes = _ast.literal_eval(r["scp_codes"])
            batch_meta.append({
                "ecg_id": int(r["ecg_id"]),
                "patient_id": str(r["patient_id"]),
                "filename_hr": r["filename_hr"],
                "label": label,
                "scp_codes": codes,
                "has_global_abn": bool(set(codes.keys()) & GLOBAL_ABN),
            })
            if len(batch) >= 64:
                flush()
        except Exception as e:
            print(f"  [skip] {r['filename_hr']}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
    flush()

    feats = np.concatenate(feats_list, axis=0)
    logits = np.concatenate(logits_list, axis=0)
    np.save(MODEL_DIR / "ptbxl_1lead_features.npy", feats.astype(np.float32))
    np.save(MODEL_DIR / "ptbxl_1lead_logits.npy", logits.astype(np.float32))
    with open(MODEL_DIR / "ptbxl_1lead_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "n": len(meta),
            "feature_dim": int(feats.shape[1]),
            "records": meta,
            "tasks": TASKS,
        }, f, ensure_ascii=False, indent=2)
    print(f"[PTBXL] saved {feats.shape} -> {MODEL_DIR / 'ptbxl_1lead_features.npy'}")


# ---------------- Real AFE ----------------
def read_ecgr(path):
    import struct
    raw = Path(path).read_bytes()
    if len(raw) < 32:
        raise ValueError(f"ECGR too short: {path}")
    dur = struct.unpack_from("<I", raw, 14)[0]
    n = struct.unpack_from("<I", raw, 18)[0]
    x = np.frombuffer(raw, dtype="<i2", count=n, offset=32).astype(np.float64) / 8000.0
    fs_eff = n / dur if dur else 0
    return x, fs_eff, dur


def ecgr_to_500(x, fs_eff):
    from fractions import Fraction
    from scipy.signal import resample_poly
    if abs(fs_eff - FS) < 1e-6:
        return x.astype(np.float64)
    ratio = Fraction(int(round(FS / fs_eff * 10000)), 10000).limit_denominator(100000)
    return resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float64)


def process_real(model):
    real_files = [
        PROJECT_ROOT / "pc_tools/ecg_dl/data/real/ecg_real_052.ecgr",
        PROJECT_ROOT / "rec_latest.ecgr",
    ]
    feats_list, logits_list, meta = [], [], []
    for path in real_files:
        if not path.exists():
            print(f"[REAL] skip missing {path}")
            continue
        x, fs_eff, dur = read_ecgr(path)
        s500 = ecgr_to_500(x, fs_eff)
        n_seg = len(s500) // SEG_LEN
        print(f"[REAL] {path.name} fs_eff={fs_eff:.2f} dur={dur}s -> 500Hz {len(s500)} samples, {n_seg} 10s segs", flush=True)
        for j in range(n_seg):
            seg = s500[j * SEG_LEN:(j + 1) * SEG_LEN]
            prep = ecgfounder_preprocess(seg)
            feats, logits = embed_batch(model, [prep])
            feats_list.append(feats[0])
            logits_list.append(logits[0])
            meta.append({
                "source_file": path.name,
                "segment_index": j,
                "start_s": round(j * 10, 3),
                "end_s": round((j + 1) * 10, 3),
                "label": "normal_real_afe",
                "fs_eff_hz": fs_eff,
            })
    if not feats_list:
        raise RuntimeError("No real AFE segments processed")
    feats = np.stack(feats_list, axis=0).astype(np.float32)
    logits = np.stack(logits_list, axis=0).astype(np.float32)
    np.save(MODEL_DIR / "real_afe_1lead_features.npy", feats)
    np.save(MODEL_DIR / "real_afe_1lead_logits.npy", logits)
    with open(MODEL_DIR / "real_afe_1lead_meta.json", "w", encoding="utf-8") as f:
        json.dump({"n": len(meta), "feature_dim": int(feats.shape[1]),
                   "segments": meta, "tasks": TASKS}, f, ensure_ascii=False, indent=2)
    print(f"[REAL] saved {feats.shape} -> {MODEL_DIR / 'real_afe_1lead_features.npy'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", action="store_true", help="提取 PTB-XL 单导联特征")
    ap.add_argument("--real", action="store_true", help="提取真实 AFE 单导联特征")
    ap.add_argument("--max-records", type=int, default=None, help="限制 PTB-XL 记录数")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    model = build_1lead_model()
    if args.ptbxl:
        process_ptbxl(model, max_records=args.max_records)
    if args.real:
        process_real(model)


if __name__ == "__main__":
    main()
