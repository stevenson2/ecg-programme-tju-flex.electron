#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_r20m_blind.py — R20M 盲切窗正例缓存（预注册 runs/R20M_PREREG.md §1）
================================================================================
从训练患者连续流生成任意相位 1s 盲切窗（R19M 失败机制的对症素材）:
  INCART 训练患者 (fs=360, lead I = p_signal[:,0], 与训练 npz 同通道)
  PTB    训练患者 (fs=1000, lead II = p_signal[:,1], 与 build_ptb 同通道)
  cinc2017 N 类训练记录 (fs=300, 通道 0)
链: corrected_deployment_chain(sig, fs) → 250Hz → 弃前 5s → 逐 1s 连续窗
→ 固件 z-score（总体 std, <1e-6 → 1）。每记录 ≤4 窗 (rng seed 42)，
三源缓存 npz 落 ECG_DATA（gitignored）。全部过 SplitGuard assert_train_only。

MIT 原始库本地缺失（§106）→ MIT 域盲窗由 INCART 替代（披露，预注册 §1）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import corrected_deployment_chain
from data.split_guard import get_guard

BASE = Path(__file__).resolve().parent
ECG_DATA = Path("/home/devcontainers/ecg_data")
OUT_NPZ = ECG_DATA / "r20m_blind_windows.npz"
AUDIT = BASE / "models" / "deploy_match" / "r20m_blind_windows_audit.json"

WIN = 250
SETTLE_S = 5.0
PER_REC = 4
SEED = 42
TARGETS = {"incart": 1000, "ptb": 1000, "cinc2017": 2000}


def blind_windows(sig, fs):
    s250 = corrected_deployment_chain(np.asarray(sig, dtype=np.float64), fs)
    start = int(SETTLE_S * 250)
    wins = []
    for i in range(start, len(s250) - WIN + 1, WIN):
        w = s250[i:i + WIN]
        mu = np.mean(w)
        std = np.sqrt(np.var(w))
        if std < 1e-6:
            std = 1.0
        wins.append((w - mu) / std)
    return wins


def main():
    import wfdb
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out = {}
    audit = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
             "policy": {"win": WIN, "settle_s": SETTLE_S,
                        "per_rec_max": PER_REC, "rng_seed": SEED,
                        "chain": "corrected_deployment_chain (D3 + HP0.5@250)"},
             "sources": {}}

    # ---------- INCART (lead I, fs=360) ----------
    from data.preprocess_incart import INCART_DIR
    g = get_guard("incart")
    rids = sorted(int(r) for r in g.train_record_ids())
    wins, used = [], []
    rec_names = [f"I{r:02d}" for r in rids]
    for name in rec_names:
        try:
            rec = wfdb.rdrecord(str(INCART_DIR / name))
            ws = blind_windows(rec.p_signal[:, 0], rec.fs)
        except Exception as e:  # noqa: BLE001
            audit["sources"].setdefault("incart_errors", []).append(
                {"rec": name, "err": str(e)[:80]})
            continue
        if not ws:
            continue
        sel = rng.choice(len(ws), min(PER_REC, len(ws)), replace=False)
        wins.extend([ws[i] for i in sel])
        used.append(name)
    out["incart_blind"] = np.asarray(
        wins[:TARGETS["incart"]], dtype=np.float32)
    audit["sources"]["incart"] = {"records_read": len(used),
                                  "windows": int(len(out["incart_blind"])),
                                  "lead": "p_signal[:,0] (Lead I, 同训练 npz)"}

    # ---------- PTB (lead II, fs=1000) ----------
    from data.preprocess_ptb import PTB_DIR, load_records as ptb_recs
    g = get_guard("ptb")
    train_rids = set(int(r) for r in g.train_record_ids())
    recs = ptb_recs()
    wins, used = [], []
    for i, name in enumerate(recs):
        rid = 400000 + i
        if rid not in train_rids:
            continue
        try:
            rec = wfdb.rdrecord(str(PTB_DIR / name))
            ws = blind_windows(rec.p_signal[:, 1], rec.fs)
        except Exception as e:  # noqa: BLE001
            audit["sources"].setdefault("ptb_errors", []).append(
                {"rec": name, "err": str(e)[:80]})
            continue
        if not ws:
            continue
        sel = rng.choice(len(ws), min(PER_REC, len(ws)), replace=False)
        wins.extend([ws[i] for i in sel])
        used.append(name)
    out["ptb_blind"] = np.asarray(wins[:TARGETS["ptb"]], dtype=np.float32)
    audit["sources"]["ptb"] = {"records_read": len(used),
                               "windows": int(len(out["ptb_blind"])),
                               "lead": "p_signal[:,1] (同 build_ptb)"}

    # ---------- cinc2017 N 类 (fs=300, 通道 0) ----------
    g = get_guard("cinc2017")
    rids = sorted(int(r) for r in g.train_record_ids())
    rng.shuffle(rids)   # 记录多样性优先
    wins, used = [], []
    for rid in rids:
        if len(wins) >= TARGETS["cinc2017"]:
            break
        name = "A%05d" % (rid - 200000)
        try:
            rec = wfdb.rdrecord(str(ECG_DATA / "cinc2017" / "training2017"
                                    / name))
            ws = blind_windows(rec.p_signal[:, 0], rec.fs)
        except Exception as e:  # noqa: BLE001
            audit["sources"].setdefault("cinc_errors", []).append(
                {"rec": name, "err": str(e)[:80]})
            continue
        if not ws:
            continue
        sel = rng.choice(len(ws), min(PER_REC, len(ws)), replace=False)
        wins.extend([ws[i] for i in sel])
        used.append(name)
    out["cinc_blind"] = np.asarray(wins[:TARGETS["cinc2017"]], dtype=np.float32)
    audit["sources"]["cinc2017"] = {"records_read": len(used),
                                    "windows": int(len(out["cinc_blind"]))}

    np.savez(OUT_NPZ, **out)
    audit["npz"] = str(OUT_NPZ)
    audit["wall_s"] = round(time.time() - t0, 1)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    for k, v in out.items():
        print(f"[R20M-BLIND] {k}: {v.shape}", flush=True)
    print(f"[R20M-BLIND] audit -> {AUDIT} ({audit['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
