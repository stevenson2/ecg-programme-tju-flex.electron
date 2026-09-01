#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_provenance.py — 历史训练脚本患者级泄漏总审计
================================================================================
逐一复现以下脚本对公共库的随机抽样 (seed=42 RNG 流按各脚本原始调用顺序重放),
并用 data/split_guard.py 的权威患者级划分审计抽样结果是否混入测试/验证患者:

  A. 精确复现 (RNG 流可完整重建):
     - finetune_exp7c.py            ★ 板上 exp7c INT8 模型的出身 (最高优先级)
     - finetune_exp7c_mild.py
     - finetune_exp7c_hardneg.py    (先复现其 synth_hard 的 RNG 消耗)
     - finetune_exp7c_ecgfounder.py / _v2 / _v3 / _v4
     - qat_exp7c.py / qat_exp7c_v3.py / qat_exp7c_v3b.py / qat_exp7c_v4.py / qat_exp7c_v5.py
  B. 代码静态审计 (其掩码对齐方式):
     - finetune_exp7c_v4.py         (train INCART 掩码疑似错位)
     - qat_exp7c_v6_clean.py        (对照: 正确的对齐)

结论写入 models/deploy_match/provenance_leakage_audit.json。
运行环境: WSL (需 /home/devcontainers/ecg_data 或 ECG_PROCESSED_DIR)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import (
    get_guard, compute_mit_incart_split, load_arrays, LeakError,
)
from config import BEAT_WINDOW_SAMPLES

BASE = Path(__file__).resolve().parent
OUT = BASE / "models" / "deploy_match" / "provenance_leakage_audit.json"
SEED = 42

# (脚本, 抽样顺序与配额 [(tag, n_abn, n_norm), ...], 是否在公共库抽样前有 RNG 消耗)
SCRIPTS = [
    ("finetune_exp7c.py",            [("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 100)], None),
    ("finetune_exp7c_mild.py",       [("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 100)], None),
    ("finetune_exp7c_hardneg.py",    [("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 100)], "synth_hard"),
    ("finetune_exp7c_ecgfounder.py", [("mit_bih", 800, 200), ("incart", 200, 100), ("ptb", 300, 100)], None),
    ("finetune_exp7c_ecgfounder_v2.py", [("mit_bih", 800, 200), ("incart", 200, 100), ("ptb", 300, 100)], None),
    ("finetune_exp7c_ecgfounder_v3.py", [("mit_bih", 800, 200), ("incart", 200, 100), ("ptb", 300, 100)], None),
    ("finetune_exp7c_ecgfounder_v4.py", [("mit_bih", 800, 200), ("incart", 200, 100), ("ptb", 300, 100)], None),
    ("qat_exp7c.py",                 [("mit_bih", 1000, 300), ("incart", 300, 100), ("ptb", 500, 100)], None),
    ("qat_exp7c_v3.py",              [("mit_bih", 800, 200), ("incart", 200, 100), ("ptb", 800, 200)], None),
    ("qat_exp7c_v3b.py",             [("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 150)], None),
    ("qat_exp7c_v4.py",              [("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 150)], None),
    ("qat_exp7c_v5.py",              [("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 150)], None),
]


def replicate_synth_hard(rng, n):
    """逐调用复现 finetune_exp7c_hardneg.py 的 synth_hard(real[n]) RNG 消耗。"""
    for _db in (10, 20):
        rng.normal(0, 1.0, (n, BEAT_WINDOW_SAMPLES))
    for n_imp in (2, 5, 10):
        for _ in range(n_imp):
            rng.integers(0, BEAT_WINDOW_SAMPLES)
            rng.uniform(0.5, 2.0)
            rng.choice([-1, 1])
    for _frac in (0.1, 0.25):
        rng.random((n, BEAT_WINDOW_SAMPLES))


def replicate(script, specs, pre):
    rng = np.random.default_rng(SEED)
    if pre == "synth_hard":
        real1 = np.load(BASE / "data" / "real" / "real_normal_beats_exp7c.npy")
        real2 = np.load(BASE / "data" / "real" / "real_normal_beats_rec_latest.npy")
        replicate_synth_hard(rng, len(real1) + len(real2))
    picked = {}
    for tag, n_abn, n_norm in specs:
        _b, l, r = load_arrays(tag)
        l = np.asarray(l)
        r = np.asarray(r)
        ia = np.where(l == 1)[0]
        inn = np.where(l == 0)[0]
        sa = rng.choice(ia, min(n_abn, len(ia)), replace=False)
        sn = rng.choice(inn, min(n_norm, len(inn)), replace=False)
        picked[tag] = np.concatenate([r[sa], r[sn]])
    return picked


def audit_script(script, specs, pre, guards):
    picked = replicate(script, specs, pre)
    per_tag, leaked = {}, False
    for tag, sel in picked.items():
        rep = guards[tag].audit_sampled_records(sel, context=script)
        rep.pop("context", None)
        per_tag[tag] = rep
        leaked = leaked or rep["leaked"]
    return {"script": script, "rng_replicated": True,
            "pre_sampling_rng": pre, "per_tag": per_tag, "leaked": leaked}


def audit_v4_alignment():
    """finetune_exp7c_v4.py 的 INCART train 掩码对齐审计。
    该脚本把合并划分掩码按 [:len(incart_labels)] 截断后作用于 INCART,
    而合并数组前 658962 条是 MIT → 若 len(inc)<len(mit) 则取到的是 MIT 掩码。"""
    m = compute_mit_incart_split()
    merged_rids = m["merged_rids"]
    g_mi_mit = get_guard("mit_bih")
    g_mi_inc = get_guard("incart")
    n_mit, n_inc = m["n_mit"], m["n_inc"]
    merged_tr = np.concatenate([g_mi_mit.train_mask, g_mi_inc.train_mask])
    assert len(merged_tr) == len(merged_rids)

    # v4 的实际做法: mask = train_mask[:len(incart_labels)]
    v4_inc_mask = merged_tr[:n_inc]
    # 正确做法: mask = merged_tr[n_mit : n_mit + n_inc]
    correct_inc_mask = merged_tr[n_mit:n_mit + n_inc]

    inc_r = np.asarray(load_arrays("incart")[2])
    v4_kept = np.unique(inc_r[v4_inc_mask])
    correct_kept = np.unique(inc_r[correct_inc_mask])
    inc_test = set(g_mi_inc.test_record_ids().tolist())
    inc_val = set(g_mi_inc.val_record_ids().tolist())
    v4_leaked_test = sorted(rid for rid in v4_kept if rid in inc_test)
    v4_leaked_val = sorted(rid for rid in v4_kept if rid in inc_val)

    # 该掩码下可被 sample_domain 抽到的 INCART 心拍总量
    n_beats_available = int(v4_inc_mask.sum())
    return {
        "script": "finetune_exp7c_v4.py",
        "bug": ("sample_domain 用合并掩码的 [:len(incart)] 前缀充当 INCART "
                "train 掩码; 合并数组前 %d 条是 MIT 心拍, 前缀截断取到的是 "
                "MIT 的患者划分状态, 对 INCART 等价于随机过滤" % n_mit),
        "len_mit_beats": int(n_mit),
        "len_inc_beats": int(n_inc),
        "inc_records_wrongly_kept": len(v4_kept),
        "inc_records_correct_train": len(correct_kept),
        "inc_test_records_passing_filter": v4_leaked_test,
        "inc_val_records_passing_filter": v4_leaked_val,
        "inc_beats_available_under_wrong_mask": n_beats_available,
        "leaked": len(v4_leaked_test) > 0,
    }


def _to_native(o):
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, np.ndarray):
        return _to_native(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    guards = {tag: get_guard(tag) for tag in ("mit_bih", "incart", "ptb")}

    results = []
    for script, specs, pre in SCRIPTS:
        print(f"[AUDIT] {script} ...", flush=True)
        try:
            res = audit_script(script, specs, pre, guards)
        except LeakError as e:
            res = {"script": script, "error": str(e)}
        except FileNotFoundError as e:
            res = {"script": script, "error": f"数据缺失, 无法复现: {e}"}
        results.append(res)
        tags = res.get("per_tag", {})
        summary = " ".join(
            f"{t}:test={v['test']}/{v['sampled']}" for t, v in tags.items())
        print(f"        → LEAKED={res.get('leaked')} {summary}", flush=True)

    v4 = audit_v4_alignment()
    print(f"[AUDIT] finetune_exp7c_v4.py 掩码对齐: LEAKED={v4['leaked']} "
          f"(错误掩码放行的 INCART 测试记录: {v4['inc_test_records_passing_filter']})",
          flush=True)

    any_leaked = any(r.get("leaked") for r in results) or v4["leaked"]
    mi_st = get_guard("mit_bih").stats
    pt_st = get_guard("ptb").stats
    report = {
        "audit_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": ("按各脚本原始 RNG 调用顺序重放 seed=42 抽样, 用 split_guard "
                   "权威患者级划分 (合并 MIT+INCART 患者 %d: train %d/val %d/test %d; "
                   "PTB 患者 %d: train %d/val %d/test %d) 审计"
                   % (mi_st["n_patients"], mi_st["n_train"], mi_st["n_val"], mi_st["n_test"],
                      pt_st["n_patients"], pt_st["n_train"], pt_st["n_val"], pt_st["n_test"])),
        "scripts": results,
        "v4_mask_alignment": v4,
        "verdict": "LEAKED" if any_leaked else "CLEAN",
        "implication": ("板上 exp7c INT8 的训练数据出身如含测试患者, 则其 "
                        "FINAL_RESULTS 患者级指标与论文数字需重评"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_to_native(report), indent=2, ensure_ascii=False))
    print(f"[AUDIT] verdict: {report['verdict']} → {OUT}")


if __name__ == "__main__":
    main()
