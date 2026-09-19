#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_r21_ptb_corpus.py — W0: 语料 PTB 域扩容 (R21, TH §120)
================================================================================
目的: TP 保护判据去单点化 —— 既有语料的 PTB test TP 判据只有 ludb_normal
(= PTB patient001/s0010_re, §109 勘误后确认为 TP) 一条硬线 (§119 披露)。
本轮从 PTB **test** 患者另取 6 名 (3 MI + 3 非 MI 异常), 经与 ludb_normal
逐字相同的水配方 (lead II, 1000→500Hz, 5-35s, 去均值) 生成新语料 case。

段编号: seg41..46 (3 + 31 + 7 + i, corpus3 独立头文件, 仿 corpus2 先例
§114: 既有 31+7 case 逐字节不动)。

产出:
  1. experiments/.../include/signal_generator/ecg_replay_corpus3.h (int16 µV)
  2. pc_tools/ecg_dl/corpus/ptb_corpus_500hz.npy (PC 缓存, gitignored)
  3. pc_tools/ecg_dl/corpus/ptb_corpus_provenance.json (来源/划分/MD5/审计)

泄漏披露: 新 case 全部 PTB **test** 患者 (评测侧素材, 无训练接触);
划分断言经 data/split_guard.py (seed 42 权威划分) 在脚本内执行。

生成纪律 (沿 make_replay_corpus.py / make_leadoff_corpus.py):
  - RNG 不涉及 (纯载波段, 无噪声混合) → 无新 rng 流消耗;
  - 既有 corpus1/corpus2 的 int16 MD5 在本脚本内重验 (逐字节不变核验)。
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import ecg_refs as _REFS

REPO = BASE.parent.parent
HDR_DIR = (REPO / "experiments" / "esp_idf_ecg_migration" / "components" /
           "ecg_core" / "include" / "signal_generator")
CORPUS_DIR = BASE / "corpus"

FS = 500
DUR_S = 30
START_S = 5                      # 与 ludb_normal 水配方一致 (避开开头瞬态)
SEG_BASE = 3 + 31 + 7            # 41
N_CASES = 6

# 选择冻结 (R21 W0): PTB test 患者, 全部 label=1 (非 CONTROLS), MI + 非 MI。
# 诊断摘自各记录 .hea 注释; rid = 400000 + RECORDS 行号。
SELECTION = [
    # (case 名, 记录名, 患者号, 预期诊断标签)
    ("ptbtp_p003_mi",   "patient003/s0017lre",  "p003", "MI"),
    ("ptbtp_p013_mi",   "patient013/s0045lre",  "p013", "MI"),
    ("ptbtp_p025_mi",   "patient025/s0087lre",  "p025", "MI"),
    ("ptbtp_p157_dysr", "patient157/s0338lre",  "p157", "dysrhythmia (AF)"),
    ("ptbtp_p167_cmp",  "patient167/s0200_re",  "p167", "cardiomyopathy (HOCM)"),
    ("ptbtp_p217_bbb",  "patient217/s0439_re",  "p217", "bundle branch block"),
]


def verify_split_membership(record_names):
    """断言: 每条记录属 PTB test 划分 (seed 42) 且非健康对照 (label=1)。"""
    sys.path.insert(0, str(BASE))
    from data.split_guard import get_guard
    g = get_guard("ptb")
    test_rids = set(int(r) for r in g.test_record_ids())
    recs = [l.strip() for l in
            open(_REFS.ECG_DATABASE_DIR / "RECORDS") if l.strip()]
    rec2rid = {rec: 400000 + i for i, rec in enumerate(recs)}
    controls = {l.strip() for l in
                open(_REFS.ECG_DATABASE_DIR / "CONTROLS") if l.strip()}
    out = {}
    for rec in record_names:
        rid = rec2rid[rec]
        assert rid in test_rids, f"{rec} (rid {rid}) 不在 PTB test 划分"
        assert rec not in controls, f"{rec} 在 CONTROLS (label=0), 不入 TP 组"
        out[rec] = rid
    return out


def parse_diagnosis(rec):
    """从 .hea 注释解析诊断 (provenance 披露用)。"""
    txt = (_REFS.ECG_DATABASE_DIR / (rec + ".hea")).read_text(errors="ignore")
    m = re.findall(r"#\s*(Reason for admission|Additional diagnoses)"
                   r"\s*:\s*(.*)", txt)
    return {k.strip(): v.strip() for k, v in m}


def load_ptb_lead_ii(rec, start_s=START_S, dur_s=DUR_S):
    """PTB 记录 lead II, 1000Hz → 500Hz; 与 ludb_normal 水配方逐字一致。"""
    import wfdb
    from scipy.signal import resample_poly
    r = wfdb.rdrecord(str(_REFS.ECG_DATABASE_DIR / rec),
                      sampfrom=int(start_s * 1000),
                      sampto=int((start_s + dur_s) * 1000))
    assert r.fs == 1000, f"{rec}: fs={r.fs} != 1000 (水配方假设破坏)"
    ch = [s.lower() for s in r.sig_name].index("ii")
    sig = r.p_signal[:, ch].astype(np.float64)
    return resample_poly(sig, 1, 2).astype(np.float32)


def audit_existing_corpora():
    """逐字节不变核验: corpus1 (31 case) + corpus2 (7 case) 的 int16 MD5。"""
    report = {}
    c1 = np.load(CORPUS_DIR / "corpus_500hz.npy")
    q1 = np.clip(np.round(c1.astype(np.float32) * 1000.0),
                 -32768, 32767).astype(np.int16)
    p1 = json.loads((CORPUS_DIR / "replay_corpus_provenance.json")
                    .read_text(encoding="utf-8"))
    n_ok = 0
    for c in p1["cases"]:
        md5 = hashlib.md5(q1[c["index"]].tobytes()).hexdigest()
        assert md5 == c["md5_int16uv"], f"corpus1 case {c['name']} MD5 漂移"
        n_ok += 1
    report["corpus1_cases_md5_match"] = n_ok

    c2 = np.load(CORPUS_DIR / "leadoff_corpus_500hz.npy")
    q2 = np.clip(np.round(c2.astype(np.float32) * 1000.0),
                 -32768, 32767).astype(np.int16)
    p2 = json.loads((CORPUS_DIR / "leadoff_corpus_provenance.json")
                    .read_text(encoding="utf-8"))
    n_ok = 0
    for c in p2["cases"]:
        md5 = hashlib.md5(q2[c["index"]].tobytes()).hexdigest()
        assert md5 == c["md5_int16uv"], f"corpus2 case {c['name']} MD5 漂移"
        n_ok += 1
    report["corpus2_cases_md5_match"] = n_ok
    return report


def gen_header(cases, q):
    names = [c[0] for c in cases]
    lines = []
    lines.append("/* Auto-generated by pc_tools/ecg_dl/build_r21_ptb_corpus.py (R21 W0, TH §120)")
    lines.append(" * PTB test 域语料: %d case x %d 样本 (30s @500Hz), int16 微伏。" % (len(cases), q.shape[1]))
    lines.append(" * TP 保护判据去单点化: PTB test 患者 (3 MI + 3 非 MI), 评测侧素材无训练接触。")
    lines.append(" * 逐 case 来源/划分/MD5: pc_tools/ecg_dl/corpus/ptb_corpus_provenance.json")
    lines.append(" * 仅 N16R8 (16MB flash) 编入; 由 CMake ECG_REPLAY_CORPUS 宏开关。 */")
    lines.append("#ifndef ECG_REPLAY_CORPUS3_H")
    lines.append("#define ECG_REPLAY_CORPUS3_H")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#define ECG_REPLAY_CORPUS3_N %d" % len(cases))
    lines.append("#define ECG_REPLAY_CORPUS3_LEN %d" % q.shape[1])
    lines.append("")
    lines.append("/* case 名字表 (与 provenance JSON 的 cases[i].name 一一对应) */")
    lines.append("static const char *const ecg_corpus3_names[ECG_REPLAY_CORPUS3_N] = {")
    for n in names:
        lines.append('    "%s",' % n)
    lines.append("};")
    lines.append("")
    lines.append("/* int16 微伏语料数组 [case][样本] */")
    lines.append("static const int16_t ecg_corpus3[ECG_REPLAY_CORPUS3_N][ECG_REPLAY_CORPUS3_LEN] = {")
    for i in range(len(cases)):
        row = q[i]
        lines.append("    { /* %d (seg%d): %s */" % (i, SEG_BASE + i, names[i]))
        for j in range(0, len(row), 16):
            lines.append("        " + ", ".join(str(int(v)) for v in row[j:j + 16]) + ",")
        lines.append("    },")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* ECG_REPLAY_CORPUS3_H */")
    return "\n".join(lines) + "\n"


def main():
    audit = audit_existing_corpora()
    print(f"[audit] 既有语料逐字节核验: {audit}", flush=True)

    rec2rid = verify_split_membership([r for _, r, _, _ in SELECTION])
    print(f"[split] 6 记录全部 PTB test + label=1 断言通过", flush=True)

    cases = []      # (name, meta, sig_500hz_mV)
    for name, rec, pat, dx_tag in SELECTION:
        sig = load_ptb_lead_ii(rec)
        sig = sig - np.mean(sig)
        p2p = float(np.ptp(sig))
        assert 0.2 < p2p < 8.0, f"{rec}: p2p {p2p:.3f} mV 超出物理合理带"
        cases.append((name, {
            "carrier": f"PTB_{rec}_leadii_{START_S}_{START_S+DUR_S}s",
            "noise": None, "fs_src": 1000, "diagnosis_tag": dx_tag,
            "diagnosis_hea": parse_diagnosis(rec),
            "record_id": rec2rid[rec], "split": "ptb_test (seed 42)",
            "gt_label_semantics": 1,
            "erratum_note": ("PTB test 患者, 非健康对照 → 部署标签语义 "
                             "label=1 (preprocess_ptb.py 规则)"),
        }, sig))

    assert len(cases) == N_CASES

    arr = np.stack([c[2] for c in cases]).astype(np.float32)
    q = np.clip(np.round(arr * 1000.0), -32768, 32767).astype(np.int16)

    # ---- 输出 1: 固件头 corpus3 ----
    hdr_out = HDR_DIR / "ecg_replay_corpus3.h"
    hdr_out.write_text(gen_header(cases, q), encoding="utf-8")

    # ---- 输出 2/3: npy 缓存 + provenance ----
    np.save(CORPUS_DIR / "ptb_corpus_500hz.npy", arr)
    prov = {
        "generator": "build_r21_ptb_corpus.py",
        "seed": None,
        "fs": FS,
        "dur_s": DUR_S,
        "n_cases": N_CASES,
        "segment_base": SEG_BASE,
        "units": "header int16 microvolt (x0.001 -> mV numeric at runtime)",
        "recipe": ("lead II, 1000->500Hz resample_poly(1,2), "
                   f"{START_S}-{START_S+DUR_S}s, mean-removed "
                   "(identical to make_replay_corpus ludb_normal)"),
        "purpose": ("R21 W0: TP 保护判据去单点化 (ludb=s0010 单 case 硬线 → "
                    "ptb_tp_group ≥6 case)"),
        "existing_corpora_audit": audit,
        "leakage_note": ("全部 case 为 PTB test 患者 (评测侧素材, 无训练接触); "
                         "划分断言经 data/split_guard.py seed 42 在生成时执行。"),
        "cases": [],
    }
    for i, (name, meta, sig) in enumerate(cases):
        prov["cases"].append({
            "index": i, "segment": SEG_BASE + i, "name": name,
            "params": meta,
            "md5_int16uv": hashlib.md5(q[i].tobytes()).hexdigest(),
            "p2p_mV": round(float(np.ptp(sig)), 4),
            "rms_mV": round(float(np.sqrt(np.mean(sig ** 2))), 4),
        })
    (CORPUS_DIR / "ptb_corpus_provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ptb-corpus] {N_CASES} cases (seg{SEG_BASE}..{SEG_BASE+N_CASES-1}) "
          f"-> {hdr_out.name}")
    print(f"[ptb-corpus] header {hdr_out.stat().st_size/1024:.0f}KB, "
          f"payload {q.nbytes/1024:.0f}KB")
    for i, (name, _, sig) in enumerate(cases):
        print(f"  seg{SEG_BASE+i} {name}: p2p {np.ptp(sig):.2f} mV, "
              f"rms {np.sqrt(np.mean(sig**2)):.3f} mV")


if __name__ == "__main__":
    main()
