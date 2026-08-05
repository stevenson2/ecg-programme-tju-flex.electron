#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.4-4 患者级划分一致性验证 (蹊跷点 2/3/7)
==========================================
检查1 (蹊跷点3): 历史 eval 脚本 rng.choice(无放回) vs patient_split.py
  rng.permutation 在同一 seed=42 下选出的测试患者集合是否一致。
检查2 (蹊跷点7): PTB 3-beat test 拍数 13,322 > 250点 test 13,058 之谜
  (stitch_3beat 丢边界拍 → 患者集合漂移?)。
检查3 (蹊跷点2): 历史 37 条 MIT npz 是否含增强拍 (与"新旧数据域可比性"相关)。

运行 (WSL2 Ubuntu):
  cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl
  python3 verify_split_consistency.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_DIR
from data.patient_split import build_ptb_patient_map, patient_level_split
from data.preprocess_3beat import stitch_3beat

SEED = 42
TEST_FRAC = 0.2

RECORDS = next((Path(c) for c in [
    r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master\ECG-Database\RECORDS",
    "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/ECG-Database/RECORDS",
] if Path(c).exists()), None)
if RECORDS is None:
    raise RuntimeError("RECORDS 文件未找到")


def fmt(n):
    return f"{n:,}"


# ============================================================
print("=" * 64)
print("检查1 — 蹊跷点3: seed42 下 choice(无放回) vs permutation")
print("=" * 64)
recs = [l.strip() for l in open(RECORDS) if l.strip()]
patients = sorted({r.split("/")[0] for r in recs})
n = len(patients)
n_test = max(1, int(n * TEST_FRAC))
print(f"PTB 患者总数: {n}, n_test: {n_test}")

# 历史 eval_ptb_holdout.py / eval_deploy_decision.py 等 4 个脚本的逻辑
rng = np.random.default_rng(SEED)
test_choice = set(rng.choice(patients, n_test, replace=False))

# patient_split.py 的逻辑
rng2 = np.random.default_rng(SEED)
perm = rng2.permutation(n)
test_perm = set(patients[i] for i in perm[:n_test])

same = test_choice == test_perm
print(f"choice 测试患者: {len(test_choice)}, permutation 测试患者: {len(test_perm)}")
print(f"两集合完全一致: {same}  | 交集: {len(test_choice & test_perm)}/{n_test}")
if not same:
    only_choice = sorted(test_choice - test_perm)[:10]
    only_perm = sorted(test_perm - test_choice)[:10]
    print(f"  仅在 choice 侧 (前10): {only_choice}")
    print(f"  仅在 perm   侧 (前10): {only_perm}")
verdict1 = "一致" if same else f"不一致 (交集 {len(test_choice & test_perm)}/{n_test})"

# ============================================================
print()
print("=" * 64)
print("检查2 — 蹊跷点7: PTB 3-beat test 拍数 13,322 vs 13,058")
print("=" * 64)
ptb_npz = PROCESSED_DIR / "ptb_processed.npz"
d = np.load(ptb_npz)
beats, labels, rids = d["beats"], d["labels"], d["record_ids"]
print(f"ptb_processed.npz: {fmt(len(beats))} 拍, keys={list(d.keys())}")

pmap = build_ptb_patient_map()
_, _, te, stats = patient_level_split(rids, pmap)
print(f"[250点] 患者数 {stats['n_patients']} | test 患者 {stats['n_test']} | "
      f"test 拍 {fmt(te.sum())}")

x3, y3, rids3 = stitch_3beat(beats, labels, rids)
print(f"[3-beat] stitch 后序列: {fmt(len(x3))} (原 {fmt(len(beats))}, "
      f"差 {fmt(len(beats) - len(x3))})")

rid_before = set(np.unique(rids).tolist())
rid_after = set(np.unique(rids3).tolist())
vanished = sorted(rid_before - rid_after)
print(f"[3-beat] unique record: {len(rid_before)} -> {len(rid_after)}, "
      f"stitch 后整记录消失: {len(vanished)} 条")
if vanished:
    idx = {400000 + i: recs[i] for i in range(len(recs))}
    print(f"  消失记录: {[idx.get(int(r), str(r)) for r in vanished[:10]]}"
          f"{' ...' if len(vanished) > 10 else ''}")
    # 消失记录导致的患者丢失
    pat_before = {pmap.get(int(r), "?") for r in rid_before}
    pat_after = {pmap.get(int(r), "?") for r in rid_after}
    lost_pats = sorted(pat_before - pat_after)
    print(f"  因此丢失的患者: {len(lost_pats)} 个 {lost_pats[:10]}")

_, _, te3, stats3 = patient_level_split(rids3, pmap)
print(f"[3-beat] 患者数 {stats3['n_patients']} | test 患者 {stats3['n_test']} | "
      f"test 拍 {fmt(te3.sum())}")

tp_250 = set(stats["test_patients"])
tp_3b = set(stats3["test_patients"])
same_tp = tp_250 == tp_3b
print(f"两次划分 test 患者集合一致: {same_tp} | 交集: {len(tp_250 & tp_3b)}/{len(tp_250)}")
if not same_tp:
    print(f"  仅250点侧 (前10): {sorted(tp_250 - tp_3b)[:10]}")
    print(f"  仅3beat侧 (前10): {sorted(tp_3b - tp_250)[:10]}")
    verdict2 = (f"两次划分的 test 患者集合不同 (患者总数 {stats['n_patients']} vs "
                f"{stats3['n_patients']}, permutation 输入变化 → 测试患者漂移), "
                f"拍数 13,058 vs 13,322 不可直接比较")
else:
    diff = int(te3.sum()) - int(te.sum())
    verdict2 = (f"test 患者集合相同, 但 3-beat test 拍反而多 {fmt(diff)} —— "
                f"需进一步核查 (stitch 可能跨记录拼接或重复)")

# ============================================================
print()
print("=" * 64)
print("检查3 — 蹊跷点2: 历史 37 条 MIT npz 是否含增强拍")
print("=" * 64)
old_med, old_total = 0, 0
for tag, p in [("旧37条备份", PROCESSED_DIR / "mit_bih_processed_37rec_backup.npz"),
               ("新48条", PROCESSED_DIR / "mit_bih_processed.npz")]:
    if not p.exists():
        print(f"[{tag}] 文件缺失: {p.name}")
        continue
    dd = np.load(p)
    b, r = dd["beats"], dd["record_ids"]
    u, c = np.unique(r, return_counts=True)
    print(f"[{tag}] keys={list(dd.keys())} | 总拍 {fmt(len(b))} | "
          f"形状 {b.shape} {b.dtype} | 记录数 {len(u)} | "
          f"每记录拍 min/med/max = {fmt(c.min())}/{fmt(int(np.median(c)))}/{fmt(c.max())}")
    if "labels" in dd:
        lb = dd["labels"]
        print(f"         标签: N={fmt((lb == 0).sum())} A={fmt((lb == 1).sum())}")
    if tag.startswith("旧"):
        old_med = int(np.median(c))
        old_total = len(b)
print(f"推算: 无增强时 MIT 每记录约 2,000-3,000 拍; 6 倍增强应约 12,000-18,000 拍/记录")
verdict3 = (f"旧37条数据{'含' if old_med > 6000 else '不含'}增强拍 "
            f"(中位 {fmt(old_med)} 拍/记录, 总 {fmt(old_total)})")

# ============================================================
print()
print("=" * 64)
print("VERDICT")
print("=" * 64)
print(f"检查1 (seed42 语义): {verdict1}")
print(f"检查2 (3-beat 拍数): {verdict2}")
print(f"检查3 (旧npz增强):  {verdict3}")
