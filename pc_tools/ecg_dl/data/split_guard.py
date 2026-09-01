#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""split_guard.py — 患者级划分守卫 (防泄漏单一事实源)
================================================================================
背景: 多个训练/微调脚本对公共库数据做无患者过滤的随机抽样 (finetune_exp7c.py,
qat_exp7c*.py 等), 且 data/processed/patient_split.json 已陈旧 (MIT 36 患者 vs
实际 47)。本模块提供:

  1. 权威划分重算: 永远从实际 *_processed_deploy_causal_record_ids.npy 数组 +
     患者映射重新计算 (seed=42), 不信任任何已保存的 JSON。语义与
     finetune_exp7c_v4.py / qat_exp7c_v6_clean.py 的干净口径完全一致:
       - MIT+INCART 合并划分 (INCART record_id +100000 偏移)
       - PTB 独立划分
  2. SplitGuard: 每个数据集一个守卫对象, 提供
       - assert_train_only(record_ids)  采样结果若含测试患者立即抛 LeakError
       - audit_sampled_records(...)     审计已抽样本的 train/val/test 归属
  3. sample_train_only(): 先按训练患者过滤、再抽样的安全采样器
  4. CLI 自检: 划分完整性断言 + 与陈旧 patient_split.json 的差异报告

铁律 (AGENTS.md §8): 同一患者的所有记录/心拍不得跨 train/val/test。
任何训练脚本从公共库取数, 必须经过本模块的守卫或安全采样器。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.patient_split import (
    build_incart_patient_map, build_ptb_patient_map,
    patient_level_split, SEED as SPLIT_SEED,
)
from config import PROCESSED_DIR

ECG_DATA = Path(os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data"))
INCART_RID_OFFSET = 100000
SAVED_SPLIT_FILE = PROCESSED_DIR / "patient_split.json"

TAGS = ("mit_bih", "incart", "ptb")


class LeakError(RuntimeError):
    """患者级泄漏或划分完整性被破坏时抛出。"""


# ---------------- 基础加载 ----------------

def load_arrays(tag):
    """读取某数据集的因果部署链预处理数组 (mmap)。"""
    sfx = "_processed_deploy_causal"
    b = np.load(ECG_DATA / f"{tag}{sfx}_beats.npy", mmap_mode="r")
    l = np.load(ECG_DATA / f"{tag}{sfx}_labels.npy", mmap_mode="r")
    r = np.load(ECG_DATA / f"{tag}{sfx}_record_ids.npy", mmap_mode="r")
    return b, l, r


def _mit_pmap_from_rids(record_ids):
    """MIT 患者映射, 与 build_mit_patient_map 同一规则 (201/202 同患者)。
    直接从数组 record_ids 构建, 避免加载整个处理集。"""
    pmap = {}
    for rid in np.unique(record_ids):
        rid = int(rid)
        pmap[rid] = "mit_201_202" if rid in (201, 202) else "mit_%d" % rid
    return pmap


# ---------------- 权威划分 ----------------

_MERGED_CACHE = {}


def compute_mit_incart_split(seed=SPLIT_SEED):
    """MIT+INCART 合并患者级划分 (与 finetune_exp7c_v4 / qat_v6_clean 同口径)。
    返回 (masks_dict, merged_rids, stats)。masks 对齐合并数组。"""
    if seed in _MERGED_CACHE:
        return _MERGED_CACHE[seed]
    mit_b, mit_l, mit_r = load_arrays("mit_bih")
    inc_b, inc_l, inc_r = load_arrays("incart")
    n_mit, n_inc = len(mit_r), len(inc_r)

    mit_map = _mit_pmap_from_rids(mit_r)
    inc_map = build_incart_patient_map()
    if not inc_map:
        raise LeakError(
            "INCART 患者元数据 (.hea) 不可用, 无法保证患者级划分; "
            "拒绝继续 (见 patient_split.INCART_DIR)")
    pmap = dict(mit_map)
    pmap.update({rid + INCART_RID_OFFSET: "inc_" + pat
                 for rid, pat in inc_map.items()})

    merged_rids = np.concatenate([np.asarray(mit_r), np.asarray(inc_r) + INCART_RID_OFFSET])
    tr_m, va_m, te_m, stats = patient_level_split(merged_rids, pmap, seed=seed)
    tr_m, va_m, te_m = np.asarray(tr_m), np.asarray(va_m), np.asarray(te_m)

    # 完整性断言: 三分互斥且全覆盖; 每条记录都有患者归属
    assert len(tr_m) == len(merged_rids)
    assert int((tr_m & va_m).sum()) == int((tr_m & te_m).sum()) == int((va_m & te_m).sum()) == 0, \
        "患者级划分出现跨组重叠"
    assert int((tr_m | va_m | te_m).sum()) == len(merged_rids), "划分未全覆盖"
    unknown = [rid for rid in np.unique(merged_rids) if rid not in pmap]
    if unknown:
        raise LeakError(f"存在无患者映射的 record_id: {unknown[:10]} ...")

    out = {
        "masks": {"mit_bih": tr_m[:n_mit], "incart": tr_m[n_mit:]},
        "val_masks": {"mit_bih": va_m[:n_mit], "incart": va_m[n_mit:]},
        "test_masks": {"mit_bih": te_m[:n_mit], "incart": te_m[n_mit:]},
        "merged_rids": merged_rids,
        "stats": stats,
        "n_mit": n_mit, "n_inc": n_inc,
    }
    _MERGED_CACHE[seed] = out
    return out


def compute_ptb_split(seed=SPLIT_SEED):
    """PTB 独立患者级划分。返回 (tr/val/te masks 对齐 ptb 数组, stats)。"""
    ptb_b, ptb_l, ptb_r = load_arrays("ptb")
    ptb_map = build_ptb_patient_map()
    if not ptb_map:
        raise LeakError("PTB 患者元数据 (ECG-Database/RECORDS) 不可用, 拒绝继续")
    tr, va, te, stats = patient_level_split(np.asarray(ptb_r), ptb_map, seed=seed)
    tr, va, te = np.asarray(tr), np.asarray(va), np.asarray(te)
    assert int((tr & va).sum()) == int((tr & te).sum()) == int((va & te).sum()) == 0
    assert int((tr | va | te).sum()) == len(ptb_r)
    unknown = [int(rid) for rid in np.unique(ptb_r) if int(rid) not in ptb_map]
    if unknown:
        raise LeakError(f"PTB 存在无患者映射的 record_id: {unknown[:10]} ...")
    return {"train": tr, "val": va, "test": te, "stats": stats}


# ---------------- 守卫对象 ----------------

class SplitGuard:
    """某数据集的患者级泄漏守卫。

    所有公开方法的输入均为**原生** record_id (即该数据集
    {tag}_processed_deploy_causal_record_ids.npy 中存储的原始值;
    INCART 的 +100000 偏移仅是合并划分的内部细节, 调用方无需关心)。
    """

    def __init__(self, tag, seed=SPLIT_SEED):
        if tag not in TAGS:
            raise ValueError(f"unknown tag {tag!r}, expected one of {TAGS}")
        self.tag = tag
        self.seed = seed

        if tag in ("mit_bih", "incart"):
            m = compute_mit_incart_split(seed)
            self.train_mask = m["masks"][tag]
            self.val_mask = m["val_masks"][tag]
            self.test_mask = m["test_masks"][tag]
            self.stats = m["stats"]
        else:
            p = compute_ptb_split(seed)
            self.train_mask, self.val_mask, self.test_mask = p["train"], p["val"], p["test"]
            self.stats = p["stats"]

        _b, _l, self.record_ids = load_arrays(tag)
        self.record_ids = np.asarray(self.record_ids)
        if len(self.train_mask) != len(self.record_ids):
            raise LeakError(f"{tag}: 守卫掩码 {len(self.train_mask)} 与数组 "
                            f"{len(self.record_ids)} 长度不一致 (对齐错误)")

    # ---- 查询 ----

    def train_record_ids(self):
        return np.unique(self.record_ids[self.train_mask])

    def val_record_ids(self):
        return np.unique(self.record_ids[self.val_mask])

    def test_record_ids(self):
        return np.unique(self.record_ids[self.test_mask])

    # ---- 审计 / 断言 ----

    def audit_sampled_records(self, record_ids, context=""):
        """审计一次抽样的记录归属。输入为原生 record_id 数组。返回可序列化字典。"""
        r = np.asarray(record_ids)
        n_tr = int(np.isin(r, self.train_record_ids()).sum())
        n_va = int(np.isin(r, self.val_record_ids()).sum())
        n_te = int(np.isin(r, self.test_record_ids()).sum())
        n_miss = int(len(r)) - n_tr - n_va - n_te
        if n_miss:
            raise LeakError(f"{self.tag}: {n_miss} 条抽样记录不在任何划分组 "
                            f"(record_ids 异常?), context={context!r}")
        total = int(len(r))
        return {
            "tag": self.tag,
            "context": context,
            "sampled": total,
            "train": n_tr, "val": n_va, "test": n_te,
            "test_ratio": (n_te / total) if total else 0.0,
            "val_ratio": (n_va / total) if total else 0.0,
            "leaked": n_te > 0,
        }

    def assert_train_only(self, record_ids, context=""):
        """训练取数断言: 任何非 train 患者的记录 → 抛 LeakError (含记录名单)。"""
        rep = self.audit_sampled_records(record_ids, context)
        if rep["leaked"] or rep["val"] > 0:
            r = np.asarray(record_ids)
            bad = np.unique(r[np.isin(r, self.test_record_ids()) |
                              np.isin(r, self.val_record_ids())])
            raise LeakError(
                f"[LEAK] {self.tag} ({context}): 训练取数含 "
                f"{rep['test']} 条测试患者记录 / {rep['val']} 条验证患者记录。"
                f"涉及记录示例: {bad.tolist()[:12]}")

    # ---- 安全采样 ----

    def sample_train_beats(self, n_abn, n_norm, rng):
        """仅从训练患者的心拍中抽样。返回 (beat_idx_abn, beat_idx_norm)。"""
        l = np.asarray(load_arrays(self.tag)[1])
        m = self.train_mask[:len(l)]
        ia = np.where(m & (l == 1))[0]
        inn = np.where(m & (l == 0))[0]
        if len(ia) < n_abn or len(inn) < n_norm:
            raise LeakError(f"{self.tag}: 训练患者心拍不足 "
                            f"(abn {len(ia)}<{n_abn}, norm {len(inn)}<{n_norm})")
        sa = rng.choice(ia, n_abn, replace=False)
        sn = rng.choice(inn, n_norm, replace=False)
        return sa, sn


_GUARD_CACHE = {}


def get_guard(tag, seed=SPLIT_SEED):
    key = (tag, seed)
    if key not in _GUARD_CACHE:
        _GUARD_CACHE[key] = SplitGuard(tag, seed)
    return _GUARD_CACHE[key]


# ---------------- 陈旧注册表检查 ----------------

def check_saved_registry():
    """对比 data/processed/patient_split.json 与重算结果。不修改任何文件。"""
    report = {"file": str(SAVED_SPLIT_FILE), "exists": SAVED_SPLIT_FILE.exists()}
    if not report["exists"]:
        report["stale"] = None
        return report
    saved = json.loads(SAVED_SPLIT_FILE.read_text(encoding="utf-8"))
    fresh = {
        "mit": compute_mit_incart_split()["stats"],      # 合并口径, 与旧文件不可直接比
        "ptb": compute_ptb_split()["stats"],
    }
    diffs = []
    s_ptb = (saved.get("datasets") or {}).get("ptb") or {}
    if s_ptb:
        for k in ("n_patients", "n_train", "n_val", "n_test"):
            if s_ptb.get(k) != fresh["ptb"].get(k):
                diffs.append(f"ptb.{k}: saved={s_ptb.get(k)} recomputed={fresh['ptb'].get(k)}")
    report["ptb_diffs"] = diffs
    report["stale"] = bool(diffs)
    report["note"] = ("旧注册表为各数据集独立口径且已过期; "
                      "守卫永远以重算结果为准, 请勿引用该文件")
    return report


# ---------------- CLI ----------------

def main():
    print("[GUARD] ECG_DATA =", ECG_DATA)
    m = compute_mit_incart_split()
    st = m["stats"]
    print(f"[GUARD] MIT+INCART 合并: 患者 {st['n_patients']} = "
          f"train {st['n_train']} / val {st['n_val']} / test {st['n_test']}")
    print(f"[GUARD]   心拍: train {st['beats_train']} / val {st['beats_val']}"
          f" / test {st['beats_test']} (MIT {m['n_mit']} + INCART {m['n_inc']})")
    print("[GUARD]   test 患者:", ", ".join(st["test_patients"]))
    p = compute_ptb_split()
    ps = p["stats"]
    print(f"[GUARD] PTB: 患者 {ps['n_patients']} = train {ps['n_train']} / "
          f"val {ps['n_val']} / test {ps['n_test']}")

    for tag in TAGS:
        g = get_guard(tag)
        n_tr, n_va, n_te = len(g.train_record_ids()), len(g.val_record_ids()), len(g.test_record_ids())
        print(f"[GUARD] {tag}: 记录 train={n_tr} val={n_va} test={n_te}")
        # 自检: 守卫自身测试组心拍送审, 必须全部命中测试组
        rep = g.audit_sampled_records(g.record_ids[g.test_mask], context="self-check")
        assert rep["test"] == int(g.test_mask.sum()) and rep["train"] == 0 and rep["val"] == 0, \
            f"{tag}: 守卫自检失败 {rep}"
        print(f"[GUARD]   self-check ok (test 心拍 {rep['test']} 全部命中测试组)")

    reg = check_saved_registry()
    if reg.get("stale"):
        print("[GUARD] ⚠️ 已保存 patient_split.json 陈旧:", "; ".join(reg["ptb_diffs"]))
    elif reg["exists"]:
        print("[GUARD] patient_split.json 与重算一致 (仍建议以重算为准)")
    print("[GUARD] 划分完整性检查全部通过")


if __name__ == "__main__":
    main()
