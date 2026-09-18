#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_r20m_valid.py — R20M: 冻结骨干 + 盲切窗正例 valid 头
================================================================================
预注册 runs/R20M_PREREG.md (2026-09-19 冻结)。与 R19M 唯一差异: valid=1
素材增补三源盲切 1s 窗 (INCART/PTB/cinc2017 训练患者连续流, 任意相位,
build_r20m_blind.py 缓存) —— 对症 R19M 的"正例流形不覆盖评估流形"失败。

判据与 R19M 逐字一致 (C0/C1/C2/C4/C5); 失败判定: 一次迭代 (valid=0 配比
±50%) 不过即全线关闭。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_r19m_valid import (V3A_H5, MODELS, CACHE, TAP, NOISE_TRAIN_COUNTS,
                              NOISE_VAL_N, ARMB_CARRIER_N,
                              assemble_v3a_pool, build_head, train_head,
                              assemble_combined, gen_pure_noise_windows,
                              get_guard, load_arrays, augment_beats)

BASE = Path(__file__).resolve().parent
BLIND_NPZ = Path("/home/devcontainers/ecg_data/r20m_blind_windows.npz")
SEED_NOISE = 20260920


def run(seed, variant, arm, epochs, tag_suffix="", noise_scale=1.0):
    tag = f"r20m{'B' if arm == 'B' else ''}_valid_{variant}_s{seed}{tag_suffix}"
    out_h5 = MODELS / f"{tag}.h5"
    out_json = CACHE / f"train_{tag}.json"
    t0 = time.time()

    base = tf.keras.models.load_model(str(V3A_H5), compile=False)
    base.trainable = False
    fm = {"fc1": tf.keras.Model(base.input, base.get_layer("fc1").output),
          "gap": tf.keras.Model(base.input, base.get_layer("gap").output)}

    x_pool = assemble_v3a_pool(seed)
    blind = np.load(BLIND_NPZ)
    x_blind = np.concatenate([blind["incart_blind"], blind["ptb_blind"],
                              blind["cinc_blind"]], axis=0)
    rng_noise = np.random.default_rng(SEED_NOISE + seed)
    counts = {k: max(1, int(round(v * noise_scale)))
              for k, v in NOISE_TRAIN_COUNTS.items()}
    noise_train = gen_pure_noise_windows(counts, rng_noise)
    noise_val = gen_pure_noise_windows({"motion": NOISE_VAL_N // 2,
                                        "mains": NOISE_VAL_N // 2}, rng_noise)
    armB_prov = None
    if arm == "B":
        g = get_guard("mit_bih")
        _s0, s1 = g.sample_train_beats(0, ARMB_CARRIER_N, rng_noise)
        carriers = np.asarray(load_arrays("mit_bih")[0])[s1].astype(np.float32)
        corrupt, armB_prov = augment_beats(carriers, rng_noise,
                                           variants_per_beat=1,
                                           types=("mains",), snrs=(20,))
        noise_train = np.concatenate([noise_train, corrupt])
    print(f"[{tag}] pool={len(x_pool)} blind={len(x_blind)} "
          f"noise_train={len(noise_train)} noise_val={len(noise_val)}",
          flush=True)

    feats = {}
    for key, model in fm.items():
        pos = np.concatenate([model.predict(x_pool, batch_size=512, verbose=0),
                              model.predict(x_blind[..., np.newaxis],
                                            batch_size=512, verbose=0)])
        feats[key] = {
            "pos": pos,
            "neg": model.predict(noise_train[..., np.newaxis],
                                 batch_size=512, verbose=0),
            "neg_val": model.predict(noise_val[..., np.newaxis],
                                     batch_size=512, verbose=0),
            "pos_val": pos[:500],
        }

    results, heads = {}, {}
    variants = [variant] if variant else ["V1", "V2", "V3"]
    for vname in variants:
        head, best_auc, best_ep, _ = train_head(
            vname, feats[TAP[vname]]["pos"], feats[TAP[vname]]["neg"],
            feats[TAP[vname]]["neg_val"], feats[TAP[vname]]["pos_val"],
            seed, epochs)
        heads[vname] = head
        results[vname] = {"val_valid_auc": round(best_auc, 4),
                          "best_epoch": best_ep,
                          "params": int(head.count_params())}
        print(f"[{tag}] {vname}: val_valid_auc={best_auc:.4f} @ep{best_ep}",
              flush=True)

    chosen = variant
    if not chosen:
        top = max(v["val_valid_auc"] for v in results.values())
        for k in ("V1", "V2", "V3"):
            if top - results[k]["val_valid_auc"] < 0.005:
                chosen = k
                break
        print(f"[{tag}] selection -> {chosen}", flush=True)

    combined, dmax = assemble_combined(chosen, heads[chosen], x_pool[:512])
    c0_pass = (dmax == 0.0)
    print(f"[{tag}] C0 abn max|Δ|={dmax} -> "
          f"{'PASS' if c0_pass else 'FAIL'}", flush=True)
    combined.save(str(out_h5))

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prereg": "runs/R20M_PREREG.md",
        "arm": arm, "seed": seed, "chosen_variant": chosen,
        "noise_rng_seed": SEED_NOISE + seed,
        "data": {"pool_windows": int(len(x_pool)),
                 "blind_windows": int(len(x_blind)),
                 "blind_source_counts": {
                     "incart": int(len(blind["incart_blind"])),
                     "ptb": int(len(blind["ptb_blind"])),
                     "cinc2017": int(len(blind["cinc_blind"]))},
                 "noise_train": int(len(noise_train)),
                 "noise_counts": counts, "noise_scale": noise_scale,
                 "armB_mains_corrupted": (ARMB_CARRIER_N if arm == "B" else 0),
                 "armB_prov": armB_prov},
        "variants": results,
        "c0_abn_max_absdiff": dmax, "c0_pass": bool(c0_pass),
        "output_model": str(out_h5.relative_to(BASE)),
        "train_time_s": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[{tag}] saved {out_h5.name}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["select", "full", "armb"])
    ap.add_argument("--variant", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    a = ap.parse_args()
    epochs = 5 if a.smoke else 200
    suffix = "" if a.noise_scale == 1.0 else "_ns%g" % a.noise_scale
    if a.stage == "select":
        run(a.seed, "", "A", epochs)
    elif a.stage == "full":
        assert a.variant in ("V1", "V2", "V3")
        for s in (42, 43, 44):
            run(s, a.variant, "A", epochs, suffix, a.noise_scale)
    else:
        assert a.variant in ("V1", "V2", "V3")
        for s in (42, 43, 44):
            run(s, a.variant, "B", epochs, suffix, a.noise_scale)
