#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_int8_exp7c.py — INT8 量化精度核查: tflite interpreter vs float32 (MIT/PTB 缓存 + 真实拍)"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"

def run_tflite(path, beats):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    scale, zp = out["quantization"]
    probs = []
    for i in range(len(beats)):
        xf = beats[i:i+1].astype(np.float32)[..., np.newaxis]
        xq = np.clip(np.round(xf / in_scale) + in_zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], xq)
        interp.invoke()
        q = interp.get_tensor(out["index"])          # int8 输出 (1,2)
        f = (q.astype(np.float32) - zp) * scale      # 反量化 logits
        e = np.exp(f - f.max(axis=1, keepdims=True))
        probs.append(e[0, 1] / e[0].sum())
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{len(beats)}")
    return np.array(probs)

def auc(y, p):
    y = np.asarray(y).ravel(); p = np.asarray(p).ravel()
    order = np.argsort(p); y = y[order]
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)[y == 1].sum()
    return float((ranks - n1 * (n1 + 1) / 2.0) / (n1 * n0))

tfl = BASE / "models" / "ecg_model_exp7c_int8.tflite"
out = {"int8_check": {}}
for tag, f in [("mit", CACHE / "mit_deploy_causal_match.npz"),
               ("ptb", CACHE / "ptb_deploy_causal_match.npz")]:
    d = np.load(f)
    p = run_tflite(tfl, d["beats"])
    out["int8_check"][tag] = {"auc_int8": auc(d["labels"], p),
                              "n": int(len(p)), "n_abn": int((d["labels"] == 1).sum())}
    print(f"[int8] {tag}: AUC={out['int8_check'][tag]['auc_int8']:.4f} "
          f"(float32 对照: exp7c mit 0.8964 / ptb 0.8015)")

real = np.load(BASE / "data" / "real" / "real_normal_beats_exp7c.npy")
p = run_tflite(tfl, real)
out["int8_check"]["real_normal"] = {
    "mean": float(p.mean()), "median": float(np.median(p)),
    "frac_gt_0.5": float((p > 0.5).mean()), "frac_gt_0.8": float((p > 0.8).mean()),
    "n": int(len(p)),
}
print(f"[int8] real normal: mean={p.mean():.4f} median={np.median(p):.4f} "
      f"frac>0.5={(p>0.5).mean():.4f} frac>0.8={(p>0.8).mean():.4f} "
      f"(float32 对照: mean 0.4166)")
(CACHE / "int8_exp7c_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("[int8] saved int8_exp7c_check.json")
