---
name: ai-training
description: Train, fine-tune, evaluate and export the TFLite Micro INT8 1D-CNN ECG anomaly model (exp7b deployment chain, exp7c fine-tuning).
---

# AI 模型训练/微调（部署链定稿 exp6-SGD → exp7b 上板）

## 权威事实（勿与旧文档混淆）
- 部署模型：exp7b（MIT+PTB 训练，单模型 163.5KB INT8，已上板）。
- 训练链（PC 复刻）：250Hz → 梳状5抽头 → HP0.05 → LP40 → 因果 HP 0.5Hz(@250Hz)。
- 固件 AI 链已与训练链对齐（applyFilterAI，fs=250 系数 ai_hp_coeffs_fs250.txt）。
- 已知问题：真实 AFE 正常 ECG 上 AI 置信度系统性偏移到 ~0.9（域迁移，94% 在 0.8~1.0），
  调阈值无效 → 必须用真实数据微调（TH §40 B 方案）。

## 命令（WSL2，PowerShell 发起）
- 训练：`wsl -e bash -lc "cd /mnt/c/.../pc_tools/ecg_dl && python3 train.py ..."`（timeout 900000ms）
- 评估：`python3 eval_deploy_match.py`（timeout 600000ms；锚点 exp7b PTB AUC 0.7829）
- 部署链复现：`build_deploy_npz.py`、`eval_exp7b.py`、`export_exp7b.py`
- 实验脚本：`run_exp7.sh` / `run_exp7b.sh` / `launch_exp7b.sh` / `check_exp7b.sh`

## 微调流程（exp7c，交接文档已定，按此执行）
- ✅ **exp7c 已完成并上板 (2026-08-14)**: 210 真实正常拍微调 exp7b (冻结骨干
  lr=1e-5, 混 2000 异常+600 正常防遗忘)。结果: 真实拍置信度 0.732→0.417
  (frac>0.5: 81%→15%), MIT AUC 0.8769→0.8964, PTB 0.8033→0.8015 (无回退);
  INT8 部署口径 MIT 0.8979 / PTB 0.7880 / 真实拍 0.442; 真机 46 推理 0 误报。
  脚本: preprocess_real_exp7c.py / finetune_exp7c.py / eval_exp7c.py /
  export_exp7c_v2.py / check_int8_exp7c.py; 数字溯源 models/deploy_match/*.json。
- 数据: data/real/ecg_real_052.ecgr (183s, HR 73.6bpm), 有效采样率 225.68Hz
  (预处理按实测速率有理数重采样 4431/2000 → 500Hz 再走训练链)。

## 数字审计（AGENTS.md §8 铁律）
- 1.000/0/100% 完美数字必须核查；事件级断言"报警事件数 ≥ GT 事件数"；
- 混淆矩阵四项与样本总数自洽；逐类表附 n_abn/n 列；类内无负样本时只报 recall。
- 评估数字写入文档前必须能从 JSON 逐项溯源。
