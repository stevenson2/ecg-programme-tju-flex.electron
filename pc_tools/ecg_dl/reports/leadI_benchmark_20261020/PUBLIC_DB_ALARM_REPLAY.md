# 公开库 held-out 报警回放（新规则）

- 生成时间（Asia/Shanghai）：2026-10-20T00:29:01+0800
- 测试集冻结 INCART rids：`[1, 5, 11, 13, 29, 35, 36, 62, 64, 65, 70]`（patient 级，无泄漏）
- 导联：Lead I ch0；采样：重采样 250 Hz + 因果滤波（与学生契约一致）
- 报警新规则：dens 8/10，hold 12s，rel ≤1/5，cooldown 25s，thr=0.5
- ECG-FM：INCART 训练线性探针 @0.50（非原生分类头）
- 域：医院带标注心电，**几乎无运动伪影** → 若仍高 FP，更像模型错判

## 汇总排序（INCART · R 峰 · normal-only 进入次数 ↑越好越少）

| Rank | Model | R峰 normal-only 进入 | R峰 normal FP/h | Blind normal-only 进入 | Blind normal FP/h | 拍级 Se | 拍级 +P | 拍级 AUROC |
|------|-------|---------------------|-----------------|------------------------|-------------------|---------|---------|------------|
| 1 | exp5_int8 | 1 | 0.2 | 22 | 4.0 | 0.780 | 0.895 | 0.964 |
| 2 | ecg_fm_center_step | 25 | 4.5 | 0 | 0.0 | 0.810 | 0.424 | 0.888 |
| 3 | v3a_h5 | 25 | 4.5 | 13 | 2.4 | 0.483 | 0.376 | 0.769 |
| 4 | ecg_fm_mean_pool | 46 | 8.4 | 0 | 0.0 | 0.741 | 0.285 | 0.788 |
| 5 | exp0_h5 | 56 | 10.2 | 83 | 15.1 | 0.995 | 0.198 | 0.922 |

### 读表
- **normal-only 进入**：把含真实 V 的秒从阳性旗标里抹掉后再跑状态机 → 近似「没有室早上也会报」的次数。
- **pure_fp_entries**：进入瞬间的密度窗内完全没有真实 V 秒。
- 医院 Lead I 几乎无运动；若 normal-only FP/h 仍高 → **错判主导**；若接近 0 而真机很高 → **运动伪影主导**。

## 逐记录（R 峰）

| Model | Record | dur_s | V拍 | R峰 full进入 | pureFP | normal-only进入 | mean_score |
|-------|--------|-------|-----|-------------|--------|-----------------|------------|
| v3a_h5 | I01 | 1800 | 344 | 0 | 0 | 0 | 0.027 |
| v3a_h5 | I05 | 1800 | 256 | 3 | 0 | 0 | 0.173 |
| v3a_h5 | I11 | 1800 | 4 | 1 | 1 | 1 | 0.048 |
| v3a_h5 | I13 | 1800 | 230 | 0 | 0 | 0 | 0.158 |
| v3a_h5 | I29 | 1800 | 783 | 0 | 0 | 0 | 0.064 |
| v3a_h5 | I35 | 1800 | 475 | 6 | 1 | 2 | 0.149 |
| v3a_h5 | I36 | 1800 | 462 | 8 | 3 | 4 | 0.215 |
| v3a_h5 | I62 | 1800 | 807 | 6 | 5 | 9 | 0.749 |
| v3a_h5 | I64 | 1800 | 26 | 5 | 4 | 5 | 0.210 |
| v3a_h5 | I65 | 1800 | 386 | 1 | 0 | 0 | 0.169 |
| v3a_h5 | I70 | 1800 | 0 | 4 | 4 | 4 | 0.259 |
| exp0_h5 | I01 | 1800 | 344 | 3 | 0 | 8 | 0.645 |
| exp0_h5 | I05 | 1800 | 256 | 4 | 0 | 0 | 0.351 |
| exp0_h5 | I11 | 1800 | 4 | 1 | 1 | 1 | 0.957 |
| exp0_h5 | I13 | 1800 | 230 | 2 | 1 | 8 | 0.767 |
| exp0_h5 | I29 | 1800 | 783 | 5 | 0 | 10 | 0.829 |
| exp0_h5 | I35 | 1800 | 475 | 3 | 0 | 4 | 0.818 |
| exp0_h5 | I36 | 1800 | 462 | 6 | 1 | 4 | 0.772 |
| exp0_h5 | I62 | 1800 | 807 | 1 | 1 | 5 | 0.974 |
| exp0_h5 | I64 | 1800 | 26 | 3 | 3 | 3 | 0.452 |
| exp0_h5 | I65 | 1800 | 386 | 7 | 0 | 5 | 0.496 |
| exp0_h5 | I70 | 1800 | 0 | 8 | 8 | 8 | 0.611 |
| exp5_int8 | I01 | 1800 | 344 | 0 | 0 | 0 | 0.132 |
| exp5_int8 | I05 | 1800 | 256 | 2 | 0 | 0 | 0.122 |
| exp5_int8 | I11 | 1800 | 4 | 0 | 0 | 0 | 0.070 |
| exp5_int8 | I13 | 1800 | 230 | 0 | 0 | 0 | 0.183 |
| exp5_int8 | I29 | 1800 | 783 | 4 | 0 | 0 | 0.267 |
| exp5_int8 | I35 | 1800 | 475 | 0 | 0 | 0 | 0.169 |
| exp5_int8 | I36 | 1800 | 462 | 1 | 0 | 0 | 0.157 |
| exp5_int8 | I62 | 1800 | 807 | 2 | 0 | 0 | 0.289 |
| exp5_int8 | I64 | 1800 | 26 | 0 | 0 | 0 | 0.068 |
| exp5_int8 | I65 | 1800 | 386 | 0 | 0 | 0 | 0.157 |
| exp5_int8 | I70 | 1800 | 0 | 1 | 1 | 1 | 0.377 |
| ecg_fm_mean_pool | I01 | 1800 | 344 | 3 | 1 | 1 | 0.270 |
| ecg_fm_mean_pool | I05 | 1800 | 256 | 3 | 0 | 0 | 0.153 |
| ecg_fm_mean_pool | I11 | 1800 | 4 | 0 | 0 | 0 | 0.128 |
| ecg_fm_mean_pool | I13 | 1800 | 230 | 1 | 1 | 1 | 0.349 |
| ecg_fm_mean_pool | I29 | 1800 | 783 | 12 | 2 | 3 | 0.517 |
| ecg_fm_mean_pool | I35 | 1800 | 475 | 9 | 0 | 12 | 0.465 |
| ecg_fm_mean_pool | I36 | 1800 | 462 | 4 | 2 | 12 | 0.448 |
| ecg_fm_mean_pool | I62 | 1800 | 807 | 2 | 2 | 5 | 0.699 |
| ecg_fm_mean_pool | I64 | 1800 | 26 | 12 | 11 | 12 | 0.569 |
| ecg_fm_mean_pool | I65 | 1800 | 386 | 3 | 0 | 0 | 0.297 |
| ecg_fm_mean_pool | I70 | 1800 | 0 | 0 | 0 | 0 | 0.258 |
| ecg_fm_center_step | I01 | 1800 | 344 | 0 | 0 | 0 | 0.168 |
| ecg_fm_center_step | I05 | 1800 | 256 | 2 | 0 | 0 | 0.120 |
| ecg_fm_center_step | I11 | 1800 | 4 | 0 | 0 | 0 | 0.054 |
| ecg_fm_center_step | I13 | 1800 | 230 | 5 | 1 | 1 | 0.371 |
| ecg_fm_center_step | I29 | 1800 | 783 | 10 | 1 | 1 | 0.409 |
| ecg_fm_center_step | I35 | 1800 | 475 | 15 | 0 | 5 | 0.326 |
| ecg_fm_center_step | I36 | 1800 | 462 | 11 | 1 | 6 | 0.336 |
| ecg_fm_center_step | I62 | 1800 | 807 | 9 | 4 | 6 | 0.548 |
| ecg_fm_center_step | I64 | 1800 | 26 | 5 | 5 | 6 | 0.443 |
| ecg_fm_center_step | I65 | 1800 | 386 | 3 | 0 | 0 | 0.207 |
| ecg_fm_center_step | I70 | 1800 | 0 | 0 | 0 | 0 | 0.270 |

## PTB classic Lead I 正常对照（短片段）

| Model | Record | dur_s | R峰进入 | Blind进入 | R峰 FP/h |
|-------|--------|-------|---------|-----------|----------|
| v3a_h5 | patient104/s0306lre | 115 | 0 | 0 | 0.0 |
| v3a_h5 | patient105/s0303lre | 115 | 0 | 0 | 0.0 |
| v3a_h5 | patient116/s0302lre | 115 | 1 | 0 | 31.3 |
| v3a_h5 | patient117/s0291lre | 115 | 0 | 0 | 0.0 |
| v3a_h5 | patient117/s0291lre | 115 | 0 | 0 | 0.0 |
| v3a_h5 | patient121/s0311lre | 115 | 0 | 0 | 0.0 |
| exp0_h5 | patient104/s0306lre | 115 | 0 | 1 | 0.0 |
| exp0_h5 | patient105/s0303lre | 115 | 2 | 1 | 62.6 |
| exp0_h5 | patient116/s0302lre | 115 | 1 | 1 | 31.3 |
| exp0_h5 | patient117/s0291lre | 115 | 0 | 1 | 0.0 |
| exp0_h5 | patient117/s0291lre | 115 | 0 | 1 | 0.0 |
| exp0_h5 | patient121/s0311lre | 115 | 0 | 0 | 0.0 |
| exp5_int8 | patient104/s0306lre | 115 | 0 | 1 | 0.0 |
| exp5_int8 | patient105/s0303lre | 115 | 0 | 0 | 0.0 |
| exp5_int8 | patient116/s0302lre | 115 | 0 | 0 | 0.0 |
| exp5_int8 | patient117/s0291lre | 115 | 0 | 0 | 0.0 |
| exp5_int8 | patient117/s0291lre | 115 | 0 | 0 | 0.0 |
| exp5_int8 | patient121/s0311lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_mean_pool | patient104/s0306lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_mean_pool | patient105/s0303lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_mean_pool | patient116/s0302lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_mean_pool | patient117/s0291lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_mean_pool | patient117/s0291lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_mean_pool | patient121/s0311lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_center_step | patient104/s0306lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_center_step | patient105/s0303lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_center_step | patient116/s0302lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_center_step | patient117/s0291lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_center_step | patient117/s0291lre | 115 | 0 | 0 | 0.0 |
| ecg_fm_center_step | patient121/s0311lre | 115 | 0 | 0 | 0.0 |

- 耗时：2395.9 s
- JSON：`results/public_db_alarm_replay.json`

## 重要说明
- **ECG-FM 盲切嵌入已跳过（全零）**，表中 FM 的 Blind normal-only=0 不代表真实盲切表现，只看 **R 峰** 列。
- 与真机 ~2 分钟会话外推 ~30 FP/h 对比：本表医院 Lead I、5.5 小时 held-out，R 峰 normal-only FP/h 在 0.2–10 量级 → **支持「真机误报以运动伪影为主」**；但仍有非零错判（尤其 exp0 / FM-mean）。
- exp5 在本冻结集上很好，但其历史训练划分可能与本 test rids 有重叠，**不能当作无泄漏上限**；student_leadI（本流水线）才是无泄漏对照。
