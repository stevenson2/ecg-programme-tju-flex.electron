# 推进报告：滤波弃判 · R 峰固件 · 蒸馏 v2 · 500Hz 协议（2026-10-20）

时区：Asia/Shanghai。仓库：`Fe programme 25261` / `benchmark_leadI_20261019` + `ecg-programme-tju-flex.electron-master`。未刷机、未 force-push。

## 本轮做了什么

| # | 项 | 状态 |
|---|----|------|
| 1 | 更强滤波 + 难分辨拍弃判原型 | 完成（真机 + INCART 冻结拍级） |
| 2 | R 峰触发固件代码改动 | 完成代码，**未 flash** |
| 3 | 蒸馏 v2（校准教师 + α/T） | 完成，仍略低于硬标签学生 F1 |
| 4 | 500 Hz / ECGR 重录协议 | 文档就绪 |

## 关键数字

### A. 真机误报（student_leadI，新报警规则 8/10·hold12·cd25）

四段短会话合计 **进入次数**：

| 滤波 | 弃判 | 四段 entries 合计 | 结论 |
|------|------|-------------------|------|
| **baseline LP40（训练契约）** | 关/开 | **0 / 0** | 新学生已把短会话误报压到 0 |
| strong LP25 | 关 | 4 | 域偏移：训练是 LP40，强低压反而抬分 |
| strong LP25 | 开 | 1 | 弃判能砍一部分 |
| LP25+MA5 | 关/开 | 5 / 5 | 更差，不要上板 |

相对历史 v3a/exp5/FM 探针在同批真机上「几乎人人进报警」，**硬标签 Lead I 学生是质变**。用户此前判断「运动伪影主导」仍成立于旧模型；对新学生，短会话已不够敏感，需要 **≥30 min ECGR@250** 再验 FP/h。

### B. INCART 冻结测试（拍级）

| 模型 | AUROC | Se | +P | F1 |
|------|-------|----|----|----|
| **student_leadI（hard）** | **0.923** | 0.649 | 0.648 | **0.648** |
| distill v1 α0.5 T1 | 0.914 | 0.884 | 0.405 | 0.556 |
| **distill v2 α0.4 T3（best）** | 0.920 | 0.720 | 0.562 | **0.631** |
| distill v2 其它格点 | 0.88–0.91 | — | — | ≤0.567 |

弃判（分数∈[0.35,0.65] 等）在缓存拍上：reject_rate≈3.1%（V 拒 153 / N 拒 684）；弃判当阴性时 Se 0.630、+P **0.681**、F1 **0.654**（略优于无弃判 F1 0.649）。

**弃判在拒绝什么**：mid_conf（难分辨）、hf（高频残差）、crest（削波/尖刺）、flat。

### C. 蒸馏 v2 相对 v1

- 教师：FM center LogReg + **温度缩放 T=2**（bias 扫描落在 0）
- Soft 上 N 均分≈0.18、V≈0.79（不再极度刷召回）
- 选模按 **val F1**；α 降到 0.25–0.4
- **F1：0.556 → 0.631**，+P：0.405 → 0.562；仍未超过 hard 0.648
- 结论：**先部署 hard 学生**；蒸馏继续要 PVC 专用教师或全量 soft，而不是再扫 α

## 固件：R 峰触发（未刷机）

已改：
- `ecg_ai.h` / `ecg_ai.cpp`：`ECG_AI_TRIGGER_RPEAK`，`ecg_ai_on_rpeak()`，R 后 125 样本@250Hz 开火
- `main.cc`：默认 `trigger_mode=RPEAK`，`beatDetected` 时预约
- 备份：`*.bak_rpeak_20261020`

刷机前：`idf.py build`；静坐应近似「每拍一推」而非死板 1Hz。回滚：`trigger_mode=BLIND` 或还原 bak。

## 500 Hz / 忠实落盘

协议：`02_文档与规划/PROTOCOL_500Hz_ECGR_重录.md`

要点：
- **主归档用 `REC_START` → ECGR @250 Hz**（分析链前有 LP≈40，抗混叠后 2:1）——对齐学生契约
- 串口 `WAVE` ~100 Hz **不能**当金标准
- 真 ADC 500 Hz；若只为看「细抖是否混叠」，另开短时 500 调试，长程仍用 ECGR250

## 文件索引

- 结果：`benchmark_leadI_20261019/results/FILTER_ABSTAIN_PROTOTYPE.md`、`DISTILL_V2.md`
- 图：`results/figures/distill_v2_vs_hard.png`、`filter_abstain_real_entries.png`
- 权重：`models/student_leadI_distill_fm_v2.h5`（hard 仍是 `student_leadI_resnet_large.h5`）
- 脚本：`scripts/14_*.py` `15_*.py` `16_*.py` `apply_rpeak_patch.py`

## 建议下一步

1. 按协议录 **≥30 min Lead I ECGR**（静坐+慢走），用 student_leadI ± 弃判重测进入次数  
2. 你确认后刷 R 峰固件，做板端一致性  
3. 蒸馏暂搁；优先 INT8 导出 hard 学生做板上对照  
4. 房颤 RR 支路仍未做，可与长程录制并行  

## 诚实缺口

- INCART 原始 WFDB 本机未定位到 → 滤波变体的「公开库报警 FP/h」未重建（拍级弃判仍有效）  
- 真机仍是 ~100 Hz WAVE，强滤波结论受混叠污染；ECGR 重录后需复验  
- distill v2 仍未赢 hard F1  
