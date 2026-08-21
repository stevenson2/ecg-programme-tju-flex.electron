# 一次性调试脚本归档（2026-08-21 整理）

> **归档原因**：真机调试/历史实验的一次性脚本，任务已完结，且经全仓库扫描确认
> **零引用**（无任何文档、shell、其他脚本引用）。保留备查，不参与日常流水线。
>
> **恢复方法**：如需重跑，将文件移回 `pc_tools/ecg_dl/` 根目录再执行
> （部分脚本按 `__file__` 相对路径定位 `models/`、`data/` 目录，移出根目录后
> 直接运行可能找不到数据）。

| 文件 | 用途 | 所属历史任务 |
|------|------|--------------|
| analyze_real_record.py | 解析 .ecgr 真实采集记录（头部+统计+R峰+有效采样率） | TH §48 真机调试 |
| analyze_real_record2.py | 同上 v2：动态阈值+滑窗扫描+细化R峰 | TH §48 |
| analyze_real_record3.py | 分段统计 R 峰检出率相关性 | TH §48 |
| analyze_real_record4.py | 每秒分段 R 峰密度分布 | TH §48 |
| analyze_rec_hr.py | 固件同步路径 HR v6 复算 rec_latest.ecgr（70 vs 100 BPM 偏差排查） | TH §48 心率排查 |
| analyze_rec_peaks.py | 检测器分级诊断：v6 检出峰 vs 全局 MWI 峰 | TH §48 |
| rec_clean_segments.py | 洁净段 20-26s 原始/滤波数值时序对比 | TH §48 |
| rec_peak_width.py | QRS(窄)/伪峰(宽)形态区分 | TH §48 |
| rec_periodicity.py | rec_latest 周期性+节律整齐度判断 | TH §48 |
| rec_quality_hr.py | 记录质量统计+对比 v6 输出 RR | TH §48 |
| plot_rec_diag.py | rec_latest 20s 波形+v6 峰位标注图 | TH §48 |
| reconstruct_vf_ovs1.py | 从 OVS1 数据（100Hz CSV filtered 列）重建 250Hz 验证 VF v2 | TH §55 OVS 事故 |
| reconstruct_vf_paths.py | OVS1 重建两条 VF 数据路径对比训练数据 | TH §55 |
| check_filtered_signal.py | filtered 通道真实信号检查+R峰手动提取 | 真机验收 |
| check_real_signal.py | raw_standard.csv clean 通道信号检查（幅度+SQI） | 真机验收 |
| compare_model_weights.py | 4.4-4 迭代第1步：h5 权重逐层对比 | 4.4-4 患者级划分 |
| dump_json.py | JSON 结构打印小工具 | 通用 |
| show_history.py | 训练 history CSV 快速查看 | 通用 |
| check_npz.py | npz 样本数快速核对 | 通用 |
| resave_uncompressed.py | deploy npz 重存为未压缩格式（mmap 友好） | 十三章部署链重建 |
| fix_th55_encoding.py | 修复 TUNING_HISTORY 第五十五章编码损坏（UTF-8↔GBK） | 文档维护 |
| gen_disp_hp_coeffs.py | 显示用 HP 4Hz 系数生成（绘图辅助，非固件系数） | 绘图辅助 |
| gen_disp_lp_coeffs.py | 显示用 LP 系数生成（40Hz 主用/4Hz 备用） | 绘图辅助 |
| sanity_rr.py | RR 序列 sanity check：npz 心率与 .atr 标注一致性 | 数据审计 |
| smoke_mmap.py | mmap 加载冒烟测试 | 十三章 |

**整理原则**（本次全量整理的判定标准，供后续参考）：
1. 被任何 md/sh/py 引用 → 保留原位（溯源保护）
2. 被其他脚本 import → 保留原位（依赖保护）
3. 属于"数字→JSON→脚本"证据链 → 保留原位
4. 以上都不满足且任务已完结 → 移入本目录并登记
