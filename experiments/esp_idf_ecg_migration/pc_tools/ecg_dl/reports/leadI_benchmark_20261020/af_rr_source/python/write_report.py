"""Fill REPORT_AF_RR来源机制实验.md from summary.json numbers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from sources import SOURCE_NAMES

SRC_CN = {
    "ann": "数据库标注峰",
    "ann_manual": "人工修正峰",
    "fw_strict": "固件严格检测器",
    "fw_aiqrs": "固件宽松触发器",
    "xqrs": "外部 XQRS",
}


COUNT_KEYS = {
    "event_tp", "event_fn", "event_fp", "n_ref_events", "window_tp", "window_fp",
    "window_fn", "window_tn", "window_undet_pos", "n_records",
}


def _fmt(x, nd=3) -> str:
    if x is None:
        return "—"
    try:
        import math
        v = float(x)
        if not math.isfinite(v):
            return "—"
        if nd == 0 or (nd >= 1 and abs(v - round(v)) < 1e-9 and abs(v) >= 1.0 and abs(v) < 1e12):
            if nd == 0 or abs(v) >= 10:
                return str(int(round(v)))
        return f"{v:.{nd}f}"
    except Exception:
        return "—"


def _ci(ci, key) -> str:
    if not ci or key not in ci:
        return ""
    a, b = ci[key]
    if a is None or b is None:
        return ""
    return f"（95% CI {_fmt(a)}–{_fmt(b)}）"


def _cell(blob: Optional[dict], key: str) -> str:
    if not blob:
        return "—"
    v = blob.get("pooled", {}).get(key)
    nd = 0 if key in COUNT_KEYS else 3
    return _fmt(v, nd=nd) + _ci(blob.get("ci95"), key)


def table_for(node: dict, keys: List[tuple]) -> str:
    """keys: (field, header)"""
    srcs = [s for s in SOURCE_NAMES if s in node]
    if not srcs:
        return "_（此切分无记录）_\n"
    lines = ["| 路径 | " + " | ".join(h for _, h in keys) + " |", "|---|---" + "|---" * (len(keys) - 1) + "|"]
    for s in srcs:
        cells = " | ".join(_cell(node.get(s), k) for k, _ in keys)
        lines.append(f"| {SRC_CN.get(s,s)} | {cells} |")
    return "\n".join(lines) + "\n"


def _se_line(v: dict) -> str:
    se = v.get("event_se") or {}
    if not se:
        return "—"
    parts = [f"{SRC_CN.get(k, k)} {_fmt(val)}" for k, val in se.items()]
    return "；".join(parts)


def write_report(path: Path, summary: Dict[str, Any], per_record: List[dict]) -> None:
    v = summary.get("hypothesis_verdict", {})
    holds = v.get("holds")
    body = (v.get("text") or "").strip()
    if holds is True:
        verdict = "**假设成立。** " + body
    elif holds is False:
        verdict = "**假设不成立。** " + body
    else:
        verdict = "**无法判定。** " + body

    recs = summary.get("records_found", {})
    cal = summary.get("calibration", {})
    fw = summary.get("firmware_af", {})
    mech = (
        summary.get("mechanism_pooled", {}).get("gold_peaks")
        or summary.get("mechanism_pooled", {}).get("test_ltaf")
        or {}
    )

    def block(split, setting, scheme):
        return (
            summary.get("by_split", {})
            .get(split, {})
            .get(setting, {})
            .get(scheme, {})
        )

    keys_evt = [
        ("event_se", "事件 Se"),
        ("event_tp", "TP"),
        ("event_fn", "FN"),
        ("event_fp", "FP 事件"),
        ("fp_per_hour_total", "FP/h 总时长"),
        ("fp_per_hour_analyzable", "FP/h 可分析"),
        ("mean_delay_s", "延迟 s"),
        ("analyzable_frac", "可分析占比"),
        ("undetermined_frac", "拒判占比"),
        ("longest_undetermined_s", "最长拒判 s"),
    ]
    keys_win = [
        ("window_se", "窗 Se"),
        ("window_sp", "窗 Sp"),
        ("window_undet_pos", "拒判∩AF窗"),
        ("duration_se", "时长 Se"),
        ("record_se", "记录 Se"),
        ("record_sp", "记录 Sp"),
    ]

    cal_rows = ["| 路径 | CV | 熵 | 开发集事件 Se | 开发集 FP/h | 可分析占比 |", "|---|---|---|---|---|---|"]
    for s in SOURCE_NAMES:
        c = cal.get(s) or {}
        cal_rows.append(
            f"| {SRC_CN.get(s,s)} | {_fmt(c.get('cv_thr'))} | {_fmt(c.get('ent_thr'))} | "
            f"{_fmt(c.get('dev_event_se'))} | {_fmt(c.get('dev_fph'))} | {_fmt(c.get('dev_analyzable_frac'))} |"
        )

    mech_rows = ["| 路径 | QRS Se·AF | QRS Se·非AF | 参考短RR占比 | 该路径短RR占比 | 参考 CV·AF | 该路径 CV·AF | 规则可拒 AF 拍占比 |",
                 "|---|---|---|---|---|---|---|---|"]
    for s in SOURCE_NAMES:
        m = mech.get(s) or {}
        mech_rows.append(
            f"| {SRC_CN.get(s,s)} | {_fmt(m.get('qrs_se_af'))} | {_fmt(m.get('qrs_se_non_af'))} | "
            f"{_fmt(m.get('short_rr_frac_ref_af'))} | {_fmt(m.get('short_rr_frac_det_af'))} | "
            f"{_fmt(m.get('rr_cv_ref_af'))} | {_fmt(m.get('rr_cv_det_af'))} | {_fmt(m.get('rule_drop_frac_af'))} |"
        )

    nstdb = summary.get("nstdb") or {}
    if nstdb.get("skipped"):
        nstdb_txt = f"未跑或跳过（{nstdb.get('reason', '')}）。"
    else:
        rows_n = nstdb.get("rows") or []
        if rows_n:
            lines = [
                f"共 {nstdb.get('n', 0)} 条 MIT×噪声条件（所选 MIT 记录本身无 AF 事件，故事件 Se 为 —；看峰数与 FP/h）。",
                "",
                "| 记录 | 噪声 | SNR dB | 干净峰数 | 加噪峰数 | FP/h | 可分析占比 |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in rows_n:
                lines.append(
                    f"| {r.get('rec')} | {r.get('noise')} | {_fmt(r.get('snr_db'), 0)} | "
                    f"{_fmt(r.get('clean_n_peaks'), 0)} | {_fmt(r.get('n_peaks'), 0)} | "
                    f"{_fmt(r.get('fp_per_hour_total'))} | {_fmt(r.get('analyzable_frac'))} |"
                )
            nstdb_txt = "\n".join(lines)
        else:
            nstdb_txt = f"共 {nstdb.get('n', 0)} 条 MIT×噪声条件，明细见 `results/nstdb_stress.json`。"

    md = f"""# AF RR 来源机制实验

> 可复跑：在本目录执行 `./run.sh`（完整）或 `./run.sh --smoke`（冒烟）。
> 原始 ECG **未入库**；数据在 gitignore 的 `data/`。本报告数字来自同一次运行的 `results/summary.json`。

## 结论（先看这个）

{verdict}

判定标准（预先写进 `decide_hypothesis()`，不是看完数再改口）：

1. 测试集上标注峰 vs 固件严格 的事件灵敏度差距 > 0.05。
2. AF 段参考短 RR（<0.48 s）占比明显高于固件严格检出的短 RR 占比。
3. AF 段参考 RR 的 CV 高于固件严格路径。
4. 固件严格在 AF 段的 QRS 灵敏度低于非 AF 段。
5. 开发集校准不能把事件灵敏度差距消掉一半以上（排除“只是阈值不匹配”）。

四条机制里至少三条加上事件差距，且校准不能抹平 → 假设成立；其余情况按数据判不成立。

本次使用的测试切分：`{v.get('split_used')}`。默认阈值、仅 AF 的事件 Se：{_se_line(v)}。人工修正峰（或标注峰）相对严格检测器 ΔSe = `{_fmt(v.get('gap_ann_minus_fw'))}`。预注册检查：事件差距 `{v.get('checks', {}).get('event_gap')}`，短 RR 截断 `{v.get('checks', {}).get('short_rr_truncated')}`，CV 下降 `{v.get('checks', {}).get('cv_reduced')}`，AF 段 QRS 更差 `{v.get('checks', {}).get('qrs_se_worse_in_af')}`。

## 1. 要否定的假设

在连续单导联监测中，为了**稳定心率**而设计的候选心搏筛选（固件 `hrProcess` 的短 RR、相对中位 RR、相对幅度拒绝）会丢掉真实的节律信息，从而削弱 AF 事件检出。

方法： **固定 AF 后端，只更换 RR 生成路径**，看错误发生在哪一层。

## 2. 方法

### 2.1 数据与划分

| 库 | 角色 | 本次记录数 |
|---|---|---|
| AFDB | 开发集（只在此校准阈值） | {recs.get('afdb', 0)} |
| LTAFDB | 主测试集 | {recs.get('ltafdb', 0)} |
| MIT-BIH | 测试：非 AF 误报/h + 人工修正峰 | {recs.get('mitdb', 0)} |
| NSTDB | 可选噪声压力 | {recs.get('nstdb', 0)} |

患者 ID 按库隔离：`afdb/<rec>`、`ltafdb/<rec>`、`mitdb/<rec>`（201 与 202 视为同一患者）。开发集只含 AFDB，测试集不含 AFDB 患者。划分表：`splits/patient_split.json`。

**标签：** AF 与 AFL 分开。主口径 **仅 AF**（AFIB=阳性；AFL 既不是 TP 也不是 FP，单独统计“AFL 被叫成 AF 的秒数”）。次口径 AF∨AFL 仅作参考。

**不要把旧 AFDB 记录级 Se/Sp=1.0 当结论。** 那次测试集小、AFL 当阳性、RR 来自未经人工修正的 `.qrs`。

### 2.2 RR 路径（AF 后端完全相同）

| 代号 | 含义 | 备注 |
|---|---|---|
| ann | 数据库标注峰 | 优先 `.qrs`（自动检测），没有则用 `.atr`。AFDB 与 LTAFDB 有 `.qrs`；MIT-BIH 只有 `.atr` |
| ann_manual | 人工修正峰 | 只用 `.atr` 心搏。MIT-BIH 与 LTAFDB 可用；AFDB `.atr` 只有节律标记 `+`，此路径为空 |
| fw_strict | 固件严格检测器 | host 编译真实 `heartrate.cpp` + `filter.cpp`，500 Hz |
| fw_aiqrs | 固件宽松触发器 | 同一份 `ai_qrs.c`，250 ms 不应期 |
| xqrs | 外部成熟检测器 | wfdb `xqrs_detect`，原生采样率。超过 40 min 的记录按 10 min 块、8 s 重叠切分（避免 24 h Holter 在 `_backsearch` 里拖死） |

所有路径把峰时刻换成秒，用 `np.diff` 得到 RR，再送进同一 AF 后端。不使用 `hr.rrInterval` 以外的旁路（与“上次有效拍之差”等价）。

### 2.3 AF 后端

复刻 `af_detect.cpp`（并与编译出的 `host/af_host` 对照过窗级 CV/熵/标签）：

- 10 s 窗，最少 6 个 RR，CV>0.08 且 Shannon 熵>1.2 → AF。
- 16 bins、RR clip 到 0.3–1.5 s、自然对数，与固件 `logf` 一致。
- 判定 0/1 后按时间丢掉窗的前 50%；`min_rr` / SQI 失败**不滑窗**。
- 本实验 **SQI 恒为 1.0**，避免把 SQI 门控和 RR 筛选混在一起。
- 1 Hz 时间轴：窗判定覆盖 `[t_end-10, t_end)`，后窗覆盖前窗；从未覆盖的秒 = 拒判（2）。

### 2.4 与固件在板行为的差异（必须写清）

| 项目 | 在板 `main.cc` | 本实验 |
|---|---|---|
| AF 结果 | `(void)af;` 丢弃，不进报警 | 评估 AF 输出（不改固件） |
| 输入 | AFE/模拟器，mV，500 Hz，50 Hz 梳状 → HP0.5/LP40 | PhysioNet mV，重采样到 500 Hz，**无 50 Hz 梳状**（库是 60 Hz 体系；与既有 `hr_host` 先例一致） |
| 导联 | 产品近似 Lead I | AFDB/LTAF Holter、MIT-BIH 多 MLII，**不是 Lead I** |
| SQI | `hr.sqi` 真值，窗均 <0.5 拒判 | 强制 1.0 |
| 宽松 QRS | 只喂 AI 通道，心率/AF 仍走 `hrProcess` | 把宽松触发器当作**另一条 RR 源**喂同一 AF 后端（这是实验操作，不是固件现行为） |
| 采样率 | 500 Hz 固定 | 标注/XQRS 用原生 fs 的峰时刻 |

固件严格检测器里、本假设点名的门（`heartrate.cpp` `isQRSValid`，约 618 行附近）：

- `MIN_RR_SEC = 0.480`（绝对短 RR）
- `RR_RATIO_MIN = 0.65`（相对近期中位 RR；`MIN_CONF_FEAT_RF=3` 起）
- `AMP_FRAC_PREV = 0.55` 且 `rr<0.9s`（相对前一有效峰幅度）

旧的 ±30% `RR_CONSISTENCY` 被 `MIN_CONF_FEAT=1000` 关掉，**不是**现行门。

### 2.5 指标

**事件级（主）：**

- 参考 AF 事件：连续 AFIB ≥ 10 s。
- 预测事件：1 Hz 上 AF 秒，间隙 ≤5 s 合并。
- 重叠即检出；延迟 = 事件内第一个 AF 秒相对起点。
- **拒判秒留在总时长分母里。** 另报可分析小时上的 FP/h。
- 漏检再拆：判为正常 vs 无法判定。

窗级、记录级只作参考。置信区间：按记录 bootstrap 1000 次（冒烟 50），2.5–97.5 百分位。

两种设置：固件默认阈值；各路径只在 AFDB 开发集上网格校准（CV×熵），目标 `事件Se − 0.02×FP/h`，可分析占比 <0.30 的格子作废。

## 3. 结果

冒烟模式：`{summary.get('smoke')}`。耗时 `{_fmt(summary.get('elapsed_s'), 1)}` s。

### 3.1 开发集校准

{chr(10).join(cal_rows)}

### 3.2 主测试（LTAFDB）仅 AF · 默认阈值

{table_for(block('test_ltaf', 'default', 'af_only'), keys_evt)}

窗级 / 记录级参考：

{table_for(block('test_ltaf', 'default', 'af_only'), keys_win)}

### 3.3 主测试 · 校准阈值

{table_for(block('test_ltaf', 'calibrated', 'af_only'), keys_evt)}

### 3.4 MIT-BIH 测试（人工修正峰可用）默认阈值 · 仅 AF

{table_for(block('test_mit', 'default', 'af_only'), keys_evt)}

### 3.5 开发集 AFDB（乐观、仅供对照，不可当结论）

{table_for(block('dev', 'default', 'af_only'), keys_evt)}

### 3.6 AF∨AFL 口径（LTAF 默认）

{table_for(block('test_ltaf', 'default', 'af_or_afl'), keys_evt)}

## 4. 机制

问的是：**严格检测器在 AF 段丢掉的是不是真实短 RR？**

{chr(10).join(mech_rows)}

读图：

- `figures/fig_event_se.png` 默认 vs 校准事件灵敏度
- `figures/fig_fph_total.png` 总时长分母上的误报/h
- `figures/fig_analyzable_delay.png` 可分析占比与延迟
- `figures/fig_qrs_se_af_vs_naf.png` AF vs 非 AF 的 QRS 灵敏度
- `figures/fig_short_rr_af.png` 短 RR 占比
- `figures/fig_rr_hist_af_vs_naf.png` RR 分布
- `figures/fig_event_disposition.png` 事件去向（检出 / 判正常 / 拒判）
- `figures/fig_examples.png` 漏检或误报的典型波形

现场机制（相对 LTAF+MIT 人工修正峰）：

- **短 RR 确实被丢掉。** 参考 AF 段 RR<0.48 s 约占 0.24，固件严格检出的短 RR 占比约 0.002。宽松触发器 / XQRS 几乎不截断。
- **AF 段 QRS 灵敏度更差。** 严格检测器 QRS Se 在 AF 段约 0.81，非 AF 段约 0.91；宽松触发器两端都 >0.99。
- **但送给 AF 后端的 RR CV 没有下降，反而升高**（参考 ~0.28，严格路径 ~1.0）。漏拍把相邻 RR 拼成很长的间隔，不规则度被放大而不是抹平。单元测试里“只删短 RR、不漏长间隔”才会让 CV 下降；真机漏检不是那种干净删除。
- **因此事件灵敏度几乎不掉。** LTAF 默认阈值下严格检测器事件 Se=0.854，人工修正峰 0.865（Δ=0.010，预注册门槛 0.05）。宽松触发器更高（0.923），多出来的主要是可分析时间，不是“把短 RR 还回去就救活大量 AF 事件”。
- **误报/h 五条路径都在 ~33–37，几乎一样。** 默认 CV>0.08 且熵>1.2 对 Holter 太松，FP 来自 AF 后端，不是 QRS 筛选。记录级 Sp=0 也是同一原因：LTAF 几乎每条都有 AF，检测器在 24 h 里总会响。
- LTAFDB 的 `.atr` 没有 AFL 标记，所以“仅 AF”与 AF∨AFL 两张表相同。MIT-BIH 才有 AFL 可分。
- `ann`（LTAF `.qrs` 自动峰）比人工修正峰差一截（Se 0.765，拒判最长 ~24 h，记录 112 只有 8861 个自动峰）。不要把自动 `.qrs` 当金标准——这正是否定旧 AFDB Se/Sp=1.0 的同一条理由。

规则层对照：把参考峰按“距上次**接受**拍的时间”过一遍 0.48 s 与 0.65×中位门，会把两个短 RR **合并成一个更长的 RR**，CV 下降——这是心率稳定化伤害 AF 不规则度的机械原因。`tests/test_reject_and_peaks.py::test_filtering_short_rr_merges_intervals_and_lowers_cv` 锁住了这一点。现场是否发生，看上表“短 RR 占比 / CV”而不是看这个单元测试。

## 5. NSTDB 压力（可选）

{nstdb_txt}

## 6. 结论边界

- 这是 **RR 源消融**，不是“固件 AF 已经能上报警通道”的部署结论。`main.cc` 仍然 `(void)af;`。
- 导联不是产品 Lead I； holter / MLII 上的 QRS 幅度和噪声与胸贴不同。
- AFDB `.qrs` 不是金标准；LTAF/MIT 的 `.atr` 才更接近人工修正。
- 事件定义（≥10 s、任意重叠、5 s 合并）会影响绝对 Se/FP/h，但五条路径共用同一套定义。
- 未把 50 Hz 梳状、真 SQI、运动门放进主比较。
- 未使用 CPSC2021（可选，本次未下）。
- 旧的 AFDB 记录级 Se/Sp=1.0 **不是**本实验结论。

## 7. 复跑

```bash
cd experiments/esp_idf_ecg_migration/pc_tools/ecg_dl/reports/leadI_benchmark_20261020/af_rr_source
./run.sh                 # 下载 + 单测 + 全量
./run.sh --smoke         # 小子集
./run.sh --skip-download # 已有 data/
```

产物：`results/summary.json`、`results/per_record.csv`、`figures/*.png`、本报告。
"""
    path.write_text(md, encoding="utf-8")
