#!/usr/bin/env python3
"""Write the V01 synthetic counterexample note from vf_layers.

Does not replay a database and does not touch firmware thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

from vf_layers import build_v01_evidence

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "results" / "v01_score_contract.json"
MD_PATH = HERE / "REPORT_V01_评分口径.md"


def _pct(block: dict) -> str:
    if block["n"] <= 0 or block["est"] is None:
        return f"—（{block['hits']}/{block['n']}）"
    return f"{block['hits']}/{block['n']} = {block['est']:.3f}"


def render_markdown(ev: dict) -> str:
    late = ev["cases"]["late_overlap"]
    pool = ev["cases"]["pool_early_and_late"]
    budget = ev["cases"]["window_end_15s"]
    short = ev["cases"]["duration_12s_in_neither_denominator"]
    one = ev["cases"]["one_record_four_layers"]
    alone = ev["cases"]["vf_hit_alone_30s"]
    stored = ev["stored_pr2_summary"]["event_ge15"]
    delays = ev["delays_s"]
    lines = [
        "# V01：PR #2 评分口径适配与合成反例",
        "",
        f"输入提交是 PR #2 的 `{ev['input_commit']}`。",
        "阈值、RMS 下限、2.5 s 拍间门、12 s 擎住和 AI 密度门都沿用该提交里的固件，没有新门。",
        "",
        f"任务包路径 `{ev['packet_path']}` 在这台运行环境里读不到。",
        "本笔记按任务说明和该提交的 `vf_detect` / `alarmSecondTick` 实现口径。",
        "E02 对应契约没有送达，唯一 heavy 槽没有使用"
        f"（0 / {ev['heavy_slot_limit']} 条开发记录）。",
        "没有整库回放，没有改阈值，没有切换产品分支，没有刷写。",
        "",
        "## 四层",
        "",
        "| 层 | 是什么 | 不是什么 |",
        "|---|---|---|",
        "| L0 | 逻辑回归分数，以及 `score ≥ 0.15` 的当窗疑似 | 报警 |",
        "| L1 | `vfAlarm`：当前窗和上一窗都疑似。第一窗不会置位 | 最终用户报警 |",
        "| L2 | L1，并且本窗 RMS ≥ 0.05 mV，会话里见过拍，完成样本距最近拍严格大于 2500 ms | 最终用户报警。PR #2 主表的 `firmware` 列停在这一层 |",
        "| L3 | BLE 第 8 列。`alarmSecondTick` 擎住后才为 1 | `vfAlarm` 本身。VF 单独命中只打遥测；要同一秒已有别的源，或擎住已经成立，0x08 才会写进第 10 列 |",
        "",
        "## event_ge15 不是 Se15",
        "",
        "两者分母相同：可电击段时长 ≥ 15 s。",
        "`event_ge15` 只要报警窗和该段有任意重叠就算检出，时延不设上限。",
        "Se15 还要求 PR #2 的首次命中延迟 ≤ 15 s。延迟是报警窗结束时刻减去 4 ms，再减去段起点，和 `eval_vf_rules.first_hit_delay` 相同。",
        "短于 15 s 的段两边都不进分母。",
        "",
        f"晚重叠反例：报警只落在窗序号 6（30–35 s），段是 0–40 s。延迟 {delays['late_window_6']:.3f} s。",
        f"`event_ge15` {_pct(late['event_ge15'])}，Se15 {_pct(late['se15'])}。",
        f"旧 `event_counts(..., 15)` 的命中数是 {late['legacy_event_counts_hits']}，"
        f"等于 event_ge15，不等于 Se15。",
        "",
        f"早检出（窗序号 1，结束于 10 s，延迟 {delays['early_window_1']:.3f} s）和上面的晚重叠放在一起：",
        f"`event_ge15` {_pct(pool['event_ge15'])}，Se15 {_pct(pool['se15'])}。",
        "",
        f"窗结束正好落在 15 s（延迟 {delays['budget_window_2']:.3f} s）时两边都是 {_pct(budget['se15'])}。",
        f"12 s 段即使窗序号 1 已经报警，两个分母都是 {short['event_ge15']['n']}。",
        "",
        "PR #2 已提交的 `results/summary.json` 里，firmware / VFDB 的 "
        f"`event_ge15` 是 {stored['hits']}/{stored['n']}。",
        "该对象没有 `se15` 键。这里不把它改记成 Se15，也不重算整库。",
        "",
        "## 一条合成记录上的四层",
        "",
        "8 个 5 s 窗。分数只在窗序号 4 和 5 为 0.80，其余 0.01。RMS 全程 0.40 mV。",
        "唯一的 QRS 在第 0 帧，所以拍间门在后续窗上是开的。没有 RS、FLAT、LEADOFF 或 AI。",
        "",
        f"| 层 | 窗或秒上的标志 | event_ge15 | Se15 |",
        f"|---|---|---|---|",
        f"| L0 疑似 | `{one['L0_suspect']}` | 不算事件 | 无此字段 |",
        f"| L1 vfAlarm | `{one['L1_vfAlarm']}` | {_pct(one['L1']['event_ge15'])} | {_pct(one['L1']['se15'])} |",
        f"| L2 门控放行 | `{one['L2_vfHit']}` | {_pct(one['L2']['event_ge15'])} | {_pct(one['L2']['se15'])} |",
        f"| L3 用户报警 | 40 s 内置位次数 {one['L3_user_alarm_any']} | {_pct(one['L3']['event_ge15'])} | {_pct(one['L3']['se15'])} |",
        "",
        f"L1 的首次命中延迟是 {delays['trace_l1_window_5']:.3f} s，所以 L1 和 L2 的 event_ge15 为 1、Se15 为 0。",
        "L2 被下一秒的 `alarmSecondTick` 消费，时刻是第 "
        + ", ".join(str(s) for s in one["L2_consumed_at_sec"])
        + " 秒。",
        "这一秒没有别的源，擎住保持关闭，L3 的两个事件分都是 0。",
        "同一条记录上，L2 的 event_ge15 不能当成 L3，L1 的 vfAlarm 也不能当成用户报警。",
        "",
        "## vfAlarm 单独不构成用户报警",
        "",
        f"连续 {alone['n']} 秒只有 VF 命中：BLE 第 8 列之和 {alone['ble_alarm_sum']}，"
        f"第 10 列之和 {alone['ble_asrc_sum']}，每秒都只是遥测，没有一次由 VF 启动擎住。",
        "",
        "规则源可以在没有 VF 时擎住。t = 0 给 RS（0x02），t = 11 仍在 12 s 擎住内：",
        "",
        "| 秒 | 第 8 列 | 第 10 列 |",
        "|---:|---:|---:|",
    ]
    for row in ev["cases"]["rs_latch_without_vf"]:
        lines.append(f"| {row['sec']} | {row['ble_alarm']} | 0x{row['ble_asrc']:02x} |")
    lines += [
        "",
        "先由 RS 擎住，第 5 秒才出现 VF。0x08 是加进已经擎住的位图，擎住不是 VF 启动的。",
        "VF 把擎住锚点从 0 挪到 5，所以第 16 秒仍未解除，第 17 秒才解除。这是原有的延长，不是新门。",
        "",
        "| 秒 | 第 8 列 | 第 10 列 | VF 启动擎住 |",
        "|---:|---:|---:|---|",
    ]
    for row in ev["cases"]["rs_then_vf_extends_hold"]:
        lines.append(
            f"| {row['sec']} | {row['ble_alarm']} | 0x{row['ble_asrc']:02x} | {'是' if row['vf_started_latch'] else '否'} |"
        )
    same = ev["cases"]["same_second_rs_and_vf"][0]
    ai = ev["cases"]["ai_latch_without_vf"][0]
    both = ev["cases"]["ai_and_vf_same_second"][0]
    cool = ev["cases"]["ai_cooldown_blocks_relatch"]
    lines += [
        "",
        f"同一秒 RS 与 VF 一起到达：第 8 列 {same['ble_alarm']}，第 10 列 0x{same['ble_asrc']:02x}，"
        f"VF 启动擎住：{'是' if same['vf_started_latch'] else '否'}。",
        f"AI 密度 8/10 且秒内最低 SQI 0.80、没有 VF：第 8 列 {ai['ble_alarm']}，第 10 列 0x{ai['ble_asrc']:02x}。",
        f"同一秒再叠 VF：第 10 列 0x{both['ble_asrc']:02x}，VF 启动擎住：{'是' if both['vf_started_latch'] else '否'}。",
        "AI 在 t = 0 擎住，t = 12 满足解除条件后清掉，冷却到 t = 37。",
        f"t = 13 再次给出 AI 密度 8 并且 VF 命中：第 8 列 {cool['sec_13_ai_again']['ble_alarm']}。",
        "规则源不受这 25 s 冷却限制。",
        "",
        "数字来自 `python3 v01_synthetic.py`，与 `test_vf_layers.py` 使用同一套函数。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ev = build_v01_evidence()
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MD_PATH.write_text(render_markdown(ev), encoding="utf-8")
    print(JSON_PATH)
    print(MD_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
