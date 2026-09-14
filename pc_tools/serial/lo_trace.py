"""lo_trace.py — Round-H LEADOFF 板上标定/取证解析 (TH §114)

从 alarm_repro.py 的 raw 捕获 (或任意串口文本文件) 提取 TICK 尾部的
lo=<quiet>,<qual>,<rms>,<crest> 遥测 + sqimin/bpm/alarm/asrc, 输出逐秒
轨迹与分段摘要, 用于阈值标定 (A: rms 地板 / B: sqi-crest-rms 带) 与
正/负例断言证据。

用法:
  python lo_trace.py <raw.txt> [--phase 10] [--json out.json]

断言口径 (验证协议 §3):
  正例: alarm 连续 >=30s 且 asrc 含 0x10
  负例: 全程无 0x10 (既有源 0x01/0x02/0x04/0x08 允许)
"""
import argparse
import json
import re
import sys

TICK_RE = re.compile(
    r"TICK,(\d+),src=(\S+?),bpm=(\d+),sqi=([0-9.eE+-]+),.*?"
    r"alarm=(\d+),asrc=0x([0-9a-fA-F]+),seg=(\d+),sqimin=([0-9.eE+-]+)"
    r"(?:,vrms=([0-9.eE+-]+))?(?:,aidrop=(\d+))?(?:,nw=\d+)(?:,nt=\d+)"
    r"(?:,lo=(\d+),(\d+),([0-9.eE+-]+),([0-9.eE+-]+))?")
LATCH_RE = re.compile(r"\[ALARM\] (LATCH|EXTEND|CLEAR)(?: \+0x([0-9a-fA-F]+))?"
                      r"(?: src=0x([0-9a-fA-F]+))?[^ ]* ?t=(\d+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", help="alarm_repro.py 输出的 *_raw.txt")
    ap.add_argument("--phase", type=float, default=10.0,
                    help="正例 lead-in 时长 (秒, 相对捕获起点)")
    ap.add_argument("--json", default=None, help="摘要 JSON 输出路径")
    args = ap.parse_args()

    rows = []      # (t, sqimin, bpm, alarm, asrc, quiet, qual, rms, crest)
    events = []
    for line in open(args.raw, encoding="utf-8"):
        m = TICK_RE.search(line)
        if m:
            t = None
            parts = line.split(None, 1)
            try:
                t = float(parts[0])
            except ValueError:
                t = len(rows) + 1.0
            rows.append({
                "t": t, "sqimin": float(m.group(8)),
                "bpm": int(m.group(3)), "alarm": int(m.group(5)),
                "asrc": int(m.group(6), 16), "seg": int(m.group(7)),
                "quiet": int(m.group(11)) if m.group(11) is not None else None,
                "qual": int(m.group(12)) if m.group(12) is not None else None,
                "rms": float(m.group(13)) if m.group(13) is not None else None,
                "crest": float(m.group(14)) if m.group(14) is not None else None,
            })
            continue
        m = LATCH_RE.search(line)
        if m:
            events.append({"t": float(line.split(None, 1)[0]),
                           "kind": m.group(1),
                           "src": int((m.group(2) or m.group(3) or "0"), 16)})

    if not rows:
        print("no TICK rows with lo= telemetry found")
        sys.exit(2)

    # 逐秒轨迹 (紧凑)
    print("  t   sqimin bpm alarm asrc | loq loq  rms    crest")
    for r in rows:
        print("%5.0f  %.3f %3d %5d 0x%02x | %3s %3s %7.4f %6.2f" % (
            r["t"], r["sqimin"], r["bpm"], r["alarm"], r["asrc"],
            r["quiet"], r["qual"],
            r["rms"] if r["rms"] is not None else -1,
            r["crest"] if r["crest"] is not None else -1))

    # 摘要
    tail = [r for r in rows if r["t"] >= args.phase]
    seg_ids = sorted(set(r["seg"] for r in rows))
    leadoff_secs = sum(1 for r in rows if r["asrc"] & 0x10)
    alarm_secs = sum(1 for r in rows if r["alarm"])
    # alarm 最长连续 run
    best = cur = 0
    for r in rows:
        cur = cur + 1 if r["alarm"] else 0
        best = max(best, cur)
    summary = {
        "raw": args.raw,
        "ticks": len(rows),
        "segments": seg_ids,
        "alarm_secs_total": alarm_secs,
        "alarm_run_max_s": best,
        "leadoff_0x10_secs": leadoff_secs,
        "events": events,
    }
    if tail:
        rms_vals = [r["rms"] for r in tail if r["rms"] is not None]
        cr_vals = [r["crest"] for r in tail if r["crest"] is not None]
        sq_vals = [r["sqimin"] for r in tail]
        q_vals = [r["qual"] for r in tail if r["qual"] is not None]
        bpm_vals = [r["bpm"] for r in tail]
        summary["tail_after_s"] = args.phase
        summary["tail_rms_min_max"] = [round(min(rms_vals), 4), round(max(rms_vals), 4)] if rms_vals else None
        summary["tail_crest_min_max"] = [round(min(cr_vals), 2), round(max(cr_vals), 2)] if cr_vals else None
        summary["tail_sqimin_min"] = round(min(sq_vals), 3)
        summary["tail_qual_max"] = max(q_vals) if q_vals else None
        summary["tail_beat_secs"] = sum(1 for b in bpm_vals if b > 0)
        summary["tail_secs"] = len(tail)
    print("\nsummary:", json.dumps(summary, ensure_ascii=False, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print("->", args.json)


if __name__ == "__main__":
    main()
