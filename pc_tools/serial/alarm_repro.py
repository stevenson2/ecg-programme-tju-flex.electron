"""报警持续性取证 (M0/M1 基线工具): 抓串口 N 秒, 统计 AI_RESULT 异常位的连续时长分布.

背景: BLE 第 8 列跟随最后一次推理的 confirmed; ecg_ai 确认策略为
1-of-5 + cooldown 5 拍, 持续异常下 confirmed 呈 "1 秒 1 -> 5 秒 0" 交替,
App 侧仅 5 s 锁存 => 报警只响一会儿就停 (修复对象, TUNING_HISTORY §104).

口径:
  - AI_RESULT,<conf>,<raw>,<confirmed> 每行 = 1 个推理窗 (stride 250 @250Hz = 1 窗/秒)
  - run  = 连续 confirmed=1 (或 raw=1) 的推理窗数, 1 窗 ~= 1 s
  - gap  = 相邻两个 run 之间的 0 窗数
  - TICK 行提取 busy%/ovr (160 MHz 验收线: ovr 必须为 0)

用法:
  python alarm_repro.py [--port COM4] [--seconds 90] [--out 前缀] [--label 说明]
输出:
  <out>_raw.txt       原始串口 (相对时间戳逐行)
  <out>_summary.json  统计摘要 (run 分布 / gap 分布 / TICK busy/ovr)
"""
import argparse
import json
import re
import serial
import statistics
import time

AI_RE = re.compile(r"^AI_RESULT,([0-9.eE+-]+),(\d+),(\d+)(?:,(\d+))?")
TICK_RE = re.compile(
    r"^TICK,(\d+),src=(\S+?),bpm=(\d+),sqi=([0-9.eE+-]+),.*busy=(\d+)%,ovr=(\d+)"
    r"(?:,alarm=(\d+))?(?:,asrc=0x([0-9a-fA-F]+))?(?:,seg=(\d+))?")


def run_lengths(flags):
    """runs: 连续 True 段长列表; gaps: 相邻两个 run 之间的 False 个数列表."""
    runs, gaps = [], []
    cur, zeros, started = 0, 0, False
    for f in flags:
        if f:
            if started and zeros:
                gaps.append(zeros)
            zeros = 0
            started = True
            cur += 1
        else:
            if cur:
                runs.append(cur)
                cur = 0
            if started:
                zeros += 1
    if cur:
        runs.append(cur)
    return runs, gaps


def stats_of(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "max": max(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
        "all": values,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--out", default="alarm_repro", help="输出文件前缀")
    ap.add_argument("--label", default="", help="本次取证的说明 (写进 JSON)")
    ap.add_argument("--cmd", action="append", default=[],
                    help="抓取前发送的 UART 命令 (可重复, 如 --cmd 'MODE replay_flat')")
    args = ap.parse_args()

    s = serial.Serial(args.port, 460800, timeout=0.2)
    # 与 serial_monitor.py 相同的开埠手法: 避免拉 RTS/DTR 使板子复位
    s.rts = False
    s.dtr = True
    s.rts = False
    time.sleep(0.2)
    s.dtr = False
    s.rts = False
    s.reset_input_buffer()

    if args.cmd:
        for c in args.cmd:
            s.write((c + "\n").encode("ascii"))
            time.sleep(0.3)
        time.sleep(1.0)   # 等模式切换/滤波器收敛, 再开始计时取证
        # 注意: 不清缓冲 —— CMD 回复与模式切换日志留在取证头部作为证据

    t0 = time.time()
    lines = []
    while time.time() - t0 < args.seconds:
        data = s.read(4096)
        if not data:
            continue
        now = time.time() - t0
        for raw in data.split(b"\n"):
            text = raw.decode("utf-8", "replace").strip()
            if text:
                lines.append((now, text))
    s.close()

    with open(args.out + "_raw.txt", "w", encoding="utf-8") as f:
        for ts, text in lines:
            f.write("%8.3f %s\n" % (ts, text))

    ai = [(ts, float(m.group(1)), int(m.group(2)), int(m.group(3)))
          for ts, text in lines
          for m in [AI_RE.match(text)] if m]
    ai_lat = [int(m.group(4)) for _, text in lines
              for m in [AI_RE.match(text)] if m and m.group(4)]
    ticks = [(ts, m.group(2), int(m.group(5)), int(m.group(6)), m.group(7), m.group(8), m.group(9))
             for ts, text in lines
             for m in [TICK_RE.match(text)] if m]

    summary = {
        "tool": "alarm_repro.py",
        "label": args.label,
        "port": args.port,
        "seconds": args.seconds,
        "cmds": args.cmd,
        "capture_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ai_result": {},
        "tick": {},
    }
    if ai:
        dur = ai[-1][0] - ai[0][0]
        conf_runs, conf_gaps = run_lengths([r[3] == 1 for r in ai])
        raw_runs, raw_gaps = run_lengths([r[2] == 1 for r in ai])
        confs = [r[1] for r in ai]
        summary["ai_result"] = {
            "windows": len(ai),
            "rate_hz": round(len(ai) / dur, 3) if dur > 0 else None,
            "confirmed_rate": round(sum(1 for r in ai if r[3]) / len(ai), 4),
            "raw_rate": round(sum(1 for r in ai if r[2]) / len(ai), 4),
            "conf_median": round(statistics.median(confs), 4),
            "conf_mean": round(statistics.mean(confs), 4),
            "confirmed_runs_s": stats_of(conf_runs),   # 1 窗 ~= 1 s
            "confirmed_gaps_s": stats_of(conf_gaps),
            "raw_runs_s": stats_of(raw_runs),
            "raw_gaps_s": stats_of(raw_gaps),
        }
        if ai_lat:
            summary["ai_result"]["latency_us_median"] = statistics.median(ai_lat)
            summary["ai_result"]["latency_us_max"] = max(ai_lat)
    if ticks:
        srcs = sorted(set(t[1] for t in ticks))
        summary["tick"] = {
            "count": len(ticks),
            "sources": srcs,
            "busy_pct_max": max(t[2] for t in ticks),
            "busy_pct_mean": round(statistics.mean(t[2] for t in ticks), 1),
            "ovr_total": sum(t[3] for t in ticks),
            "ovr_max": max(t[3] for t in ticks),
        }
        # M1 固件起 TICK 尾部带 alarm=/asrc=/seg= (旧固件无这些字段则跳过)
        alarm_flags = [t[4] == "1" for t in ticks if t[4] is not None]
        if alarm_flags:
            alarm_runs, _ = run_lengths(alarm_flags)
            summary["alarm_latch"] = {
                "rate": round(sum(alarm_flags) / len(alarm_flags), 4),
                "runs_s": stats_of(alarm_runs),   # 1 TICK ~= 1 s
            }
        asrcs = sorted(set(t[5] for t in ticks if t[5] is not None))
        if asrcs:
            summary["alarm_latch"] = summary.get("alarm_latch", {})
            summary["alarm_latch"]["asrc_seen"] = ["0x" + a for a in asrcs]
        segs = sorted(set(t[6] for t in ticks if t[6] is not None))
        if segs:
            summary["tick"]["segments"] = segs

    with open(args.out + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["tick"].get("ovr_total", 0):
        print("!! TICK ovr>0: 2ms 节拍超时, 160MHz 不可带病继续 (回退 240MHz)")
    print("raw log -> %s_raw.txt, summary -> %s_summary.json" % (args.out, args.out))


if __name__ == "__main__":
    main()
