#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""board_corpus_eval.py — 板上噪声语料闭环评估 (M2/M4, TH §106)
================================================================================
经 M1 UART 命令通道全自动驱动: 逐 case `MODE replay_normal` + `REPLAY <seg>`,
抓取 ~36s, 解析 AI_RESULT/TICK, 输出逐 case 报告:
  - raw_rate: 逐窗超阈值率 (模型原始行为, 主指标)
  - conf_median / conf_mean: 置信度分布
  - confirmed_rate / alarm_latch_rate / asrc_seen: 报警擎住行为
  - sqi_mean: 板上 hr.sqi 对该 case 的读数 (M4 与噪声强度相关性用)
前置: 固件需已编入语料段 (REPLAY fail (0..2) 说明语料未编入, 停止)。
录制: 评估期间 REC_AUTO 0 (避免 SPIFFS 写入尖峰污染 busy/ovr 读数),
结束时恢复 REC_AUTO 1。
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import serial

AI_RE = re.compile(r"^AI_RESULT,([0-9.eE+-]+),(\d+),(\d+)(?:,(\d+))?")
TICK_RE = re.compile(
    r"^TICK,(\d+),src=(\S+?),bpm=(\d+),sqi=([0-9.eE+-]+),.*busy=(\d+)%,ovr=(\d+)"
    r"(?:,alarm=(\d+))?(?:,asrc=0x([0-9a-fA-F]+))?(?:,seg=(\d+))?")
SETTLE_S = 5.0     # 模式切换后丢弃前 5s (滤波/AI 窗瞬态)


def open_port(port):
    s = serial.Serial(port, 460800, timeout=0.2)
    s.rts = False
    s.dtr = True
    s.rts = False
    time.sleep(0.2)
    s.dtr = False
    s.rts = False
    s.reset_input_buffer()
    return s


def send(s, cmd):
    s.write((cmd + "\n").encode("ascii"))
    time.sleep(0.25)


def capture(s, seconds):
    t0 = time.time()
    lines = []
    while time.time() - t0 < seconds:
        d = s.read(4096)
        if not d:
            continue
        now = time.time() - t0
        for raw in d.split(b"\n"):
            t = raw.decode("utf-8", "replace").strip()
            if t:
                lines.append((now, t))
    return lines


def parse_case(lines, seg):
    ai = [(ts, float(m.group(1)), int(m.group(2)), int(m.group(3)))
          for ts, t in lines
          for m in [AI_RE.match(t)] if m and ts >= SETTLE_S]
    ticks = [(ts, float(m.group(4)), m.group(7))
             for ts, t in lines
             for m in [TICK_RE.match(t)] if m and ts >= SETTLE_S]
    out = {"windows": len(ai), "raw_rate": None, "conf_median": None,
           "conf_mean": None, "confirmed_rate": None, "alarm_rate": None,
           "asrc_seen": [], "sqi_mean": None, "seg_seen": None}
    segs = sorted({m.group(9) for ts, t in lines
                   for m in [TICK_RE.match(t)] if m and m.group(9)})
    out["seg_seen"] = segs
    if ai:
        out["raw_rate"] = round(sum(r[2] for r in ai) / len(ai), 4)
        out["confirmed_rate"] = round(sum(r[3] for r in ai) / len(ai), 4)
        confs = [r[1] for r in ai]
        out["conf_median"] = round(statistics.median(confs), 4)
        out["conf_mean"] = round(statistics.mean(confs), 4)
    if ticks:
        alarms = [1 if a == "1" else 0 for _, _, a in ticks if a is not None]
        if alarms:
            out["alarm_rate"] = round(sum(alarms) / len(alarms), 4)
        out["sqi_mean"] = round(statistics.mean(q for _, q, _ in ticks), 3)
    out["asrc_seen"] = sorted({m.group(8) for ts, t in lines
                               for m in [TICK_RE.match(t)] if m and m.group(8)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--seconds", type=float, default=36.0,
                    help="每 case 抓取时长 (默认 36 = 30s case + 瞬态余量)")
    ap.add_argument("--model", default="v3a", help="模型标签 (写进报告名)")
    ap.add_argument("--only", default="", help="逗号分隔的 case index, 默认全部")
    ap.add_argument("--provenance",
                    default=str(Path(__file__).resolve().parent.parent /
                                "ecg_dl" / "corpus" /
                                "replay_corpus_provenance.json"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent /
                                              "captures"))
    args = ap.parse_args()

    prov = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    cases = prov["cases"]
    only = {int(x) for x in args.only.split(",")} if args.only else None

    s = open_port(args.port)
    # 语料可用性探测
    send(s, "REPLAY 99")
    time.sleep(0.3)
    s.reset_input_buffer()
    send(s, "PING")
    probe = capture(s, 1.5)
    probe_txt = " ".join(t for _, t in probe)
    # REPLAY 99 的应答在 reset 前发出, 改为直接探测: REPLAY 3 若 fail 则无语料
    s.reset_input_buffer()
    send(s, "MODE replay_normal")
    send(s, "REPLAY 3")
    reply = capture(s, 1.0)
    if any("REPLAY fail" in t for _, t in reply):
        print("!! 固件未编入语料段 (REPLAY fail); 需 16MB 构建再跑")
        s.close()
        return 1

    send(s, "REC_AUTO 0")   # 评估期关闭自动录制, 减 SPIFFS 写尖峰

    results = []
    t_start = time.time()
    for c in cases:
        if only is not None and c["index"] not in only:
            continue
        seg = c["segment"]
        send(s, "MODE replay_normal")
        send(s, "REPLAY %d" % seg)
        time.sleep(0.5)
        s.reset_input_buffer()
        lines = capture(s, args.seconds)
        r = parse_case(lines, seg)
        r.update({"index": c["index"], "segment": seg, "name": c["name"],
                  "params": c["params"]})
        results.append(r)
        print("[%2d/%2d] seg%-2d %-32s raw=%.3f conf=%.3f alarm=%s sqi=%.2f" % (
            len(results), len(cases) if only is None else len(only),
            seg, c["name"],
            r["raw_rate"] if r["raw_rate"] is not None else -1,
            r["conf_median"] if r["conf_median"] is not None else -1,
            r["alarm_rate"], r["sqi_mean"]), flush=True)

    send(s, "MODE sim")
    send(s, "REC_AUTO 1")
    s.close()

    out = {
        "tool": "board_corpus_eval.py",
        "model": args.model,
        "port": args.port,
        "seconds_per_case": args.seconds,
        "settle_s": SETTLE_S,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_s": round(time.time() - t_start, 1),
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "corpus_eval_%s_%s" % (args.model, time.strftime("%Y%m%d_%H%M"))
    (out_dir / (stem + ".json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-> %s.json" % stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
