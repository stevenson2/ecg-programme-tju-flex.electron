"""真实 AFE 帧率隔离测试: 同一会话内先测当前 OVS(默认4), 再切 OVS=1 测一轮.

用法:
    python serial_monitor_afe_ovs.py [port] [seconds] [outA] [outB]

与 serial_monitor_afe.py 相同复位/切 AFE 流程; 第二阶段发送 'DIAG OVS 1'
(2026-08-16 新增命令, 不重烧录), 用于隔离 AFE analogRead 过采样开销。
注意: 全程串口保持打开, 不要同时开其他串口脚本; 测帧率时手机蓝牙请关闭。
"""
import re
import sys
import time

import serial

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
out_a = sys.argv[3] if len(sys.argv) > 3 else "esp_timer_check_afe_ovs4.txt"
out_b = sys.argv[4] if len(sys.argv) > 4 else "esp_timer_check_afe_ovs1.txt"

s = serial.Serial(port, 460800, timeout=0.2)
s.rts = False
s.dtr = True
s.rts = False
time.sleep(0.2)
s.dtr = False
s.rts = False
time.sleep(3.0)
s.reset_input_buffer()
s.write(b"m\nm\n")
s.flush()


def capture(secs):
    end = time.time() + secs
    buf = b""
    while time.time() < end:
        d = s.read(4096)
        if d:
            buf += d
    return buf.decode("utf-8", "replace")


def report(tag, text):
    lines = text.splitlines()
    csv_n = sum(1 for l in lines if re.match(r"^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},", l))
    sample_n = [l for l in lines if "[SAMPLE]" in l]
    drops = sum(int(m.group(1)) for l in sample_n for m in [re.search(r"dropped=(\d+)", l)])
    hr_n = sum(1 for l in lines if "[心率]" in l)
    mode = [l for l in lines if "真实AFE" in l]
    print(f"[{tag}] CSV={csv_n} ({csv_n/dur:.1f}Hz 串口 → {csv_n*5/dur:.1f}Hz 主循环) "
          f"SAMPLE行={len(sample_n)} 总drop={drops} 心率行={hr_n} 真实AFE={len(mode)}")


print("[PHASE A] OVS 当前值(默认4) 捕获 %.0fs ..." % dur)
a = capture(dur)
with open(out_a, "w", encoding="utf-8") as f:
    f.write(a)
report("A", a)

print("[PHASE B] 发送 DIAG OVS 1 ...")
s.reset_input_buffer()
s.write(b"DIAG OVS 1\n")
s.flush()
time.sleep(3.0)
b0 = capture(0.5)
print("  回复:", " | ".join(l.strip() for l in b0.splitlines() if "DIAG OVS" in l))
print("[PHASE B] OVS=1 捕获 %.0fs ..." % dur)
b = capture(dur)
with open(out_b, "w", encoding="utf-8") as f:
    f.write(b)
report("B", b)
s.close()
print("outputs:", out_a, ",", out_b)
