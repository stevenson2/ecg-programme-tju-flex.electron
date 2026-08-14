"""启动序列 (Windows 怪癖版): 每次 dtr 赋值后必须跟一次 rts 赋值才会传播.
实测极性: DTR=True=EN低=复位保持; RTS=True=IO0低=下载模式.
序列: rts=False(IO0高) -> dtr=True+rts=False(按住EN) -> dtr=False+rts=False(释放) = RUN 启动.
用法: python serial_boot_read.py [port] [read_seconds]
"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

s = serial.Serial(port, 460800, timeout=0.2)
s.rts = False          # IO0 高 (run 拉线), 同时传播当前 DTR
s.dtr = True           # EN 低 (按住复位)
s.rts = False          # 触发传播: EN 低 + IO0 高
time.sleep(0.2)
s.dtr = False          # EN 释放
s.rts = False          # 触发传播: EN 高 + IO0 高 -> RUN 启动

end = time.time() + dur
buf = b""
while time.time() < end:
    d = s.read(4096)
    if d:
        buf += d
s.close()
text = buf.decode("utf-8", "replace")
print(text if text else "(no output)")
print("bytes:", len(buf))
