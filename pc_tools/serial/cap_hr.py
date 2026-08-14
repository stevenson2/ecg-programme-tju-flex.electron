import serial, time
s = serial.Serial('COM4', 460800, timeout=0.2)
s.rts = False; s.dtr = True; s.rts = False
time.sleep(0.2); s.dtr = False; s.rts = False
time.sleep(3); s.reset_input_buffer()
buf = b''; end = time.time() + 15
while time.time() < end:
    d = s.read(4096)
    if d: buf += d
s.close()
text = buf.decode('utf-8', 'replace')
for l in text.splitlines():
    if '心率' in l: print(l.strip())
