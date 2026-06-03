"""查找 ESP32 串口，打印端口名 (如 COM5)，找不到则打印空行"""
import serial.tools.list_ports

for p in serial.tools.list_ports.comports():
    desc = p.description.lower()
    if any(kw in desc for kw in ["ch340", "ch341", "cp210", "ftdi", "usb-serial", "esp32", "espressif"]):
        print(p.device)  # e.g. COM5
        exit(0)
    if p.vid in [0x1A86, 0x10C4, 0x0403, 0x303A]:
        print(p.device)
        exit(0)

# 没找到，尝试最后手段：取最后一个 COM 口
ports = [p.device for p in serial.tools.list_ports.comports() if p.device.startswith("COM")]
if ports:
    print(ports[-1])
