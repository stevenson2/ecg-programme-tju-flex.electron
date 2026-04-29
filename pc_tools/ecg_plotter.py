#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import argparse
import time


DEFAULT_BAUD = 115200
WINDOW_SIZE = 500
MAX_DATA_POINTS = 2000
UPDATE_INTERVAL_MS = 40


data_clean = deque(maxlen=MAX_DATA_POINTS)
data_noisy = deque(maxlen=MAX_DATA_POINTS)
data_filtered = deque(maxlen=MAX_DATA_POINTS)
time_data = deque(maxlen=MAX_DATA_POINTS)

serial_port = None
running = True
sample_count = 0
current_bpm = 0
current_true_bpm = 0
bpm_confidence = 0.0


def serial_reader(port, baud):
    global serial_port, running, sample_count, current_bpm, current_true_bpm, bpm_confidence
    try:
        serial_port = serial.Serial(port, baud, timeout=1)
        print("Connected: " + port + " @ " + str(baud) + " bps")
        time.sleep(1)
        serial_port.reset_input_buffer()
    except Exception as e:
        print("Error: cannot open " + port + ": " + str(e))
        running = False
        return

    while running:
        try:
            line = serial_port.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if not line[0].isdigit() and line[0] != "-":
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                clean_val = float(parts[0].strip())
                noisy_val = float(parts[1].strip())
                filtered_val = float(parts[2].strip())
                data_clean.append(clean_val)
                data_noisy.append(noisy_val)
                data_filtered.append(filtered_val)
                time_data.append(sample_count)
                sample_count += 1

                # 第4列: 检测 BPM (可选)
                if len(parts) >= 4:
                    try:
                        bpm_val = int(parts[3].strip())
                        if bpm_val > 0:
                            current_bpm = bpm_val
                            bpm_confidence = min(1.0, bpm_confidence + 0.125)
                    except ValueError:
                        pass

                # 第5列: 真实 BPM (模拟器模式, 可选)
                if len(parts) >= 5:
                    try:
                        true_val = int(parts[4].strip())
                        current_true_bpm = true_val
                    except ValueError:
                        pass
        except ValueError:
            pass
        except serial.SerialException:
            print("Error: serial disconnected")
            running = False
            break


def update_plot(frame):
    if len(data_clean) < 2:
        return
    start = max(0, len(data_clean) - WINDOW_SIZE)
    t = list(time_data)[start:]
    clean = list(data_clean)[start:]
    noisy = list(data_noisy)[start:]
    filtered = list(data_filtered)[start:]
    t_sec = [x / 250.0 for x in t]
    line_clean.set_data(t_sec, clean)
    line_noisy.set_data(t_sec, noisy)
    line_filtered.set_data(t_sec, filtered)
    all_data = clean + noisy + filtered
    if all_data:
        y_min = min(all_data)
        y_max = max(all_data)
        y_range = y_max - y_min
        if y_range < 0.01:
            y_range = 1.0
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)
    if t_sec:
        current_end = t_sec[-1]
        current_start = max(0, current_end - WINDOW_SIZE / 250.0)
        ax.set_xlim(current_start, current_end + 0.1)

    # BPM 显示 (左上角白底文本框)
    if current_bpm > 0 and bpm_confidence >= 0.3:
        if current_true_bpm > 0:
            bpm_str = "♥ %d BPM  (真实 %d BPM)" % (current_bpm, current_true_bpm)
        else:
            bpm_str = "♥ %d BPM" % current_bpm
    else:
        bpm_str = "♥ -- BPM"
    bpm_text.set_text(bpm_str)

    txt = "Samples: %d | Window: %.1fs | Rate: %dms" % (
        sample_count, WINDOW_SIZE/250.0, UPDATE_INTERVAL_MS)
    status_text.set_text(txt)
    return line_clean, line_noisy, line_filtered, status_text, bpm_text


def on_key(event):
    global WINDOW_SIZE, running, ani
    if event.key == "right":
        WINDOW_SIZE = min(MAX_DATA_POINTS, WINDOW_SIZE + 100)
        print("Window: %.1fs (%d pts)" % (WINDOW_SIZE/250.0, WINDOW_SIZE))
    elif event.key == "left":
        WINDOW_SIZE = max(50, WINDOW_SIZE - 100)
        print("Window: %.1fs (%d pts)" % (WINDOW_SIZE/250.0, WINDOW_SIZE))
    elif event.key == "up":
        ymin, ymax = ax.get_ylim()
        center = (ymin + ymax) / 2
        rh = (ymax - ymin) / 2 * 0.7
        if rh > 0.01:
            ax.set_ylim(center - rh, center + rh)
        print("Y zoom in: %.2f V" % (ax.get_ylim()[1]-ax.get_ylim()[0]))
    elif event.key == "down":
        ymin, ymax = ax.get_ylim()
        center = (ymin + ymax) / 2
        rh = (ymax - ymin) / 2 / 0.7
        ax.set_ylim(center - rh, center + rh)
        print("Y zoom out: %.2f V" % (ax.get_ylim()[1]-ax.get_ylim()[0]))
    elif event.key == "1":
        line_clean.set_visible(not line_clean.get_visible())
        ax.legend()
    elif event.key == "2":
        line_noisy.set_visible(not line_noisy.get_visible())
        ax.legend()
    elif event.key == "3":
        line_filtered.set_visible(not line_filtered.get_visible())
        ax.legend()
    elif event.key in ("r", "R"):
        ax.autoscale(True)
        ax.relim()
        print("View reset")
    elif event.key == " ":
        if ani.event_source.running:
            ani.event_source.stop()
            print("Paused")
        else:
            ani.event_source.start()
            print("Resumed")
    elif event.key in ("q", "Q"):
        print("Quitting...")
        running = False
        plt.close()
    fig.canvas.draw_idle()


def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        if any(kw in desc for kw in ["ch340", "ch341", "ch343", "cp210", "ftdi", "usb-serial"]):
            return port.device
        if "esp32" in desc or "espressif" in desc:
            return port.device
        if port.vid in [0x1A86, 0x10C4, 0x0403, 0x303A]:
            return port.device
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", help="COM port")
    parser.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("-w", "--window", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()

    port = args.port or find_esp32_port()
    if not port:
        print("Error: no ESP32 found. Use -p COMx")
        sys.exit(1)

    WINDOW_SIZE = args.window
    reader_thread = threading.Thread(target=serial_reader, args=(port, args.baud), daemon=True)
    reader_thread.start()
    time.sleep(0.5)
    if not running:
        sys.exit(1)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.canvas.manager.set_window_title("ESP32-ECG-Serial-Plotter")

    line_clean, = ax.plot([], [], "g-", lw=1.8, alpha=0.85, label="Clean(noise-free)")
    line_noisy, = ax.plot([], [], "r-", lw=1.0, alpha=0.5, label="Noisy(raw)")
    line_filtered, = ax.plot([], [], "b-", lw=1.8, label="Filtered")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("ESP32-ECG Real-time Signal Monitor (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # BPM 显示: 图内左上角 + 白底不挡波形
    bpm_text = ax.text(0.02, 0.95, "♥ -- BPM",
                       transform=ax.transAxes, fontsize=20, color="red",
                       ha="left", va="top", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor="lightgray", alpha=0.85))

    status_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                          fontsize=9, color="gray", va="bottom")

    fig.canvas.mpl_connect("key_press_event", on_key)
    ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL_MS, blit=False, cache_frame_data=False)

    print("="*50)
    print("  ESP32-ECG Serial Plotter v2.0")
    print("  (with onboard BPM detection)")
    print("="*50)
    print("  ->/<- : Time axis   up/down : Y axis")
    print("  1/2/3 : Toggle curves      R : Reset")
    print("  Space : Pause/Resume       Q : Quit")
    print("="*50)

    plt.tight_layout()
    plt.show()

    running = False
    if serial_port and serial_port.is_open:
        serial_port.close()
