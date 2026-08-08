#!/usr/bin/env python
"""Capture ECG serial output for heartrate debug analysis.

Captures: clean, noisy, filtered, bpm, true_bpm, sqi, motion
Logs to CSV and prints summary every second.
"""
import serial
import time
import sys
import os

def find_esp32_port():
    """Auto-detect ESP32 port or use COM7."""
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "COM7" in p.device or ("Silicon" in p.description and "CP210" in p.description):
            return p.device
    # Fallback: try COM7
    return "COM7"

def main():
    port = find_esp32_port()
    print(f"[Capture] Opening {port} @ 460800...")
    
    ser = serial.Serial(port, 460800, timeout=1)
    time.sleep(1)  # Wait for ESP32 to settle
    
    # Flush startup messages
    ser.reset_input_buffer()
    
    # Create output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(os.path.dirname(__file__), f"..", f"ecg_capture_{timestamp}.csv")
    with open(filename, 'w') as f:
        f.write("ts,clean,noisy,filtered,bpm,true_bpm,sqi,motion\n")
    
    print(f"[Capture] Logging to {filename}")
    print("[Capture] Press Ctrl+C to stop. Collecting data...")
    print(f"{'Time':>8s} {'BPM':>4s} {'TrueBPM':>7s} {'SQI':>6s} {'Motion':>6s} {'RRms':>6s} {'Beats':>5s}")
    print("-" * 55)
    
    start_time = time.time()
    sample_count = 0
    last_bpm = 0
    bpm_jumps = 0
    bpm_locked_count = 0
    last_summary = time.time()
    
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            if not line:
                continue
            
            # Skip non-data lines (text/debug messages)
            if not line or line[0] not in '-0123456789.':
                continue
            
            parts = line.split(',')
            if len(parts) < 7:
                continue
            
            try:
                clean = float(parts[0])
                noisy = float(parts[1])
                filtered = float(parts[2])
                bpm = int(parts[3])
                true_bpm = int(parts[4])
                sqi = float(parts[5])
                motion = int(parts[6])
            except ValueError:
                continue
            
            sample_count += 1
            elapsed = time.time() - start_time
            
            # Log to file
            with open(filename, 'a') as f:
                f.write(f"{elapsed:.3f},{clean:.4f},{noisy:.4f},{filtered:.4f},{bpm},{true_bpm},{sqi:.3f},{motion}\n")
            
            # Track BPM jumps (change > 15 in one frame)
            if last_bpm > 0:
                jump = abs(bpm - last_bpm)
                if jump > 15:
                    bpm_jumps += 1
                    print(f"[JUMP] t={elapsed:.1f}s BPM: {last_bpm}→{bpm} (Δ={jump}) true={true_bpm}")
                if bpm == last_bpm and bpm != true_bpm and true_bpm > 0:
                    bpm_locked_count += 1
            
            # Print summary every second
            if time.time() - last_summary >= 5.0:
                print(f"{elapsed:>7.1f}s {bpm:>4d} {true_bpm:>7d} {sqi:>5.2f} {motion:>6d} {0:>6d} {sample_count//250:>5d}")
                last_summary = time.time()
            
            last_bpm = bpm
            
    except KeyboardInterrupt:
        print("\n\n[Capture] Stopped.")
        print(f"[Capture] Total samples: {sample_count} ({sample_count/250:.1f}s @250Hz)")
        print(f"[Capture] BPM jumps (>15): {bpm_jumps}")
        print(f"[Capture] BPM locked frames: {bpm_locked_count} ({(bpm_locked_count/max(1,sample_count))*100:.1f}%)")
        print(f"[Capture] Data saved to: {filename}")
        
        if bpm_jumps > 0:
            print("\n[Analysis] BPM jump issue detected - check log for timestamp details")
        if bpm_locked_count > sample_count * 0.3:
            print(f"[Analysis] BPM appears LOCKED (>30% frames same value despite true_bpm change)")
        
        ser.close()

if __name__ == "__main__":
    main()