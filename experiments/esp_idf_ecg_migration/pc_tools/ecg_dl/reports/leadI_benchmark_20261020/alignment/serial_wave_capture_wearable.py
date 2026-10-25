"""Serial WAVE capture. afe_confirmed proves ADC readiness/reads, not ECG quality."""
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

STATUS_KEYS = ("ready", "valid", "init", "read", "ok", "fail", "status", "cal", "ovs")


def parse_afe_status(text):
    """Require one complete, unambiguous device status reply."""
    matches = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("CMD "):
            line = line[4:]
        if not line.startswith("AFE_STATUS "):
            continue
        fields = {}
        for item in line.split()[1:]:
            if "=" not in item:
                return None
            key, value = item.split("=", 1)
            if key in fields:
                return None
            fields[key] = value
        try:
            status = {"mode": fields["mode"]}
            status.update({key: int(fields[key], 0) for key in STATUS_KEYS})
        except (KeyError, ValueError):
            return None
        if any(not 0 <= status[key] <= 0xffffffff for key in STATUS_KEYS):
            return None
        matches.append(status)
    return matches[0] if len(matches) == 1 else None


def confirm_afe(mode_reply, before, after):
    """Compare fresh snapshots; missing telemetry in legacy firmware fails closed."""
    replies = {line.strip().removeprefix("CMD ") for line in mode_reply.splitlines()}
    if "MODE ok AFE_REAL" not in replies or any(s.startswith("MODE fail") for s in replies):
        return False, "MODE AFE request not acknowledged successfully"
    if before is None or after is None:
        return False, "missing or malformed AFE_STATUS"
    for state in (before, after):
        if state["mode"] != "AFE_REAL" or state["ready"] != 1 or state["valid"] != 1:
            return False, "AFE not ready or latest sample invalid"
        if state["init"] != 0 or state["read"] != 0:
            return False, "AFE initialization/read error"
        if state["status"] not in (0, 1, 2):
            return False, "AFE status has no valid ADC sample"
        if state["cal"] not in (0, 1) or not 1 <= state["ovs"] <= 16:
            return False, "invalid calibration/oversampling telemetry"
    if after["ok"] <= before["ok"]:
        return False, "valid read count did not advance (or counter reset/wrapped)"
    if after["fail"] != before["fail"]:
        return False, "failed read count changed"
    return True, "AFE_REAL ready; valid reads advanced; no new read failures"


def parse_wave(line):
    parts = line.strip().split(",")
    if len(parts) != 10:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    return values if all(math.isfinite(v) for v in values) else None


def main():
    import serial  # importing confirmation helpers must never open a device
    port = sys.argv[1] if len(sys.argv) > 1 else "COM15"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a positive finite number")
    inbox = Path(r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\real_captures\inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("serial_%Y%m%d_%H%M%S_%f_wearable_single_lead_AD8232")
    csv_path = inbox / (session_id + ".csv")
    meta_path = inbox / (session_id + "_meta.json")
    log_path = inbox / (session_id + "_log.txt")
    lines, evidence, statuses = [], [], []

    def log(tag, message):
        entry = f"[{tag}] {message}"
        lines.append(entry)
        print(entry, flush=True)

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout, ser.write_timeout = port, 460800, 0.25, 2
    ser.dsrdtr = ser.rtscts = ser.dtr = ser.rts = False
    meta = {
        "session_id": session_id, "capture_method": "serial_wave_csv",
        "port": port, "baud": 460800, "afe": "AD8232",
        "lead": "wearable_single_lead (placement unverified)",
        "afe_confirmed": False, "afe_evidence": evidence, "afe_status_snapshots": statuses,
        "confirmation_scope": "ADC path readiness and valid reads; not electrode contact, waveform fidelity or diagnosis",
        "duration_s_requested": duration, "csv_path": str(csv_path),
        "label_status": "unlabeled_raw_capture", "do_not_use_onboard_ai_as_gold": True,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai", "pid": os.getpid(),
    }

    def read_for(seconds):
        end, buf = time.monotonic() + seconds, bytearray()
        while time.monotonic() < end:
            buf.extend(ser.read(8192))
        return buf.decode("utf-8", "replace")

    def send(command, wait=0.7):
        ser.write((command + "\n").encode("ascii"))
        ser.flush()
        return read_for(wait)

    code = 3
    try:
        ser.open()
        time.sleep(0.3)
        ser.reset_input_buffer()
        log("OPEN", f"{port} @ 460800 dtr=False")
        send("WAVE 0")
        mode_reply = send("MODE afe")
        log("MODE", mode_reply.strip())
        first = parse_afe_status(send("AFE_STATUS"))
        second = parse_afe_status(send("AFE_STATUS"))
        statuses.extend((first, second))
        confirmed, reason = confirm_afe(mode_reply, first, second)
        if not confirmed:
            meta.update(status="aborted_afe_unverified", failure_reason=reason)
            log("AFE_FAIL", reason)
            return code
        evidence.append(reason)
        meta["firmware_status"] = send("STATUS").strip()
        send("WAVE 1", 0.3)
        n_csv, n_lines, hist = 0, 0, {}
        pending = b""
        start = last_progress = time.monotonic()
        previous_status = second
        observation_ok = True
        log("CAPTURE", f"start {duration}s -> {csv_path}")
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("host_ts_s", "raw_line"))
            while time.monotonic() - start < duration:
                pending += ser.read(8192)
                records = pending.split(b"\n")
                pending = records.pop()
                for raw in records:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    n_lines += 1
                    if line.startswith(("CMD AFE_STATUS ", "AFE_STATUS ")):
                        state = parse_afe_status(line)
                        statuses.append(state)
                        valid, reason = confirm_afe(mode_reply, previous_status, state)
                        observation_ok = observation_ok and valid
                        if not valid:
                            log("AFE_FAIL", reason)
                        previous_status = state
                        continue
                    values = parse_wave(line)
                    if values is not None:
                        n_csv += 1
                        key = str(int(values[4])) if values[4].is_integer() else str(values[4])
                        hist[key] = hist.get(key, 0) + 1
                        writer.writerow((f"{time.monotonic()-start:.3f}", line))
                if len(pending) > 4096:
                    raise ValueError("unframed/oversized serial line")
                if time.monotonic() - last_progress >= 10:
                    log("PROG", f"t={time.monotonic()-start:.0f}s csv={n_csv}")
                    ser.write(b"AFE_STATUS\n")
                    ser.flush()
                    last_progress = time.monotonic()
                    stream.flush()
        elapsed = time.monotonic() - start
        # WAVE command can drain a pending periodic status: keep that evidence too.
        tail = send("WAVE 0")
        for line in tail.splitlines():
            if line.strip().startswith(("CMD AFE_STATUS ", "AFE_STATUS ")):
                state = parse_afe_status(line)
                statuses.append(state)
                valid, reason = confirm_afe(mode_reply, previous_status, state)
                observation_ok = observation_ok and valid
                previous_status = state
        final = parse_afe_status(send("AFE_STATUS"))
        statuses.append(final)
        valid, reason = confirm_afe(mode_reply, previous_status, final)
        observation_ok = observation_ok and valid
        # Simulator fields are an additional rejection check, never the proof.
        observation_ok = observation_ok and bool(n_csv) and set(hist) == {"0"}
        meta.update(
            status="captured" if observation_ok else "captured_afe_unverified",
            afe_confirmed=observation_ok, true_bpm_hist=hist, csv_rows=n_csv,
            total_lines_seen=n_lines, duration_s_actual=round(elapsed, 2),
            firmware_note="Requires MODE success and fresh AFE_STATUS counters before/during/after capture",
        )
        if observation_ok:
            evidence.append("Valid read counts advanced with unchanged failure counters throughout observation")
        else:
            meta["failure_reason"] = reason if not valid else "AFE observation failed or non-AFE mode leaked into WAVE"
        code = 0 if observation_ok and n_csv > 100 else 4
        log("DONE", f"csv={n_csv} afe_confirmed={observation_ok}")
        return code
    except Exception as exc:
        meta.update(status="capture_error", afe_confirmed=False, failure_reason=str(exc))
        log("ERROR", str(exc))
        return 5
    finally:
        try:
            if ser.is_open:
                ser.write(b"WAVE 0\n")
        except Exception as exc:
            log("CLOSE", str(exc))
        finally:
            if ser.is_open:
                try:
                    ser.close()
                except Exception as exc:
                    log("CLOSE", str(exc))
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
