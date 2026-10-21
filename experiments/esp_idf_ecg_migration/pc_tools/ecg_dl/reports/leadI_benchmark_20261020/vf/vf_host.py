"""Subprocess client for host/fw_replay_host. Little-endian int32 / float32."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

WIN_N = 1250


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"host closed after {len(buf)}/{n} bytes")
        buf += chunk
    return bytes(buf)


class ReplayHost:
    def __init__(self, binary: Path, stderr_path: Path | None = None):
        if sys.byteorder != "little":
            raise RuntimeError("host protocol is little-endian")
        self.binary = Path(binary)
        self.stderr_path = stderr_path
        self.p: subprocess.Popen | None = None
        self._err = None

    def _ensure(self) -> None:
        if self.p is not None and self.p.poll() is None:
            return
        if self.stderr_path is not None:
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._err = open(self.stderr_path, "ab", buffering=0)
        self.p = subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            bufsize=0,
        )

    def close(self) -> None:
        if self.p is not None and self.p.poll() is None:
            try:
                self.p.stdin.write(np.int32(0).tobytes())
                self.p.stdin.close()
                self.p.wait(timeout=5)
            except Exception:
                self.p.kill()
        self.p = None
        if self._err is not None:
            self._err.close()
            self._err = None

    def _write(self, blob: bytes) -> None:
        self._ensure()
        assert self.p is not None and self.p.stdin is not None
        self.p.stdin.write(blob)
        self.p.stdin.flush()

    def score_window(self, x: np.ndarray) -> dict:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (WIN_N,):
            raise ValueError("window must be 1250 float samples")
        self._write(np.int32(1).tobytes() + x.tobytes())
        raw = read_exact(self.p.stdout, 9 * 4)
        arr = np.frombuffer(raw, dtype=np.float32)
        return {
            "score": float(arr[0]),
            "rms": float(arr[1]),
            "suspect": int(arr[2] >= 0.5),
            "feat": arr[3:9].astype(np.float32).copy(),
        }

    def vf_stream(self, x: np.ndarray) -> np.ndarray:
        """250 Hz samples -> (nwin, 4) float32 score, rms, suspect, alarm."""
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        self._write(np.int32(3).tobytes() + np.int32(x.size).tobytes() + x.tobytes())
        nwin = int(np.frombuffer(read_exact(self.p.stdout, 4), dtype=np.int32)[0])
        if nwin == 0:
            return np.zeros((0, 4), dtype=np.float32)
        raw = read_exact(self.p.stdout, nwin * 4 * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(nwin, 4).copy()

    def frontend(self, x500: np.ndarray):
        """500 Hz samples -> (windows (nwin,4), beat indices int32)."""
        x = np.ascontiguousarray(x500, dtype=np.float32).reshape(-1)
        self._write(np.int32(2).tobytes() + np.int32(x.size).tobytes() + x.tobytes())
        nwin = int(np.frombuffer(read_exact(self.p.stdout, 4), dtype=np.int32)[0])
        if nwin:
            wins = np.frombuffer(read_exact(self.p.stdout, nwin * 16), dtype=np.float32).reshape(nwin, 4).copy()
        else:
            wins = np.zeros((0, 4), dtype=np.float32)
        nb = int(np.frombuffer(read_exact(self.p.stdout, 4), dtype=np.int32)[0])
        if nb:
            beats = np.frombuffer(read_exact(self.p.stdout, nb * 4), dtype=np.int32).copy()
        else:
            beats = np.zeros(0, dtype=np.int32)
        return wins, beats
