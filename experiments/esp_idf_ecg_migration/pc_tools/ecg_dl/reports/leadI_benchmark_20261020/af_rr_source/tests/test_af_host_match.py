import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from af_backend import FirmwareAF

HOST = Path(__file__).resolve().parents[1] / "host" / "af_host"


@pytest.mark.skipif(not HOST.exists(), reason="af_host not built")
def test_python_backend_matches_compiled_af_detect():
    rng = np.random.default_rng(7)
    rr = rng.uniform(0.34, 0.95, size=80)
    payload = "".join(f"{r:.6f} 1.0\n" for r in rr).encode()
    proc = subprocess.run(
        [str(HOST)], input=payload, capture_output=True, check=True
    )
    c_rows = []
    for line in proc.stdout.decode().strip().splitlines():
        p = line.split()
        c_rows.append((float(p[0]), int(p[1]), float(p[2]), float(p[3]), int(p[5])))

    af = FirmwareAF()
    py = []
    for w in af.evaluate_rr(rr):
        if w.window_ready or w.n_rr > 0:
            py.append((w.t_end, w.n_rr, w.cv, w.entropy, w.label))

    assert len(c_rows) == len(py), (len(c_rows), len(py), proc.stdout.decode()[:400])
    for c, p in zip(c_rows, py):
        assert c[1] == p[1]
        assert c[4] == p[4]
        assert abs(c[2] - p[2]) < 2e-4
        assert abs(c[3] - p[3]) < 2e-4
