"""Fake-clock test: plotter waits for PONG and does not need a DTR pulse."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pc_tools"))

from plotter_handshake import wait_until_pong


class Port:
    def __init__(self, lines):
        self.t = 0.0
        self.writes = []
        self.lines = list(lines)

    def clock(self):
        return self.t

    def write(self, data):
        self.writes.append(data)

    def readline(self):
        self.t += 0.05
        if not self.lines:
            self.t += 0.25
            return ""
        return self.lines.pop(0)


def test_pong_then_ready():
    port = Port(["boot\n", "CMD PONG 10\n"])
    ok = wait_until_pong(port.write, port.readline, port.clock, timeout_s=3.0)
    assert ok, "expected PONG"
    assert port.writes[0] == "PING\n"
    assert port.t < 3.0


def test_timeout_without_pong():
    port = Port([])
    ok = wait_until_pong(port.write, port.readline, port.clock, timeout_s=1.0)
    assert not ok
    assert port.writes, "pings were sent"
    assert all(w == "PING\n" for w in port.writes)


if __name__ == "__main__":
    test_pong_then_ready()
    test_timeout_without_pong()
    print("PASS plotter handshake")
