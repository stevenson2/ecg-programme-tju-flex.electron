"""Serial handshake for the PC plotter.

Opening the port must not pulse DTR/RTS (that resets typical auto-program
boards). After the port is open, ping until the firmware answers PONG, then
the caller sends WAVE 1.
"""


def wait_until_pong(write, readline, clock, timeout_s=3.0, ping_interval_s=0.3):
    """Send PING about every ping_interval_s until a line contains PONG.

    write(str), readline() -> str, clock() -> seconds. readline is expected
    to advance time (a serial timeout, or a test double). Returns True when
    PONG is seen before timeout_s.
    """
    start = clock()
    deadline = start + timeout_s
    next_ping = start
    while True:
        now = clock()
        if now >= deadline:
            return False
        if now >= next_ping:
            write("PING\n")
            next_ping = now + ping_interval_s
        line = readline()
        if line and "PONG" in line:
            return True
