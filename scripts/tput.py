# One-direction TCP throughput with an explicit per-socket busy-poll setting.
# server: tput.py --serve PORT [--busy USEC]   client: tput.py --client HOST:PORT --secs N
import socket, sys, time, argparse, os
SO_BUSY_POLL = 46
ap = argparse.ArgumentParser()
ap.add_argument("--serve"); ap.add_argument("--client"); ap.add_argument("--secs", type=float, default=6)
ap.add_argument("--busy", type=int, default=-1, help="SO_BUSY_POLL usec on the receiving socket (-1 = leave sysctl default)")
ap.add_argument("--bufs", type=int, default=4_000_000)
a = ap.parse_args()
def tune(s):
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, a.bufs)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, a.bufs)
if a.serve:
    ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", int(a.serve))); ls.listen(1)
    while True:
        c, peer = ls.accept(); tune(c)
        if a.busy >= 0: c.setsockopt(socket.SOL_SOCKET, SO_BUSY_POLL, a.busy)
        buf = bytearray(1 << 20); n = 0; t0 = time.time(); last = t0
        while True:
            r = c.recv_into(buf)
            if not r: break
            n += r
        dt = time.time() - t0
        print(f"recv {n/1e9:.2f} GB in {dt:.2f}s = {8*n/dt/1e9:.2f} Gbit/s busy={a.busy}", flush=True)
        c.close()
else:
    host, port = a.client.rsplit(":", 1)
    s = socket.create_connection((host, int(port))); tune(s)
    payload = b"\x5a" * (1 << 20); n = 0; t0 = time.time()
    while time.time() - t0 < a.secs:
        s.sendall(payload); n += len(payload)
    dt = time.time() - t0; s.close()
    print(f"sent {n/1e9:.2f} GB in {dt:.2f}s = {8*n/dt/1e9:.2f} Gbit/s", flush=True)
