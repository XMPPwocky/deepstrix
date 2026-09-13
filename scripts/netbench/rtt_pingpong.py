"""Application-level round-trip benchmark for the box-to-box link: the
engine's real per-layer pattern is one request + one reply of a few tens of
KB over a persistent TCP connection, 40 times per decode token.

  server:  python3 rtt_pingpong.py --serve 0.0.0.0:5201
  client:  python3 rtt_pingpong.py --client <peer>:5201 --bytes 32768 --iters 2000

Prints p50 / p90 / p99 / max round-trip in microseconds and the implied
40-RTT-per-token cost (PLAN.md §6 budgets ≤ 100 µs → 4 ms/token).
"""
import argparse
import socket as _sk


def _split(addr):
    """host:port, with IPv6 as [addr%scope]:port or addr%scope:port (last colon splits)."""
    host, port = addr.rsplit(":", 1)
    host = host.strip("[]")
    fam = _sk.AF_INET6 if ":" in host else _sk.AF_INET
    return fam, host, int(port)

import socket
import struct
import time


SPIN = False
BUFS = 0


def tune(sock):
    if BUFS:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFS)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFS)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        except OSError:
            pass


def recv_exact(sock, n):
    if SPIN:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            try:
                k = sock.recv_into(view[got:], n - got, socket.MSG_DONTWAIT)
            except BlockingIOError:
                continue
            if k == 0:
                raise ConnectionError("peer closed")
            got += k
        return buf
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed")
        got += k
    return buf


def serve(addr):
    fam, host, port = _split(addr)
    ls = socket.socket(fam)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind((host, port))
    ls.listen(1)
    print(f"listening on {addr}")
    while True:
        conn, peer = ls.accept()
        tune(conn)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("client", peer)
        try:
            while True:
                (n,) = struct.unpack("<I", recv_exact(conn, 4))
                payload = recv_exact(conn, n)
                conn.sendall(struct.pack("<I", n) + payload)  # echo (same size reply)
        except (ConnectionError, struct.error):
            conn.close()


def client(addr, nbytes, iters):
    fam, host, port = _split(addr)
    s = socket.socket(fam)
    # getaddrinfo resolves the IPv6 scope ("%thunderbolt0") into the sockaddr
    tune(s)
    s.connect(socket.getaddrinfo(host, port, fam, socket.SOCK_STREAM)[0][4])
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    payload = bytes(nbytes)
    hdr = struct.pack("<I", nbytes)
    for _ in range(50):  # warm-up
        s.sendall(hdr + payload)
        recv_exact(s, 4 + nbytes)
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        s.sendall(hdr + payload)
        recv_exact(s, 4 + nbytes)
        lat.append((time.perf_counter_ns() - t0) / 1e3)
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
    print(f"{nbytes} B request/reply x {iters}: p50 {q(0.5):.0f} us  p90 {q(0.9):.0f} us  "
          f"p99 {q(0.99):.0f} us  max {lat[-1]:.0f} us  -> 40 RTT/token = {40 * q(0.5) / 1e3:.2f} ms (p50)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve")
    ap.add_argument("--client")
    ap.add_argument("--bytes", type=int, default=32768)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--bufs", type=int, default=0, help="SO_SNDBUF/SO_RCVBUF bytes (0 = kernel default/autotune); also sets TCP_QUICKACK")
    ap.add_argument("--spin", action="store_true", help="busy-poll non-blocking sockets instead of blocking recv (no scheduler wakeups)")
    a = ap.parse_args()
    globals()["SPIN"] = a.spin
    globals()["BUFS"] = a.bufs
    if a.serve:
        serve(a.serve)
    else:
        client(a.client, a.bytes, a.iters)
