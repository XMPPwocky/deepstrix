#!/usr/bin/env python3
"""Run ON box 2: stream-filter b2_req records for WALL-clock UTC ranges (converted to this box's CLOCK_MONOTONIC_RAW
here, so no cross-box clock mapping) via evtfilt.py; prints evtfilt's text output (small).
  b2win.py EVTFILT_PY EVT_FILE 2026-10-10T21:38:00/2026-10-10T21:41:46[,...] [KINDS]"""
import datetime, subprocess, sys, time

evtfilt, path, spans = sys.argv[1], sys.argv[2], sys.argv[3]
kinds = sys.argv[4] if len(sys.argv) > 4 else "b2_req"
now_raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
now_wall = datetime.datetime.now(datetime.timezone.utc).timestamp()


def raw(iso):
    w = datetime.datetime.fromisoformat(iso).replace(tzinfo=datetime.timezone.utc).timestamp()
    return int(now_raw - (now_wall - w) * 1e9)


rng = ",".join(f"{raw(a)}:{raw(b)}" for a, b in (s.split("/") for s in spans.split(",")))
print(f"# ranges raw {rng}", flush=True)
subprocess.run([sys.executable, "-I", evtfilt, path, rng, kinds], check=False)
