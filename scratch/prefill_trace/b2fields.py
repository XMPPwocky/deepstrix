"""Run ON box 2: list evtrace record kinds, counts and field names of one .evt file (+2 sample records per kind).
usage: python3 -I b2fields.py EVTRACE_PY EVT_FILE"""
import sys, importlib.util

spec = importlib.util.spec_from_file_location('evtrace', sys.argv[1])
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
hdr, recs = ev.read_file(sys.argv[2])
for k, v in sorted(recs.items(), key=lambda kv: -len(kv[1])):
    print(f"{k}: {len(v)} records; fields {sorted(v[0].keys()) if v else []}")
    for r in v[:2]:
        print("   ", {f: r[f] for f in sorted(r)})
