#!/usr/bin/env python3
"""One line per hub_req text file (prefill requests, b >= 64): count, misses/req, page ms/req, srv ms/req, pf words/req.
  hubreq_sum.py TXT [TXT...]"""
import os, sys
for p in sys.argv[1:]:
    F, R = None, []
    for line in open(p):
        if line.startswith("#F hub_req "):
            F = line.split()[2:]
        elif line.startswith("hub_req ") and F:
            r = dict(zip(F, (float(x) for x in line.split()[1:])))
            if r["b"] >= 64:
                R.append(r)
    n = max(len(R), 1)
    s = lambda k: sum(r[k] for r in R)
    print(f"{os.path.basename(p):22} reqs {len(R):5d}  misses/req {s('n_miss') / n:5.1f}  page ms/req {s('page_us') / n / 1e3:6.1f}"
          f"  srv ms/req {s('srv_us') / n / 1e3:6.1f}  pf words/req {s('n_pf_words') / n:5.1f}")
