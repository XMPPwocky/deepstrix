"""LRU expert-residency simulation on a V4.1 routing trace taken from an oracle
dump (layer_NN_topk_ids.pt, [T, 6] per layer): misses per token vs resident
fraction, global slot pool (any layer's expert can occupy any slot), warm LRU.
Teacher-forced prefill routing stands in for decode routing.

  python3 v41_lru_from_dump.py ~/.cache/deepstrix/v41/agentic/main
"""
import glob
import os
import re
import sys
from collections import OrderedDict

import torch


def main(dump):
    files = sorted(glob.glob(os.path.join(dump, "layer_*_topk_ids.pt")))
    layers = [int(re.search(r"layer_(\d+)_topk", f).group(1)) for f in files]
    ids = torch.stack([torch.load(f).reshape(-1, 6) for f in files], dim=1).long()  # [T, L, 6]
    T, L, K = ids.shape
    n_expert = 384
    total = L * n_expert
    print(f"trace: T={T} layers={L} picks/token={L * K}; distinct experts touched: "
          f"{len(set((l, int(e)) for t in range(T) for l in range(L) for e in ids[t, l]))} of {total}")
    # picks concentration
    from collections import Counter
    cnt = Counter((l, int(e)) for t in range(T) for l in range(L) for e in ids[t, l])
    top = sum(c for _, c in cnt.most_common(int(0.05 * total)))
    print(f"top-5% experts carry {top / (T * L * K):.1%} of picks")
    for frac in (0.50, 0.60, 0.66, 0.75, 0.85):
        slots = int(frac * total)
        lru = OrderedDict()
        misses = []
        for t in range(T):
            m = 0
            for l in range(L):
                for e in ids[t, l].tolist():
                    key = (l, e)
                    if key in lru:
                        lru.move_to_end(key)
                    else:
                        m += 1
                        lru[key] = None
                        if len(lru) > slots:
                            lru.popitem(last=False)
            misses.append(m)
        warm = misses[T // 2:]  # second half = warm cache
        print(f"resident {frac:.0%} ({slots} slots): misses/token warm {sum(warm) / len(warm):.2f} "
              f"(cold-start half {sum(misses[:T // 2]) / (T // 2):.2f}), max {max(warm)}")


if __name__ == "__main__":
    main(sys.argv[1])
