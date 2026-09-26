#!/usr/bin/env python3
"""Patch the copied harness2 into the reviewer harness: bmax 8 -> 16, extra correctness shapes
(odd row counts, odd tails in both sections, dense store short rows, b=16 = max production b,
and an n_total=641 overflow probe beyond the candidate's 640-key cap)."""
p = 'attn_harness_rv.cpp'
s = open(p).read()
assert 'const unsigned bmax = 8, raw_slots = 256' in s
s = s.replace('const unsigned bmax = 8, raw_slots = 256', 'const unsigned bmax = 16, raw_slots = 256')
old = '''            mk({128, 0},              {512, 0},              false, "b=2 with an EMPTY row (n_total=0)"),
        };'''
new = '''            mk({128, 0},              {512, 0},              false, "b=2 with an EMPTY row (n_total=0)"),
            // ---- reviewer additions ----
            mk({128, 128, 128, 3, 128, 128, 128}, {512, 511, 1, 0, 17, 512, 509}, false, "RV b=7 odd rows, odd tails gathered"),
            mk({127}, {497}, false, "RV b=1 127+497 odd tails both sections"),
            mk({1}, {1}, false, "RV b=1 1+1"),
            mk({128, 33}, {512, 5}, true, "RV b=2 dense comp store, short row"),
            mk(std::vector<int>(16, 128), std::vector<int>(16, 512), false, "RV b=16 128+512 gathered (max production b)"),
            mk({128, 128, 64, 128, 100, 128, 128, 128, 128, 17, 128, 128, 128, 128, 128, 5}, {512, 512, 512, 129, 300, 512, 512, 512, 512, 33, 512, 512, 512, 511, 512, 0}, true, "RV b=16 dense comp store mixed"),
            mk({129, 128}, {512, 512}, false, "RV OVERFLOW PROBE b=2 row0 n_total=641 (beyond cand cap 640; expected WRONG)"),
        };'''
assert old in s
s = s.replace(old, new)
open(p, 'w').write(s)
print('patched')
