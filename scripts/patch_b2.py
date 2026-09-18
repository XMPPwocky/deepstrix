p = "/home/claude-code/b2_run_expertd.sh"
s = open(p).read()

old_pl = "PLACEMENT=${PLACEMENT-/home/claude-code/box2_placement.txt}"
new_pl = """# DEFAULT SINCE 2026-09-16: UNIFORM assignment, NO placement file.
#
# Was: --experts L0-L19:116-383,L20-L39:344-383 plus a frequency-ranked
# placement file. That gave DECODER layers only 40 slots each against a B=6
# verify union of up to 36, so those regions turned over almost completely every
# step, while encoder layers sat on 268.
#
# Two reasons to drop the static placement:
#  * It does not generalise. Measured: a static frequency table retains only
#    ~30% of its in-sample coverage held-out, and a prose-fit table covers just
#    35.2% of CODE picks.
#  * It is no longer load-bearing. `enable_paging` seeds the LRU from whatever
#    `load` placed, so the file only decides BOOT residency; eviction has been
#    global since V41_B2_GLOBAL_POOL (default on) with V41_B2_POOL_FLOOR=0, i.e.
#    capacity already migrates between layers for decode. The per-layer number
#    now only sets boot residency and the prefill-sweep containment region.
#
# L0-L39:230-383 = 154 experts/layer x 40 = 6160 slots, the same total capacity
# as the old split (268x20 + 40x20), just spread evenly. Decoder 40 -> 154 (3.9x).
#
# NOT YET BENCHMARKED head-to-head: the A/B that would have scored this raced on
# its readiness check and the uniform arm died on connection refused. This is a
# decision on the generalisation evidence above, not on a measured win.
#
# Roll back with:
#   EXPERTS='L0-L19:116-383,L20-L39:344-383' PLACEMENT=/home/claude-code/box2_placement.txt
PLACEMENT=${PLACEMENT-}"""
assert s.count(old_pl) == 1, "placement default not found"
s = s.replace(old_pl, new_pl, 1)

old_e = '--experts "${EXPERTS:-L0-L19:116-383,L20-L39:344-383}"'
new_e = '--experts "${EXPERTS:-L0-L39:230-383}"'
assert s.count(old_e) == 1, "experts default not found"
s = s.replace(old_e, new_e, 1)

open(p, "w").write(s)
print("patched ok")
