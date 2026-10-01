# KV prefix store analysis scripts

The analyses behind `docs/v41/KV_PREFIX_STORE_DESIGN.md` (section 1), committed as run on
2026-10-01. Inputs are the hub logs and the snapshot root named in the doc's section 1.0;
most scripts take their paths as arguments or constants at the top. They are one-off
analyses, not tools: expect to edit paths before re-running.

| Script | Author | What it computes |
|---|---|---|
| `cache_stats.py` | owner session | snapshot store contents: entries, bytes, duplicate prefixes |
| `cache_lcp2.py` | owner session | longest common prefix of each admission vs the stored snapshots |
| `cache_head.py` | owner session | where cold prompts diverge from stored ones (prompt head) |
| `snap_diverge.py` | owner session | first divergence position between consecutive turns' snapshots |
| `sim.py` | design writer | corpus replay of the chunked store (hit rate, bytes, by K) |
| `logstats.py`, `logstats2.py`, `logstats3.py` | design writer | admissions, prefill time, checkpoint saves from the hub log |
| `ck.py`, `ck2.py` | reviewer | checkpoint save bandwidth and the tokens= counting bug; reads `ckpt_counts.txt` (a grep of the hub log's checkpoint lines; regenerate it) |
| `sess.py` | reviewer | session_id presence in snapshot entries, eviction reasons |
| `bpt.py` | reviewer | bytes per position of a checkpoint |
| `anchor.py`, `anchor2.py` | reviewer | anchor suffix lengths and anchor-prefix reuse over first turns |
