# KV prefix store analysis scripts

The analyses behind `docs/v41/KV_PREFIX_STORE_DESIGN.md` (section 1), committed as run on
2026-10-01. Inputs are the hub logs, the snapshot root `~/.cache/deepstrix/snapshots-v41`
(named in the design doc's header, not section 1.0; corrected 2026-10-04) and `ckpt_counts.txt`
(below). Paths are constants at the top of each script. They are one-off analyses, not tools:
expect to edit paths before re-running.

| Script | Author | What it computes |
|---|---|---|
| `cache_stats.py` | owner session | snapshot store contents: entries, bytes, duplicate prefixes |
| `cache_lcp2.py` | owner session | longest common prefix of each admission vs the stored snapshots |
| `cache_head.py` | owner session | where cold prompts diverge from stored ones (prompt head) |
| `snap_diverge.py` | owner session | first divergence position between consecutive turns' snapshots |
| `sim.py` | design writer | corpus replay of the chunked store (hit rate, bytes, by K) |
| `logstats.py`, `logstats2.py`, `logstats3.py` | design writer | admissions, prefill time, checkpoint saves from the hub log |
| `ck.py`, `ck2.py` | reviewer | checkpoint save bandwidth and the tokens= counting bug; parse the hub logs directly (globs at the top) |
| `sess.py` | reviewer | session_id presence in snapshot entries, eviction reasons |
| `bpt.py` | reviewer | bytes per position of a checkpoint |
| `anchor.py`, `anchor2.py` | reviewer | anchor suffix lengths and anchor-prefix reuse over first turns |

## `ckpt_counts.txt` (corrected 2026-10-04)

Read by `sim.py`, `cache_lcp2.py`, `cache_head.py`, `anchor.py`, `anchor2.py`, `bpt.py` and
`sess.py` (not by `ck.py` / `ck2.py`), from the hardcoded path
`/home/claude-code/.claude/jobs/16b63e08/tmp/ckpt_counts.txt`: a session job's temp dir, not in
the repo (still on box 1 on 2026-10-04, 142 values, but liable to be cleaned). Format: one integer
per line, sorted, unique. Each is the token count of a checkpoint snapshot; a snapshot whose
`tokens.bin` length is in the set is treated as a checkpoint. Most scripts drop those to keep
prompt-end snapshots only; `bpt.py` and `cache_lcp2.py` split on it.
(`cache_head.py` also puts that dir on `sys.path` but imports nothing from it.)

The grep that built it was not recorded. Reconstruction from the code: the hub logs
`multistream: prefill checkpoint saved tokens=N ...` and `multistream: prefill cancelled; partial
snapshot saved tokens=N ...` (`multistream.rs`, `tokens = tokens_saved.len()`, the length
`snapshot::save` writes to `tokens.bin`), the same two line kinds `ck.py` filters on:

    sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log* \
      | grep -a -E 'prefill checkpoint saved|partial snapshot saved' \
      | grep -o 'tokens=[0-9]*' | cut -d= -f2 | sort -nu > ckpt_counts.txt

The 2026-09-28..10-01 logs the original came from have rotated out of `~/logs`, so a file
regenerated today covers different snapshots and will not reproduce the design doc's numbers
exactly. Point the scripts' `CK = ...` line at the new file.
