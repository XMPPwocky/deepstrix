# P1 / P2a GPU runbook (written 2026-10-10 while the GPU was busy; nothing here has been run)

Branch `worktree-igpu-wmma-moe`. Design: `LM_LAYER_OUTER_PREFILL_DESIGN.md`. Everything is default-OFF; rollback of any step = unset its knob
and restart (P1/P2a knobs are read at startup / window open; box 2 needs no change until P3).

## 0. Before you start
* `miss_par` on box 2 has been `4` since 2026-10-10 18:21 UTC (backup `~/expertd-knobs.txt.pre-misspar4-20261010`, revert = `miss_par=1`,
  live, ~1 s). It is UNMEASURED. For a clean P1 A/B revert it first, or run the A/B with it as the shared baseline and say so.
* Build the worktree binaries (own target dir; per `project_v41_server_restart_procedure.md`):
  `cd <worktree> && CARGO_TARGET_DIR=target-v41 nix develop -c cargo build --release -p deepstrix-server --features v41`
  (box 2 needs a new `deepstrix-expertd` only for the P3 landing-rank commit 455c40aa; P1/P2a are hub-only).
* Two-box restart order if both restart: box 1 down, box 2 down, box 2 up, box 1 up (`feedback_two_box_restart_order.md`); a hub restart alone
  leaves box 2 warm.

## 1. GATE A -- P2a bit-identity (server DOWN, one weight load; box 2 not needed by the test)
```
V41_LM_PREFILL=1 V41_LM_LAYER_OUTER=1 cargo test --release -p v4flash-kernels --features v41 \
    --test lm_prefill_bitexact -- --ignored --nocapture
```
Cases `fresh,multi,restored,lazy` (env `LM_CASES` selects; `LM_ROWS` = window rows, default 4096). PASS = every case bit-identical to chunked prefill
(KV bytes + logits), the chunked-vs-chunked null clean, and `lm_windows_run() > 0` in the log. Also run once WITHOUT `V41_LM_LAYER_OUTER` to confirm
group-major still passes (regression guard for the shared edits). Split and NO_DENSE stay OFF (the test has no box-2 client; invariant I1 is for a fixed partition).
If it fails ONLY in `restored` (device seed at pos0 > 0) or `lazy` (window prediction + whole-window inputs) those are the two paths no CPU test covers.

## 2. GATE B -- P2a live window with box 2 attached (per-task remote ticket ordering is untested by Gate A)
Hub env (append to the hub env file; backup it): `V41_LM_LAYER_OUTER=1` (`V41_LM_PREFILL=1` is already production). Restart the hub only.
Run ONE long cold prompt (>= 64K tokens) twice, layer-outer vs group-major, and compare the last-token logits with
`V41_PREFILL_LOGITS_DUMP=<path>` (appends raw f32 per prompt; byte-compare). Log lines to see: `prefill_lm_window open ... layer_outer=true`,
`prefill_lm_window done ... layer_outer = true`, and the one-time `V41_LM_PREFETCH is a no-op for layer-outer windows` warning.
Watch decode p90 of other streams while it runs (a pass is atomic, ~1-2 s).

## 3. A/B -- P1 frozen split (+ reclaim)
Hub env: `V41_PREFILL_SPLIT=1`, `V41_PAGER_NO_DENSE=1` (needs `V41_PREFILL_UNIFIED_POOL=1`, already production), optional `V41_PREFILL_B1_FILLER=64`
(the effective filler is `min(knob, live free LRU slots / 20)`). Restart the hub.
The job-start log line `prefill split: frozen encoder-layer expert split for a long job` prints warm, lru_slots, owned_now, filler, head_min/max,
share1, share2, share2_per_layer, usable2, expected_rereads_per_window. Expect (design sec 4): with the reclaim, filler 65/41 (top/interleave),
share2 216/228 per layer = 4,320/4,560 vs ~4,570 usable. PASS criteria: steady 4096-row window wall (baseline 8.02 s = 511 rows/s), box-2 reads
per window (baseline ~1,146) via the trace scripts below, `n_missing` distribution, hub `ensure` stall in window 1, and decode p90 over the
60 s after a long job vs the same traffic before.
Gate: golden/KLD for the partition change (f32 association only) before leaving it on.

## 4. Trace analysis (scratch/prefill_trace/, stdlib python; the 2026-10-10 pipeline)
Files: hub `~/logs/evtrace/hub-*.evt` (box 1), box 2 `~/logs/evtrace/b2-*.evt` (run the box-2 filters ON box 2, nice/ionice; do not copy GB files).
1. `hdr.py FILE` (header + kinds), `rng.py FILE DATE HH:MM:SS HH:MM:SS` (raw-clock range of a wall-clock span).
2. `evtfilt.py FILE LO:HI KIND[,KIND] > extract.txt` (stream filter with byte bisection; kinds hub_req,hub_step,hub_phase,b2_req,b2_read,b2_ensure,b2_write,sys).
3. `mkwin.py SERVERLOG... > windows.txt` (windows per job from `prefill_lm_window` log lines; marks windows overlapping a decode burst).
4. `split3.py HUBEXTRACT LOG WINFILE OUT.json` (per-window wall / exposed wait / hub busy / gaps / misses), `agg3.py OUT.json` (class means:
   steady <=100 misses, mid, heavy >=800, decode-interleaved).
5. `join3.py`, `cond.py`, `runvar.py`, `depth3.py`, `hubwork.py`, `wire2.py`, `rereads2.py`, `share.py`, `b2split.py` as documented in each docstring.
CAVEAT: these assume GROUP-MAJOR windows (`gid(h)` layer groups, "G0 prep gaps"). Under layer-outer there are no groups: the hub_req layer/lane
fields still work for per-request decomposition, `split3.py`/`agg3.py` window walls and misses still work, but the G0/boundary gap attribution does
not apply (the gap categories need re-deriving from pass boundaries: look at `prefill_job_progress` / `prefill_lm_window` timestamps).
`V41_EVTRACE_DEV` (short prefill-only window, Tier B regressed decode 10-02) is needed for the dGPU-chain vs iGPU-MoE split inside the hub's time.

## 5. Order of operations for one GPU window (suggested)
Gate A (group-major, then layer-outer) -> restart hub with P2a on -> Gate B -> restart with P1 on (and P2a) -> A/B on a long prompt -> if good, leave
P1 on and run the golden/KLD. Revert `miss_par` before step 3 unless intentionally shared.
