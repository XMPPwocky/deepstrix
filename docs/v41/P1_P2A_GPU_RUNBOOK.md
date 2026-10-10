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

## 2. LIVE ARMS WITH ONE HUB RESTART (owner 10-10: "stay up as long as possible")
The four P1/P2a knobs are LIVE (commit after 9460a08e): `V41_LM_LAYER_OUTER`, `V41_LM_QUANTUM_LAYERS`, `V41_PREFILL_SPLIT`, `V41_PREFILL_B1_FILLER`.
Put them in the hub knob file `/dev/shm/deepstrix-knobs.txt` (`NAME=value` lines, re-read every second, every change is logged as `knob changed`,
the effective table is written to `<file>.effective`; write a temp file and `mv` it, an empty file is ignored). Layer-outer is decided per WINDOW
(next window onwards), the split per LONG JOB (next long job onwards). So after ONE restart onto the new binary (with NO new env vars) every
arm below is a file edit, no restart:
* **Arm 0 baseline:** knob file without the new keys = today's behaviour (group-major, decode partition).
* **Gate B (live, box 2 attached) -- a STATISTICAL gate, not bit identity (review r8).** On the live hub the expert->box partition
  is the box-1 hot set, which `multistream` refreshes from live decode picks every `V41_B1_HOT_REFRESH` (500, a STATIC knob) ticks and
  swaps ids under interleave; box 2's same-layer merge can also coalesce differently run to run. So two dumps of the same prompt taken
  minutes apart legitimately differ in f32 summation order whatever the prefill driver: a byte-compare would fail a correct layer-outer.
  Byte identity is Gate A's job (`lm_prefill_bitexact`, fixed partition). Gate B instead: with `V41_PREFILL_LOGITS_DUMP=<path>` set in the
  single restart's env, run the SAME >= 64K-token cold prompt THREE times back to back, as little other traffic in between as you can
  arrange: arm 0 (file without the key), arm 0 again (the NULL), then `V41_LM_LAYER_OUTER=1`. Compare per-prompt last-token logits as
  KLD and argmax: PASS = KLD(layer-outer, arm 0) within the null's KLD(arm 0, arm 0') spread and the same argmax (the null must match
  the marginal: if the null itself shows a large KLD, the comparison has no power; rerun quieter). Log lines to see:
  `prefill_lm_window open ... layer_outer=true`, `... done ... layer_outer = true`, and the one-time
  `V41_LM_PREFETCH is a no-op for layer-outer windows`. Watch decode p90 of other streams during it (a pass is atomic, ~1-2 s).
* **P1 arm:** add `V41_PREFILL_SPLIT=1` (and optionally `V41_PREFILL_B1_FILLER=n`) and run a long cold prompt; the job-start line
  `prefill split: frozen encoder-layer expert split for a long job` prints warm, lru_slots, owned_now, filler, head_min/max, share1, share2,
  share2_per_layer, usable2, expected_rereads_per_window. Without `V41_PAGER_NO_DENSE=1` (STATIC, a startup env: add it only at a restart you
  would do anyway) the filler is 46/22 per layer and box 2's share overflows by 130-400 experts: expect most, not all, of the gain.
* **Combine:** `V41_LM_LAYER_OUTER=1` + `V41_PREFILL_SPLIT=1` together, same prompt.
Revert any arm by deleting its line. If a layer-outer window misbehaves, delete `V41_LM_LAYER_OUTER` and the NEXT window runs group-major.
Gate: golden/KLD for the partition change (f32 association only) before leaving the split on.

## 3. (merged into section 2)

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

## 5. Order of operations (minimising downtime)
0. (hub UP, niced/pinned, memory-watched) pre-build the release test binary and server+expertd into `target-gate/` (see the build log).
1. Server DOWN, ~17-20 min: Gate A with `LM_CASES=lazy,restored` (the two cases that exercise the window prediction, multi-window, the saved-indexer
   flag in both states, foreign interleave, and the device seed at pos0 > 0; `fresh`/`multi` are covered by `lazy`'s eager layer-major run). The full four
   cases are ~31 min (measured 10-01); run them at a quieter time if you want the complete gate.
2. ONE hub restart onto the new binary (3-5 min, box 2 stays warm), then everything in section 2 live, no more restarts. Revert `miss_par` on box 2
   (live, ~1 s) before the P1 arm unless you intentionally keep it as the shared baseline.

## 6. RESULTS 2026-10-10 (window 20:42-20:56 UTC, live arms 20:57-21:18 UTC)

* **Gate A PASSED** (14.5 min down incl. restart): `lazy` (2 windows, lazy + foreign interleave) and `restored` (1 window,
  pos0 1500) BIT-IDENTICAL to chunked prefill (state + logits + 3 decode steps, null clean). Layer-outer verified by pass
  counts: 13 / 6 units (group-major would be 11 / 5: 5 groups vs 6 quanta per window). Hub on `fc67a363` (b8323821).
* **Live arms** (one 82,056-token cold prompt per arm; `arms_p1p2a.py`, `arm_windows.py`; window = 4096 rows):

  | arm | wall s | median window s | rows/s | decode stream live |
  |---|---|---|---|---|
  | a0 (group-major) | 161.7 | 6.18 | 663 | no |
  | a0n (null) | 262.0 | 9.49 | 432 | yes |
  | lo (layer-outer) | 139.7 | 4.95 | 828 | yes |
  | split (P1) | 216.3 | 7.49 | 547 | yes |
  | split_lo (P1 + layer-outer) | 119.4 | **3.98** | **1029** | yes |
  | a0e (group-major) | 253.4 | 9.33 | 439 | yes |

  Group-major windows run ~50% slower whenever another stream is decoding (a0 alone 6.2 s vs 9.3-9.5 s); layer-outer
  arms held 4.0-4.9 s under the same load. Wall times also carry 20-30 s queueing behind a decode burst and a 30 s burst
  mid-prefill (scheduler 120 s prefill / 30 s decode). ONE run per arm: repeat before defaulting on.
  P1 split at job start: lru_slots 5041, owned_now 4485, filler 27 (capped by free LRU; `V41_PAGER_NO_DENSE` would raise
  it), head 107-115, share2 4894 vs usable2 4570, expected re-reads 324/window (vs ~1146 measured group-major).
* **Gate B has NO POWER on the live hub.** The arm-0 null itself: KLD(a0 || a0n) 0.275 nats, a0e 0.360, argmax flips;
  lo 0.306 (inside the null), split_lo 0.532 (above it; one sample). Prefill is not run-to-run reproducible on the live
  hub by far more than f32 re-association: the leading suspect is box 2's f16 PREFILL replies (`V41_PREFILL_F16_REPLIES`,
  default on since 10-01, never gated) combined with the live partition moving between runs (it decides which picks get
  f16-rounded). No inf/NaN logged. Substitution is decode-only. Test: a restart with `V41_PREFILL_F16_REPLIES=0`, then the
  same null pair. Gate A stays the fidelity proof for layer-outer.
* Harness bug fixed mid-run: same-prompt snapshots share hashes, so the per-arm trash dir name must be unique (arm suffix).

## 7. Restart 21:32-21:36 UTC + repeat arms + PRODUCTION ON (owner OK, 2026-10-10)

* Hub `f9fa7f87` (main 316182e3: f16-WMMA arm wired, default off) with `V41_PAGER_NO_DENSE=1` (pager: `prefill dense = 0`,
  LRU = all 5425 slots) and `V41_PREFILL_F16_REPLIES=0` (box-2 prefill partials f32). 3 min 17 s down: a `# comment` line
  appended to the env file was exec'd by `env -i` and the supervisor died at once (fixed: `launch_hub.sh`).
* **Repeat arms** (same 82K prompt; `--out ~/gateb_20261010b`):

  | arm | median window s | rows/s |
  |---|---|---|
  | a0_2 / a0n_2 / a0_3 (group-major) | 6.70 / 9.94 / 8.21 | 611 / 412 / 499 |
  | split_lo_2 / split_lo_3 | 3.83 / 4.03 | 1069 / 1015 |
  | split_lo_f16 (+ `V41_MOE_WMMA_F16=1`, hub only) | 3.49 | 1172 (one sample, lighter load) |

* **Fidelity, live:** the `split` (group-major) and `split_lo` (layer-outer) records of the first session are
  BIT-IDENTICAL (dump records 6 and 7, ~3 min apart, live traffic): with the same frozen partition, layer-outer reproduces
  group-major exactly on the live hub. The ~0.3-nat run-to-run spread is NOT the f16 replies (with f32 replies the null
  is still 0.29 / 0.47 nats): any partition change (hot-set refresh between jobs; the split is built from the live hot set
  at job start) perturbs f32 association, and at 82K tokens a near-tie in the router top-6 / indexer top-k flips and
  compounds. A live KLD gate at long context has no power below ~0.3 nats; bit-identity under a fixed partition is the gate.
  `scratch/prefill_trace/dump_find.py` recovers a prompt's records from a busy dump by top-5 signature.
* **PRODUCTION 22:0x UTC:** `V41_LM_LAYER_OUTER=1`, `V41_PREFILL_SPLIT=1` in `/dev/shm/deepstrix-knobs.txt` AND the hub env
  file (code defaults stay off). Revert: delete the two lines from the knob file (next window / next long job) and the
  env file (`.pre-lo-split-20261010` backup). `V41_PREFILL_LOGITS_DUMP` is on this launch only (~0.5 MB per prefill).
