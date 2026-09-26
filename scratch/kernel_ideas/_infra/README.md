# Kernel-ideas sweep 2026-09-26 — shared infrastructure and rules

Worktree: `/home/claude-code/deepstrix/.claude/worktrees/kernel-ideas-2026-09-26`, branch
`worktree-kernel-ideas-2026-09-26`, based on **361d4f9 = the code the production hub runs**
(deployed 2026-09-26 02:18 UTC). Nothing here gets merged; winners are tracked in the
ledger for a later, deliberate integration.

## HARD RULES — box 1 is the LIVE production hub

1. **Never** kill / signal / restart `deepstrix-server` or `deepstrix-expertd`, never write
   sysfs (DPM clocks etc.), never ssh to box 2 (10.99.0.2), never `pkill` anything you did not
   start. Another job is A/B-testing iGPU clocks on production; `gpu_run.sh` waits for it.
2. **Every command that touches a GPU goes through the scheduler** (`_infra/gpu_submit.sh` /
   `gpu_wait.sh`, or `gpu_run.sh` = both) with an honest `--mb` (device memory you allocate).
   The scheduler refuses a job that would not leave the production hub its memory (dGPU: free
   VRAM − 270 MB; iGPU: host MemAvailable − 3 GB). While the hub is DOWN (as on 2026-09-26
   evening) that is ~15 GB / ~80 GB, but keep jobs ≤ 4 GB dGPU / ≤ 16 GB iGPU so a hub restart
   (which takes ~15.5 GB of VRAM and ~80 GB of host RAM) does not strand your harness: still
   rotate over a few copies of an expert rather than allocating 384 of them.
3. **No weight loads, no cargo.** Do not run `cargo build`/`cargo test`/the server/bench
   binaries of the workspace (each is a 50-100 s, 86 GB load + ~150 hipcc jobs; RAM cannot take
   it). Work with standalone harnesses on synthetic data at production shapes.
4. Write only under `scratch/kernel_ideas/<your-family>/`. Do not edit in-tree kernels or Rust.
   Express a proposed in-tree change as a candidate `.hip` in your dir (and optionally a
   `.patch`). No git commit/stash/checkout/reset — the orchestrator commits.
5. This session's worktree guard rejects inline shell with variables in `nix`/`find`/`sed`,
   `source` inside `bash -c`, and `git -C`. Put multi-step shell in a script file and run
   `bash path/to/script.sh`. Use `bash _infra/in_env.sh <cmd>` for one-off tool calls.

## Tools (all paths relative to `scratch/kernel_ideas/`)

| tool | use |
|---|---|
| `_infra/env.sh` | sourced by the others: hipcc on PATH, `$KFLAGS_V41` (= exact build.rs flags: `-O3 -DDEEPSTRIX_V41=1 -DMHC_N_EMBD=5120 -DMHC_HC_DIM=20480 -DROUTER_MAX_EXPERTS=512`), `$KERNELS_DIR`, `$ROCPROFV3`, `$ATT_DECODER_DIR` |
| `_infra/kcc.sh <hipcc args>` | hipcc behind a 2-slot compile semaphore (RAM). Baseline code object: `kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1151 $KERNELS_DIR/foo.hip -o base_gfx1151.hsaco` |
| `_infra/gpu_submit.sh --dev igpu\|dgpu --mb N --label <family>/<idea> [--timeout S] -- cmd` | **enqueue** a GPU job on the per-device scheduler; prints a ticket at once. Runs later from your cwd with the GPU as HIP device 0 |
| `_infra/gpu_wait.sh TICKET...` | block until done; prints the job's output; exits with its rc |
| `_infra/gpu_run.sh …` (same args as submit) | = submit + wait, for when you need the number now |
| `_infra/gpu_queue.sh` | who is running / waiting per device, device-seconds per family in the last 10 min |

**How the GPUs are shared (read this).** Two GPUs, seven engineers. A scheduler per device
(`gpu_sched.sh`, run by the orchestrator) executes ONE job at a time per device and picks the next
job from the family that has used the least device time recently (FIFO within a family), so nobody
can monopolise a GPU and nobody's timing is contaminated by a concurrent kernel. Consequences for
you: (1) **batch**: put all shapes/variants of a measurement into ONE harness process and ONE job
(a job costs a queue round-trip + ~0.3 s ROCm init; 20 one-shape jobs are 20x the overhead and
20x the queueing); (2) **submit and keep working**: `gpu_submit.sh` returns immediately — write the
next candidate, read ISA, update NOTES.md, then `gpu_wait.sh`; only use `gpu_run.sh` when the next
step really needs the number; (3) check `gpu_queue.sh` before a long job — if your family already
has jobs queued or the queue is deep, do CPU work first; (4) keep jobs short (`--timeout` <= 300 s,
hard cap 900): a >300 s job blocks every other family on that device; (5) never run a GPU binary
outside the scheduler (`ROCR_VISIBLE_DEVICES`, direct `./harness`, rocprofv3) — a bare run is
invisible to the scheduler and contaminates someone else's measurement. The scheduler also enforces:
memory guards, **NO rocprofv3 ATT on the iGPU** (refused, rc 66), and at most ONE rocprofv3
session machine-wide — both because the box hung during the first attempt of this sweep
(2026-09-26 04:01 UTC, 15 h of downtime) while an iGPU ATT and a dGPU ATT ran concurrently.
| `_infra/kbench.h` | harness helpers: `kb::Module` (load .hsaco), `kb::launch`, random fills, `kb::ab` (interleaved A/B, graph mode, flush hook, warm-up spin, med/p10/p90, GB/s, `KBJSON` lines), `kb::compare_f32` / `print_cmp` |
| `_infra/isa.sh file.hsaco gfx1151 [--dis substr]` | VGPR/SGPR/LDS/scratch/spills per kernel + disassembly |
| `_infra/in_env.sh cmd` | run llvm-objdump / llvm-readelf / hipcc etc. in the dev env |
| `_infra/example/` | complete template: build.sh (baseline hsaco from the UNMODIFIED in-tree source + candidate + harness), harness.cpp, run.sh |

Inside a harness the GPU is always **HIP device 0** (`gpu_run.sh` hides the other one).
Device facts: dGPU gfx1201 = RX 9070 XT, 64 CUs (HIP reports 32 WGPs), 640 GB/s DRAM,
8 MB L2 + 64 MB MALL, f16 WMMA ~194 TF peak / 35-50 TF realised, f32 VALU ~49 TF.
iGPU gfx1151 = Strix Halo 8060S, 40 CUs (HIP reports 20 WGPs), 256 GB/s theoretical,
**214-231 GB/s measured achievable**, 32 MB MALL, 64 KB LDS. Graph node / direct launch
2.2-2.8 us, each graph launch +6.7 us, event pair 4.6 us (dGPU).

## Measurement methodology (non-negotiable)

* **Baseline = the production code object**: the unmodified in-tree `.hip` compiled with
  `$KFLAGS_V41 --genco`, launched with the SAME grid/block/args/template the Rust dispatch
  uses (read the wrapper in `crates/v4flash-kernels/src/…`). A baseline that differs from
  production invalidates the comparison.
* **Production shapes and regime**: decode is multistream, 1-8 rows (B=1 and B=4 at least);
  prefill chunks are 512/1024 rows. State the cache regime and make it match production:
  weights streamed once per token are COLD (use `kb::Flusher` or rotate over several weight
  copies bigger than the MALL); small activations are warm. A warm-cache number for a
  cold-weight kernel is a bound, not a result.
* **Interleaved A/B** (`kb::ab`), then **>= 3 separate process runs** for any claimed win.
  This box throws 1.4-1.6x outliers and production load moves under you: a delta below ~5%
  (iGPU) / ~3% (dGPU) needs >= 5 runs with consistent medians AND p10s before you call it.
  Graph mode (`o.graph = true`) for latency-bound decode kernels (production decode stages
  are graph-captured).
* **Roofline first**: write bytes/flops per call and the ceiling-implied time next to every
  measurement. If the baseline is already at >= 90% of its roof, say so and move on.
* **Correctness**: every candidate is compared to the baseline's output on identical inputs
  (`kb::compare_f32`). Bit-exact is best; otherwise report max_abs / rel_rmse and whether the
  change re-associates a reduction, changes precision, etc. A non-bit-exact kernel needs the
  project's fidelity gate (KLD vs golden CPU reference with pinned routing) before it could
  ever merge — note it, do not run it. Test tails/odd shapes: a prefill-tuned kernel was once
  silently wrong below its tile size.
* Save raw outputs: `…/results/<idea>_<run>.txt`.

## Profiling (re-verified 2026-09-26)

* **ATT (per-instruction stalls) works on BOTH GPUs now** (the old iGPU hang is gone):
  `gpu_run.sh --dev igpu --mb N --label fam/idea -- bash _infra/prof.sh att igpu <kernel-regex> OUTDIR -- ./harness …`
  (`att dgpu` for the dGPU; prof.sh maps to the PHYSICAL `--att-gpu-index`, 0 = dGPU,
  1 = iGPU, which ignores gpu_run's device mask). Give the harness a short mode (a few
  calls; ATT is ~20x overhead). Never `--att-consecutive-kernels` (hangs). Rank stalls:
  `python3 ~/scripts/att_top.py OUTDIR/stats_ui_output_*.csv --by stall --top 25`.
  gfx1151 waits: `s_waitcnt vmcnt/lgkmcnt`; gfx1201: `s_wait_loadcnt/dscnt`.
* **PMC counters**: iGPU has live SQ_INSTS_VALU / SQ_INSTS_LDS / SQ_WAVES / GRBM_GUI_ACTIVE /
  LDSBankConflict / MemUnitBusy…; the dGPU only SQ_WAVES / GRBM_GUI_ACTIVE / SQ_BUSY_CYCLES
  (SQ_INSTS_* read 0). **Counters are device-global — the live server's kernels add to
  them**: check SQ_WAVES equals your dispatch's wave count and discard polluted dispatches.
  <= 4 counters per pass: `… -- bash _infra/prof.sh pmc "SQ_WAVES SQ_INSTS_VALU …" OUTDIR -- ./harness`.
* `bash _infra/prof.sh trace OUTDIR -- ./harness` (under gpu_run.sh) for plain per-dispatch durations.
