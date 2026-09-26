# B_moe_prefill — routed MoE at prefill batch (iGPU gfx1151), 2026-09-26

Owner: kernel-ideas sweep, family B. Production kernels (commit 361d4f9):

| kernel | file | production launch (B rows, n_wi work items) |
|---|---|---|
| `mxfp4_pair_matvec_fused_swiglu_kwide` (gate+up) | K/mxfp4_pair_matvec.hip:395 | grid (2304/8=288, n_wi_bound) x 256; chunk 32; n_blocks 20; clamp 10; n_wi_dev set (V41_MOE_WI_DEVCOUNT) |
| `mxfp4_matvec_par_by_expert_kwide2` (down) | K/mxfp4_matvec.hip:219 | grid (5120/16=320, n_wi_bound) x 256; xq_slot_stride 9*292=2628; n_blocks_in 9 |
| `q8_k_quantize` | K/q8_k_quantize.hip:23 | (20B) x 256 pre, (54B) x 256 mid |
| `moe_group_builder_hetsplit` | K/moe_group_builder.hip:73 | (ceil(6B/512)) x 512 |
| `moe_work_items_builder` | K/moe_work_items_builder.hip:26 | (ceil(384/256)=2) x 256 |
| `q2_k_reduce_partials_hetsplit` | K/q2_k_accumulate_matvec_par.hip:737 | (ceil(5120B/256)) x 256 |

n_wi_bound = moe_wi_upper_bound(6B, 384, 32, cap) = min(6B, min(6B,384) + ceil(6B/32)):
B=512 -> 384+96 = 480; B=1024 -> 384+192 = 576 (dispatch.rs:436-440). Work-groups past
the device count exit at once.

## Harness regime (what differs from production, and why it does not matter)

* Physical expert copies: N_EXP (default 96) x 18.8 MB = 1.8 GB (iGPU cap 3 GB). Routing is
  drawn EXACTLY like production: every row picks 6 DISTINCT experts uniformly over the 384
  virtual experts; picks landing on virtual experts >= N_EXP become the -1 sentinel (the
  production builders skip e < 0). So each active expert sees the production member-count
  distribution (Binomial(6B, 1/384): mean 8 at B=512, 16 at B=1024, 32 at B=2048), the
  activation rows touched are spread over the full [B x 5840 B] / [B x 6 x 2628 B] buffers,
  and per launch 96 experts x 18.8 MB = 1.8 GB stream through a 32 MB MALL: the weights are
  COLD on every call without any rotation. Times scale by 384/96 = 4 to a full layer.
* Grid.y is the production upper bound (576 at B=1024) with a device count, so the empty
  work-group tail is included (I also time the exact-grid variant to price it).
* Q8_K activations are produced by the production `q8_k_quantize` from random f32 in
  [-1, 1]; the down kernel consumes the gate/up kernel's real output (production chain).
* Weights: random nibbles; E8M0 scales uniform in [116, 128] (2^-12 .. 2^0). Real expert
  scales sit around 2^-10 .. 2^-4.

## Log

### 2026-09-26 03:5x UTC (first attempt, before the box hang) — what is on disk
* `results/base_B1024_run1.txt`: B=1024, 64 experts, hub LIVE (iGPU shared with decode):
  kwide 9086 us = 90 GB/s, kwide2 9584 us = 44 GB/s (x6 = 54.5 / 57.5 ms per full layer —
  much slower than the 39.1 / 31.8 ms in PREFILL_100K_PROFILE, so that run is contaminated by
  production decode traffic; re-measured below with the hub down).
* `results/grid_B1024_run1.txt`: exact grid vs the production upper-bound grid: -5.7% (kwide),
  -7.8% (kwide2). The 512 empty tail WGs (576 bound vs 64 used at n_exp=64) cost ~0.5 ms; at
  n_exp=384 the tail is ~192 of 576 WGs, so the production cost of the bound is smaller.
* PMC (prof_pmc1-3_kwide/, one 64-expert launch at B=1024, hub live):
  kwide : SQ_WAVES 1.357e6 (= 288*576*8 = 1.327e6 + q8k etc., ok), SQ_INSTS_VALU 8.07e8,
          SQ_INSTS_SALU 3.36e8, SQ_INSTS_LDS 1.26e8, SQ_BUSY_CYCLES 4.10e8, SQ_WAVE_CYCLES 1.74e10,
          MemUnitBusy 75.6%.  24.5 GMAC / (32 lanes*4) = 1.9e8 sudot4 wave-instrs -> only 24% of
          the VALU instructions are the dot; 4.2 VALU per sudot4.
  kwide2: SQ_INSTS_VALU 5.69e8, SQ_INSTS_SALU 3.93e8 (!), SQ_INSTS_LDS 1.09e8, MemUnitBusy 88%.
  GRBM_GUI_ACTIVE reads a constant 24789 for every kernel on the iGPU = broken; SQ_INST_CYCLES_*,
  SQ_WAIT_INST_ANY, VALUBusy, MemUnitStalled do not exist on gfx1151 (pmc3/pmc4 logs).
  VALU utilisation estimate: 8.07e8 / 40 CU = 2.0e7 wave-instr per CU; at the documented
  6.5 ms per 64 experts and 2.9 GHz that is 2.0e7 / (2 SIMD * 1.9e7 cyc) = 53% of one
  VALU issue per SIMD-cycle. Not saturated, but not idle either (and the iGPU clock under load
  is unknown: pp_dpm_sclk shows 600/1100/2900 levels).
* ISA (isa.sh): kwide 119 VGPR / 70 SGPR / 0 scratch -> 12 waves/SIMD; kwide2 173 VGPR -> 8 waves/SIMD.
  LDS kwide: s_q8v 16 KiB + s_yd 256 B + 128 B -> 3 WGs per 64 KiB (24 waves/CU).
* ATT (prof_att_kwide*/ at 19:00 UTC, B=1024, 32 experts; results/att_*_top.txt):
  kwide : stall/latency 55.5%; top stall = `s_waitcnt vmcnt(3)` 29% of all stall (the weight
          loads at the top of each superblock-pair iteration), then lgkmcnt waits (LDS LUT +
          staging) ~15% spread over many sites, then the dot phase itself
          (v_movrels/v_movreld = the runtime-indexed lane_pacc[mi] arrays, v_dot4, v_mul).
  kwide2: stall/latency 65.3%; vmcnt(3)+vmcnt(3)+vmcnt(5) = 50% of all stall = the per-member
          global q8 loads (activations re-read per row pair from L2), lgkmcnt ~12%.
  Reading: both kernels alternate "issue loads -> wait -> compute" with no software pipelining;
  the memory system is idle during the dot phase and the VALU is idle during the wait. Neither
  the VALU nor DRAM is saturated; it is a latency/overlap problem plus a fat VALU inner loop
  (30 MACs per VALU instruction in kwide).

### 2026-09-26 19:1x UTC — resumed (hub DOWN, both GPUs idle)
Plan: (1) clean baseline B=128/512/1024 under gpu_submit; (2) ablation variants of kwide
(no-dot / no-weight-load / no-stage) to price VALU vs VMEM vs LDS; (3) int8-WMMA candidates for
gate+up and down (see IDEAS below); (4) cheap knobs (prefetch next superblock's weights).

### Clean baseline (hub down), 96 expert copies = 1.2 GB gate+up + 0.6 GB down per launch, run2
`results/base_B{128,512,1024}_run2.txt` (bash run_base.sh 2 under gpu_submit --dev igpu --mb 2100):

| B | members/expert | kwide gate+up | GB/s | roof@230 | kwide2 down | GB/s | roof@230 | q8k pre | q8k mid | reduce |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 1.9 | 5526 us | 184 | 4.43 ms (80%) | 3096 us | 165 | 2.23 ms (72%) | 32 | 57 | 77 |
| 512 | 7.9 | 8139 us | 149 | 5.28 ms (65%) | 5619 us | 110 | 2.69 ms (48%) | 122 | 254 | 404 |
| 1024 | 16.2 | 11838 us | 104 | 5.33 ms (45%) | 9591 us | 66 | 2.77 ms (29%) | 250 | 530 | 821 |

Time grows ~linearly with members/expert at constant weight bytes: (11.84-5.53)/(16.2-1.9) =
0.44 ms per unit of mean members for gate+up -> at B=1024 roughly 6.5 of 11.8 ms is per-member
compute, i.e. the production shape is issue-bound, not DRAM-bound. Scaled x4 to 384 experts:
47 / 38 ms per layer (documented bench: 39.1 / 31.8 — that bench probably ran at a pinned clock;
DPM auto here, cannot write sysfs; relative A/Bs are what count).

### kwide ablations (results/ablate_B{1024,512}_run1.txt; timing-only kernels, wrong results)
B=1024: full copy 1.002x (sanity ok); nodot (7 of 8 sudot4 removed) 0.823x; noload (NO weight
VMEM at all, bytes synthesised) 0.871x; nolut (LUT -> shift) 0.958x. B=512: 0.898 / 0.741 / 0.952.
=> removing ALL weight traffic saves only 13% at B=1024: the kernel is not DRAM-bound there; the
per-member dot loop (VALU + LDS + movrel indexing) and the staging are the time. The first
`nostage` number (0.256x) was an artefact: LLVM folded loads of never-written LDS to undef and
DCE'd the weight path (391 GB/s apparent); fixed by staging once (run2 pending).

### CANDIDATE B_moe_prefill/wmma_i8 — cand_wmma.hip (FIRST RESULT, run1)
int8 WMMA (v_wmma_i32_16x16x16_iu8) with A = Q8_K activations (16 members x 16 k from LDS),
B = MXFP4 weights unpacked per lane with v_perm to w' = 12 + kvalue (unsigned), accumulator
pre-loaded with -12 * bsum32 (from the Q8_K bsums the production quantizer writes) so the int32
result is the exact production integer dot; per block acc += (2^(e-128) * yd) * (float)C.
128 rows per WG (8 waves x 16 rows), grid (n_rows/128, n_work_items), <= 32 members = 1-2 M tiles.
ISA: gate+up 230 VGPR / 10.1 KB LDS / 0 spills (6 waves/SIMD); down 133 VGPR (11 waves/SIMD).
Correctness (results/wmma_cmp_B*_run1.txt, B = 1024/512/128/37, bound grid AND exact grid):
  gate+up vs kwide  : rel_rmse 5.4e-7, max_abs 3.8e-4, 11% of elements differ in bits
  down vs kwide2    : rel_rmse 1.7e-7, max_abs 6.8e-3 (values ~1e2), 21% differ in bits
  = f32 reassociation only (identical integer sums). NOT bit-exact -> fidelity gate before merge.
Timing (results/wmma_ab_B{1024,512}_run1.txt, interleaved, 20 rounds x 2):
  B=1024 gate+up 12035 -> 7331 us (0.609x, 167 GB/s);  down 9640 -> 3374 us (0.350x, 189 GB/s)
  B=512  gate+up  8400 -> 6668 us (0.794x, 182 GB/s);  down 5823 -> 3005 us (0.516x, 206 GB/s)
Down is at 82-90% of the 230 GB/s roof. Gate+up at 73-79%: headroom ~1.3x left (230 VGPRs ->
6 waves/SIMD; the whole 136-B super-block of both matrices is held in 68 VGPRs during compute).
Run 2 (separate process, results/wmma_ab_B*_run2.txt): B=1024 gate+up 0.602x, down 0.356x; B=512 0.810x /
0.524x; B=128 0.976x / 0.764x. Chunked members (B=2048, >32 members per expert -> 2 work items,
results/wmma_cmp_B2048_run2.txt): rel_rmse 5.4e-7 / 1.7e-7, same as the other shapes. Logged to WINS.log.

### PMC of the WMMA kernels vs production (results/pmc_wmma_summary.txt, B=1024, 96 experts, 1 launch)
| kernel | SQ_INSTS_VALU | SALU | LDS | waves | MemUnitBusy | time |
|---|---|---|---|---|---|---|
| kwide (prod) | 1.17e9 | 4.9e8 | 1.78e8 | 1.33e6 | 73% | 12.1 ms |
| gu_wmma (cand_wmma) | 5.76e8 (-51%) | 7.5e6 (-98%) | 1.6e7 (-91%) | 82944 | 47% | 7.3 ms |
| kwide2 (prod) | 7.46e8 | 5.65e8 | 1.35e8 | 1.47e6 | 89% | 9.7 ms |
| down_wmma (cand_wmma) | 2.90e8 (-61%) | 7.7e6 | 1.6e7 | 184320 | 49% | 3.4 ms |
gu_wmma VALU per wave per super-block = 2084 = 8 blocks x 2 mats x (84 unpack + 4 scale + ~1.5
tiles x 24 cvt/mul/fma) — the v_perm nibble unpack is ~65% of its VALU; VALU busy only ~34% of
SIMD-cycles, WMMA pipe ~12%, DRAM 73%, MemUnitBusy 47%: nothing saturated -> latency/overlap
(6 waves/SIMD at 230 VGPRs). down_wmma: 36% VALU, 185-209 GB/s -> at the practical roof.

### cand_wmma2 (pipelined loads, gate/up alternating units) = DEAD END for gate+up
gate+up compiled to 256 VGPR + 534 spills (1572 B scratch): 2.1x SLOWER than kwide
(results/cand_wmma2_ab_*_run1.txt). __builtin_amdgcn_sched_barrier(0) around the prefetch made it
worse (886 spills, 3.06x slower, run2); unconditional (clamped) loads no help (try_compile.sh).
The two rotating 34-VGPR buffers + 32 acc + 16 yd + unpack temporaries do not fit when the two
matrices are computed in separate passes. Down variant (2 super-block buffers, 157 VGPR): neutral
(0.361x vs 0.350-0.356x for the non-pipelined) -> the down kernel is not latency-bound at 11 waves/SIMD.

### Probe: RDNA3 wave32 WMMA reads the B fragment from BOTH 16-lane halves
cand_wmma built with -DWM_PROBE_UPPER (lanes 16-31 hold garbage in B): results wrong (max_abs 99).
So the upper-half replication of the unpacked weights is REQUIRED; halving the unpack via
v_permlanex16 would need a swap + v_cndmask per dword (16 ops per block vs 42 saved) — untested.

### cand_wmma3: split-matrix waves (waves 0-3 gate, 4-7 up, 64 rows/WG) + 2-buffer pipelined loads
159 VGPR (9 waves/SIMD), 0 spills, LDS 10.1 KB. Up waves park acc in LDS for the gate waves' SwiGLU.
Submitted: run_cand.sh cand_wmma3 1 all 64 1024 512 128.

### cand_wmma3 (split-matrix waves, 64 rows/WG, 159 VGPR, pipelined): correct, NOT faster
results/cand_wmma3_ab_B*_run1.txt: B=1024 0.634x (7776 us) vs cand_wmma 0.602-0.609x (7269-7331);
B=512 0.804x (=); B=128 0.991x. Same knob without the pipeline (tmp_w3np, 134 VGPR): 0.622x.
=> 9-11 waves/SIMD and one-super-block prefetch buy nothing; the split itself costs ~5%
(2x the A-fragment/C-init LDS reads, 2x staging per row).

### Ablations of cand_wmma gate+up (B=1024 / B=512, results/tmp_wNC_*, tmp_wNL_*)
| variant | B=1024 | B=512 |
|---|---|---|
| full cand_wmma | 7.27-7.33 ms | 6.67-6.71 ms |
| loads+staging only, no unpack/WMMA (WM_ABL_NOCOMPUTE) | 6.80 ms = 180 GB/s | 6.78 ms |
| compute only, weights synthesised (WM_ABL_NOLOAD) | 5.81 ms | 4.50 ms |
| down: loads-only / compute-only | 3.08 (207 GB/s) / 2.94 | 2.97 / 2.27 |
The gate+up kernel is within 7% of its own loads-only time: overlap is fine, the ACCESS PATTERN
tops out at ~180 GB/s (two 136-B streams per row, 256 per WG) while the one-matrix down pattern
reaches 207 GB/s. Compute-only (5.8 ms) is close behind: VALU 2.5 ms of issue + WMMA ~0.9 ms +
LDS/barrier/dependency latency at 6 waves/SIMD.
Knobs on the access pattern:
* __builtin_nontemporal_load on the weight loads (tmp_wNT): gate+up 10.5 ms (0.87x vs kwide only),
  down 5.0 ms — MUCH WORSE (dword-split loads + no cache-line reuse). DEAD.
* lanes 0-15 / 16-31 loading blocks 0-3 / 4-7 of the same row (no duplicate addresses, loads-only,
  tmp_wNCS): 8.35 ms = 147 GB/s, WORSE than the duplicated pattern (6.8 ms). Down: unchanged
  (3.10 ms). So fewer, wider-spread load instructions hurt; the duplicated 16-B-per-lane pattern
  keeps more requests in flight. This also kills cand_wmma4 (half-row loads + v_permlanex16
  exchange to halve the unpack): it spilled anyway (256 VGPR, 793 / 215 spills with / without the
  2-buffer prefetch) — recorded as untested/blocked.

### cand_wmma5: matrix-sequential K loops (gate all K, then up all K) — one stream per row
134 VGPR (11 waves/SIMD) plain; 173 VGPR (8 waves) with -DWM5_PIPE (2-buffer prefetch). Submitted.

### cand_wmma5 (matrix-sequential K loops) results (results/tmp_w5_*, tmp_w5p_*)
plain (134 VGPR): B=1024 0.593x (7192 us), B=512 0.780x (6581) — 1-3% better than cand_wmma
(0.602-0.609 / 0.794-0.810), within the ~5% iGPU noise band; runs 2-3 queued. Pipelined
(-DWM5_PIPE, 173 VGPR): 0.616x / 0.767x — mixed, no gain. So one stream per row does not
change the picture much either: the loads-only floor of this access pattern is ~180 GB/s.

## IDEAS (ranked by expected gain x confidence / effort at the time of writing; status after)
1. int8 WMMA arm (A = Q8_K activations, B = MXFP4 -> uint8 via v_perm, +12 offset, bsums fold)
   -> cand_wmma.hip. MEASURED WIN: gate+up 1.66x, down 2.81x at B=1024 (see above).
2. Software-pipelined weight loads (next super-block issued before compute) -> cand_wmma2/3/5p.
   MEASURED NEUTRAL/LOSS (spills, or no gain at 8-11 waves/SIMD): the kernels are not latency-bound.
3. Matrix-sequential K loops (one weight stream per row) -> cand_wmma5. MEASURED ~+2% (noise band).
4. Halve the unpack VALU via v_permlanex16 exchange of half-row loads -> cand_wmma4. BLOCKED:
   spills (256 VGPR) AND the loads-only ablation shows the split access pattern is 1.2x slower.
5. Non-temporal weight loads (knob). MEASURED LOSS (1.4x slower).
6. Fused down + reduce (atomicAdd f32 into out[b][row] instead of [B x 6 x 5120] partials +
   q2_k_reduce_partials): saves the 0.8 ms reduce + 126 MB partials write/read per 1024 rows
   (~7% of the new down time), but float atomics make the sum order non-deterministic -> the
   project's determinism rule (score expert-cache changes with determinism) says no; a
   deterministic variant needs a fixed slot order = the reduce kernel we have. UNTESTED.
7. Interleaved gate/up weight layout (row r: gate sb, up sb adjacent -> one 272-B stream per
   row) to lift the access-pattern floor from 180 towards the 207-230 GB/s the one-stream
   kernels reach. Pager/pool layout change (mxfp4_repack + expert pool + snapshots): LARGE
   integration, expected <= +15% on gate+up. UNTESTED.
8. Cheaper nibble unpack: 2-perm 16-entry LUT + select, or f16 WMMA with the "1024+k magic"
   dequant (iq2_s style) — same op count within +-10% on paper; compute-only time (5.8 ms) is
   below the loads-only time (6.8 ms), so unpack VALU is not the wall anymore. UNTESTED.
9. Prefetch next super-block in the PRODUCTION kwide (cheap knob): noload ablation bounds the
   gain at <= 13%; superseded by the WMMA arm. UNTESTED.
10. Exact grid instead of the upper bound (drop V41_MOE_WI_DEVCOUNT tail): -5.7% / -7.8% on the
    production kernels at n_exp=64 (results/grid_B1024_run1.txt); the WMMA kernels have 6x fewer
    WGs per work item so their tail is proportionally smaller. Known, not a kernel change.

## FINAL SUMMARY (2026-09-26 20:3x UTC)
cand_wmma.hip, all separate-process runs (3 at B=1024/512, 2 at B=128), 96 expert copies,
interleaved vs the production kernels (ratio = cand/prod median, [min..max] over runs):
| B | gate+up prod | gate+up wmma | ratio | down prod | down wmma | ratio |
|---|---|---|---|---|---|---|
| 1024 | 12035 us | 7269 us (169 GB/s) | 0.609 [0.602..0.613] = 1.64x | 9640 us | 3374 us (189 GB/s) | 0.356 [0.350..0.359] = 2.81x |
| 512 | 8281 | 6673 (182 GB/s) | 0.810 [0.794..0.819] = 1.23x | 5807 | 3005 (206 GB/s) | 0.524 [0.516..0.535] = 1.91x |
| 128 | 5651 | 5614 (184 GB/s) | 0.993 [0.976..1.011] = 1.0x | 3232 | 2479 (209 GB/s) | 0.767 = 1.30x |
cand_wmma5 (matrix-sequential): 3 runs, B=1024 7192 [7109..7344] (+1%), B=512 6582 [6581..6682]
(+1.4%) vs cand_wmma — inside the noise band: NEUTRAL; keep cand_wmma as the reference candidate
(fewer moving parts), cand_wmma5 as the lower-VGPR alternative (134 vs 230) if occupancy matters
when co-scheduled with other kernels.
Correctness: identical integer dot products; f32 accumulation order differs (rel_rmse 5.4e-7
gate+up, 1.7e-7 down; max_abs 3.8e-4 / 6.8e-3 on O(1)/O(1e2) values) at B = 37/128/512/1024/2048,
bound and exact grids. NOT bit-exact -> fidelity gate (KLD vs golden CPU ref, pinned routing)
before any merge; mid goes through q8_k_quantize so a 1-ulp change can flip a Q8_K code.

Leverage (arithmetic): documented production per-layer times at B=1024 (all 384 experts):
gate+up 39.1 ms, down 31.8 ms -> x0.609 / x0.356 -> 23.8 + 11.3 = 35.1 ms vs 70.9 ms = 2.02x on
the MoE kernel chain (q8_k_quantize/builders/reduce add ~1.6 ms unchanged). Per token (40 layers /
1024 rows): 2.77 ms -> 1.37 ms of MoE kernel time = -1.40 ms/token of iGPU kernel time (0.60 gate+up
+ 0.80 down), split across box 1 and box 2 by expert residency. Prefill wall: box-2 MoE GPU time
is ~52% of the 100K wall (82.9 of 160 s) and box-1 iGPU ~27%; if box 2 stays the critical path
and its MoE kernels take 2.0x less: ~ -24% wall (gate+up alone ~ -11%, down alone ~ -15%; upper
bounds — read/write overhead, paging and the hub's own share do not shrink).
Decode (family A territory): box 2 runs kwide/kwide2 for every b>=2 request; at 1-2 members per
expert (B=128 here) the WMMA gate+up is neutral and the WMMA down 1.3x — worth a look there.

Caveats vs production: DPM-auto clocks (my x4-scaled baseline is 47/38 ms per layer vs the
documented 39.1/31.8), 96 expert copies scaled x4, random nibbles / scales in [2^-12, 2^0],
routing drawn like production (6 distinct of 384, Binomial member counts), bound grid + device
count exactly as dispatch.rs; the harness chain runs q8_k_quantize -> builders -> gate+up ->
q8_k_quantize -> down -> reduce on the same buffers production uses.

## REPRO
cd scratch/kernel_ideas/B_moe_prefill && bash build.sh          # baseline hsacos + candidates + harness
bash ../_infra/gpu_run.sh --dev igpu --mb 2100 --label B_moe_prefill/base -- bash run_base.sh 9
bash ../_infra/gpu_run.sh --dev igpu --mb 2100 --label B_moe_prefill/ablate -- bash run_ablate.sh 9
bash ../_infra/gpu_run.sh --dev igpu --mb 2600 --label B_moe_prefill/wmma -- bash run_cand.sh cand_wmma 9 all 128 1024 512 128
bash ../_infra/gpu_run.sh --dev igpu --mb 2600 --label B_moe_prefill/wmma5 -- bash run_cand.sh cand_wmma5 9 all 128 1024 512
bash ../_infra/gpu_run.sh --dev igpu --mb 2600 --label B_moe_prefill/wmma3 -- bash run_cand.sh cand_wmma3 9 all 64 1024 512
bash ../_infra/gpu_run.sh --dev igpu --mb 2300 --label B_moe_prefill/pmc -- bash run_pmc_wmma.sh
Knob/ablation builds: bash try_compile.sh cand_wmma.hip wNC_gfx1151 "-DWM_ABL_NOCOMPUTE" (then
run_cand.sh tmp_wNC ...); -DWM_ABL_NOLOAD, -DWM_NT_LOAD, -DWM_ABL_NOCOMPUTE -DWM_ABL_SPLITLOAD,
-DWM_PROBE_UPPER; cand_wmma3.hip -DWM3_NO_PIPE; cand_wmma5.hip -DWM5_PIPE.
