# iGPU prefill MoE: what is left after the int8-WMMA arm (plan, 2026-10-10)

Status (updated same day): the f16 kernel is BUILT and measured in the B_moe_prefill harness on the iGPU,
beside the live hub (owner: "slot beside live traffic"; ~2.7 GB, seconds per run, no profiler).
**Results are at the end of this file.** No production code changed; not wired into dispatch.

CORRECTION: the first compile table below ("3 tiles 233 VGPR, free") was measured on a generated file whose
3-tile dispatch was dead (it only existed under `WM_NTMAX >= 4`, so NTMAX=3 silently ran the 1-tile path).
The real numbers are in the results section.

## Where we are

The int8-WMMA MXFP4 arm already ships (`mxfp4_moe_wmma.hip`, `V41_MOE_WMMA_GATEUP` / `_DOWN`, default ON,
sweep ledger #27/#28: gate+up 1.56x at B=1024, down 2.7x). Measured iGPU matrix peak is ~55-59 TOPS
(`project_igpu_moe_wmma_f16`; dot4 and WMMA share it). The shipped arm realises about 12 TOPS-equivalent
at B=1024 (gate+up 7.3 ms / 96 experts, down 3.4 ms), i.e. ~20% of peak; the prefill roofline needs ~35.

## Why it stops at ~20% (from the kernel + the ledger's PMC table)

1. Weight-stationary over <= 32 members. `WM_MAX_CHUNK = 32`, one work item per (expert, 32-member chunk).
   An expert with 64 members is read from DRAM twice, with 1024 members 32 times. At 18.8 MB per
   chunk of 32 members the DRAM roof is 32 x 70.8 MF / 18.8 MB x 230 GB/s = **27.7 TOPS** even with
   perfect compute; layer-major windows (N/64 members per expert on average) make this the typical case.
2. Per-(block, tile) float epilogue. `acc += (sc*yd) * (float)C` is ~24 VALU per 2 WMMA (64 clk) per tile per
   block, linear in tiles, and WMMA occupies the VALU (nothing overlaps; ISA 7.9, measured in the f16 work).
   Plus the nibble unpack: 84 VALU per 32-weight block per matrix.
   Cycle model, fraction of the WMMA-pipe peak: 2 tiles 48%, 3 tiles 55%, 4 tiles 58%, infinite 73%.

## Compile-only data (gfx1151, this tree, `cand_wide.hip` = `mxfp4_moe_wmma.hip` with `-DWM_NTMAX=n`)

| tiles (members/work item) | gate+up VGPR / LDS / scratch | down VGPR |
|---|---|---|
| 2 (32, shipped) | 230 / 10.1 KB / 0 | 133 |
| 3 (48) | 233 / 15.2 KB / 0 | 128 |
| 4 (64) | 256 / 20.2 KB / 500 B (spills) | 231 |

So 48 members per work item is free on registers; 64 spills slightly on gate+up. Wider chunks alone buy the
DRAM side (27.7 -> 41 TOPS roof at 48 members) but not the VALU side (55% of peak).

## Proposal: f16 WMMA with the E8M0 scale folded into the dequant LUT (not built)

Replace the int8 dot + per-block float epilogue with an **f16 x f16 -> f32 WMMA chain over the whole 256-K
super-block**, so there is no per-block epilogue at all:

* Weights as f16 `kv * 2^(e8-128)` (kv = {0,1,2,3,4,6,8,12} x sign). Every e2m1 magnitude has f16 low byte 0, so
  the pattern is one hi byte: `(exp<<2 | mant?2:0)`. Scaling by 2^s is `+4*s` on that byte. Build the 8-byte
  perm LUT per (row, block) with **2 VALU** (add `4*(s+14)*0x01010101`, mask the zero entry), then per 4 weights:
  and, perm (LUT select), and+shift-or (sign), 2 spread perms = ~6 VALU. Block cost ~56 VALU vs 84+epilogue.
* Activations as exact f16 `q` (int8 values are exact in f16); `yd` is applied once per super-block per tile
  (`acc += yd * t`, 8 fma per 8 blocks). No `-12*bsum` offset trick (signed weights). Staging converts int8 -> f16
  once per WG (shared by 8 waves; ~2% of loop cost), LDS x2 (528 B/member row).
* Cycle model, per block per matrix: 64*NT clk WMMA + ~58 VALU -> NT=2 69%, NT=3 77%, NT=4 82% of the WMMA peak
  (vs 48/55/58% now): **1.4-1.7x in the compute-bound regime**, on top of the member-chunk widening.
* Numerics: products q*w are exact in f32 (<= 11 x 4 significant bits), the sum order differs from the int path
  only by f32 re-association inside WMMA/accumulation: the same class as the shipped arm (rel_rmse ~5e-7),
  so it goes through the same combined golden/KLD gate. Not bit-exact with kwide.

### De-risk evidence (all CPU, done today)

* **Scale range is narrow.** Read-only scan of the HF checkpoint (shards 05 and 10 = layers 2 and 7, first 1 MB of
  each of 40/60 expert scale tensors, 14.7M + 22.1M scales): e8 in 119..123, 99.99% in 120..121 (2^-8, 2^-7).
  f16 range needs e8-128 in [-14, 9]; huge margin. (`scan_scales.py` is in the job dir, not committed; rerun on
  more layers before the real build. Layers 20+ and the DSpark/MTP experts are unscanned.)
* **LUT math is exact.** `f16lut_emul.py`: emulates v_perm_b32, builds the LUT for every e8 in 100..139 (valid
  subset), all 16 nibbles x 4 lanes, compares with `kv*2^(e8-128)`: 1536 values, 0 mismatches, lane order correct.

## Build plan (each step gated; nothing GPU until a window is agreed)

1. (CPU/compile) kernel `mxfp4_moe_wmma_f16.hip`: gate+up (MATS=2) and down (MATS=1), NT in {2,3}, chunk 48,
   host work-item builder for chunk 48 behind a new knob (default OFF). Compile gate: no spills, VGPR <= 232.
2. (CPU) static ISA check: VALU/WMMA ratio per super-block vs the model above (`isa.sh`).
3. (GPU, needs a window) oracle vs CPU f32 reference, then `harness.cpp` A/B vs `mxfp4_moe_wmma` at B=512/1024/4096
   with the zipf routing; accept if >= 1.3x at B=1024 and the DRAM/compute balance moves as modelled.
4. (GPU, server down) golden/KLD gate, then e2e prefill A/B with layer-major windows.
5. Independently, and probably first in value: **MoE across sub-chunks** (LM phase 4). Layer-major today still calls
   the MoE per 512-row sub-chunk (8 members/expert on average); the kernel work above only pays when the
   dispatch sees >= ~2K rows. This is a driver change, not a kernel change.

## Risks / unknowns

* The model assumes WMMA and VALU fully serialize and DRAM at 230 GB/s; measured kernels sit at 73-79% (gate+up)
  and 82-90% (down) of that, so the realistic ceiling is ~35 TOPS, not 48.
* 2 mats x 3 tiles at 233 VGPR is already at the occupancy edge (earlier pipelining attempts spilled badly:
  ledger #74/#75/#76).
* Box 2's iGPU is identical, but it runs the same kernels through the same dispatch; the xq rows it receives
  already carry the data this needs (int8 q + yd).
* NPU / dGPU are not touched by this.

## RESULTS (2026-10-10, gfx1151 iGPU, harness `scratch/kernel_ideas/B_moe_prefill`, beside live hub traffic)

Files: `mid_f16.inc` (the kernel arm) + `cand_wide.hip` (shared scaffold) -> `cand_f16.hip` via
`mkf16.py` (`build_f16abl.sh` calls it and builds the timing-only ablations). `cand_f16.hip` is committed too.
Same entry names/args as the shipped arm, so the harness swaps it in with `KB_CAND`. Harness additions: `KB_CHUNK`
(candidate work-item chunk), `abx` mode (shipped arm vs two candidates interleaved in one process), `run_f16.sh`.

**Resources (hipcc gfx1151, no spills anywhere):**

| | gate+up VGPR / LDS | down VGPR |
|---|---|---|
| shipped int8 arm (2 tiles) | 230 / 10.1 KB | 133 |
| f16, 2 tiles (32 members) | 179 / 17.3 KB | 123 |
| f16, 3 tiles (48 members) | 224 / 25.9 KB | 159 |

Static mix per super-block (`isa_mix.py`): VALU per WMMA 51.2 -> 29.6 (gate+up), 49.1 -> 28.5 (down) at 2 tiles.

**Correctness** (`cmp`, vs production kwide/kwide2, real harness inputs): gate+up rel_rmse 3.9e-7, down 1.1e-7
at 2 tiles B=512, and the same at 3 tiles / chunk 48 / B=3072 (max_group 68, so chunking is exercised). The shipped
int8 arm is 5.4e-7 / 1.7e-7, so the f16 arm is at least as close. The baseline itself matches the CPU reference
to 4e-6 / 1.7e-6. Not bit-exact (f32 reassociation, same class as the shipped arm): still needs the golden/KLD gate.

**Speed, interleaved in one process** (`abx`, B=3072, 48 experts = 48.5 members/expert, 12 rounds; the hub is
live so absolute numbers move 9-16 ms between runs for the SAME kernel; only same-process ratios are usable):

| comparison (vs shipped int8 arm) | gate+up med (p10) | down med (p10) |
|---|---|---|
| f16, 2 tiles, chunk 32 | 9.42 vs 15.99 ms = **1.70x** (9.05 vs 12.82 = 1.42x) | 4.66 vs 5.39 = 1.16x (4.56 vs 5.02) |
| f16, 3 tiles, chunk 48 | 7.57 vs 11.22 ms = **1.48x** (7.28 vs 10.57 = 1.45x) | 5.57 vs 5.44 = tie (3.43 vs 4.02 = 1.17x) |
| (f16 3 tiles vs f16 2 tiles, from the two runs) | ~1.24x | ~1.0x |

So the scale-in-LUT f16 arm is a real **~1.4-1.7x on gate+up** over what ships, and ~1.15-1.2x on down (down was
already near its DRAM roof). Wider chunks add a further ~1.2x on gate+up. At B=1024 / 16.5 members per expert the
two arms tie on gate+up (memory/latency bound there), as predicted: this lever only matters at big windows.

**Ablation (f16, 2 tiles, gate+up, B=3072):** removing the activation global loads: 8.9 ms (vs 9.2); weight loads:
8.1 (p10 6.8); both: 8.7 (p10 6.8). So even with ALL global loads removed the kernel sits at ~6.8 ms = 16 TFLOP/s =
28% of the ~57 TOPS peak: the WMMA pipe is ~1/3 busy, and it is NOT instruction count (VALU/WMMA fell 42% for ~1.4x)
nor memory. Suspects, untested: two `__syncthreads` per super-block with stage->barrier->compute serialisation
(no overlap inside a WG), LDS fragment reads (4 ds_read_b128 per tile-block), only 4 independent accumulator chains
per wave. Next experiments: double-buffered staging (stage sb+1 while computing sb), PMC (`SQ_BUSY_CYCLES`,
`SQ_INSTS_LDS`, ONE counter per pass, iGPU PMC only via the flake-pinned rocprofiler, NEVER ATT on the iGPU).

**Not yet done:** driver integration (a `V41_MOE_WMMA_F16` knob in `het::dispatch`, chunk 48 into the work-item builder,
rust wrappers, `build.rs` registration, expected default OFF), the golden/KLD gate (needs the hub down), e2e prefill
A/B, box 2 (same binary), and a scan of the remaining layers' E8M0 range (only layers 2 and 7 sampled; the kernel
clamps to e8 in [114,137], a weight outside that would silently flush/saturate).
