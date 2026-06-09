# M52 — dGPU prefill (long-context) journal

Context: after M51 (kwide MoE, 320 tok/s @4K), the depth curve sagged at long
ctx (290.8 @64K, 261.1 @96K) — the dGPU attention/indexer side stopped being
hidden behind the now-faster iGPU. All profiling per user guidance at LONG
contexts via FAKE_PREFILL_POS (dGPU is invisible at short ctx).

## 2026-06-09 — Phase 0: kernel-trace + pftrace at 96K

rocprofv3 kernel-trace (FAKE_PREFILL_POS=98304, PIPELINE_LANES=2, B=1024/iter,
per 1024-tok chunk): iGPU busy 3231 ms; dGPU busy 1812 ms; wall 3922.
Top dGPU: q8_0_gemm_wmma_lds_tiled 415, **indexer_score_wmma_batched 402
(28.7 ms/launch)**, f16_matvec_pair 238, wmma_grouped 141.

pftrace gaps (per 512-tok lane-chunk): iGPU busy 1568 + 388 idle (9 ms/layer
after q2k_down waiting on dGPU); dGPU busy 1411 + 527 idle (12.4 ms/layer
waiting on iGPU MoE). Perfect packing floor = iGPU busy = 326 tok/s; ~835 ms
of unfilled per-layer ping-pong stalls. Steady-state check (4096 tok/iter):
261.6 — identical to 2-chunk number → stalls are per-layer structural, NOT
pipeline edges.

indexer_score roofline: B=1024 × n_comp 24.5K × 64 heads × 128 dim ≈ 412
GFLOP ≈ 3.4 ms at WMMA peak; measured 28.7 — 8× over. Cause (read from
kernel): 1-wave WGs, each re-staging the token's FULL Q (8192 f32→f16, 256
serial iters/lane) per 128 cols → ~6.3 GB redundant Q reads/launch + 4×
staging:WMMA instruction ratio.

## 2026-06-09 — indexer_score_wmma_batched_mw SHIPPED as default

Design: 8 waves/WG, Q+hw staged cooperatively ONCE per 1024 cols; NO K LDS
(B-fragments straight from global — K is MALL-resident); single barrier, so
per-wave early-outs can't deadlock. Grid (ceil(cols/1024), B), block 256.

- Oracle: **bit-exact** on [0, n_comp); -INF tail contract on
  [n_comp, stride) (the 1-wave kernel leaves cols past its n_idx_max-sized
  grid UNWRITTEN — fine, topk only reads [0, n_comp)).
- Isolated (B=512, n_idx=24576): **32.6 → 7.45 ms (4.4×)**.
- E2e cold (back-to-back):

| depth | sw (1-wave) | mw | Δ |
|---|---|---|---|
| 4096  | 319.7 | 318.6 | noise |
| 32768 | 312.0 | 311.6 | noise |
| 65536 | 291.8 | **314.6** | +7.8% |
| 98304 | 262.3 | **318.0** | +21.2% |

Wall saved at 96K (684 ms) > kernel time saved (~310 ms): the shorter dGPU
per-layer critical path also dissolved most of the per-layer packing stalls.
**Depth penalty eliminated — 96K runs at 4K speed.** Rollback:
INDEXER_SCORE_VARIANT=sw.

## Remaining (next levers at long ctx)

- Depth curve now flat 4K-96K at ~312-320 → the wall is back to the iGPU MoE
  side everywhere; M51's "remaining headroom" list applies at all depths
  (B_MAX=2048, q2k traffic halving, FAST_FULL hotlist → ~400).
- mw kernel could still drop K MALL traffic ~2× (2 tokens/WG sharing
  B-fragments) if indexer ever re-surfaces in a trace.
- q8_0_gemm_wmma_lds_tiled (415 ms/chunk dGPU, depth-independent) is the next
  dGPU item if packing ever exposes it; WMMA-i8 idea from [[q8-lds-tiled-wmma]].

# M53 — iGPU round 2 (no hot-expert cache)

## 2026-06-09 — kwide staging ISA audit

The xq staging loop compiles terribly: per dword — 64-bit addr chain, exec-mask
branching for the ii==0 yd case, and `s_waitcnt vmcnt(0)` before EVERY
ds_store (zero MLP); members copied sequentially with only 130/256 threads.
Plan: branch-free run-per-thread staging (all members concurrent, 16
pipelined loads/thread), then q2k row-pair reuse, then chunk=64 sequential
halves.

## 2026-06-09 — M53.1 staging rewrite SHIPPED: iq2 50.6 → 30.0 ms (−41%)

Branch-free run-per-thread staging (all members concurrent, 16 pipelined
loads/thread, yd by separate thread subset). Oracle unchanged (1.1e-4).
The serialized staging stalled all 8 waves at the barrier — its true cost was
~2× its PMC instruction share.

E2e: **4K 419.9 / 32K 409.8 / 96K 384.0 tok/s** (was 318/312/318).
iq2 now ≈ at the (conservative-estimate) dp4a compute roofline.
96K dips again as the iGPU shrinks → dGPU partially re-exposed (expected).
Next: q2k row-pair (q2k now ~36% of iGPU busy).
