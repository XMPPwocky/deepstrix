# Brief: how does V4.1-Flash decode reach 30 tok/s WITHOUT DSpark?

Question for review. DSpark (speculative decode, `dspark_target_layer_ids [37,38,39]`,
128 experts, top-3) is NOT yet built and we want the non-speculative path to 30 first.

## Hardware
* Box 1 "lumi-brain": Strix Halo, 96 GB unified, gfx1151 iGPU (~256 GB/s LPDDR5X)
  + RX 9070 XT (gfx1201, ~640 GB/s) over OCuLink. Model weights on a YMTC PC411 NVMe
  behind dm-crypt; measured 4.4 GB/s single-thread O_DIRECT, ~4.8 GB/s aggregate in
  production.
* Box 2 "lumi-brain2": Strix Halo, 128 GB, iGPU only, over USB4 (~724 MB/s, RTT 436 us).
  Runs `deepstrix-expertd --paged`, owning `L0-L19:116-383,L20-L39:344-383` (6,160
  resident expert slots, ~116 GB). Weights NEVER cross USB4 — only activations.

## Model (config.json)
40 layers, hidden 5120, 384 routed experts, top-6, moe_intermediate 2304, 1 shared
expert. 64 attn heads, head_dim 512, qk_rope 64. Engram layers [1,14]. CED: prefill
runs encoder layers 0-19, then replays a 128-token window through decoder layers 20-39.

## Measured decode (today, back-to-back, routing verifier clean)
* **15.8 tok/s at `--ctx 8192`**, `decode_hit=1.0000`, ZERO expert disk I/O in steady
  state. Path: two-box split + T2 catch-all (box 1 computes only what it already holds
  and reassigns every miss to box 2, which pages from its OWN disk).
* **5.44 / 5.56 tok/s at `--ctx 130688`** with only ~293 ACTUAL tokens in context,
  `decode_hit=1.0, decode_misses=0`. Same weights, same box-2 spec, no paging in either
  arm. So decode appears to cost ~2.8x more purely because the context was ALLOCATED
  larger, not used. UNVERIFIED as a regression — being re-checked against ctx=8192
  back-to-back — but if real it is the single largest decode lever we have found.
* History: 0.8 -> 4.5 -> 15.8 over two days (residency work).

## What is already known DEAD or measured-negative (do not re-propose without new evidence)
* `V41_MHC_SPLIT=1` (mHC mixes on a side stream): decode 6.95 -> 6.22 tok/s, and
  `sel_sync_us` 37,333 -> 67,394. Defaulted OFF.
* Static hot-expert placement on V4-Flash was INERT; on V4.1 routing is Zipfian so a
  dGPU hot-expert tier simulates at ~-19% misses, but decode already has ~0 misses at
  8K, so it can only help long-context/cold cases.
* Gather work-capping: regressed decode 14.4 -> 11.9 (warm-bench conclusion applied to
  cold-NVMe reads).
* Capacity: `V41_REPLAY_OFFLOAD=1` is unreachable — box 2 sizes its pool by OWNERSHIP
  even with `--paged`, and the algebra forces box 1 to 89 GB against a 52 GB pool.

## Structural facts that may matter
* Decode is SERIAL per layer; the iGPU MoE leg is on the critical path (prefill is
  pipelined, decode is not).
* The CSA2 **sparse indexer is NOT ported**. `forward_layer.rs` gates the sparse path on
  `ratio == 4 && n_index_comp > INDEXER_TOP_K`; V4.1's ratios are 1 and 2, so V4.1
  attention scores DENSELY over the whole compressed store at every layer >= 2. Costed
  at +4-5 ms/token at 8K and +10-18 ms/token at 32K, growing linearly. Sparse would be
  flat (512 rows + 128 window) from ~1K onwards.
* Decode hard-codes `attn_n_comp = INDEXER_TOP_K` at `forward_layer.rs:1367`.
* Prefill for contrast is 586-648 tok/s at 32K; attention is only 7% of prefill stage
  time there and `igpu.routed_moe` is 56%.

## The question
Rank the concrete paths from 15.8 to 30 tok/s decode, non-speculative. For each: the
mechanism, the expected ms/token it removes, the measurement that would confirm or kill
it BEFORE implementation, and what could make it a mirage. Be adversarial about the
ctx-allocation finding above — say what else could produce that 2.8x besides "decode
scales with allocated ctx", and name the cheapest experiment that discriminates.
