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
