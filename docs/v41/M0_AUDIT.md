# M0 audit — raw grep classification (generated 2026-09-12)

## literal 256 as N_EXPERT candidates (src, non-test)
crates/v4flash-kernels/src/config.rs:30:pub const N_EXPERT: u32 = 256;
crates/v4flash-kernels/src/embed.rs:196:        let mut out = [0f32; 256];
crates/v4flash-kernels/src/embed.rs:223:            let mut ours = [0f32; 256];
crates/v4flash-kernels/src/embed.rs:228:            for i in 0..256 {
crates/v4flash-kernels/src/router_topk.rs:19:pub const ROUTER_MAX_EXPERTS: u32 = 256;
crates/v4flash-kernels/src/het/batch_scratch.rs:509:pub const HOT_MAX_EXPERTS: usize = 256;
crates/v4flash-kernels/src/het/mod.rs:10://! * **iGPU (Strix Halo, gfx1151)** — router + routed MoE (256 experts × 6
crates/v4flash-kernels/src/het/weights.rs:5://! * [`IgpuLayerWeights`] — routed MoE (gate/up/down for 256 experts).
crates/v4flash-kernels/src/het/weights.rs:104:    /// Routed experts. All 256 by default; with `IGPU_DEDUP_HOT` only the
crates/v4flash-kernels/src/het/weights.rs:105:    /// `256 - n_hot` experts that are NOT dGPU-resident, packed dense
crates/v4flash-kernels/src/het/weights.rs:416:/// Format: `N_LAYER * N_EXPERT` (43 × 256) f32 little-endian, layer-major
crates/v4flash-kernels/src/het/weights.rs:895:        let mut remap_host = vec![-1i32; 256];
crates/v4flash-kernels/src/het/weights.rs:947:        let mut remap = DeviceBuffer::<i32>::new(device_id, 256)?;
crates/v4flash-kernels/src/weight_contract.rs:124:        "blk.N.ffn_gate_exps.weight" | "blk.N.ffn_up_exps.weight" => vec![e, f, 256],
crates/v4flash-kernels/src/weight_contract.rs:125:        "blk.N.ffn_down_exps.weight" => vec![f, e, 256],
crates/v4flash-kernels/src/weight_contract.rs:133:        "blk.N.ffn_gate_inp.weight" => vec![e, 256],
crates/v4flash-kernels/src/het/engine.rs:604:                    Option<std::sync::Mutex<(Vec<[u64; 256]>, u64)>>,
crates/v4flash-kernels/src/het/engine.rs:607:                        std::sync::Mutex::new((vec![[0u64; 256]; N_LAYER as usize], 0u64))
crates/v4flash-kernels/src/het/engine.rs:612:                static HOTSETS: std::sync::LazyLock<Option<Vec<[bool; 256]>>> =
crates/v4flash-kernels/src/het/engine.rs:623:                                    let mut m = [false; 256];
crates/v4flash-kernels/src/het/engine.rs:680:                        if (0..256).contains(&e) {
crates/v4flash-kernels/src/het/engine.rs:687:                            if (0..256).contains(&e) && hs[layer][e as usize] {
crates/v4flash-kernels/src/het/engine.rs:746:                                        let mut idx: Vec<usize> = (0..256).collect();
crates/v4flash-kernels/src/het/prefill_stats.rs:29://! predicts ~1.9× reuse at B=64 and ~6× at B=256; this instrumentation
crates/v4flash-kernels/src/mxfp4.rs:94:        if remap.len() < 256 {
crates/v4flash-kernels/src/mxfp4.rs:95:            return Err(eyre!("mxfp4 hetsplit: remap len {} < 256", remap.len()));
crates/v4flash-kernels/src/het/forward_prefill.rs:66:/// Offload cost of the cap is ~nil: with ~8 of 256 experts resident,
crates/v4flash-kernels/src/het/forward_prefill.rs:2625:        // wide when n_rows >= 64 (N_EXPERT=256 ≥ 64); a future wide-
crates/v4flash-kernels/src/het/forward_prefill.rs:2634:            // GEMM-tile via LDS-WMMA: M=N_EXPERT=256, K=N_EMBD=4096, N=B.

## router kernels
crates/v4flash-kernels/kernels/router_topk.hip:20:// Single block, 256 threads (= N_EXPERT). Top-K is computed by thread
crates/v4flash-kernels/kernels/router_topk.hip:25:#define ROUTER_MAX_EXPERTS 256
crates/v4flash-kernels/kernels/router_topk.hip:43:    __shared__ float s_probs[ROUTER_MAX_EXPERTS];
crates/v4flash-kernels/kernels/router_topk.hip:44:    __shared__ float s_select[ROUTER_MAX_EXPERTS];
crates/v4flash-kernels/kernels/router_topk_par.hip:4:// thread 0 of a 256-thread block, leaving 99.6% of the wavefront i
crates/v4flash-kernels/kernels/router_topk_par.hip:14:#define ROUTER_MAX_EXPERTS 256
crates/v4flash-kernels/kernels/router_topk_par.hip:32:    __shared__ float s_probs[ROUTER_MAX_EXPERTS];
crates/v4flash-kernels/kernels/router_topk_par.hip:33:    __shared__ float s_scores[ROUTER_MAX_EXPERTS]; // mutable; win
crates/v4flash-kernels/kernels/router_topk_par.hip:34:    __shared__ float buf[ROUTER_MAX_EXPERTS];
crates/v4flash-kernels/kernels/router_topk_par.hip:35:    __shared__ int   idxs[ROUTER_MAX_EXPERTS];
crates/v4flash-kernels/kernels/router_topk_par.hip:60:    // [n_expert, ROUTER_MAX_EXPERTS).
crates/v4flash-kernels/kernels/router_topk_par.hip:72:        // Tree reduce: stride-halving. For ROUTER_MAX_EXPERTS=256
crates/v4flash-kernels/kernels/router_topk_par.hip:74:        for (unsigned int stride = ROUTER_MAX_EXPERTS / 2; stride 
crates/v4flash-kernels/src/router_topk.rs:19:pub const ROUTER_MAX_EXPERTS: u32 = 256;
crates/v4flash-kernels/src/router_topk.rs:69:        if n_expert == 0 || n_expert > ROUTER_MAX_EXPERTS {
crates/v4flash-kernels/src/router_topk.rs:71:                "router_topk: n_expert {n_expert} must be in [1, {ROUTER_MA
crates/v4flash-kernels/src/router_topk.rs:148:        if n_expert == 0 || n_expert > ROUTER_MAX_EXPERTS {
crates/v4flash-kernels/src/router_topk.rs:150:                "router_topk: n_expert {n_expert} must be in [1, {ROUTER_M

## odd-layer / swap parity
crates/v4flash-kernels/src/het/engine.rs:766:            std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
crates/v4flash-kernels/src/het/engine.rs:799:        // N_LAYER (43) is odd, so 43 in-loop swaps leave residual /
crates/v4flash-kernels/src/het/engine.rs:800:        // residual_next inverted from token start. Without an extra swap
crates/v4flash-kernels/src/het/engine.rs:805:        // alternating tokens). The extra swap restores the initial state
crates/v4flash-kernels/src/het/engine.rs:808:        std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
crates/v4flash-kernels/src/het/forward_prefill.rs:258:            std::mem::swap(&mut batch_dgpu.residual, &mut batch_dgpu.residual_next);
crates/v4flash-kernels/src/het/forward_prefill.rs:431:            std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
crates/v4flash-kernels/src/het/forward_prefill.rs:449:            std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);
crates/v4flash-kernels/src/het/forward_prefill.rs:475:        std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
crates/v4flash-kernels/src/het/forward_prefill.rs:477:        std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);

## width caps (n > 4096 etc.)
crates/v4flash-kernels/src/rms_norm.rs:69:        if n > 4096 {
crates/v4flash-kernels/src/rms_norm.rs:110:        if n > 4096 || n % 32 != 0 {
crates/v4flash-kernels/src/rms_norm.rs:158:        if n > 4096 {
crates/v4flash-kernels/src/rms_norm.rs:159:            return Err(eyre!("rms_norm_weighted_batched: n={n} > 4096"));

## kernel #define of dims (guarded?)
crates/v4flash-kernels/kernels/hc_sigmoid_bias.hip:8:#define DS4_HC_EPS 1.0e-6f
crates/v4flash-kernels/kernels/mhc_pre_fused.hip:30:#define MHC_HC_DIM      16384
crates/v4flash-kernels/kernels/mhc_pre_fused.hip:31:#define MHC_N_EMBD      4096
crates/v4flash-kernels/kernels/mhc_pre_fused.hip:32:#define MHC_HC_MIX_DIM  24
crates/v4flash-kernels/kernels/mhc_pre_fused.hip:33:#define MHC_N_HC        4
crates/v4flash-kernels/kernels/mhc_pre_fused.hip:34:#define MHC_BLOCK       512
crates/v4flash-kernels/kernels/mhc_pre_fused.hip:35:#define MHC_WAVES       16
crates/v4flash-kernels/kernels/mhc_pre_fused.hip:36:#define MHC_LANES       32
crates/v4flash-kernels/kernels/laguna_ops.hip:281:#define ROUTER_WRO_MAXJ 16
crates/v4flash-kernels/kernels/router_topk.hip:25:#define ROUTER_MAX_EXPERTS 256
crates/v4flash-kernels/kernels/router_topk.hip:26:#define ROUTER_MAX_USED    8
crates/v4flash-kernels/kernels/router_topk_par.hip:14:#define ROUTER_MAX_EXPERTS 256
crates/v4flash-kernels/kernels/router_topk_par.hip:15:#define ROUTER_MAX_USED    8
0

## literal 43 / N_LAYER arrays
crates/v4flash-kernels/src/config.rs:76:pub const COMPRESS_RATIOS: [u32; 43] = [
crates/deepstrix-server/src/engine_worker.rs:991:            let in_block = (10..14).contains(&i) || (40..43).contains(&

## divisibility guards (count by modulus)
     59 8
     17 2
     16 16
     12 32
      7 64
      5 4
      3 4==0
      3 128
