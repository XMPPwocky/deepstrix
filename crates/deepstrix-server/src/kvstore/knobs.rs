//! The knob classification and the knob hash (design 4.4).
//!
//! Every file header records the effective numerics generation: the code
//! constant [`super::KV_NUMERICS_GEN`] AND a hash over the values of every env
//! knob that can change prefill numerics. A knob flipped only in the launch env
//! changes no commit, so the constant alone would hide it (R5-1).
//!
//! The classification below covers EVERY string literal of a knob name
//! (`V41_*`, `DEEPSTRIX_*`, `VIT_*`) in the kernels, core, vision and server
//! crates, not only `std::env::var` calls: reads hide in wrappers whose key is a
//! variable (`env_f` / `env_u` in `expert_pager.rs`, `env_usize` in
//! `multistream.rs`). [`tests::knob_literals_are_classified`] fails on any
//! literal not listed here and on any listed name that no longer occurs.
//!
//! Classifying a new knob: `N` if it can change the bits of anything the
//! prefill stores (compressed rows, index keys, accumulators, the encoder and
//! decoder windows, the DSpark ring seeded at finish), including kernel arm
//! choices, chunk / lane sizes (kernels pick arms by batch size, E3), the
//! box-1/box-2 split, Engram and the vision tower. `P` for decode-only,
//! scheduling, logging, paths and pure IO. When in doubt, `N`: over-inclusion
//! only adds a generation pair to the headers, under-inclusion hides drift.
//!
//! Known gap (M2): knobs read by the box-2 daemon (`V41_B2_MERGE`,
//! `V41_B2_FAST_CHAIN`, ...) are hashed from the HUB's environment, which may
//! not be box 2's. Box 2 would have to report its values (e.g. in HELLO).
//!
//! The initial classification (2026-10-01) read each literal's use sites; the
//! reasons are one line each.

/// Can this knob change prefill numerics?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Numerics,
    Plain,
}

use Class::{Numerics as N, Plain as P};

/// `(name, class, why)`, sorted by name.
pub const KNOBS: &[(&str, Class, &str)] = &[
    ("DEEPSTRIX_ALLOW_IMAGE_DIRS", P, "image path allow-list for requests; no compute effect"),
    ("DEEPSTRIX_ATTN_LEGACY_STRIDE", N, "rollback of prefill attention-scores stride/scratch layout (doubt: cheap N)"),
    ("DEEPSTRIX_BIAS_VL_FILE", N, "router bias_vl sidecar for image rows; weights file outside namespace fingerprint"),
    ("DEEPSTRIX_COMP_GATHER", N, "compressor pooling gather vs serial loop on the prefill path"),
    ("DEEPSTRIX_COMP_GEMM", N, "compressor kv/score projection WMMA GEMM (not bit-exact) vs matvec"),
    ("DEEPSTRIX_COMP_TILED", N, "compressor projection batch-tiled matvec pair vs plain matvec"),
    ("DEEPSTRIX_DUMP_DIR", P, "test/fixture only (tests/)"),
    ("DEEPSTRIX_DUMP_RESIDUAL_DIR", P, "residual dump dir; observe only"),
    ("DEEPSTRIX_DUMP_SUBTENSOR_DIR", P, "subtensor dump dir; observe only (adds syncs, disables arena graphs)"),
    ("DEEPSTRIX_DUMP_SUBTENSOR_LAYER", P, "subtensor dump layer (legacy name); observe only"),
    ("DEEPSTRIX_DUMP_SUBTENSOR_LAYERS", P, "subtensor dump layer list; observe only"),
    ("DEEPSTRIX_EXPERT_LOAD_PROFILE", P, "load-time read/copy timing; observe only"),
    ("DEEPSTRIX_EXPERT_READ_THREADS", P, "expert load read threads; IO only, same bytes"),
    ("DEEPSTRIX_EXPERT_STATS", P, "single-token path expert pick stats; observe only"),
    ("DEEPSTRIX_EXPERT_TRACE", P, "single-token path expert pick trace; observe only"),
    ("DEEPSTRIX_F32_SCORES", N, "attention scores in f32 instead of f16 (prefill batched attention kernels)"),
    ("DEEPSTRIX_FAST_EXPERT_LOAD", P, "expert load mechanism; IO only, same bytes"),
    ("DEEPSTRIX_FAST_LOAD", P, "weight load mechanism; byte-identity tested (fast_load_bytes_match)"),
    ("DEEPSTRIX_GFX_TARGETS", P, "compile-time; changes need a code/build change"),
    ("DEEPSTRIX_GGUF", P, "test/fixture only (tests/)"),
    ("DEEPSTRIX_GTT_NODE", P, "test/fixture only (tests/)"),
    ("DEEPSTRIX_HANG_DEADLINE_MS", P, "server hang watchdog timing"),
    ("DEEPSTRIX_HEARTBEAT_TOKENS", P, "heartbeat log interval"),
    ("DEEPSTRIX_HF_THREADS", P, "HF safetensors reader threads; IO only"),
    ("DEEPSTRIX_IQ2S_BLOB_DIR", P, "test/fixture only (tests/)"),
    ("DEEPSTRIX_IQ2S_KW_MAX_CHUNK", P, "compile-time; changes need a code/build change"),
    ("DEEPSTRIX_IQ3S_BLOB", P, "test-only blob path (cfg(test) in iq3_s_ref.rs)"),
    ("DEEPSTRIX_IQ3S_BLOB_DIR", P, "test/fixture only (tests/)"),
    ("DEEPSTRIX_IQ3S_KW_MAX_CHUNK", P, "compile-time; changes need a code/build change"),
    ("DEEPSTRIX_KERNEL_CFLAGS", P, "compile-time; changes need a code/build change"),
    ("DEEPSTRIX_MMPROJ", P, "vision tower weights path; tower fingerprint is in the namespace"),
    ("DEEPSTRIX_PREFILL_PROFILE", P, "prefill stage profiling; observe only (serialises lanes, timing)"),
    ("DEEPSTRIX_SEL_STATS", P, "router selection stats; observe only"),
    ("DEEPSTRIX_SNAPSHOT_KEEP_PER_SESSION", P, "snapshot-store retention"),
    ("DEEPSTRIX_SNAPSHOT_REUSE", P, "snapshot-store reuse on/off"),
    ("DEEPSTRIX_SUBSTITUTE_RESIDUAL", N, "debug residual substitution alters computation (forward_token path; cheap N)"),
    ("DEEPSTRIX_TOKEN_PROFILE", P, "per-token HIP event profiling; observe only"),
    ("DEEPSTRIX_TRACE_TOKENS", P, "token trace logging"),
    ("DEEPSTRIX_VISION_DEVICE", P, "test/fixture only (tests/)"),
    ("V41_", P, "log-filter prefix literal, not a knob"),
    ("V41_ALLOW_MISSING_FLOORS", P, "test/fixture only (tests/)"),
    ("V41_ATTN_COMP_CAP", N, "debug clamp of compressed attention rows (forward_layer path; cheap N)"),
    ("V41_ATTN_DEC_FUSED", N, "fused attention kernel for lanes <= 16 rows; short prefill lanes take it"),
    ("V41_ATTN_DEC_SCORE", N, "attention score kernel for lanes <= 16 rows (short prefill lanes)"),
    ("V41_ATTN_DEC_SCORE_BLK128", N, "attention score kernel variant for lanes <= 16 rows (short prefill lanes)"),
    ("V41_ATTN_META_FILL", P, "attention metadata upload via kernel args; device bytes identical"),
    ("V41_B1_HOT", N, "box-1 hot-set ownership on/off vs hash split; changes MoE box split"),
    ("V41_B1_HOT_HYST", N, "box-1 hot-set hysteresis; ownership decides box1/box2 MoE split"),
    ("V41_B1_HOT_MASS", N, "box-1 hot-set mass cap; ownership decides box1/box2 MoE split"),
    ("V41_B1_HOT_MAX_CHANGE", N, "box-1 hot-set churn cap; ownership decides box1/box2 MoE split"),
    ("V41_B1_HOT_MIN_PICKS", N, "box-1 hot-set warm threshold; ownership decides box1/box2 MoE split"),
    ("V41_B1_HOT_PER_LAYER", N, "box-1 hot-set size; ownership decides box1/box2 MoE split"),
    ("V41_B1_HOT_REFRESH", N, "box-1 hot-set refresh period; ownership decides box1/box2 MoE split"),
    ("V41_B1_HOT_SLACK", N, "box-1 hot-set size slack; ownership decides box1/box2 MoE split"),
    ("V41_B1_PAGE_MISSES", N, "box 1 pages its partition misses itself instead of handing them to box 2"),
    ("V41_B1_PREFETCH", N, "box-1 misses go to box 2 by residency in the prefill pick loop"),
    ("V41_B1_PREFETCH_ADMIT", N, "box-1 residency fill rate; residency decides box under B1_PREFETCH"),
    ("V41_B1_PREFETCH_MIN_TOUCH", N, "box-1 residency admission gate; residency decides box under B1_PREFETCH"),
    ("V41_B2_ASSERT_NO_SURPRISE", P, "pin-ledger assertion (panic); no compute effect"),
    ("V41_B2_ASSERT_PINNED", P, "box-2 pin assertion (panic); no compute effect"),
    ("V41_B2_COALESCE", N, "box-2 coalesced expert read; documented corruption history (cheap N)"),
    ("V41_B2_COALESCE_CHECK", P, "byte-compare of coalesced reads; logs only"),
    ("V41_B2_DBG", P, "box-2 debug logging"),
    ("V41_B2_DECODE_BUSY_POLL_US", P, "socket busy-poll timing"),
    ("V41_B2_DECODE_DOWN", N, "box-2 batched pass runs decode down kernel per token (diagnostic arm)"),
    ("V41_B2_EARLY_PAGE", P, "box-2 early paging of queued requests; residency/timing only"),
    ("V41_B2_ENCODER_VICTIMS_FIRST", P, "box-2 eviction order; residency only"),
    ("V41_B2_FAST_CHAIN", N, "box-2 short MoE chain for b <= 16 (claimed bit-identical; compute variant)"),
    ("V41_B2_GLOBAL_POOL", P, "box-2 pool eviction scope; residency only"),
    ("V41_B2_GPU_REPACK", P, "box-2 MXFP4 byte repack on GPU vs CPU; byte permutation, same bytes"),
    ("V41_B2_HITS_FIRST", N, "box-2 two-pass hits-first MoE (claimed bit-identical; compute variant)"),
    ("V41_B2_KNOBS", N, "path of box-2 runtime knobs file that can flip merge/fast_chain/coalesce"),
    ("V41_B2_MERGE", N, "box-2 merges two lanes into one MoE pass; row count picks WMMA arms"),
    ("V41_B2_MERGE_WAIT_US", N, "merge partner wait; decides whether passes merge (row-count arms)"),
    ("V41_B2_MISS_PAR", P, "box-2 concurrent miss reads; IO only"),
    ("V41_B2_MODE_EVICT", P, "box-2 mode-aware eviction; residency only"),
    ("V41_B2_ODIRECT", P, "box-2 O_DIRECT reads; IO only, same bytes"),
    ("V41_B2_PARK", P, "box-2 serves queued requests while one pages; scheduling only"),
    ("V41_B2_PARK_LOG", P, "box-2 park logging"),
    ("V41_B2_PIN", P, "box-2 pin ledger on/off; residency only"),
    ("V41_B2_PIN_DECAY_STEPS", P, "box-2 pin ledger decay; residency only"),
    ("V41_B2_PIN_HEADROOM", P, "box-2 pin headroom; residency only"),
    ("V41_B2_PIN_PREFILL_BAND", P, "box-2 pin prefill band; residency only"),
    ("V41_B2_PIN_PREFILL_BAND_FILE", P, "box-2 pin prefill band (file); residency only"),
    ("V41_B2_PIN_RESERVE", P, "box-2 unpinnable reserve; residency only"),
    ("V41_B2_PIN_RESTORE", P, "box-2 pin restore; residency only"),
    ("V41_B2_PIN_RESTORE_PER_REQ", P, "box-2 pin restore rate; residency only"),
    ("V41_B2_PIN_WANTS", P, "box-2 pin ranking input; residency only"),
    ("V41_B2_POOL_FLOOR", P, "box-2 per-layer pool floor; residency only (bit-identical measured)"),
    ("V41_B2_PREFETCH_PAR", P, "box-2 prefetch readers; IO only"),
    ("V41_B2_PREFETCH_RESERVE", P, "box-2 prefetch reader reserve; IO only"),
    ("V41_B2_PREFETCH_SETS", P, "box-2 prefetch staging sets; IO only"),
    ("V41_B2_PREFILL_BUDGET", P, "box-2 decode victims per prefill phase; residency only"),
    ("V41_B2_PREFILL_ROUTE", P, "box-2 drive routing for prefill reads; IO only"),
    ("V41_B2_PREFILL_STAGE", P, "box-2 prefill staging band size; residency only"),
    ("V41_B2_RESTORE", P, "box-2 restore of evicted decode experts; residency only"),
    ("V41_B2_ROUTE", P, "box-2 drive routing (split/urgency); IO only"),
    ("V41_B2_SCAN_CLASS", P, "box-2 two-class LRU; residency only"),
    ("V41_B2_SPEC_CHUNK_KB", P, "box-2 background read chunking; IO only"),
    ("V41_BATCH_BUSY_POLL_US", P, "socket busy-poll timing"),
    ("V41_BENCH_LAYER", P, "test/fixture only (tests/)"),
    ("V41_CANDIDATE_POOL", N, "hierarchical candidate pool changes attention selection (prefill and decode)"),
    ("V41_CAND_THRESH_ILP", N, "candidate threshold kernel variant for small batches (prefill tails)"),
    ("V41_CED", N, "CED encoder/decoder split changes which layers prefill computes"),
    ("V41_COMP_POSITIONAL", N, "=1 overrides compressor n_comp_after with the positional formula"),
    ("V41_COMP_POS_DBG", P, "compressor positional-count mismatch logging"),
    ("V41_COMP_ROLLBACK", P, "compressor speculative rollback (DSpark verify, decode only)"),
    ("V41_DECODE_BUSY_POLL_US", P, "socket busy-poll timing"),
    ("V41_DECODE_LOGITS_DUMP", P, "decode logits dump; observe only"),
    ("V41_DECODE_PRESUBMIT", P, "single-token decode shared-expert reorder (forward_layer)"),
    ("V41_DEC_FUSE", N, "fused launch chains for lanes <= 16 rows (short prefill lanes)"),
    ("V41_DEC_SKIP_DEAD", N, "skips launches judged dead on the batched path (any b; cheap N)"),
    ("V41_DSPARK", P, "legacy DSpark enable (drafting/verify, decode)"),
    ("V41_DSPARK_CONF_MIN", P, "legacy DSpark K policy (decode)"),
    ("V41_DSPARK_DEBUG_MOE", P, "drafter MoE debug; drafting only"),
    ("V41_DSPARK_DENSE_RING", P, "legacy decode-time ring writes"),
    ("V41_DSPARK_DRAFT_TIMING", P, "drafter timing"),
    ("V41_DSPARK_FORCE_N0", P, "legacy DSpark accept debug (decode output only)"),
    ("V41_DSPARK_K", P, "legacy DSpark draft length (decode)"),
    ("V41_DSPARK_LAYER_TIMING", P, "drafter layer timing (mode 2 syncs; timing only)"),
    ("V41_DSPARK_NO_ATTN", P, "drafter ablation; drafts only, ring rows unaffected"),
    ("V41_DSPARK_NO_ROUTED", P, "drafter ablation; drafts only, ring rows unaffected"),
    ("V41_DSPARK_RING_FAST", P, "legacy decode-time batched ring writes"),
    ("V41_DSPARK_SEED_RING", P, "legacy path ring seeding on/off (legacy ring is not stored)"),
    ("V41_DSPARK_STEP_TIMING", P, "legacy DSpark step timing"),
    ("V41_DUMP_CACHES", P, "test/fixture only (tests/)"),
    ("V41_DUMP_FIRST_LOGITS", P, "first logits dump; observe only"),
    ("V41_DUMP_VERIFY_DIST", P, "verify distribution dump; observe only"),
    ("V41_ENGRAM_CHUNK128", N, "Engram wkv prefill pass rows 128 vs 64 (designed bit-identical; listed in 4.4)"),
    ("V41_ENGRAM_DIR", P, "Engram tables path; Engram identity is in the namespace"),
    ("V41_ENGRAM_GEMV", N, "Engram wkv prefill projection on Q8 GEMV fallback"),
    ("V41_ENGRAM_I8X", N, "Engram prefill GEMM kernel variant"),
    ("V41_EVTRACE_DIR", P, "trace output dir only"),
    ("V41_EVTRACE_KEEP", P, "trace retention"),
    ("V41_EVTRACE_MAX_MB", P, "trace size cap"),
    ("V41_EVTRACE_SYS_MS", P, "trace system sampling period"),
    ("V41_EXEC_MODE", P, "test/fixture only (tests/)"),
    ("V41_EXIT_GRACE_S", P, "shutdown grace period"),
    ("V41_EXPERT_MIRROR_DIR", P, "expert mirror drive dir; IO only, same bytes"),
    ("V41_EXPERT_MIRROR_FRAC", P, "expert read split across drives; IO only"),
    ("V41_EXPERT_ODIRECT", P, "expert reads O_DIRECT; IO only"),
    ("V41_EXPERT_PREAD_THREADS", P, "expert pread threads; IO only"),
    ("V41_F16X_256", N, "q_b / wo_a prefill GEMM tile variant"),
    ("V41_F16X_DB_BN64", N, "kv / q_a prefill GEMM tile variant"),
    ("V41_F16_MV_Z16", N, "f16 matvec z16 kernel variant (indexer q, compressor, replay router)"),
    ("V41_FIXTURE_EXPERTS", P, "test/fixture only (tests/)"),
    ("V41_FIXTURE_GGUF", P, "test/fixture only (tests/)"),
    ("V41_GEMV_BPACK", N, "Q8 GEMV b-packed kernel for b <= 16 (prefill tails, replay)"),
    ("V41_GEMV_BPACK_Z16", N, "Q8 GEMV z16 kernel for 16 < b <= 64 (replay lanes)"),
    ("V41_GEMV_TB", N, "Q8 GEMV compile-time-batch twin (small lanes incl. prefill tails)"),
    ("V41_GRID_PAD", N, "padded grids on replay-size launches (claimed bit-identical; cheap N)"),
    ("V41_GROUP_AUDIT", P, "het-split builder audit; observe only"),
    ("V41_GROUP_AUDIT_VERBOSE", P, "audit logging"),
    ("V41_HF_BENCH", P, "test/fixture only (tests/)"),
    ("V41_HF_DIR", P, "test/fixture only (tests/)"),
    ("V41_IDX_GATHER_B128", N, "indexer gather kernel variant"),
    ("V41_IDX_SCORE_QREG", N, "indexer score kernel variant"),
    ("V41_IDX_TOPK_HYBRID", N, "indexer top-k kernel variant"),
    ("V41_INDEXER_DBG", P, "indexer debug logging"),
    ("V41_INDEXER_DUMP", P, "indexer dump; observe only"),
    ("V41_INDEXER_FORCE", N, "forces the indexer path in prefill attention"),
    ("V41_INDEX_K", N, "index keys on/off changes stored E2M1 keys and attention selection"),
    ("V41_ISOLATED", P, "test/fixture only (tests/)"),
    ("V41_KV_F16_ROUNDTRIP", N, "restores f16 roundtrip node on the KV write path (claimed no-op)"),
    ("V41_LAYER", P, "test/fixture only (tests/)"),
    ("V41_LAYER_HOST_TIMING", P, "host timing counters"),
    ("V41_LAYER_MAJOR", P, "test/fixture only (tests/)"),
    ("V41_LAYER_MAJOR_BF16RES", P, "test/fixture only (tests/)"),
    ("V41_LAYER_MAJOR_RESEED", P, "test/fixture only (tests/)"),
    ("V41_LAYER_MAJOR_VERBOSE", P, "test/fixture only (tests/)"),
    ("V41_LAYER_MISS_HIST", P, "per-layer miss histogram logging"),
    ("V41_LM_PREFETCH", P, "layer-major box-2 prefetch words; residency only"),
    ("V41_LM_PREFETCH_PER_REQ", P, "layer-major box-2 prefetch pacing; residency only"),
    ("V41_LM_PREFILL", N, "layer-major prefill driver (designed bit-identical; GPU-gated)"),
    ("V41_LM_ROWS", N, "layer-major window rows (window/sub-chunk planning)"),
    ("V41_LOCAL_CLAIM_MAX", P, "single-token decode local claim cap (forward_layer)"),
    ("V41_LOCAL_PICKS", P, "single-token decode local picks override (forward_layer)"),
    ("V41_LOOKAHEAD_DEPTH", P, "look-ahead prefetch hint depth; hints only"),
    ("V41_LOOKAHEAD_PREFETCH", P, "look-ahead routing prefetch (arena rows); hints only"),
    ("V41_LOOKAHEAD_TOPK", P, "look-ahead prefetch hint rank cut; hints only"),
    ("V41_MASK_DBG", P, "remote submit mask logging"),
    ("V41_MAX_TOKENS", P, "test/fixture only (tests/)"),
    ("V41_MHC_ARENA_FUSED", N, "fused mHC mix kernels (b <= 8 pre-attn, b <= 64 pre-ffn)"),
    ("V41_MHC_FAST", N, "one-launch mHC sub-block kernels on small lanes"),
    ("V41_MHC_FFN_LATE", N, "V41_MHC_* family per design 4.4 (arena-gated reorder; cheap N)"),
    ("V41_MHC_GEMM_NARROW", N, "prefill mHC pre-mix GEMM narrow kernel (f16.rs)"),
    ("V41_MHC_NARROW", N, "mHC pre-mix narrow f32 matvec vs WMMA GEMM threshold"),
    ("V41_MHC_PRE_SCALED", N, "decode-exact mHC mix for b <= 8 vs batched form"),
    ("V41_MHC_SPLIT", N, "V41_MHC_* family per design 4.4 (single-token side-stream mixes)"),
    ("V41_MISS_HIST", P, "pager miss histogram; observe only"),
    ("V41_MISS_HIST_EVERY", P, "miss histogram log interval"),
    ("V41_MODEL", P, "test/fixture only (tests/, cfg(test))"),
    ("V41_MODEL_DIR", P, "test/fixture only (tests/)"),
    ("V41_MOE_DOWN_DN2", N, "MoE down small-b kernel variant (rows <= 8)"),
    ("V41_MOE_WI_DEVCOUNT", N, "MoE work-item count on device, upper-bound grid (kernel launch variant)"),
    ("V41_MOE_WMMA_DOWN", N, "MoE down int8-WMMA arm at >= 128 rows (not bit-exact)"),
    ("V41_MOE_WMMA_GATEUP", N, "MoE gate/up int8-WMMA arm at >= 256 rows (not bit-exact)"),
    ("V41_MS_AGING_S", P, "scheduler aging"),
    ("V41_MS_BURST_SCALE", P, "scheduler burst scaling"),
    ("V41_MS_CHECKPOINT_EVERY", N, "mid-prefill checkpoint resets decoder ring counters and lending (doubt)"),
    ("V41_MS_CHECKPOINT_MIN_ROWS", P, "cancel-path checkpoint threshold; cancelled job does not continue"),
    ("V41_MS_CHUNK_ROWS", N, "prefill chunk size (busy); chunk partition changes kernel arms"),
    ("V41_MS_CHUNK_ROWS_IDLE", N, "prefill chunk size (idle); chunk partition changes kernel arms"),
    ("V41_MS_CTX_ROWS", P, "arena context capacity; allocation only"),
    ("V41_MS_DECODE_BURST_MIN_MS", P, "scheduler decode burst"),
    ("V41_MS_DECODE_BURST_MS", P, "scheduler decode burst"),
    ("V41_MS_DSPARK", P, "DSpark enable; ring bits keyed by drafter fingerprint, prefill unchanged"),
    ("V41_MS_DSPARK_COST", P, "DSpark K policy cost ladder"),
    ("V41_MS_DSPARK_COST2", P, "DSpark K policy cost ladder"),
    ("V41_MS_DSPARK_COST_LIVE", P, "DSpark K policy live fit"),
    ("V41_MS_DSPARK_COST_MEMORY", P, "DSpark K policy fit memory"),
    ("V41_MS_DSPARK_DRAFTS", P, "DSpark draft sampling mode"),
    ("V41_MS_DSPARK_DRAFT_MS", P, "DSpark draft cost"),
    ("V41_MS_DSPARK_EXPLORE", P, "DSpark K exploration"),
    ("V41_MS_DSPARK_EXPLORE_SEED", P, "DSpark K exploration seed"),
    ("V41_MS_DSPARK_K", P, "DSpark draft length"),
    ("V41_MS_DSPARK_KMAX", P, "DSpark max draft length"),
    ("V41_MS_DSPARK_MIN_GAIN", P, "DSpark draft-or-not gate"),
    ("V41_MS_DSPARK_RING", P, "which decode rows enter the ring (seed is unconditional)"),
    ("V41_MS_DSPARK_RING_ASYNC", N, "ring writes without sync incl. prefill seed (race-sensitive; cheap N)"),
    ("V41_MS_ENGRAM_THREADS", P, "host Engram gather threads; IO only"),
    ("V41_MS_ENGRAM_THREADS_FILE", P, "host Engram gather threads (file); IO only"),
    ("V41_MS_FINISH_GROUP", P, "scheduler: finish layer-major group before decode burst"),
    ("V41_MS_GRAPHS", P, "arena (decode) per-stage HIP graphs only (cap_ok)"),
    ("V41_MS_HEAD_CANDS", P, "sampling head candidates (logits to token)"),
    ("V41_MS_HEAD_CANDS_FILE", P, "sampling head candidates (file)"),
    ("V41_MS_KV_GROW", P, "arena KV growth step; allocation only"),
    ("V41_MS_KV_GROW_AT", P, "arena KV growth policy; allocation only"),
    ("V41_MS_KV_HEADROOM", P, "arena KV reservation headroom; allocation only"),
    ("V41_MS_KV_SPARE", P, "arena KV spare; allocation only"),
    ("V41_MS_LANES", P, "decode step lane count"),
    ("V41_MS_LANES3_MIN_ROWS", P, "decode step three-lane threshold"),
    ("V41_MS_LANE_C_ROWS", P, "decode third-lane scratch rows"),
    ("V41_MS_LM_FILE", N, "runtime layer-major on/off for new prefill jobs"),
    ("V41_MS_MHC_SPLIT", P, "arena (decode) mHC side stream only (cap_ok)"),
    ("V41_MS_PIPELINE", P, "decode step two-lane pipelining"),
    ("V41_MS_PIPELINE_MIN_ROWS", P, "decode step two-lane threshold / K pricing"),
    ("V41_MS_PREFILL_BURST_MIN_MS", P, "scheduler prefill burst"),
    ("V41_MS_PREFILL_BURST_MS", P, "scheduler prefill burst"),
    ("V41_MS_PREFILL_JOBS", P, "concurrent prefill scratch states; scheduling only"),
    ("V41_MS_PROFILE", P, "multistream profiling"),
    ("V41_MS_PROFILE_EVERY", P, "multistream profiling interval"),
    ("V41_MS_SLOTS", P, "arena slot count; allocation only"),
    ("V41_MS_SPEC_LANES", P, "decode spec verify lanes"),
    ("V41_MS_SPEC_LANES_FILE", P, "decode spec verify lanes (file)"),
    ("V41_MS_STAGGER", P, "decode step lane driver"),
    ("V41_MS_STARVE_S", P, "scheduler starvation guard"),
    ("V41_MTP_EXPERT_STATS", P, "drafter expert stats; observe only"),
    ("V41_MTP_GRAPH", P, "drafter forward graph replay; drafting only"),
    ("V41_MTP_KV_QUANT", N, "drafter ring KV quantizer; changes seeded ring bytes"),
    ("V41_MTP_MOE_GROUPED", P, "drafter grouped MoE; drafting only"),
    ("V41_MULTISTREAM", N, "selects multistream vs legacy prefill driver (chunking/lanes differ)"),
    ("V41_MXFP4_PAIR_WARPS", N, "MXFP4 pair kernel WG geometry; 1-row passes may run it (cheap N)"),
    ("V41_ORACLE_BINS", P, "test/fixture only (tests/)"),
    ("V41_PAGED_EXPERTS", N, "paged vs fully resident experts changes MoE path and box split"),
    ("V41_PAGER", P, "test/fixture only (tests/)"),
    ("V41_PAGER_BATCH_MISS", P, "single-token decode batched misses; IO only"),
    ("V41_PAGER_COALESCE", N, "box-1 coalesced expert read; same class as B2 corruption (cheap N)"),
    ("V41_PAGER_COALESCE_CHECK", P, "byte-compare of coalesced reads; logs only"),
    ("V41_PAGER_DBG", P, "pager debug logging"),
    ("V41_PAGER_DECODE_FRAC", N, "box-1 pool split sets hot-set capacity (ownership, MoE box split)"),
    ("V41_PAGER_GPU_REPACK", P, "box-1 MXFP4 byte repack on GPU vs CPU; byte permutation, same bytes"),
    ("V41_PAGER_MISS_PAR", P, "box-1 concurrent miss reads; IO only"),
    ("V41_PAGER_MISS_THREADS", P, "box-1 miss read threads; IO only"),
    ("V41_PAGER_NOGRAPH", P, "single-token decode pager staging (forward_layer)"),
    ("V41_PAGER_POOL_GB", N, "box-1 pool size sets hot-set capacity (ownership, MoE box split)"),
    ("V41_PAGER_READ_BATCH", P, "box-1 read batch staging; IO only"),
    ("V41_PAGER_READ_THREADS", P, "box-1 read threads; IO only"),
    ("V41_PAGER_RESIDENT", P, "single-token decode residency debug logging"),
    ("V41_PAGER_SLOTS", N, "box-1 pool slots set hot-set capacity (ownership, MoE box split)"),
    ("V41_PAGER_STRIDE", N, "box-1 window stride sets decode LRU / hot-set capacity"),
    ("V41_PAGER_SYNC_AFTER_ENSURE", N, "device drain after paging; race-sensitive MoE weights (cheap N)"),
    ("V41_PAGER_SYNC_IGPU", N, "iGPU drain before eviction; =0 risks wrong expert weights"),
    ("V41_PAGER_UNION", N, "union vs dense prefill paging; box-1 residency decides box (cheap N)"),
    ("V41_PAGER_WINDOWS", N, "box-1 dense windows set decode LRU / hot-set capacity"),
    ("V41_PARTITION_BOX1_SHARE", N, "T2 hash partition share decides which box computes each expert"),
    ("V41_PERFETTO_OUT", P, "perfetto trace output; observe only"),
    ("V41_PICK_TRACE", P, "router pick trace; observe only"),
    ("V41_PREFILL_F16_REPLIES", N, "box-2 prefill MoE partials returned as f16 (not bit-exact)"),
    ("V41_PREFILL_F32_MATVEC", N, "f32 matvec vs f16 WMMA projection arm by lane size"),
    ("V41_PREFILL_LANE_DEBUG", P, "prefill lane split logging"),
    ("V41_PREFILL_LOGITS_DUMP", P, "prefill logits dump; observe only"),
    ("V41_PREFILL_PRESUBMIT", N, "defers prefill shared expert past remote submit (unmeasured on KLD)"),
    ("V41_PREFILL_READAHEAD", P, "page-cache readahead hints; IO only"),
    ("V41_PREFILL_READAHEAD_DEPTH", P, "page-cache readahead hint depth; IO only"),
    ("V41_PREFILL_SCAN_SLOTS", N, "bounds prefill misses in box-1 pool; residency decides box (cheap N)"),
    ("V41_PREFILL_SINGLE_LANE_MAX", N, "single-lane vs two-lane prefill cut changes lane sizes"),
    ("V41_PREFILL_UNIFIED_POOL", N, "prefill on sparse LRU residency vs dense windows (MoE path)"),
    ("V41_PROBE_DUMP", P, "probe dump path; observe only"),
    ("V41_PROBE_FPRINT", P, "legacy verify probe fingerprints; observe only"),
    ("V41_PROBE_SRC", P, "probe dump layers; observe only"),
    ("V41_PROFILE_KERNEL_STAGES", P, "kernel stage profiling; observe only"),
    ("V41_PUSH_XQ", N, "iGPU uses dGPU-quantised Q8_K activations (cross-device; GPU-test only)"),
    ("V41_Q8_QUANT_GRID_PAD", N, "Q8 activation quantize grid pad (claimed bit-identical; cheap N)"),
    ("V41_Q8_QUANT_WAVE", N, "Q8 activation quantize wave kernel (all-NaN block differs)"),
    ("V41_RB_PACK", P, "router readback via pack kernel vs async copies; same bytes"),
    ("V41_RB_PACK_MAX_ROWS", P, "router readback pack row cap; copy mechanism, same bytes"),
    ("V41_REMOTE_ADDR", N, "box 2 present or not: two-box MoE split vs local-only"),
    ("V41_REMOTE_BATCHED_MULTI", N, "box-2 batched vs decode chain for b 2..4 (different sum order)"),
    ("V41_REMOTE_BUSY_POLL_US", P, "socket busy-poll timing"),
    ("V41_REMOTE_DBG", P, "remote debug logging"),
    ("V41_REMOTE_NOMASK", N, "debug mask override alters remote picks (forward_layer path; cheap N)"),
    ("V41_REMOTE_QUICKACK", P, "socket quickack"),
    ("V41_REMOTE_SPLIT", N, "two-box split mode for prefill MoE (exclude/add partial)"),
    ("V41_REMOTE_SPLIT_DECODE", P, "single-token decode remote split mode (forward_layer)"),
    ("V41_REPLAY_F16X", N, "CED replay q_b/wo projections f16x WMMA vs dp4a (not bit-exact)"),
    ("V41_REPLAY_OFFLOAD", N, "CED replay decoder MoE computed wholly on box 2"),
    ("V41_REQUIRE_FIXTURES", P, "test/fixture only (cfg(test))"),
    ("V41_RESET_ZERO", N, "zero-fills KV and compressor caches at reset (debug that alters state)"),
    ("V41_RMS_FAST", N, "RMS norm kernel variant (n = 512/1280/5120, prefill included)"),
    ("V41_ROUTER_ALTS", N, "router top-k emits alternates (kernel variant; picks claimed identical)"),
    ("V41_ROUTER_MV_H20", N, "router matvec kernel variant (replay/small b)"),
    ("V41_ROUTER_WMMA", N, "router logits WMMA vs fp32 matvec by batch size"),
    ("V41_ROUTE_PROBE", P, "route probe dump; observe only"),
    ("V41_SHARED_FUSED", N, "fused shared-expert kernel for lanes <= 5 rows (short prefill lanes)"),
    ("V41_SINGLE_LANE_AB", N, "sets the prefill single-lane threshold at runtime (A/B)"),
    ("V41_SLACK_PROBE", P, "drafter slack timing probe"),
    ("V41_SMALL_B_CATCHALL_AB", N, "sets the small-B catch-all threshold at runtime (A/B)"),
    ("V41_SMALL_B_CATCHALL_DET", N, "small-B catch-all hands every pick to box 2"),
    ("V41_SMALL_B_CATCHALL_HALF", N, "small-B catch-all layer half (box split)"),
    ("V41_SMALL_B_CATCHALL_MAX", N, "small-B catch-all threshold (box split for small lanes)"),
    ("V41_SMALL_B_DENSE_DP4A", N, "dp4a GEMV vs WMMA for small-b dense projections"),
    ("V41_SMALL_B_DENSE_MAX", N, "dp4a/WMMA crossover for small-b dense projections"),
    ("V41_SPARSE_REMAP_SYNC", N, "sync against shared remap race (wrong expert weights)"),
    ("V41_SPARSE_VERIFY_RESIDENCY", N, "=0 reverts sparse residency path (MoE path, residency box split)"),
    ("V41_SUB", P, "substitution / cache prior mode: decode (Arena) rows only (design E2)"),
    ("V41_SUB_ADMIT", P, "miss substitution admissions: decode (Arena) rows only"),
    ("V41_SUB_ADMIT_GATE", P, "miss substitution: decode (Arena) rows only"),
    ("V41_SUB_DRY", P, "miss substitution dry run: decode (Arena) rows only"),
    ("V41_SUB_INCOMING", P, "miss substitution: decode (Arena) rows only"),
    ("V41_SUB_LAMBDA", P, "cache prior lambda: decode (Arena) rows only"),
    ("V41_SUB_LAMBDA_FILE", P, "cache prior lambda (file): decode (Arena) rows only"),
    ("V41_SUB_MAX_W", P, "miss substitution: decode (Arena) rows only"),
    ("V41_SUB_MIN_RANK", P, "miss substitution: decode (Arena) rows only"),
    ("V41_SUB_PENDING", P, "miss substitution: decode (Arena) rows only"),
    ("V41_SUB_PROTECT", P, "miss substitution: decode (Arena) rows only"),
    ("V41_SWA_MIXED", N, "ratio-0 layers on batched WMMA mixed attention vs attention_swa_batched"),
    ("V41_T2_CATCHALL", N, "catch-all: box-1 misses to box 2; =2 constant split (box split)"),
    ("V41_T2_PARTITION", N, "T2 partition decides which box computes each expert"),
    ("V41_TOKENIZER_JSON", P, "test/fixture only (tests/)"),
    ("V41_TOPK_SELECT_ILP", N, "indexer top-k select kernel variant"),
    ("V41_TOPK_WFRED", N, "router top-k kernel variant for small b (prefill tails)"),
    ("V41_VDP_TRACE", P, "legacy verify decode-path trace logging"),
    ("V41_VERIFY_BATCHED", P, "legacy verify probe mode (decode)"),
    ("V41_VERIFY_DECODE_ATTN", N, "decode attention chain per row for b <= 16 (short prefill lanes)"),
    ("V41_VERIFY_DECODE_MOE", N, "decode MoE kernels for b <= 16 (short prefill lanes)"),
    ("V41_VERIFY_DECODE_PATH", P, "legacy verify probe path (decode)"),
    ("V41_VERIFY_PROBE", P, "legacy verify probe (decode, rolled back)"),
    ("V41_VICTIM_CACHE", N, "box-1 decode LRU fill policy; residency decides box (cheap N)"),
    ("V41_WINDOW_DBG", P, "SWA window debug logging"),
    ("V41_XCHECK_POISON", N, "debug poison row in legacy verify probe alters computation (cheap N)"),
    ("V41_XCHECK_ROWS", P, "legacy verify cross-check probe (decode)"),
    ("VIT_GEMM", N, "vision tower WMMA vs scalar GEMM changes image-row tower outputs"),
    ("VIT_PROFILE", P, "vision tower profiling"),
];

/// The knob hash over `get(name)` for every numerics knob, in table order.
/// An unset knob and a knob set to its default hash differently: that only
/// over-separates (a new pair, no cold start).
pub fn knob_hash_with(get: impl Fn(&str) -> Option<String>) -> [u8; 16] {
    let mut h = blake3::Hasher::new_derive_key("deepstrix kvstore v1 knob hash");
    for (name, class, _) in KNOBS {
        if *class != N {
            continue;
        }
        h.update(name.as_bytes());
        match get(name) {
            Some(v) => {
                h.update(&[1]);
                h.update(&(v.len() as u32).to_le_bytes());
                h.update(v.as_bytes());
            }
            None => {
                h.update(&[0]);
            }
        }
    }
    let mut out = [0u8; 16];
    out.copy_from_slice(&h.finalize().as_bytes()[..16]);
    out
}

/// The knob hash of this process's environment.
pub fn knob_hash_from_env() -> [u8; 16] {
    knob_hash_with(|k| std::env::var(k).ok())
}

/// The numerics knobs that are set, for the `kv.gen_new` log line.
pub fn numerics_knobs_set() -> Vec<(&'static str, String)> {
    KNOBS
        .iter()
        .filter(|(_, c, _)| *c == N)
        .filter_map(|(n, _, _)| std::env::var(n).ok().map(|v| (*n, v)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};

    const PREFIXES: [&[u8]; 3] = [b"V41_", b"DEEPSTRIX_", b"VIT_"];
    const SCANNED: [&str; 4] = ["crates/v4flash-kernels", "crates/v4flash-core", "crates/v4flash-vision", "crates/deepstrix-server"];

    /// Every `"NAME"` literal whose whole content is a knob-shaped name.
    fn literals(src: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < src.len() {
            if src[i] == b'"' {
                let rest = &src[i + 1..];
                if PREFIXES.iter().any(|p| rest.starts_with(p)) {
                    let n = rest.iter().take_while(|b| b.is_ascii_alphanumeric() || **b == b'_').count();
                    if rest.get(n) == Some(&b'"') {
                        out.push(String::from_utf8_lossy(&rest[..n]).into_owned());
                        i += n + 2;
                        continue;
                    }
                }
            }
            i += 1;
        }
        out
    }

    fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if e.file_type().is_ok_and(|t| t.is_dir()) {
                if !name.starts_with("target") && !name.starts_with('.') {
                    rust_files(&p, out);
                }
            } else if name.ends_with(".rs") {
                out.push(p);
            }
        }
    }

    #[test]
    fn literal_scanner() {
        let src = br#"let a = env("V41_FOO"); b("DEEPSTRIX_X_Y2") "V41_" "V41_not a knob" "VIT_GEMM"x "V41_A=1""#;
        assert_eq!(literals(src), vec!["V41_FOO", "DEEPSTRIX_X_Y2", "V41_", "VIT_GEMM"]);
    }

    #[test]
    fn table_is_sorted_and_unique() {
        for w in KNOBS.windows(2) {
            assert!(w[0].0 < w[1].0, "KNOBS not sorted/unique at {} / {}", w[0].0, w[1].0);
        }
    }

    /// G5 / 4.4: every knob-name literal in the scanned crates is classified,
    /// and every classified name still occurs.
    #[test]
    fn knob_literals_are_classified() {
        let ws = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let this = Path::new(file!()).file_name().unwrap();
        let mut found: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for c in SCANNED {
            let mut files = Vec::new();
            rust_files(&ws.join(c), &mut files);
            assert!(!files.is_empty(), "no sources under {c}");
            for f in files {
                // The table itself names every knob.
                if f.file_name() == Some(this) && f.parent().is_some_and(|p| p.ends_with("kvstore")) {
                    continue;
                }
                let src = std::fs::read(&f).unwrap();
                for lit in literals(&src) {
                    let rel = f.strip_prefix(ws).unwrap_or(&f).display().to_string();
                    found.entry(lit).or_default().insert(rel);
                }
            }
        }
        let table: BTreeSet<&str> = KNOBS.iter().map(|k| k.0).collect();
        let missing: Vec<String> = found
            .iter()
            .filter(|(n, _)| !table.contains(n.as_str()))
            .map(|(n, f)| format!("{n} (in {})", f.iter().cloned().collect::<Vec<_>>().join(", ")))
            .collect();
        assert!(
            missing.is_empty(),
            "knob literals not classified in kvstore/knobs.rs (N if they can change prefill numerics, \
             else P; see the module doc):\n  {}",
            missing.join("\n  ")
        );
        let stale: Vec<&str> = table.iter().filter(|n| !found.contains_key(**n)).copied().collect();
        assert!(stale.is_empty(), "classified knobs that no longer occur anywhere (remove them): {stale:?}");
    }

    #[test]
    fn knob_hash_tracks_numerics_values_only() {
        let base = knob_hash_with(|_| None);
        let n_knob = KNOBS.iter().find(|k| k.1 == N).unwrap().0;
        let p_knob = KNOBS.iter().find(|k| k.1 == P).unwrap().0;
        assert_ne!(base, knob_hash_with(|k| (k == n_knob).then(|| "1".into())));
        assert_ne!(
            knob_hash_with(|k| (k == n_knob).then(|| "1".into())),
            knob_hash_with(|k| (k == n_knob).then(|| "2".into()))
        );
        assert_eq!(base, knob_hash_with(|k| (k == p_knob).then(|| "1".into())), "a plain knob is not hashed");
        // The empty string is a value, distinct from unset.
        assert_ne!(base, knob_hash_with(|k| (k == n_knob).then(String::new)));
        for name in ["V41_PREFILL_F32_MATVEC", "V41_MS_CHUNK_ROWS", "VIT_GEMM", "V41_T2_CATCHALL", "V41_MHC_GEMM_NARROW"] {
            assert_eq!(KNOBS.iter().find(|k| k.0 == name).map(|k| k.1), Some(N), "{name} must be numerics (4.4)");
        }
    }
}
