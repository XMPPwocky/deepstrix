//! Heterogeneous dGPU + iGPU orchestrator (M13).
//!
//! The single-device baseline in [`crate::forward`] stays untouched as
//! the regression reference. This module builds a parallel implementation
//! that splits the V4-Flash per-token forward across two HIP devices:
//!
//! * **dGPU (9070 XT, gfx1201)** — attention, mHC, shared expert, head.
//!   Holds attention LoRAs, mHC weights, shared expert, globals, KV cache,
//!   compressor state. ~9 GiB resident weights.
//! * **iGPU (Strix Halo, gfx1151)** — router + routed MoE (256 experts × 6
//!   selected). Holds routed expert weights + router weights. ~52 GiB
//!   resident.
//!
//! M13.1 exit criterion: a serial het execution (`ExecMode::HetSingleStream`,
//! one `.synchronize()` per kernel, no event-based overlap) passes the
//! forward_full_logits oracle. M13.4 turns on real concurrency via
//! `ExecMode::HetParallel`. M13.5 migrates the compressor to iGPU.

pub mod batch_scratch;
pub mod dispatch;
pub mod engine;
pub mod forward_head;
pub mod forward_layer;
pub mod forward_prefill;
pub mod graph_cache;
pub mod perfetto;
pub mod prefill_stats;
pub mod scratch;
pub mod state;
pub mod sync;
pub mod trace;
pub mod weights;

pub use batch_scratch::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, BatchScratch, B_MAX,
};
pub use prefill_stats::{LayerStats, PerChunkReuse, PrefillStats};
pub use engine::{DeviceEngine, ExecMode, HeterogeneousEngine, SampleMode};
pub use scratch::{DgpuScratch, IgpuScratch};
pub use state::{HetCompressorState, HetLayerState, HetModelState};
pub use weights::{
    DgpuLayerWeights, HetGlobalWeights, HetModelWeights, IgpuLayerWeights,
};
