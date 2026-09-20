//! Single OS-thread worker that owns the V4-Flash inference engine and
//! per-session state.
//!
//! Phase 2 model: the worker emits a `WorkerEvent` stream per request.
//! Both streaming and non-streaming HTTP handlers consume the same
//! event stream — the non-streaming handler just accumulates events
//! into a single response before returning.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use color_eyre::eyre::{self, eyre, WrapErr};
use tokio::sync::{mpsc, oneshot};
use v4flash_core::tokenizer::BpeVocab;
#[cfg(any(test, not(feature = "v41")))]
use v4flash_core::MappedGguf;
#[cfg(feature = "v41")]
use v4flash_core::V41HfWeights;
use v4flash_core::WeightSrc;
use v4flash_hip::Device;
use v4flash_kernels::config::{COMPRESS_RATIOS, HC_DIM, N_EMBD, N_HC, N_VOCAB};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, DgpuScratch, ExecMode,
    HetModelState, HetModelWeights,
    HeterogeneousEngine, IgpuScratch, SampleMode, B_MAX,
};
use v4flash_kernels::sampler::SamplerRng;
use v4flash_kernels::RopeParams;

use crate::embed::{build_gpt2_byte_decoder, embed_lookup, gpt2_decode_token};
use crate::rope_for_layer;
use crate::snapshot::{self, ModelFingerprint, SnapshotIndex};
use crate::tokens::{
    is_turn_end, TOK_ASSISTANT, TOK_EOS, TOK_THINK_BEGIN, TOK_THINK_END, TOK_USER,
};
use crate::vision_prompt::{
    shift_spans, span_hash_at, synthetic_token_bytes, ImageSpan, PreparedImage,
};

/// V4.1 Engram context: the n-gram hasher and one table handle per Engram layer
/// (1 and 14). The tables themselves are 189 GiB on SSD and are never resident —
/// each token gathers `ENGRAM_COLS` rows per layer through `gather_position`.
#[cfg(feature = "v41")]
pub struct EngramCtx {
    pub hasher: v4flash_core::EngramHash,
    pub tables: Vec<v4flash_core::EngramTable>,
    /// Compressed ids of every token forwarded so far, in position order. The
    /// hash for position `p` reads the last four entries, so this has to track
    /// the real sequence rather than just the current token.
    pub compressed: Vec<i32>,
}

#[cfg(feature = "v41")]
impl EngramCtx {
    /// Append `token` and gather its Engram rows, one `Vec<f32>` per Engram layer
    /// in `ENGRAM_LAYERS` order. `pos` must be the token's KV position.
    pub(crate) fn rows_for(
        &mut self,
        src: &v4flash_core::SafetensorsDir,
        token: i32,
        pos: u32,
    ) -> eyre::Result<Vec<Vec<f32>>> {
        // Positions must arrive in order; a rewind (snapshot reuse / retry) just
        // truncates back to the branch point.
        let p = pos as usize;
        self.compressed.truncate(p);
        if self.compressed.len() != p {
            return Err(eyre!(
                "engram: position {p} but {} tokens tracked",
                self.compressed.len()
            ));
        }
        let c = self.hasher.compress(token);
        self.compressed.push(c);
        let hashes = self.hasher.hash_ids(&self.compressed, p);
        let mut out = Vec::with_capacity(self.tables.len());
        for (li, tbl) in self.tables.iter().enumerate() {
            let mut rows = vec![0f32; v4flash_kernels::config::ENGRAM_IN as usize];
            // A dead (image-span) position takes no Engram contribution: all-zero
            // rows → wkv (no bias) → value 0 → `h += gate · 0`. Exactly the
            // reference's zeroed gate (`engram_mask`), without touching the kernel.
            if c != v4flash_core::engram_hash::DEAD {
                tbl.gather_position(src, &hashes[li], &mut rows)?;
            }
            out.push(rows);
        }
        Ok(out)
    }

    /// Rebuild the compressed-id sequence from a token list, for a KV snapshot
    /// restore. The sequence is a pure function of the token ids
    /// (`hasher.compress` per token, image tokens -> DEAD), so a restore that
    /// skips the prefill can regenerate exactly what that prefill would have
    /// appended. Without this, the first `rows_for*` after a restore sees
    /// `compressed.len() != pos` and the request fails -- which is why snapshot
    /// reuse was disabled under v41 (17 s of prefill + CED replay per request
    /// for an 86-token prompt, measured 2026-09-17).
    fn rebuild(&mut self, tokens: &[i32]) {
        self.compressed.clear();
        self.compressed.reserve(tokens.len());
        for &t in tokens {
            let c = self.hasher.compress(t);
            self.compressed.push(c);
        }
    }

    /// Batched twin of [`Self::rows_for`]: append `tokens` (starting at KV position
    /// `pos0`) and gather their Engram rows for the whole chunk.
    ///
    /// Returns one flattened `[B * ENGRAM_IN]` buffer per Engram layer, which is
    /// exactly what `stage_engram_rows_batch` consumes. Batched prefill needs this
    /// because the per-token path it replaced staged rows one token at a time.
    fn rows_for_chunk(
        &mut self,
        src: &v4flash_core::SafetensorsDir,
        tokens: &[i32],
        pos0: u32,
    ) -> eyre::Result<Vec<Vec<f32>>> {
        let ein = v4flash_kernels::config::ENGRAM_IN as usize;
        let p0 = pos0 as usize;
        self.compressed.truncate(p0);
        if self.compressed.len() != p0 {
            return Err(eyre!(
                "engram: chunk at position {p0} but {} tokens tracked",
                self.compressed.len()
            ));
        }
        let mut out = vec![vec![0f32; tokens.len() * ein]; self.tables.len()];

        // BATCHED GATHER. This used to call `gather_position` per (token, layer),
        // and `gather_position` passes `threads = ENGRAM_COLS = 24` for exactly 24
        // ids — so `per = 1` and it spawned **24 scoped OS threads each doing one
        // row** (2 preads + 256 MACs), twice per token. MEASURED 2026-09-13: 48
        // spawns/token = 289,056 for a 6k prompt, and a replica of that shape cost
        // **3.17 s warm** of which **3.01 s was pthread create/join with zero I/O**.
        // It is LINEAR at ~0.53 ms/token, so it does not amortise — a 100K prefill
        // would spend ~53 s here before the first GPU kernel launches. It sits
        // outside `prefill_start`, which is why it never showed up in any prefill
        // stage timing.
        //
        // Now: hash the whole chunk first, then issue ONE gather per layer per RUN
        // of live positions — 289,056 spawns -> ~64. Bit-identical by construction:
        // `hash_ids` only ever reads `c[pos - s]` (strictly backward, engram_hash.rs
        // :148), so pre-pushing the whole chunk cannot change any position's hashes;
        // the same ids land in the same destination slots in the same order through
        // the same `gather`; DEAD positions stay all-zero because runs skip them.
        debug_assert_eq!(ein, v4flash_core::engram_hash::ENGRAM_COLS * v4flash_core::engram_hash::ENGRAM_ROW_DIM, "ENGRAM_IN must be COLS x ROW_DIM");
        let mut hashes: Vec<Option<[[i64; v4flash_core::engram_hash::ENGRAM_COLS]; v4flash_core::engram_hash::ENGRAM_LAYERS]>> = Vec::with_capacity(tokens.len());
        for (i, &tok) in tokens.iter().enumerate() {
            let c = self.hasher.compress(tok);
            self.compressed.push(c);
            // Image-span rows (synthetic ids → DEAD) stay all-zero: no Engram
            // contribution, and `hash_ids` blocks later look-backs at them.
            hashes.push(if c == v4flash_core::engram_hash::DEAD {
                None
            } else {
                Some(self.hasher.hash_ids(&self.compressed, p0 + i))
            });
        }
        // One sustained, deep gather beats 24 one-row threads: this box's NVMe
        // peaks around 32 readers (see safetensors.rs), and the old shape put a
        // join barrier after every single position.
        const GATHER_THREADS: usize = 32;
        let mut i = 0usize;
        while i < tokens.len() {
            if hashes[i].is_none() {
                i += 1;
                continue;
            }
            let start = i;
            while i < tokens.len() && hashes[i].is_some() {
                i += 1;
            }
            for (li, tbl) in self.tables.iter().enumerate() {
                let flat: Vec<i64> = hashes[start..i]
                    .iter()
                    .flat_map(|h| h.as_ref().expect("run contains only live positions")[li])
                    .collect();
                tbl.gather(src, &flat, &mut out[li][start * ein..i * ein], GATHER_THREADS)?;
            }
        }
        Ok(out)
    }
}

/// Drive one decode token, routing through the M7 expert pager when it is active.
///
/// Written as a macro rather than a method because the pager and the scratch
/// buffers are disjoint fields of the same `WorkerState`: a method would have to
/// borrow all of `self` mutably and conflict with itself.
#[cfg(feature = "v41")]
macro_rules! forward_one {
    ($state:expr, $residual:expr, $pos:expr, $tok:expr) => {
        if let Some(pg) = $state.pager.as_mut() {
            // Token boundary: nothing is reading the pool, so admit the
            // background-read experts now (`V41_B1_PREFETCH`).
            pg.drain_prefetched()?;
            // Gather this token's Engram rows before the forward: the tables are
            // SSD-resident and the gather needs the same HF source the pager owns.
            // Timed into `phase::CALLER_ENGRAM_NS` -> `engram_us` on the next
            // `het.token.summary`: 2 layers x 24 spawned threads x 2 preads
            // against a 98 GB NVMe table, all on this thread's critical path.
            let engram_rows = match $state.engram.as_mut() {
                Some(ec) => {
                    let t = std::time::Instant::now();
                    let rows = ec.rows_for(pg.raw(), $tok, $pos)?;
                    v4flash_kernels::het::trace::phase::add(
                        &v4flash_kernels::het::trace::phase::CALLER_ENGRAM_NS,
                        t.elapsed().as_nanos() as u64,
                    );
                    Some(rows)
                }
                None => None,
            };
            $state.engine.forward_token_paged(
                &mut $state.dgpu_scratch,
                &mut $state.igpu_scratch,
                &mut $state.state,
                &$state.weights,
                &$residual,
                $pos,
                $tok,
                pg,
                engram_rows.as_deref(),
            )
        } else {
            $state.engine.forward_token(
                &mut $state.dgpu_scratch,
                &mut $state.igpu_scratch,
                &mut $state.state,
                &$state.weights,
                &$residual,
                $pos,
                $tok,
            )
        }
    };
}

#[cfg(not(feature = "v41"))]
macro_rules! forward_one {
    ($state:expr, $residual:expr, $pos:expr, $tok:expr) => {
        $state.engine.forward_token(
            &mut $state.dgpu_scratch,
            &mut $state.igpu_scratch,
            &mut $state.state,
            &$state.weights,
            &$residual,
            $pos,
            $tok,
        )
    };
}

/// Verify-vs-decode argmax agreement (see `V41_DSPARK_XCHECK`).
#[cfg(feature = "v41")]
static XCHECK_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "v41")]
static XCHECK_TOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "v41")]
static XCHECK_COS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
#[cfg(feature = "v41")]
static XCHECK_COS_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// KL(decode || verify) in nats x 1e6, summed over row-0 softmax distributions.
/// The principled "same distribution" measure — cosine on raw logits is
/// scale-sensitive and ignores the softmax the tokens are actually drawn from.
static XCHECK_KLD: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// `V41_SMALL_B_CATCHALL_AB=<max>`: interleave the small-B catch-all ON/OFF by
/// step parity and bucket the xcheck by arm. Under the SHADOW probe this is a
/// true control — decode drives the text, so the arm cannot change what is
/// generated, only whether the batched verify reproduces it.
/// `V41_SINGLE_LANE_AB=<max>`: interleave the prefill single-lane threshold
/// ON/OFF by step parity, bucketing the xcheck by arm. Tests whether the
/// batched verify's divergence from decode comes from the two-lane split.
fn single_lane_ab() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("V41_SINGLE_LANE_AB").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    })
}

/// `V41_XCHECK_POISON=1`: see the row-independence probe in the verify probe.
fn xcheck_poison() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("V41_XCHECK_POISON").as_deref() == Ok("1"))
}
/// An ordinary in-vocabulary token, deliberately unrelated to the context.
const POISON_TOKEN: i32 = 1000;

fn small_b_catchall_ab() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("V41_SMALL_B_CATCHALL_AB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}
/// Per-arm [off, on]: agreements, totals, summed cosine x1e6, cosine count,
/// summed probe wall-microseconds.
/// Arms 0/1 for the flag A/Bs; when bucketing by probe WIDTH the arm IS the
/// width, so this is indexed by `k` up to `XCHECK_ARMS - 1`.
const XCHECK_ARMS: usize = 17;
static XCHECK_ARM_OK: [std::sync::atomic::AtomicU64; XCHECK_ARMS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; XCHECK_ARMS];
static XCHECK_ARM_TOT: [std::sync::atomic::AtomicU64; XCHECK_ARMS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; XCHECK_ARMS];
static XCHECK_ARM_COS: [std::sync::atomic::AtomicI64; XCHECK_ARMS] =
    [const { std::sync::atomic::AtomicI64::new(0) }; XCHECK_ARMS];
static XCHECK_ARM_COS_N: [std::sync::atomic::AtomicU64; XCHECK_ARMS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; XCHECK_ARMS];
static XCHECK_ARM_US: [std::sync::atomic::AtomicU64; XCHECK_ARMS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; XCHECK_ARMS];

/// Everything the DSpark drafter needs, loaded once at startup.
#[cfg(feature = "v41")]
pub struct MtpCtx {
    /// Layers + entry projection, iGPU.
    pub w: v4flash_kernels::het::weights::MtpWeights,
    /// Final norm, markov head, confidence — dGPU, beside the tied head.
    pub xw: v4flash_kernels::het::weights::MtpExitWeights,
    pub state: v4flash_kernels::het::mtp::MtpState,
    pub exit: v4flash_kernels::het::mtp::MtpExit,
    pub capture: v4flash_kernels::het::mtp::MtpCapture,
    /// Markov EMBEDDING half, host-side like `token_embd` (M57).
    pub markov_embd: Vec<u8>,
    pub markov_dtype: v4flash_core::gguf::GgufType,
    /// `embed_lookup` of `dspark_noise_token_id`, `[HC_DIM]`. Constant.
    pub noise_row: Vec<f32>,
    /// Shadow scoring: drafts produced, keyed by the position the FIRST draft
    /// predicts, plus the tokens actually generated from `actual_base` on.
    /// Accept mode: drafts produced for the CURRENT head token, the tokens a
    /// verify has already confirmed AND ingested into KV (so the decode loop
    /// must not forward them again), and the `main_hidden` of whichever row
    /// became the new head.
    pub pending: Option<[i32; v4flash_kernels::het::mtp::MTP_BLOCK]>,
    /// `DSparkConfidenceHead` score for each pending draft, from the same
    /// drafter pass. Calibrated against actual acceptance by
    /// `dspark_stats::record_conf`; gates `k` once a threshold is set.
    pub pending_conf: [f32; v4flash_kernels::het::mtp::MTP_BLOCK],
    pub confirmed: std::collections::VecDeque<i32>,
    /// Upcoming tokens (including the current `next`) that a verify already
    /// appended to KV — the decode loop must advance `pos` past them without
    /// forwarding again.
    pub ingested: usize,
    /// The model's own correction after the accepted prefix. Not in KV.
    pub next_after: Option<i32>,
    pub main_hidden: Vec<f32>,
    pub accept_steps: u64,
    pub accept_tokens: u64,
    pub drafts: Vec<(u32, [i32; v4flash_kernels::het::mtp::MTP_BLOCK])>,
    /// Same batches with the markov bias omitted — the ablation.
    pub drafts_plain: Vec<(u32, [i32; v4flash_kernels::het::mtp::MTP_BLOCK])>,
    pub actual: Vec<i32>,
    pub actual_base: u32,
}

/// Per-request input.
pub struct GenerateReq {
    pub tokens: Vec<i32>,
    /// Vision-Exp: the preprocessed images spliced into `tokens`, in
    /// stream order. Empty for text-only requests. `images[i]` is the
    /// image whose block sits at `image_spans[i]`.
    pub images: Vec<PreparedImage>,
    /// `[IMAGE_START ..= IMAGE_END]` spans, in `tokens` index space
    /// (NOT KV positions — the worker rebases them per prefill call).
    pub image_spans: Vec<ImageSpan>,
    pub max_new: usize,
    pub temperature: f32,
    pub min_p_rel: f32,
    /// Nucleus cutoff in (0, 1]. `1.0` = no truncation (the pre-top_p
    /// sampler chain, bit-for-bit). The HTTP layer resolves the request's
    /// `top_p` (or the server default) and clamps it before it gets here,
    /// and `finish_decode` re-clamps defensively — a `0.0` (the field's
    /// natural zero value) or a NaN degrades instead of failing the turn.
    pub top_p: f32,
    pub seed: u64,
}

/// Generation outcome — sent as the final `WorkerEvent::Done`.
#[derive(Debug, Clone, Copy)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_openai(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// Events emitted by the worker during a generation.
#[derive(Debug)]
pub enum WorkerEvent {
    /// One token's raw decoded bytes, with the token's id (so the
    /// handler's DSML scanner can react to TOK_DSML structurally) and
    /// a flag for whether we're inside a `<think>…</think>` block.
    /// Bytes are NOT UTF-8-validated — BPE can split a multi-byte
    /// UTF-8 character (e.g. `─` = E2 94 80) across tokens, so
    /// per-token bytes are often a fragment. The handler maintains
    /// the cross-chunk UTF-8 buffer for JSON-safe output.
    Chunk {
        token_id: i32,
        bytes: Vec<u8>,
        reasoning: bool,
    },
    /// Generation finished.
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
        finish: FinishReason,
    },
    /// Fatal error mid-generation.
    Error(String),
}

pub enum EngineRequest {
    /// Stream of generation events. `tx` is closed by the worker on
    /// completion (Done) or error (Error).
    Generate {
        req: GenerateReq,
        tx: mpsc::Sender<WorkerEvent>,
        /// Optional sessionId hint from the client. Used as a fast-path
        /// for the on-disk snapshot lookup.
        session_id: Option<String>,
        /// Flipped by the HTTP handler when the client disconnects.
        /// The worker polls it between forward_token calls and breaks
        /// out as soon as it sees true.
        cancel: Arc<AtomicBool>,
    },
    /// Save any dirty live state to disk and shut down cleanly.
    Shutdown {
        ack: oneshot::Sender<()>,
    },
}

/// Maximum number of generation requests queued behind the one
/// currently executing. Each request takes seconds-to-minutes; a deep
/// queue would just be requests timing out client-side anyway. When
/// full, `submit()` returns `SubmitError::Busy` which the handler maps
/// to HTTP 503 + Retry-After.
pub const ENGINE_QUEUE_CAP: usize = 8;

#[derive(Debug)]
pub enum SubmitError {
    /// Engine queue is full (more than ENGINE_QUEUE_CAP requests
    /// already waiting). Client should back off and retry.
    Busy,
    /// Engine worker thread has terminated. Fatal — server is gone.
    WorkerDead,
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::Busy => write!(
                f,
                "engine busy: queue depth {} reached, retry in a moment",
                ENGINE_QUEUE_CAP
            ),
            SubmitError::WorkerDead => write!(f, "engine worker channel closed"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Forward-progress signal for the hang watchdog and /readyz.
///
/// The worker thread `pet()`s after each meaningful progress unit:
/// one decoded token emitted, or one prefill chunk completed. The
/// watchdog thread observes `last_pet_ms` and aborts the process if
/// it goes stale while `inflight` is set.
///
/// All fields are Arc-wrapped so the watchdog and /readyz handler
/// can share them with the worker by cloning the WorkerProgress
/// (which only clones two Arcs).
#[derive(Clone, Default)]
pub struct WorkerProgress {
    /// Unix ms of last forward progress. 0 until first pet.
    pub last_pet_ms: Arc<AtomicI64>,
    /// True between request-begin and request-end. The watchdog only
    /// fires when this is set — if no request is in-flight, the worker
    /// is legitimately idle.
    pub inflight: Arc<AtomicBool>,
}

impl WorkerProgress {
    pub fn pet(&self) {
        self.last_pet_ms.store(unix_now_ms(), Ordering::Relaxed);
    }
    pub fn begin(&self) {
        self.pet();
        self.inflight.store(true, Ordering::Relaxed);
    }
    pub fn end(&self) {
        self.inflight.store(false, Ordering::Relaxed);
    }
    /// ms since last pet. Saturates at 0 if the clock moved backwards
    /// or no pet has happened yet.
    pub fn stale_ms(&self) -> i64 {
        let last = self.last_pet_ms.load(Ordering::Relaxed);
        if last == 0 {
            return 0;
        }
        unix_now_ms().saturating_sub(last).max(0)
    }
}

fn unix_now_ms() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Watchdog: if a request is in-flight and no forward progress has
/// happened in `deadline_ms`, abort the process so the supervisor
/// restarts us. Polls every `poll_ms`.
///
/// abort() is intentional: by the time we're here the engine is
/// usually stuck on a GPU stream sync that no Rust-level cancellation
/// can unstick. A clean shutdown would just hang too. The on-disk KV
/// snapshot is persisted at every turn-end (`save_live_if_dirty`),
/// so the next start replays from there cheaply.
pub fn run_watchdog(progress: WorkerProgress, deadline_ms: i64, poll_ms: u64) {
    loop {
        std::thread::sleep(Duration::from_millis(poll_ms));
        if !progress.inflight.load(Ordering::Relaxed) {
            continue;
        }
        let stale = progress.stale_ms();
        if stale > deadline_ms {
            tracing::error!(
                stale_ms = stale,
                deadline_ms,
                "engine wedged — no forward progress in deadline; aborting for supervisor restart"
            );
            // Flush logs before abort.
            std::thread::sleep(Duration::from_millis(50));
            std::process::abort();
        }
    }
}

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineRequest>,
    /// The engine thread, kept so `shutdown` can JOIN it.
    ///
    /// It used to be detached. The worker acks `Shutdown` and only THEN drops
    /// `WorkerState` — streams, modules, events, device buffers. With the
    /// handle dropped, `main` returned as soon as the ack landed and libc began
    /// running static destructors, including libamdhip64's, while the worker was
    /// still calling into HIP. That race is what produced every teardown crash
    /// we saw: SIGSEGV in `hip::Device::RemoveStream`, a throwing
    /// `amd::roc::Kernel::~Kernel` during `hipModuleUnload`, SIGSEGV in
    /// `hip::setCurrentDevice`, and `malloc_consolidate(): unaligned fastbin
    /// chunk` on the runs where the damage surfaced as heap corruption instead.
    worker: Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
    pub vocab: Arc<BpeVocab>,
    pub model_name: Arc<String>,
    /// Total KV-cache capacity in tokens — surfaced on `/v1/models` so
    /// clients (notably letta) can size requests to fit. Matches
    /// `WorkerConfig.n_kv_max`.
    pub n_kv_max: u32,
    /// Forward-progress signal. Shared with the watchdog thread and
    /// the /readyz HTTP handler.
    pub progress: WorkerProgress,
    /// True when the worker loaded a Vision-Exp tower (`--mmproj`) AND
    /// the text vocab carries `<｜deepseek_image｜>`. The HTTP handler rejects
    /// image parts with 400 when this is false.
    pub vision_enabled: bool,
    /// Vocab id of `<｜deepseek_image｜>`, or `None` when the loaded GGUF has
    /// no such token. `prompt::render_prompt` needs it to render image
    /// parts.
    pub image_placeholder_id: Option<i32>,
    /// Which image sources the HTTP layer may read. Default-deny for
    /// absolute local paths; see `vision_prompt::ImagePolicy`.
    pub image_policy: Arc<crate::vision_prompt::ImagePolicy>,
    /// Nucleus cutoff applied when the request omits `top_p`
    /// (`--default-top-p`). 0.95 = DeepSeek's agent recipe for this model.
    /// Sampling-only: it does not change the rendered prompt, so on-disk KV
    /// snapshots stay valid across a change.
    pub default_top_p: f32,
    /// What an absent `reasoning` / `reasoning_effort` request field means
    /// (`--default-reasoning-effort`). Defaults to the historical `Low`.
    /// Changing it DOES change the rendered prompt and invalidates every
    /// cached prefix — see `prompt::from_request_fields_with_default`.
    pub default_reasoning_effort: crate::prompt::ReasoningEffort,
}

impl EngineHandle {
    /// Submit a generation request. Returns the worker-event stream
    /// and a cancellation handle the caller can flip to ask the worker
    /// to stop mid-decode.
    ///
    /// Returns `SubmitError::Busy` when the engine queue is full so
    /// the caller can return HTTP 503 instead of blocking the HTTP
    /// task on a long backlog.
    pub fn submit(
        &self,
        req: GenerateReq,
        session_id: Option<String>,
    ) -> Result<(mpsc::Receiver<WorkerEvent>, Arc<AtomicBool>), SubmitError> {
        let (tx, rx) = mpsc::channel(64);
        let cancel = Arc::new(AtomicBool::new(false));
        match self.tx.try_send(EngineRequest::Generate {
            req,
            tx,
            session_id,
            cancel: cancel.clone(),
        }) {
            Ok(()) => Ok((rx, cancel)),
            Err(mpsc::error::TrySendError::Full(_)) => Err(SubmitError::Busy),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(SubmitError::WorkerDead),
        }
    }

    /// Block until the worker has saved its dirty live state and exited.
    /// Used during graceful shutdown.
    ///
    /// Uses `send().await` rather than `try_send`: shutdown must
    /// actually reach the worker, and waiting briefly for queue
    /// slack is fine here (no other request will be admitted —
    /// shutdown is initiated only on process termination).
    pub async fn shutdown(&self) -> eyre::Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .send(EngineRequest::Shutdown { ack: ack_tx })
            .await
            .map_err(|_| eyre!("engine worker channel closed"))?;
        ack_rx
            .await
            .map_err(|_| eyre!("engine worker dropped shutdown ack"))?;
        // The ack only means the worker LEFT its request loop; it is still
        // dropping `WorkerState` (and with it every HIP object) as the thread
        // unwinds. Join before returning, or `main` returns into libc's static
        // destructors and tears the HIP runtime out from under that drop.
        let handle = self.worker.lock().ok().and_then(|mut g| g.take());
        if let Some(h) = handle {
            tokio::task::spawn_blocking(move || h.join())
                .await
                .map_err(|e| eyre!("joining engine thread panicked: {e}"))?
                .map_err(|_| eyre!("engine thread panicked during teardown"))?;
        }
        Ok(())
    }
}

/// Configuration for starting the worker.
pub struct WorkerConfig {
    pub gguf_path: String,
    pub n_kv_max: u32,
    pub model_name: String,
    /// Directory where on-disk KV snapshots live. Created if missing.
    pub snapshot_root: std::path::PathBuf,
    /// Soft cap for the snapshot cache. LRU eviction kicks in above this.
    pub snapshot_cap_bytes: u64,
    /// Vision-Exp mmproj GGUF (ViT + aligner). `None` disables image
    /// input; requests carrying images then get HTTP 400.
    pub mmproj_path: Option<std::path::PathBuf>,
    /// Directories under which absolute local image paths may be read
    /// (`--allow-image-dir`). Empty = local paths rejected outright.
    pub allow_image_dirs: Vec<std::path::PathBuf>,
    /// Nucleus cutoff for requests that omit `top_p` (`--default-top-p`).
    pub default_top_p: f32,
    /// Effort level for requests that omit `reasoning`/`reasoning_effort`
    /// (`--default-reasoning-effort`).
    pub default_reasoning_effort: crate::prompt::ReasoningEffort,
}

pub fn spawn(cfg: WorkerConfig) -> eyre::Result<EngineHandle> {
    let (tx, rx) = mpsc::channel::<EngineRequest>(ENGINE_QUEUE_CAP);
    let (ready_tx, ready_rx) =
        std::sync::mpsc::sync_channel::<eyre::Result<WorkerReady>>(1);
    let n_kv_max = cfg.n_kv_max;
    let default_top_p = cfg.default_top_p;
    let default_reasoning_effort = cfg.default_reasoning_effort;
    let image_policy = Arc::new(crate::vision_prompt::ImagePolicy::from_dirs(
        cfg.allow_image_dirs.clone(),
    ));
    if !image_policy.allow_local_dirs.is_empty() {
        tracing::warn!(
            dirs = ?image_policy.allow_local_dirs,
            "local image paths ENABLED: any client of this HTTP API can read              image files under these directories"
        );
    }
    let progress = WorkerProgress::default();
    let progress_worker = progress.clone();

    let worker = std::thread::Builder::new()
        .name("deepstrix-engine".into())
        .spawn(move || worker_main(cfg, rx, ready_tx, progress_worker))
        .map_err(|e| eyre!("failed to spawn engine thread: {e}"))?;

    let WorkerReady {
        vocab,
        model_name,
        vision_enabled,
    } = ready_rx
        .recv()
        .map_err(|_| eyre!("engine thread dropped ready channel"))??;
    let image_placeholder_id = vocab.lookup_token_id(v4flash_vision::IMAGE_PLACEHOLDER);
    Ok(EngineHandle {
        tx,
        worker: Arc::new(std::sync::Mutex::new(Some(worker))),
        vocab,
        model_name,
        n_kv_max,
        progress,
        vision_enabled: vision_enabled && image_placeholder_id.is_some(),
        image_placeholder_id,
        image_policy,
        default_top_p,
        default_reasoning_effort,
    })
}

/// What the worker thread hands back once initialization succeeded.
struct WorkerReady {
    vocab: Arc<BpeVocab>,
    model_name: Arc<String>,
    /// The worker loaded a vision tower.
    vision_enabled: bool,
}

fn worker_main(
    cfg: WorkerConfig,
    mut rx: mpsc::Receiver<EngineRequest>,
    ready_tx: std::sync::mpsc::SyncSender<eyre::Result<WorkerReady>>,
    progress: WorkerProgress,
) {
    let mut state = match initialize_state(&cfg) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    state.progress = progress;
    let vocab = state.vocab.clone();
    let model_name = Arc::new(cfg.model_name.clone());
    let vision_enabled = state.tower.is_some();
    let _ = ready_tx.send(Ok(WorkerReady {
        vocab,
        model_name,
        vision_enabled,
    }));

    worker_loop(state, &mut rx);
}

pub struct WorkerState {
    pub dgpu: Device,
    pub igpu: Device,
    pub engine: HeterogeneousEngine,
    pub weights: HetModelWeights,

    /// M7 expert tier: pages V4.1's ~276 GiB of routed experts on demand instead
    /// of making them resident. Owns the HF source, so the mmap it reads experts
    /// from lives as long as the model. `None` when `V41_PAGED_EXPERTS` is unset
    /// (full residency, which only fits for V4-Flash).
    #[cfg(feature = "v41")]
    pub pager: Option<v4flash_kernels::het::ExpertPager>,

    /// V4.1 Engram: the n-gram hasher plus one table handle per Engram layer.
    /// The 189 GiB of tables stay on SSD and are row-gathered per token, so this
    /// holds only the hash parameters and the tensor descriptors.
    #[cfg(feature = "v41")]
    pub engram: Option<EngramCtx>,
    pub vocab: Arc<BpeVocab>,
    pub token_embd_bytes: Vec<u8>,
    pub token_embd_dtype: v4flash_core::gguf::GgufType,

    /// DSpark drafter. `Some` only under `V41_DSPARK=1` — it costs 7.93 GB of
    /// iGPU residency, so it is not loaded unless asked for. The layer stack
    /// and its state live on the iGPU; the exit and the residual capture live
    /// on the dGPU, beside the tied `output` head.
    #[cfg(feature = "v41")]
    pub mtp: Option<MtpCtx>,
    pub byte_decoder: std::collections::HashMap<char, u8>,

    /// Vision-Exp ViT + aligner, resident on the iGPU. `None` when the
    /// server was started without `--mmproj` — image requests are then
    /// rejected by the HTTP handler (`EngineHandle::vision_enabled`).
    pub tower: Option<v4flash_vision::Tower>,

    /// Bounded memo of `Tower::encode_rows` outputs, keyed by
    /// `PreprocessedImage::content_hash`, most-recent first.
    ///
    /// OpenAI chat completions are stateless: a client replays the whole
    /// `messages` array (images included) on every turn. Without this,
    /// turn N re-ran the full ViT for all N images even when their KV
    /// blocks were already resident and `prefill_suffix` never asked for
    /// a single row — ~454 ms of iGPU time per 1080p image per turn,
    /// which is exactly the TTFT the snapshot cache exists to remove.
    /// Entries are `n_llm * 4096` f32 (≤ 6.3 MiB each).
    pub vit_rows: Vec<([u8; 32], std::sync::Arc<Vec<f32>>)>,

    pub state: HetModelState,
    pub dgpu_scratch: DgpuScratch,
    pub igpu_scratch: IgpuScratch,
    pub bd_a: BatchDgpuScratch,
    pub bi_a: BatchIgpuScratch,
    pub bd_b: BatchDgpuScratch,
    pub bi_b: BatchIgpuScratch,
    /// Shared prefill scratch (one instance for both lanes).
    pub sd: BatchDgpuShared,
    pub si: BatchIgpuShared,

    pub n_kv_max: u32,

    /// The conversation currently resident in the KV cache.
    ///
    /// Invariant: `live.pos == live.tokens.len() as u32`. When set, the
    /// engine's `HetModelState` reflects exactly these tokens having
    /// been prefilled / forwarded in order, including any per-turn
    /// trailing EOS we force into the cache at turn end.
    pub live: Option<LiveSession>,

    /// On-disk snapshot index. Loaded at startup, mutated as sessions
    /// switch.
    pub snapshot_index: SnapshotIndex,
    pub model_fingerprint: ModelFingerprint,

    /// M62: on-disk expert-selection aggregate + its sidecar path.
    /// Harvested from the engine's device banks and flushed at the same
    /// points snapshots save (turn end / session switch / shutdown).
    pub expert_stats: crate::expert_stats::ExpertStatsAgg,
    pub expert_stats_path: std::path::PathBuf,

    /// Forward-progress signal. The worker pet()s this after every
    /// emitted decode token and after every completed prefill chunk;
    /// the watchdog thread aborts the process if it goes stale.
    pub progress: WorkerProgress,
}

#[derive(Debug, Clone)]
pub struct LiveSession {
    pub tokens: Vec<i32>,
    /// Image spans inside `tokens` (live-token-index space). Empty for
    /// text-only sessions. Carried so the byte-aligned LCP and the
    /// snapshot key both see each image's content hash.
    pub image_spans: Vec<ImageSpan>,
    pub pos: u32,
    /// Set when `live.tokens` advanced past the last save point on disk.
    /// Triggers a save when the session is about to be evicted.
    pub dirty: bool,
    /// sessionId hint provided by the client (if any). Carried through
    /// so that when we save we can update the index's sessionId hint.
    pub session_id: Option<String>,
}

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 (9070 XT) device found"))
}

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 (Strix iGPU) device found"))
}

fn initialize_state(cfg: &WorkerConfig) -> eyre::Result<WorkerState> {
    // Model source: a GGUF (V4-Flash) or the HF safetensors dir (V4.1). Both
    // present their tensors through `WeightSrc`, so everything downstream of
    // `src` is source-agnostic (compile-time model selection, see ENGINE_PORT §0).
    #[cfg(not(feature = "v41"))]
    let src_owner = {
        tracing::info!(gguf = %cfg.gguf_path, "loading GGUF");
        MappedGguf::open(&cfg.gguf_path)?
    };
    #[cfg(feature = "v41")]
    let src_owner = {
        tracing::info!(dir = %cfg.gguf_path, "loading V4.1 HF safetensors");
        V41HfWeights::open(&cfg.gguf_path, None)?
    };
    let src = WeightSrc::from(&src_owner);

    #[cfg(not(feature = "v41"))]
    let vocab = BpeVocab::from_gguf(src_owner.gguf())?;
    #[cfg(feature = "v41")]
    let vocab = {
        // DeepSeek V4/V4.1 use the "joyai-llm" pre-tokenizer; the tokenizer.json
        // table is id-for-id identical to the GGUF one (tokenizer.rs).
        let tj = std::path::Path::new(&cfg.gguf_path).join("tokenizer.json");
        BpeVocab::from_tokenizer_json(&tj, Some("joyai-llm".to_string()))?
    };
    tracing::info!(
        vocab_size = vocab.vocab_size(),
        dsml_id = ?vocab.dsml_id,
        "vocab loaded"
    );

    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    tracing::info!(dgpu=%dgpu_arch, dgpu_id=dgpu.id, igpu=%igpu_arch, igpu_id=igpu.id, "selected devices");

    let token_embd_t = src
        .tensor("token_embd.weight")
        .ok_or_else(|| eyre!("missing token_embd.weight"))?;
    if !v4flash_kernels::weight_contract::TOKEN_EMBD_ALLOWED.contains(&token_embd_t.dtype) {
        return Err(eyre!(
            "token_embd dtype {:?} unsupported (allowed: {:?})",
            token_embd_t.dtype,
            v4flash_kernels::weight_contract::TOKEN_EMBD_ALLOWED
        ));
    }
    let token_embd_dtype = token_embd_t.dtype;
    let token_embd_bytes = src.read_tensor(token_embd_t)?;

    let rope = |layer: i32| -> eyre::Result<RopeParams> { Ok(rope_for_layer(layer)) };

    // M62: if a derived placement file from a previous run exists and the
    // user hasn't pinned one, point the hot-expert loader at it BEFORE the
    // weights load — restarts then self-improve from accumulated stats.
    if std::env::var("DGPU_HOT_EXPERTS_FILE").is_err() {
        let placement = deepstrix_server_placement_path(&cfg.snapshot_root);
        if placement.exists() {
            tracing::info!(path = %placement.display(), "using accumulated expert-stats placement");
            std::env::set_var("DGPU_HOT_EXPERTS_FILE", &placement);
        }
    }

    tracing::info!("loading het weights (dGPU ~9 GiB + iGPU ~52 GiB)");
    let t0 = std::time::Instant::now();
    // V4.1's routed experts are ~276 GiB against 96 GB here, so they cannot be
    // resident. `V41_PAGED_EXPERTS=1` leaves them out of the per-layer iGPU
    // weights and an ExpertPager (built below) pages the router's actual picks on
    // demand. Without the flag `load_all` still tries full residency and OOMs.
    #[cfg(feature = "v41")]
    if v4flash_kernels::het::weights::v41_paged_experts() {
        tracing::info!("V4.1: paged expert tier ON — routed experts are NOT resident");
    } else {
        tracing::warn!(
            "V4.1: V41_PAGED_EXPERTS unset — load_all will try to make ALL routed experts \
             resident (~276 GiB) and will OOM on this box"
        );
    }
    #[cfg_attr(not(feature = "v41"), allow(unused_mut))]
    let mut weights = HetModelWeights::load_all(src, dgpu, igpu, &rope)?;
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "weights loaded");

    // DSpark drafter (`V41_DSPARK=1`). 7.93 GB of iGPU residency, so opt-in.
    #[cfg(feature = "v41")]
    let mtp: Option<MtpCtx> = if matches!(
        std::env::var("V41_DSPARK").as_deref(),
        Ok("1") | Ok("on") | Ok("shadow") | Ok("accept")
    ) {
        use v4flash_kernels::het::mtp::{MtpCapture, MtpExit, MtpState, MTP_NOISE_TOKEN};
        use v4flash_kernels::het::weights::{MtpExitWeights, MtpWeights};
        let t = std::time::Instant::now();
        let w = MtpWeights::load(src, igpu, v4flash_kernels::config::N_LAYER as usize)?;
        let xw = MtpExitWeights::load(src, dgpu)?;
        let mk = src
            .tensor("mtp.2.markov_embd.weight")
            .ok_or_else(|| eyre!("mtp.2.markov_embd.weight not found"))?;
        let markov_dtype = mk.dtype;
        let markov_embd = src.read_tensor(mk)?;
        let mut noise_row = vec![0.0f32; v4flash_kernels::config::HC_DIM as usize];
        embed_lookup(&token_embd_bytes, token_embd_dtype, MTP_NOISE_TOKEN, &mut noise_row);
        tracing::info!(
            elapsed_s = t.elapsed().as_secs_f64(),
            "DSpark drafter loaded (layers+entry on iGPU, exit on dGPU)"
        );
        Some(MtpCtx {
            w,
            xw,
            state: MtpState::alloc(igpu.id)?,
            exit: MtpExit::alloc(dgpu.id)?,
            capture: MtpCapture::alloc(dgpu.id)?,
            markov_embd,
            markov_dtype,
            noise_row,
            pending: None,
            pending_conf: [0.0; v4flash_kernels::het::mtp::MTP_BLOCK],
            confirmed: std::collections::VecDeque::new(),
            ingested: 0,
            next_after: None,
            main_hidden: Vec::new(),
            accept_steps: 0,
            accept_tokens: 0,
            drafts: Vec::new(),
            drafts_plain: Vec::new(),
            actual: Vec::new(),
            actual_base: 0,
        })
    } else {
        None
    };

    let mut engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    // `V41_PERFETTO_OUT=<path>`: device-time trace of every kernel stage on all
    // four streams (dgpu/igpu x compute/xfer) plus the host-time
    // `remote.expert` track for the two-box round trips. Off unless set —
    // attaching turns on HIP event timing in both pools, which is not free.
    if let Ok(path) = std::env::var("V41_PERFETTO_OUT") {
        engine.attach_perfetto(&path)?;
        tracing::info!(path = %path, "perfetto device trace attached");
    }
    let engine = engine;
    let dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let igpu_scratch = IgpuScratch::alloc(igpu)?;
    // Decode attention writes `n_raw + n_comp` scores per head into
    // `DgpuScratch.attn_scores`, which is sized at ATTN_MIXED_MAX_KEYS.
    // A --ctx past that cap used to be accepted and then fail mid-decode.
    //
    // The old check divided --ctx by 4, which silently assumed the model's
    // smallest ungathered compressed store is n_kv/4. True for V4-Flash
    // (ratio-4 layers are gathered to INDEXER_TOP_K, ratio-128 layers give
    // n_kv/128) and FALSE for V4.1, whose layers 20-39 are ratio 1 with no
    // indexer: it accepted `--ctx 328704` while decode died at ~82K.
    // `attn_max_scored_keys` is the derivation both models share.
    {
        let raw_window = if cfg.mmproj_path.is_some() {
            v4flash_kernels::het::image_spans::IMAGE_RAW_WINDOW_MAX
        } else {
            v4flash_kernels::config::SWA_WINDOW
        };
        // TWO bounds, and the indexer's is the binding one under v41.
        // `attn_max_scored_keys` asks what ATTENTION scores after the gather,
        // and `scored_keys_are_gathered` reads `V41_INDEX_K` at runtime — so
        // with the indexer on it answers 640 at ANY context and admits a --ctx
        // whose comp-indexed scratch does not exist. The indexer still has to
        // score the whole store to pick that top-k. Check both; the max wins.
        let attn_need =
            v4flash_kernels::attention::attn_max_scored_keys(cfg.n_kv_max, raw_window);
        let index_need =
            v4flash_kernels::attention::indexer_max_scored_keys(cfg.n_kv_max, raw_window);
        let need = attn_need.max(index_need);
        let cap = v4flash_kernels::ATTN_MIXED_MAX_KEYS;
        if need > cap {
            let max_ctx = v4flash_kernels::attention::attn_max_ctx_for_keys(cap, raw_window)
                .min(v4flash_kernels::attention::indexer_max_ctx_for_keys(cap, raw_window));
            return Err(eyre!(
                "--ctx {} needs {} keys per head (attention {}, indexer {}) but \
                 ATTN_MIXED_MAX_KEYS is {} (raise it in attention.rs, or use --ctx <= {})",
                cfg.n_kv_max,
                need,
                attn_need,
                index_need,
                cap,
                max_ctx,
            ));
        }
    }
    let state = HetModelState::alloc(dgpu, igpu, cfg.n_kv_max)?;
    // Two-lane pipelined prefill: each lane holds at most ceil(B_MAX/2)
    // rows of a chunk (forward_prompt_batch_v2_pipelined), so size the
    // per-lane scratch at that instead of the full chunk. The shared set
    // (one instance, alternating between lanes) is sized the same.
    let lane_rows = B_MAX.div_ceil(2);
    let bd_a = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let bi_a = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    // The batched attention scores scratch is the one prefill buffer that
    // scales with context, and on a model with no sparse indexer it scales
    // FAST (V4.1: 32 KiB per token of --ctx). Size it from --ctx instead of
    // the ratio>=4-shaped ATTN_SCORES_STRIDE constant, and say what it cost.
    {
        let keys = v4flash_kernels::het::batch_scratch::attn_scores_capacity_keys(
            lane_rows,
            cfg.n_kv_max,
        );
        tracing::info!(
            n_kv_max = cfg.n_kv_max,
            lane_rows,
            attn_scores_mib = (keys * 2) / (1024 * 1024),
            "attention scores scratch sized from --ctx"
        );
    }
    let sd = BatchDgpuShared::alloc_rows_ctx(dgpu, lane_rows, cfg.n_kv_max)?;
    let si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    tracing::info!(n_kv_max = cfg.n_kv_max, "KV cache allocated");

    let byte_decoder = build_gpt2_byte_decoder();

    // Vision tower (V4-Flash: Vision-Exp mmproj GGUF; V4.1: the HF snapshot
    // dir, `vision_v41`). Lives on the iGPU (host RAM): ~0.9 GiB of f16
    // weights plus a small per-image activation workspace. Loaded here so
    // a bad `--mmproj` fails startup rather than the first image request.
    let tower = match cfg.mmproj_path.as_deref() {
        None => None,
        Some(path) => {
            if vocab
                .lookup_token_id(v4flash_vision::IMAGE_PLACEHOLDER)
                .is_none()
            {
                return Err(eyre!(
                    "--mmproj {} was given but the text vocab has no `{}` token \
                     (not a vision checkpoint)",
                    path.display(),
                    v4flash_vision::IMAGE_PLACEHOLDER
                ));
            }
            // V4.1 carries `bias_vl` in the checkpoint itself: derive the sidecar
            // the engine reads from the presented `exp_probs_b_vl.bias` tensors.
            #[cfg(feature = "v41")]
            crate::vision_v41::ensure_bias_vl(
                &mut weights,
                &src,
                std::path::Path::new(&cfg.gguf_path),
                dgpu,
            )?;
            // The router needs the `bias_vl` sidecar for every image row.
            // Check it HERE: otherwise the first image request dies deep
            // inside layer-0 prefill as a 500, after part of the chunk's
            // KV has already been appended.
            if !weights.has_bias_vl() {
                let where_ = v4flash_kernels::het::weights::bias_vl_sidecar_path(std::path::Path::new(
                    &cfg.gguf_path,
                ));
                return Err(eyre!(
                    "--mmproj {} was given but the `bias_vl` routing-bias sidecar was not \
                     loaded (expected at {}). Generate it with scripts/fetch_bias_vl.py, or \
                     point DEEPSTRIX_BIAS_VL_FILE at it. Image requests cannot be routed \
                     without it.",
                    path.display(),
                    where_
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "<unresolvable>".into()),
                ));
            }
            // The attention score scratch is sized for text raw windows;
            // V4-Flash image rows widen them. Refuse at launch rather than
            // mid-prefill if --ctx makes that overrun. (V4.1 image tokens
            // attend causally — no widening, nothing to check.)
            #[cfg(not(feature = "v41"))]
            v4flash_kernels::attention::check_vision_ctx_fits(cfg.n_kv_max)?;
            tracing::info!(mmproj = %path.display(), "loading vision tower (iGPU)");
            let t0 = std::time::Instant::now();
            #[cfg(not(feature = "v41"))]
            let mut tower = v4flash_vision::Tower::load(path, igpu)
                .map_err(|e| eyre!("loading mmproj {}: {e:#}", path.display()))?;
            #[cfg(feature = "v41")]
            let mut tower = crate::vision_v41::load_tower(path, igpu)?;
            // Host mirror is only needed for requantisation experiments;
            // drop it so the worker doesn't sit on ~0.9 GiB of host RAM.
            tower.drop_host();
            tracing::info!(
                mib = tower.device_bytes() / (1024 * 1024),
                elapsed_s = t0.elapsed().as_secs_f64(),
                "vision tower loaded"
            );
            dgpu.set_current()?;
            Some(tower)
        }
    };

    // Compute model fingerprint and load (or create) the snapshot index.
    let fingerprint =
        ModelFingerprint::compute(vocab.vocab_size() as u32, &token_embd_bytes, src.tensors());
    if !cfg.snapshot_root.exists() {
        std::fs::create_dir_all(&cfg.snapshot_root)
            .map_err(|e| eyre!("create snapshot root {:?}: {e}", cfg.snapshot_root))?;
    }
    let snapshot_index = SnapshotIndex::load(
        cfg.snapshot_root.clone(),
        fingerprint.clone(),
        cfg.snapshot_cap_bytes,
    )?;
    let expert_stats_path = crate::expert_stats::ExpertStatsAgg::path_for(&cfg.snapshot_root);
    let expert_stats =
        crate::expert_stats::ExpertStatsAgg::load_or_fresh(&expert_stats_path, &fingerprint);

    // M7 expert pager. Takes ownership of the HF source: its mmap must outlive the
    // model, since every routed expert is read from it on demand for the life of
    // the process. Built last so the `src` borrow above (load_all, fingerprint) is
    // finished. Slot count auto-sizes from V41_PAGER_POOL_GB.
    #[cfg(feature = "v41")]
    let pager = if v4flash_kernels::het::weights::v41_paged_experts() {
        let t0 = std::time::Instant::now();
        let mut pg = v4flash_kernels::het::ExpertPager::new(src_owner, igpu, 0)?;
        if v4flash_kernels::het::expert_pager::b1_prefetch() {
            pg.start_prefetcher(std::path::Path::new(&cfg.gguf_path))?;
        }
        if v4flash_kernels::het::expert_pager::prefill_readahead() {
            pg.start_readahead(std::path::Path::new(&cfg.gguf_path))?;
        }
        tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "expert pager ready");
        Some(pg)
    } else {
        None
    };

    // V4.1 Engram. The hash parameters (token map, multipliers, pad id) come from
    // a dumped dir; the tables are read straight out of the checkpoint. Both
    // Engram layers must resolve or the forward fails at layer 1, so this is a
    // hard error rather than a silent skip.
    #[cfg(feature = "v41")]
    let engram = match pager.as_ref() {
        None => None,
        Some(pg) => {
            let dir = std::env::var("V41_ENGRAM_DIR").unwrap_or_else(|_| {
                format!(
                    "{}/.cache/deepstrix/v41/engram",
                    std::env::var("HOME").unwrap_or_default()
                )
            });
            let hasher = v4flash_core::EngramHash::load(std::path::Path::new(&dir))
                .map_err(|e| eyre!("engram hash params at {dir}: {e:#} (set V41_ENGRAM_DIR)"))?;
            let mut tables = Vec::new();
            for &l in v4flash_kernels::config::ENGRAM_LAYERS {
                tables.push(v4flash_core::EngramTable::open(pg.raw(), l as usize)?);
            }
            tracing::info!(dir = %dir, layers = ?v4flash_kernels::config::ENGRAM_LAYERS, "engram ready (tables stay on SSD)");
            Some(EngramCtx { hasher, tables, compressed: Vec::new() })
        }
    };

    Ok(WorkerState {
        #[cfg(feature = "v41")]
        pager,
        #[cfg(feature = "v41")]
        engram,
        dgpu,
        igpu,
        engine,
        weights,
        vocab: Arc::new(vocab),
        token_embd_bytes,
        token_embd_dtype,
        #[cfg(feature = "v41")]
        mtp,
        byte_decoder,
        tower,
        vit_rows: Vec::new(),
        state,
        dgpu_scratch,
        igpu_scratch,
        bd_a,
        bi_a,
        bd_b,
        bi_b,
        sd,
        si,
        n_kv_max: cfg.n_kv_max,
        live: None,
        snapshot_index,
        model_fingerprint: fingerprint,
        expert_stats,
        expert_stats_path,
        // worker_main overwrites this with the real progress handle
        // that's shared with EngineHandle + watchdog. Default here so
        // initialize_state stays a self-contained fn.
        progress: WorkerProgress::default(),
    })
}

/// Text-aware longest common prefix of two token-id sequences. When
/// token IDs differ at a position, decode both via the BPE vocab + GPT-2
/// byte decoder and compare raw bytes — same-text-different-id pairs
/// (e.g. two encodings of "." that differ in BPE rank) count as a
/// match.
///
/// This is the smallest fix to the canonical BPE-roundtrip problem:
/// when the model samples a non-canonical encoding for some text, the
/// re-encoded form of that same text on the next request will produce
/// different IDs at the same position, but the decoded bytes match. We
/// can safely treat that as a cache hit because:
///   * Token COUNT is identical — ROPE position numbering lines up.
///   * The KV state at that position was derived from the actual
///     sampled token; subsequent attention from later positions reads
///     the same byte-content's K,V — semantically equivalent context
///     even if the exact token-id differs.
///
/// What this does NOT handle: divergences where the BPE produces a
/// different NUMBER of tokens for the same text (e.g. ["hello", "!"]
/// vs ["hello!"]). Those still terminate the LCP and trigger reset.
/// In practice the per-position case covers most observed mismatches
/// (punctuation, single-character tokens).
/// Outcome of a [`byte_aligned_lcp`] computation. `live_tokens` and
/// `req_tokens` count the largest *clean alignment* — token boundaries
/// where the decoded byte streams have matched up to exactly the same
/// position on both sides. `bridged_tokens` counts positions where the
/// token IDs differed but bytes still matched within the same boundary
/// (telemetry only).
///
/// The bytes-not-tokens framing makes us robust to tokenizer
/// non-determinism — if the model samples `["foo", "bar"]` and the
/// next request re-encodes the same text as `["foob", "ar"]`, the byte
/// stream "foobar" matches and we keep the in-VRAM KV state.
#[derive(Debug, Clone, Copy)]
struct AlignedLcp {
    live_tokens: usize,
    req_tokens: usize,
    bridged_tokens: usize,
    first_bridge: Option<(i32, i32)>,
}

/// Text-only convenience wrapper (the shape the pre-vision tests use).
#[cfg(test)]
fn byte_aligned_lcp(
    live: &[i32],
    req: &[i32],
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
) -> AlignedLcp {
    byte_aligned_lcp_vl(live, &[], req, &[], vocab, byte_decoder)
}

/// [`byte_aligned_lcp`] for streams that may contain image blocks.
///
/// Synthetic image ids have no vocab text, so a plain decode would make
/// EVERY image block contribute zero bytes and two different pictures
/// would compare equal. Both sides therefore decode through
/// `synthetic_token_bytes`, which emits a per-type marker and folds the
/// image's content hash in at its IMAGE_START. The id fast-path gets the
/// same treatment: two IMAGE_START tokens are only "the same token" when
/// their spans carry the same hash.
fn byte_aligned_lcp_vl(
    live: &[i32],
    live_spans: &[ImageSpan],
    req: &[i32],
    req_spans: &[ImageSpan],
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
) -> AlignedLcp {
    let mut out = AlignedLcp {
        live_tokens: 0,
        req_tokens: 0,
        bridged_tokens: 0,
        first_bridge: None,
    };
    let decode_at = |spans: &[ImageSpan], idx: usize, id: i32| -> Vec<u8> {
        if let Some(b) = synthetic_token_bytes(id, span_hash_at(spans, idx)) {
            return b;
        }
        vocab
            .token_text(id)
            .map(|b| gpt2_decode_token(b, byte_decoder))
            .unwrap_or_default()
    };

    let mut li = 0usize;
    let mut ri = 0usize;
    let mut live_buf: Vec<u8> = Vec::new();
    let mut req_buf: Vec<u8> = Vec::new();
    // last sync point: token indices where both buffers were empty AND
    // all bytes up to here were equal.
    let mut sync_live = 0usize;
    let mut sync_req = 0usize;
    let mut bridged_in_round = 0usize;
    let mut first_bridge: Option<(i32, i32)> = None;

    loop {
        // Fast path: both buffers empty AND same token id ⇒ advance both.
        if live_buf.is_empty() && req_buf.is_empty() {
            // Commit sync point at the start of every clean round.
            sync_live = li;
            sync_req = ri;
            out.bridged_tokens += bridged_in_round;
            bridged_in_round = 0;
            if li >= live.len() || ri >= req.len() {
                break;
            }
            // Same id AND (for image slots) the same image. Only
            // IMAGE_START carries a hash, so this is a no-op everywhere
            // else.
            if live[li] == req[ri]
                && span_hash_at(live_spans, li) == span_hash_at(req_spans, ri)
            {
                li += 1;
                ri += 1;
                continue;
            }
            // Different ids — start byte-buffering both sides.
            let (l_id, r_id) = (live[li], req[ri]);
            live_buf = decode_at(live_spans, li, l_id);
            req_buf = decode_at(req_spans, ri, r_id);
            li += 1;
            ri += 1;
            if first_bridge.is_none() {
                first_bridge = Some((l_id, r_id));
            }
            bridged_in_round += 1;
            continue;
        }

        // Compare what we have. Any byte-prefix mismatch ends the LCP
        // at the last sync point.
        let prefix = live_buf.len().min(req_buf.len());
        if live_buf[..prefix] != req_buf[..prefix] {
            break;
        }
        if live_buf.len() == req_buf.len() {
            // Buffers exactly consume each other — clean round done.
            live_buf.clear();
            req_buf.clear();
            continue;
        }
        // One side is shorter — extend it by consuming the next token,
        // and drop the just-matched prefix from the other.
        if live_buf.len() < req_buf.len() {
            req_buf.drain(..live_buf.len());
            live_buf.clear();
            if li >= live.len() {
                break;
            }
            let l_id = live[li];
            live_buf.extend(decode_at(live_spans, li, l_id));
            li += 1;
            bridged_in_round += 1;
        } else {
            live_buf.drain(..req_buf.len());
            req_buf.clear();
            if ri >= req.len() {
                break;
            }
            let r_id = req[ri];
            req_buf.extend(decode_at(req_spans, ri, r_id));
            ri += 1;
            bridged_in_round += 1;
        }
    }

    out.live_tokens = sync_live;
    out.req_tokens = sync_req;
    out.first_bridge = first_bridge;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loads the real BPE vocab. Gated since the GGUF is large.
    fn load_vocab() -> Option<BpeVocab> {
        let path = "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";
        if !std::path::Path::new(path).exists() {
            return None;
        }
        let gguf = MappedGguf::open(path).ok()?;
        BpeVocab::from_gguf(gguf.gguf()).ok()
    }

    /// Find an aliased pair: two distinct token ids whose
    /// `gpt2_decode_token(token_text(id))` bytes are equal. The DeepSeek
    /// vocab has these because BPE can have multiple paths to the same
    /// surface text. We scan a small range; on the V4-Flash vocab a
    /// pair shows up easily within the first few thousand ids.
    fn find_alias_pair(
        vocab: &BpeVocab,
        dec: &std::collections::HashMap<char, u8>,
    ) -> Option<(i32, i32)> {
        let mut by_text: std::collections::HashMap<Vec<u8>, i32> =
            std::collections::HashMap::new();
        for id in 0..vocab.vocab_size() as i32 {
            // Skip special tokens — those don't alias to plain text.
            if id == 0 || (id >= 128800 && id <= 128900) {
                continue;
            }
            if let Some(bytes) = vocab.token_text(id) {
                let decoded = gpt2_decode_token(bytes, dec);
                if decoded.is_empty() {
                    continue;
                }
                if let Some(&other) = by_text.get(&decoded) {
                    return Some((other, id));
                }
                by_text.insert(decoded, id);
            }
        }
        None
    }

    #[test]
    #[ignore]
    fn aligned_lcp_bridges_text_aliased_token() {
        let Some(vocab) = load_vocab() else { return };
        let dec = build_gpt2_byte_decoder();
        let Some((a, b)) = find_alias_pair(&vocab, &dec) else {
            // No alias found in scanned range — vocab is unusually
            // canonical; aligned_lcp would behave the same as plain
            // prefix match. Nothing to test, skip.
            eprintln!("no alias pair found; skipping aligned LCP bridge test");
            return;
        };
        let aa = vocab
            .token_text(a)
            .map(|b| gpt2_decode_token(b, &dec))
            .unwrap_or_default();
        let bb = vocab
            .token_text(b)
            .map(|b| gpt2_decode_token(b, &dec))
            .unwrap_or_default();
        assert_eq!(aa, bb, "alias pair must decode to the same bytes");
        eprintln!(
            "alias: id {a} ↔ id {b}, both decode to {:?}",
            String::from_utf8_lossy(&aa)
        );
        // Identical prefix, then the alias at position 3, then more matching.
        let live = vec![100, 200, 300, a, 1];
        let req = vec![100, 200, 300, b, 1, 128803];
        let res = byte_aligned_lcp(&live, &req, &vocab, &dec);
        // 5 live tokens cleanly aligned (3 identical + 1 aliased + 1 trailing EOS).
        assert_eq!(res.live_tokens, 5);
        assert_eq!(res.req_tokens, 5);
        assert_eq!(res.bridged_tokens, 1);
        assert_eq!(res.first_bridge, Some((a, b)));
    }

    #[test]
    #[ignore]
    fn aligned_lcp_does_not_match_different_text() {
        let Some(vocab) = load_vocab() else { return };
        let dec = build_gpt2_byte_decoder();
        // id 16 = "." (1 byte), id 603 = ".\n" (2 bytes) in V4-Flash —
        // these are NOT aliases, the text differs by a newline. The
        // LCP should stop at the divergent position, not bridge.
        let live = vec![100, 200, 300, 16, 1];
        let req = vec![100, 200, 300, 603, 1];
        let res = byte_aligned_lcp(&live, &req, &vocab, &dec);
        assert_eq!(res.live_tokens, 3, "LCP must stop where bytes diverge");
        assert_eq!(res.req_tokens, 3);
        assert_eq!(res.bridged_tokens, 0);
    }

    // ---- vision: per-request encoded rows + span rebasing --------

    fn span(start: u32, len: u32, seed: u8) -> ImageSpan {
        ImageSpan { start, len, hash: [seed; 32] }
    }

    #[test]
    fn encoded_images_row_at_covers_exactly_the_block() {
        let n_embd = N_EMBD as usize;
        // Two blocks: [10, 14) and [40, 43).
        let vl = EncodedImages {
            blocks: vec![
                (10, 4, (0..4 * n_embd).map(|i| i as f32).collect()),
                (40, 3, (0..3 * n_embd).map(|i| -(i as f32)).collect()),
            ],
            spans: vec![span(11, 3, 1), span(40, 3, 2)],
        };
        for i in 0..50usize {
            let in_block = (10..14).contains(&i) || (40..43).contains(&i);
            assert_eq!(vl.row_at(i).is_some(), in_block, "idx {i}");
        }
        // Rows are handed out in block order, one per block token.
        assert_eq!(vl.row_at(10).unwrap()[0], 0.0);
        assert_eq!(vl.row_at(11).unwrap()[0], n_embd as f32);
        assert_eq!(vl.row_at(13).unwrap()[0], (3 * n_embd) as f32);
        assert_eq!(vl.row_at(41).unwrap()[0], -(n_embd as f32));
        assert_eq!(vl.row_at(10).unwrap().len(), n_embd);
        assert!(EncodedImages::default().row_at(0).is_none());
    }

    #[test]
    fn spans_rebase_to_absolute_kv_positions() {
        use crate::vision_prompt::spans_in_range;
        let spans = vec![span(10, 4, 1), span(40, 3, 2)];
        // Suffix req.tokens[8..50] prefilled at KV pos 100: block starts
        // move to 100 + (10 - 8) = 102 and 100 + (40 - 8) = 132.
        let (base, pos0) = (8usize, 100u32);
        let abs: Vec<(u32, u32)> = spans_in_range(&spans, base, 50)
            .unwrap()
            .iter()
            .map(|s| (s.start + pos0, s.len))
            .collect();
        assert_eq!(abs, vec![(102, 4), (132, 3)]);
        // Spans entirely before the range drop out.
        let abs2: Vec<(u32, u32)> = spans_in_range(&spans, 20, 50)
            .unwrap()
            .iter()
            .map(|s| (s.start, s.len))
            .collect();
        assert_eq!(abs2, vec![(20, 3)]);
        // A boundary cutting a block is a hard error, both edges.
        assert!(spans_in_range(&spans, 12, 50).is_err());
        assert!(spans_in_range(&spans, 0, 42).is_err());
    }

    #[test]
    fn spans_from_keeps_only_the_suffix_blocks() {
        let spans = vec![span(10, 4, 1), span(40, 3, 2)];
        assert_eq!(spans_from(&spans, 0), spans);
        assert_eq!(spans_from(&spans, 11), vec![span(40, 3, 2)]);
        assert_eq!(spans_from(&spans, 41), Vec::<ImageSpan>::new());
        assert_eq!(shift_spans(&spans_from(&spans, 11), -5), vec![span(35, 3, 2)]);
    }

    /// The interesting case: live and req represent the same TEXT but
    /// with different token splits. `byte_aligned_lcp` should align all
    /// the way through; the (live, req) counts may differ.
    #[test]
    #[ignore]
    fn byte_aligned_lcp_bridges_split_divergence() {
        let Some(vocab) = load_vocab() else { return };
        let dec = build_gpt2_byte_decoder();
        // Find a string the BPE splits into 2+ tokens.
        let s = "Hello world.";
        let canonical = vocab.encode(s);
        if canonical.len() < 2 { return; }
        // Construct a "live" sequence that decodes to the same bytes
        // but has a different split. We don't have a way to force a
        // non-canonical split short of running the model; instead test
        // the trivial identity case (same on both sides), plus the
        // alias case for which we already have coverage above. So this
        // test just sanity-checks the identity path.
        let live = canonical.clone();
        let req = canonical.clone();
        let res = byte_aligned_lcp(&live, &req, &vocab, &dec);
        assert_eq!(res.live_tokens, live.len());
        assert_eq!(res.req_tokens, req.len());
    }
}

fn worker_loop(mut state: WorkerState, rx: &mut mpsc::Receiver<EngineRequest>) {
    #[cfg(feature = "v41")]
    if crate::multistream::enabled() {
        return crate::multistream::worker_loop_ms(state, rx);
    }
    while let Some(msg) = rx.blocking_recv() {
        match msg {
            EngineRequest::Generate {
                req,
                tx,
                session_id,
                cancel,
            } => {
                // Bracket the whole request with begin/end so the
                // watchdog only fires when we're actually supposed to
                // be making progress. RAII: end runs even on panic-unwind
                // or early return inside handle_generate_stream. The
                // guard owns its own clone (two Arcs) so it doesn't
                // co-borrow `state` with handle_generate_stream below.
                struct InflightGuard(WorkerProgress);
                impl Drop for InflightGuard {
                    fn drop(&mut self) {
                        self.0.end();
                    }
                }
                state.progress.begin();
                let _guard = InflightGuard(state.progress.clone());
                #[cfg(feature = "v41")]
                let pager_c0 = state
                    .pager
                    .as_ref()
                    .map(|p| p.counters())
                    .unwrap_or_default();
                let t_req = std::time::Instant::now();
                if let Err(e) = handle_generate_stream(&mut state, req, session_id, cancel, &tx) {
                    let _ = tx.blocking_send(WorkerEvent::Error(format!("{e:#}")));
                    // A mid-layer failure leaves box-2 requests in flight; drain them
                    // or every later request dies on a ticket/seq mismatch.
                    let n = state.engine.remote_drain_in_flight();
                    if n > 0 {
                        tracing::warn!(drained = n, "remote expert client: drained in-flight tickets after request error");
                    }
                    // The error may BE the link dying (a box-2 error closes the
                    // socket, the writer thread exits, and `tx_req` is closed for
                    // good). Redial here, at the request boundary with nothing in
                    // flight, or every later request 500s until a manual restart.
                    match state.engine.remote_reconnect_if_dead() {
                        Ok(true) => tracing::warn!("remote expert client: link was dead, reconnected"),
                        Ok(false) => {}
                        Err(e) => tracing::error!(error = %e, "remote expert client: reconnect FAILED; later requests will fail until box 2 is reachable"),
                    }
                }
                // Engine-side end-to-end wall for the request (prefill + decode
                // loop + snapshot saves). `e2e_ms - decode.loop.summary.loop_ms`
                // is the per-request fixed cost a curl-wall harness amortises
                // over completion_tokens.
                tracing::info!(
                    e2e_ms = t_req.elapsed().as_millis() as u64,
                    "request.summary"
                );
                // Box-1's two per-call thread hops. `link_us` cannot see them:
                // `rtt_us` includes the submit->writer hop and excludes the
                // reader->caller one. Against a MEASURED 16.6 us raw TCP round
                // trip on this link, anything here above a few us is the
                // scheduler, and it is paid 80x per token.
                #[cfg(feature = "v41")]
                {
                    let (to_write, wake, slack, n_blocked, n) =
                        v4flash_kernels::het::remote_experts::take_hop_stats();
                    if n > 0 {
                        // `exposed_pct` is the number that matters: the fraction
                        // of calls where we were already blocked waiting for the
                        // reply. The rest finished behind local work and cost
                        // nothing, so per-call link latency must NOT be
                        // multiplied by 80 to get a per-token figure.
                        tracing::info!(
                            submit_to_write_us = to_write,
                            wake_us = wake,
                            slack_us = slack,
                            exposed_pct = 100.0 * n_blocked as f64 / n as f64,
                            blocked = n_blocked,
                            calls = n,
                            "remote.hop.summary"
                        );
                    }
                }
                // tx is dropped here, signaling end of stream.
                // Return free heap pages to the kernel and log the heap
                // shape: in_use vs free is the fragmentation-vs-leak
                // discriminator (see v4flash_core::heap). Cheap (~ms)
                // relative to a request.
                // M7: cumulative pager hit rate. Pool size only moves this number,
                // never correctness, so it is the knob for decode throughput.
                #[cfg(feature = "v41")]
                if let Some(pg) = state.pager.as_ref() {
                    // PREFILL and DECODE are reported separately: prefill's dense
                    // path issues 384 requests per (layer, chunk) against decode's
                    // <= 6 per (layer, token), so a merged "hit rate" is a prefill
                    // number with decode rounded away. Both a per-request delta and
                    // the cumulative totals, since the LRU warms across requests.
                    let cum = pg.counters();
                    let d = cum - pager_c0;
                    let rate = |m: u64, r: u64| if r > 0 { 1.0 - m as f64 / r as f64 } else { f64::NAN };
                    let ra = pg.readahead_stats();
                    tracing::info!(
                        prefill_requests = d.prefill_requests,
                        prefill_misses = d.prefill_misses,
                        prefill_hit = rate(d.prefill_misses, d.prefill_requests),
                        prefill_read_ms = d.prefill_read_ns / 1_000_000,
                        prefill_h2d_ms = d.prefill_h2d_ns / 1_000_000,
                        // Read 0 on every real prefill until 2026-09-18:
                        // `ExpertPager::ensure` filed its time under `decode_*`
                        // while its miss COUNT went to `prefill_misses`. The
                        // per-miss cost is the number that decides whether
                        // overlapping prefill's paging is worth building, so
                        // emit it next to decode's twin rather than making the
                        // reader divide two fields that were not comparable.
                        // Read-ahead is ADVISORY: nothing fails if the kernel
                        // ignores it, so it must be observable or you cannot
                        // tell a working hint from a dead one. `ra_ranges` is
                        // fadvise calls the kernel accepted; `ra_dropped` is
                        // hints the queue refused (prefill never blocks on one).
                        ra_queued = ra.map(|r| r.0).unwrap_or(0),
                        ra_dropped = ra.map(|r| r.1).unwrap_or(0),
                        ra_ranges = ra.map(|r| r.2).unwrap_or(0),
                        prefill_ms_per_miss = if d.prefill_misses > 0 {
                            (d.prefill_read_ns + d.prefill_h2d_ns) as f64
                                / d.prefill_misses as f64
                                / 1e6
                        } else {
                            f64::NAN
                        },
                        decode_requests = d.decode_requests,
                        decode_misses = d.decode_misses,
                        decode_hit = rate(d.decode_misses, d.decode_requests),
                        decode_read_ms = d.decode_read_ns / 1_000_000,
                        decode_h2d_ms = d.decode_h2d_ns / 1_000_000,
                        decode_ms_per_miss = if d.decode_misses > 0 {
                            (d.decode_read_ns + d.decode_h2d_ns) as f64 / d.decode_misses as f64 / 1e6
                        } else { f64::NAN },
                        // Miss-phase split (M8 measurement E), decode only.
                        decode_alloc_ms = d.decode_alloc_ns / 1_000_000,
                        decode_pread_ms = d.decode_pread_ns / 1_000_000,
                        decode_repack_ms = d.decode_repack_ns / 1_000_000,
                        decode_pread_gbps = if d.decode_pread_ns > 0 {
                            d.decode_pread_bytes as f64 / d.decode_pread_ns as f64
                        } else { 0.0 },
                        // Which of the two reads is slow? The weight read is
                        // O_DIRECT (~5.9 MB); the scale read is BUFFERED (~368 KB).
                        // `setup` is whatever pread_ns has left after both.
                        decode_w_ms = d.decode_weight_ns / 1_000_000,
                        decode_w_gbps = if d.decode_weight_ns > 0 {
                            d.decode_weight_bytes as f64 / d.decode_weight_ns as f64
                        } else { 0.0 },
                        decode_sc_ms = d.decode_scale_ns / 1_000_000,
                        decode_sc_gbps = if d.decode_scale_ns > 0 {
                            d.decode_scale_bytes as f64 / d.decode_scale_ns as f64
                        } else { 0.0 },
                        decode_setup_ms = d.decode_pread_ns.saturating_sub(
                            d.decode_weight_ns + d.decode_scale_ns) / 1_000_000,
                        // Raw byte counters: `weight_bytes + scale_bytes` MUST equal
                        // `pread_bytes` (every site adds wt.len / sc.len to both), so
                        // a mismatch means a read path is bypassing the split.
                        decode_pread_mb = d.decode_pread_bytes / 1_048_576,
                        decode_w_mb = d.decode_weight_bytes / 1_048_576,
                        decode_sc_mb = d.decode_scale_bytes / 1_048_576,
                        // Which read function served these misses?
                        n_layout = d.decode_n_layout,
                        n_runs = d.decode_n_runs,
                        n_direct = d.decode_n_direct,
                        n_raw = d.decode_n_raw,
                        decode_slots = pg.decode_slots(),
                        dense_windows = pg.dense_windows(),
                        "expert pager (request)"
                    );
                    // `V41_MISS_HIST=1`: the SHAPE of this request's misses.
                    // The rate alone cannot separate capacity from policy —
                    // see `ExpertPager::miss_hist`. `total` MUST equal this
                    // request's prefill+decode misses; if it does not, the
                    // histogram is miscounting and its shape means nothing,
                    // so the mismatch is logged rather than silently trusted.
                    // Window = `V41_MISS_HIST_EVERY` requests (default 20). Over one
                    // request `distinct == total` is nearly tautological; the ratio
                    // only becomes informative once pairs have had a chance to recur.
                    // See `ExpertPager::take_miss_shape`.
                    static MH_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let every: u64 = std::env::var("V41_MISS_HIST_EVERY")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(20);
                    let nth = MH_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let closing = every > 0 && nth % every == 0;
                    if let Some(ms) = pg.take_miss_shape(closing) {
                        if closing && ms.total > 0 {
                            // Cross-check only holds for a 1-request window; across a
                            // longer one the counter delta covers just the last request.
                            let want = if every == 1 { d.prefill_misses + d.decode_misses } else { ms.total };
                            let by_layer: Vec<String> =
                                ms.by_layer.iter().map(|v| v.to_string()).collect();
                            tracing::info!(
                                window_requests = every,
                                miss_total = ms.total,
                                counter_misses = want,
                                agrees = ms.total == want,
                                distinct_pairs = ms.distinct,
                                // >1 means pairs MISS AGAIN after being dropped: capacity.
                                // ~1 means each is fetched once and never returns: compulsory.
                                refetch_ratio = format!("{:.2}", ms.total as f64 / ms.distinct.max(1) as f64),
                                top16_pct = format!("{:.1}", ms.top16_pct),
                                by_layer = by_layer.join(","),
                                "expert pager miss SHAPE"
                            );
                        }
                    }
                    tracing::info!(
                        prefill_requests = cum.prefill_requests,
                        prefill_misses = cum.prefill_misses,
                        prefill_hit = rate(cum.prefill_misses, cum.prefill_requests),
                        decode_requests = cum.decode_requests,
                        decode_misses = cum.decode_misses,
                        decode_hit = rate(cum.decode_misses, cum.decode_requests),
                        "expert pager (cumulative)"
                    );
                    // Per-miss phase split (M8 measurement E): the pager's host read
                    // is two cached preads of the HF shard plus a scalar HF->ggml
                    // MXFP4 repack; only the pread is SSD-bound.
                    let (calls, alloc_ns, pread_ns, repack_ns, bytes) =
                        v4flash_core::hf_v41::expert_read_profile();
                    tracing::info!(
                        role_reads = calls,
                        alloc_ms = alloc_ns / 1_000_000,
                        pread_ms = pread_ns / 1_000_000,
                        repack_ms = repack_ns / 1_000_000,
                        pread_gb = bytes as f64 / 1e9,
                        pread_gbps = if pread_ns > 0 { bytes as f64 / pread_ns as f64 } else { 0.0 },
                        us_per_role_alloc = if calls > 0 { alloc_ns / calls / 1000 } else { 0 },
                        us_per_role_pread = if calls > 0 { pread_ns / calls / 1000 } else { 0 },
                        us_per_role_repack = if calls > 0 { repack_ns / calls / 1000 } else { 0 },
                        "expert read phases (cumulative)"
                    );
                }
                let hs = v4flash_core::heap::trim_and_stats();
                tracing::info!(
                    rss_mib = v4flash_core::heap::rss_bytes() >> 20,
                    heap_in_use_mib = hs.in_use >> 20,
                    heap_free_mib = hs.free >> 20,
                    heap_arena_mib = hs.arena >> 20,
                    heap_mmapped_mib = hs.mmapped >> 20,
                    trimmed = hs.trimmed,
                    "host heap after request"
                );
            }
            EngineRequest::Shutdown { ack } => {
                tracing::info!("worker received shutdown");
                save_live_if_dirty(&mut state);
                let _ = ack.send(());
                break;
            }
        }
    }
    tracing::info!("engine worker channel closed; shutting down");
    let _ = state.engine.shutdown();
}

pub(crate) fn short_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Compact forensic fingerprint of the worker's KV-related state.
/// Host-side only — no GPU work. Used at restore / cancel / decode-end
/// transitions to pin down wrong-KV-attended bugs by correlating
/// snapshot hashes with the in-VRAM counters that should match them.
/// `V41_PROBE_FPRINT=1`: content hashes of the KV state that a rollback is
/// supposed to restore, taken before a speculative probe and after its
/// rollback. Any component that differs is state the rollback does NOT restore
/// — which is a correctness bug, because decode reads it next.
///
/// Hashes only the ACTIVE region of each buffer: rows a probe wrote past
/// `n_raw` / `n_comp` are expected to differ and are never read.
#[cfg(feature = "v41")]
fn probe_fingerprint(state: &WorkerState) -> eyre::Result<Vec<(String, u64)>> {
    let mut out = Vec::new();
    let mut h = |name: String, v: &[u8]| out.push((name, u64::from_le_bytes(
        blake3::hash(v).as_bytes()[..8].try_into().unwrap(),
    )));
    for (li, l) in state.state.layers.iter().enumerate() {
        // Raw SWA window, active region only.
        let total_rows = v4flash_kernels::het::state::KV_CACHE_ROWS;
        let row = l.kv_cache.len() / total_rows.max(1);
        if row > 0 && l.n_raw > 0 {
            let off = (l.raw_off as usize) * row;
            let len = (l.n_raw as usize) * row;
            if off + len <= l.kv_cache.len() {
                let mut buf = vec![0u16; len];
                l.kv_cache.slice_view(off, len).copy_to_host(&mut buf)?;
                let bytes: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
                h(format!("L{li}.kv_raw"), &bytes);
            }
        }
        for (tag, cs) in [("main", l.compressor.as_ref()), ("idx", l.indexer_compressor.as_ref())]
        {
            let Some(cs) = cs else { continue };
            let mut sk = vec![0f32; cs.state_kv.len()];
            cs.state_kv.copy_to_host(&mut sk)?;
            let b: Vec<u8> = sk.iter().flat_map(|v| v.to_le_bytes()).collect();
            h(format!("L{li}.{tag}.state_kv"), &b);
            let mut ss = vec![0f32; cs.state_score.len()];
            cs.state_score.copy_to_host(&mut ss)?;
            let b: Vec<u8> = ss.iter().flat_map(|v| v.to_le_bytes()).collect();
            h(format!("L{li}.{tag}.state_score"), &b);
            h(format!("L{li}.{tag}.n_comp"), &cs.n_comp.to_le_bytes());
        }
    }
    Ok(out)
}

pub(crate) fn state_fingerprint(state: &WorkerState) -> String {
    let (live_h, live_pos, live_toks) = match &state.live {
        Some(l) => {
            let bytes: Vec<u8> = l.tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
            let h = blake3::hash(&bytes);
            (
                short_hex(&h.as_bytes()[..4]),
                Some(l.pos),
                Some(l.tokens.len()),
            )
        }
        None => ("none".to_string(), None, None),
    };
    let layers = &state.state.layers;
    let l0_n_raw = layers.first().map(|l| l.n_raw).unwrap_or(0);
    let (l2_n_raw, l2_n_comp) = layers
        .get(2)
        .map(|l| {
            (
                l.n_raw,
                l.compressor.as_ref().map(|c| c.n_comp).unwrap_or(0),
            )
        })
        .unwrap_or((0, 0));
    let (sum_raw, sum_comp) = layers.iter().fold((0u64, 0u64), |(r, c), layer| {
        (
            r + layer.n_raw as u64,
            c + layer
                .compressor
                .as_ref()
                .map(|c| c.n_comp as u64)
                .unwrap_or(0),
        )
    });
    format!(
        "liveH={} live={:?}/{:?} L0.n_raw={} L2={}/{} sum={}/{}",
        live_h, live_toks, live_pos, l0_n_raw, l2_n_raw, l2_n_comp, sum_raw, sum_comp
    )
}

/// M62: derived placement path from the snapshot root (sidecar dir).
fn deepstrix_server_placement_path(snapshot_root: &std::path::Path) -> std::path::PathBuf {
    crate::expert_stats::ExpertStatsAgg::placement_path(
        &crate::expert_stats::ExpertStatsAgg::path_for(snapshot_root),
    )
}

/// M62: harvest the engine's device-side expert-selection banks into the
/// on-disk aggregate. Cheap (2 × 88 KB readback) and skipped when nothing
/// accumulated. Called from the same points that save snapshots.
pub(crate) fn flush_expert_stats(state: &mut WorkerState) {
    let ((pc, pt), (dc, dt)) = match state.engine.harvest_sel_stats() {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("expert_stats harvest failed: {e}");
            return;
        }
    };
    if pt == 0 && dt == 0 {
        return;
    }
    // THIS REQUEST's working set, before it is merged into the cumulative file.
    //
    // The on-disk aggregate is fingerprint-keyed and merged across every run, so
    // reading "distinct experts per layer" out of IT answers a union over
    // hundreds of varied prompts -- which saturates at 384 and says nothing
    // about what a cache has to hold for one workload. This harvest is the
    // per-request set, which is the number that decides whether box 2's slots
    // are enough.
    if dt > 0 || pt > 0 {
        // Harvests fire several times per request (each covers 2-3 tokens), so
        // OR them into a per-request touched-set; the request-end site reports
        // it. A single harvest is not the working set.
        {
            let mut acc = REQ_TOUCHED.lock().unwrap();
            if acc.is_empty() {
                acc.resize(dc.len(), 0u8);
            }
            for (i, &v) in dc.iter().enumerate() {
                if v > 0 {
                    acc[i] = 1;
                }
            }
            // A DSpark verify runs through the PREFILL driver, so its expert
            // picks land in `pc`, not `dc`. Measuring only `dc` counts the
            // handful of non-speculative decode steps and misses the bulk of
            // the work -- which is what box 2 is actually serving.
            let mut accp = REQ_TOUCHED_PF.lock().unwrap();
            if accp.is_empty() {
                accp.resize(pc.len(), 0u8);
            }
            for (i, &v) in pc.iter().enumerate() {
                if v > 0 {
                    accp[i] = 1;
                }
            }
        }
        let ne = v4flash_kernels::config::N_EXPERT as usize;
        let nl = v4flash_kernels::config::N_LAYER as usize;
        let mut distinct: Vec<usize> = Vec::with_capacity(nl);
        let mut cov154: Vec<f64> = Vec::with_capacity(nl);
        for l in 0..nl {
            let row = &dc[l * ne..(l + 1) * ne];
            let tot: u64 = row.iter().map(|&v| v as u64).sum();
            if tot == 0 {
                continue;
            }
            distinct.push(row.iter().filter(|&&v| v > 0).count());
            let mut s: Vec<u64> = row.iter().map(|&v| v as u64).collect();
            s.sort_unstable_by(|a, b| b.cmp(a));
            let top: u64 = s.iter().take(154).sum();
            cov154.push(top as f64 / tot as f64);
        }
        if !distinct.is_empty() {
            distinct.sort_unstable();
            cov154.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = |v: &Vec<usize>| v[v.len() / 2];
            tracing::info!(
                decode_tokens = dt,
                distinct_per_layer_median = med(&distinct),
                distinct_per_layer_min = distinct[0],
                distinct_per_layer_max = distinct[distinct.len() - 1],
                top154_coverage_median = format!("{:.3}", cov154[cov154.len() / 2]),
                "expert.working_set (THIS request, not the cumulative file)"
            );
        }
    }
    state.expert_stats.merge_harvest(&pc, pt, &dc, dt);
    if let Err(e) = state.expert_stats.save(&state.expert_stats_path) {
        tracing::warn!("expert_stats save failed: {e}");
        return;
    }
    // Derived placement for the NEXT server start (loader-format txt).
    let alpha: f64 = std::env::var("DGPU_HOT_ALPHA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    let placement = crate::expert_stats::ExpertStatsAgg::placement_path(&state.expert_stats_path);
    if let Err(e) = state.expert_stats.write_placement(&placement, alpha) {
        tracing::warn!("expert_stats placement write failed: {e}");
    }
    tracing::debug!(
        prefill_tokens = state.expert_stats.prefill.tokens,
        decode_tokens = state.expert_stats.decode.tokens,
        "expert_stats flushed"
    );
}

/// If the live session has uncommitted state, persist it to disk and
/// clear its dirty flag. Called before any operation that would evict
/// the live state (conversation switch, shutdown).
pub(crate) fn save_live_if_dirty(state: &mut WorkerState) {
    flush_expert_stats(state);
    let Some(live) = &state.live else { return };
    if !live.dirty {
        return;
    }
    let tokens = live.tokens.clone();
    let image_spans = live.image_spans.clone();
    let session_id = live.session_id.clone();
    match snapshot::save(
        &state.state,
        &tokens,
        &image_spans,
        state.dgpu,
        state.igpu,
        &state.model_fingerprint,
        state.snapshot_index.root(),
        state.vocab.as_ref(),
        &state.byte_decoder,
        session_id.as_deref(),
    ) {
        Ok(entry) => {
            let hash = entry.hash;
            let n = entry.token_count;
            state.snapshot_index.insert(entry);
            if let Some(sid) = session_id {
                state.snapshot_index.session_to_hash.insert(sid, hash);
            }
            if let Some(l) = &mut state.live {
                l.dirty = false;
            }
            let fp = state_fingerprint(state);
            tracing::info!(
                tokens = n,
                snap_hash = %short_hex(&hash[..4]),
                total_disk = state.snapshot_index.total_bytes(),
                fp = %fp,
                "saved live to disk"
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "snapshot.save failed; live not persisted");
        }
    }
}

pub(crate) fn handle_generate_stream(
    state: &mut WorkerState,
    mut req: GenerateReq,
    session_id: Option<String>,
    cancel: Arc<AtomicBool>,
    tx: &mpsc::Sender<WorkerEvent>,
) -> eyre::Result<()> {
    if req.tokens.is_empty() {
        return Err(eyre!("generate: empty tokens"));
    }
    if (req.tokens.len() as u32) >= state.n_kv_max {
        return Err(eyre!(
            "generate: prompt length {} >= n_kv_max {}",
            req.tokens.len(),
            state.n_kv_max
        ));
    }

    // Pre-flight cancel check. The request may have sat in the
    // engine queue while a long generation occupied the worker; the
    // HTTP client could have disconnected in the meantime. Bail
    // before doing any GPU work.
    if cancel.load(Ordering::Relaxed) {
        tracing::info!("generate: cancelled before prefill (client gone)");
        return Ok(());
    }

    // Forensic: per-request entry fingerprint. Correlates inbound
    // req.tokens.len() with the in-VRAM state inherited from the
    // previous turn. Wrong-KV-attended bugs show up here as a live
    // hash + counter set that doesn't match the previous turn's exit
    // fingerprint, or a non-None live whose hash doesn't byte-prefix
    // req.
    tracing::debug!(
        req_len = req.tokens.len(),
        sid = ?session_id,
        fp = %state_fingerprint(state),
        "generate: entry"
    );

    // Strip the trailing `<think>`/`</think>` marker from the rendered
    // prompt. The marker is "transient" — letta's history-replay of
    // this turn always starts with `</think>` (never `<think>`), so
    // saving a snapshot whose bytes include the trailing `<think>`
    // would byte-diverge against every subsequent turn's request.
    //
    // Strategy: prefill everything UP TO the marker, save the
    // snapshot here (canonical bytes match letta's future replay),
    // then forward the marker manually as the first sampling input.
    // The model still thinks/responds; we just don't bake the marker
    // into the snapshot's saved tokens. See [[think-cache-design]].
    let prompt_tokens = req.tokens.len() as u32;
    let trailing_marker: Option<i32> = req
        .tokens
        .last()
        .copied()
        .filter(|&t| t == TOK_THINK_BEGIN || t == TOK_THINK_END);
    if trailing_marker.is_some() {
        req.tokens.truncate(req.tokens.len() - 1);
    }

    // Vision-Exp: run the ViT + aligner once for every image in this
    // request, up front. The result is a 4096-d row per BLOCK token
    // (aligner rows at IMAGE slots, sentinel vectors at START / PAD /
    // NEWLINE / END), which `prefill_suffix` broadcasts into HC_DIM the
    // same way `embed_lookup` does for text. Doing it here — before any
    // cache decision — means a tower failure is reported before we
    // disturb the KV cache. Repeat turns of the same conversation do NOT
    // re-run the ViT: `encode_request_images` memoises the aligner rows
    // by content hash (`WorkerState::vit_rows`), so a replayed image
    // costs only the host-side `place_rows` scatter.
    let vl = encode_request_images(state, &req)?;

    // KV-cache reuse decision. There are three cases:
    //   1. No live session, or new request diverges from live mid-prefix:
    //        reset in place, full prefill from pos=0.
    //   2. New request strictly extends live (lcp == live.tokens.len()):
    //        prefill only the suffix at pos0 = live.pos.
    //   3. New request equals live exactly (lcp == req.len()):
    //        no prefill — sample from existing logits in dgpu_scratch.
    // Compute LCP at the BYTE level — robust to tokenizer non-determinism
    // (model samples one tokenization; letta re-encodes the same text
    // as a different split). We track both sides separately because
    // a `(live, req)` byte-aligned match can have different token counts.
    let (lcp_live, lcp_req, live_len) = match &state.live {
        Some(live) => {
            let res = byte_aligned_lcp_vl(
                &live.tokens,
                &live.image_spans,
                &req.tokens,
                &req.image_spans,
                state.vocab.as_ref(),
                &state.byte_decoder,
            );
            if res.bridged_tokens > 0 {
                tracing::debug!(
                    bridged = res.bridged_tokens,
                    first_bridge = ?res.first_bridge,
                    live = res.live_tokens,
                    req = res.req_tokens,
                    "byte-aligned LCP bridged tokenization divergence"
                );
            }
            (res.live_tokens, res.req_tokens, live.tokens.len())
        }
        None => (0, 0, 0),
    };

    // The byte-aligned LCP may end with `lcp_live < live_len` even when
    // the byte stream up to `lcp_live` matches the request exactly —
    // that just means live had extra tokens beyond the shared bytes.
    // We treat `lcp_live == live_len` as the "live fully covers a
    // prefix of req" case (extend / exact); otherwise fall back to
    // disk or full reprefill.
    if state.live.is_none() || lcp_live < live_len {
        // Try the sessionId hot-cache first.
        let disk_hit_session = session_id
            .as_deref()
            .and_then(|sid| state.snapshot_index.lookup_session(sid, &req.tokens));
        let disk_hit_walk = state.snapshot_index.find_longest_prefix(
            &req.tokens,
            &req.image_spans,
            TOK_EOS,
            TOK_ASSISTANT,
            TOK_USER,
            state.vocab.as_ref(),
            &state.byte_decoder,
        );
        let disk_hit = match (disk_hit_session, disk_hit_walk) {
            (Some(a), Some(b)) => Some(if a.0 >= b.0 { a } else { b }),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        // Disk snapshot lookup. We can't actually USE the in-VRAM
        // partial match (lcp_req bytes) here — without truncation
        // support we're about to throw all of it away and full-reset.
        // So restoring from disk is worthwhile whenever the snapshot
        // covers MORE THAN ZERO req tokens (with a small threshold to
        // avoid the ~1-2 s restore overhead for tiny snapshots that
        // wouldn't pay for themselves).
        const DISK_RESTORE_MIN_TOKENS: usize = 64;
        // V4.1: a KV snapshot does NOT carry the Engram n-gram state. EngramCtx keeps a running
        // compressed-id sequence over every token seen, and restoring a snapshot skips the prefill
        // that would have appended those ids — so the next decode asks for rows at position N
        // while the sequence still holds only the ids from earlier requests
        // ("engram: position 965 but 44 tokens tracked"), and the request fails. The sequence is a
        // pure function of the token ids, so the real fix is to rebuild it on restore (or persist
        // it beside the KV). FIXED 2026-09-17: `EngramCtx::rebuild` regenerates the sequence from
        // the restored token list right after the byte-prefix verification below, so reuse is on.
        // `DEEPSTRIX_SNAPSHOT_REUSE=0` forces the full-prefill path (the A/B control for a restore).
        let disk_hit = if std::env::var("DEEPSTRIX_SNAPSHOT_REUSE").as_deref() == Ok("0") {
            None
        } else {
            disk_hit
        };
        if let Some((snap_req_tokens, snap_hash, snap_dir)) = disk_hit {
            if snap_req_tokens >= DISK_RESTORE_MIN_TOKENS {
                // The window between picking a request up and its `prefill` line
                // measured 12.2% of all engine-busy time (p50 0.31 s, MEAN 8.70 s,
                // max 302 s, scaling with context: 15.05 s above 50k tokens vs
                // 6.06 s below 20k). That window was bounded by LOG LINES, so it
                // lumps restore together with prompt render, tokenisation and LCP
                // matching, and the attribution was a guess. Time the two halves
                // directly instead: the reset preamble and `restore_vl` itself.
                let t_reset = std::time::Instant::now();
                save_live_if_dirty(state);
                state.state.reset_in_place(state.dgpu, state.igpu)?;
                // The drafter's KV ring is process-lifetime state and is NOT part of
                // `HetModelState`, so resetting the main KV leaves it holding the
                // previous conversation's positions. See `MtpState::reset_ring`.
                reset_drafter_ring(state);
                // `state.live` no longer describes the KV cache from
                // here on. Clear it BEFORE the restore + suffix prefill
                // so an error out of either (missing bias_vl, a span
                // straddle, a tower row-count mismatch) can't leave the
                // next request computing its LCP against a session that
                // was thrown away — the full-prefill branch below does
                // the same thing.
                state.live = None;
                let reset_ms = t_reset.elapsed().as_millis() as u64;
                let t_restore = std::time::Instant::now();
                let restored = snapshot::restore_vl(
                    &mut state.state,
                    &snap_dir,
                    state.dgpu,
                    state.igpu,
                    &state.model_fingerprint,
                    snapshot::RestoreKernels {
                        fp8: &state.engine.dgpu.comp_kv_fp8,
                        stream: &state.engine.dgpu.compute,
                    },
                );
                let restore_ms = t_restore.elapsed().as_millis() as u64;
                let restored = match restored {
                    Ok(r) => Some(r),
                    Err(e) => {
                        // A snapshot that cannot be loaded (truncated
                        // file, refused format conversion) is a cache
                        // MISS, not a failed request: wipe whatever was
                        // partially written, evict the entry so the next
                        // matching request does not re-read it, and fall
                        // through to the full prefill below (same shape
                        // as the byte-prefix verification failure).
                        tracing::warn!(
                            snap_hash = %short_hex(&snap_hash[..4]),
                            error = %e,
                            "snapshot restore failed; evicting and prefilling from scratch"
                        );
                        state.state.reset_in_place(state.dgpu, state.igpu)?;
                        // The drafter's KV ring is process-lifetime state and is NOT part of
                        // `HetModelState`, so resetting the main KV leaves it holding the
                        // previous conversation's positions. See `MtpState::reset_ring`.
                        reset_drafter_ring(state);
                        state.live = None;
                        state.snapshot_index.evict(&snap_hash, "restore failed");
                        None
                    }
                };
                if let Some(crate::snapshot::RestoredSnapshot {
                    tokens: loaded,
                    image_spans: loaded_spans,
                }) = restored
                {
                    let loaded_len = loaded.len() as u32;
                    // Verify the snapshot's BYTE stream is actually a prefix
                    // of req. With byte-hashed keys (format v2) this should
                    // always be true when find_longest_prefix returned this
                    // entry; the session-hint path bypasses that check so
                    // we re-verify here.
                    let verify = byte_aligned_lcp_vl(
                        &loaded,
                        &loaded_spans,
                        &req.tokens,
                        &req.image_spans,
                        state.vocab.as_ref(),
                        &state.byte_decoder,
                    );
                    if verify.live_tokens != loaded.len() {
                        tracing::warn!(
                            loaded_len = loaded.len(),
                            verify_live = verify.live_tokens,
                            verify_req = verify.req_tokens,
                            snap_hash = %short_hex(&snap_hash[..4]),
                            "restored snapshot bytes are NOT a prefix of the request; falling back"
                        );
                        state.state.reset_in_place(state.dgpu, state.igpu)?;
                        // The drafter's KV ring is process-lifetime state and is NOT part of
                        // `HetModelState`, so resetting the main KV leaves it holding the
                        // previous conversation's positions. See `MtpState::reset_ring`.
                        reset_drafter_ring(state);
                        state.live = None;
                    } else {
                        let _ = state.snapshot_index.touch(&snap_hash);
                        // The snapshot carries the KV, not the Engram n-gram sequence; the
                        // suffix prefill / first decode will ask for rows at `loaded_len`
                        // and needs `compressed.len() == loaded_len` (see `rows_for`).
                        #[cfg(feature = "v41")]
                        if let Some(ec) = state.engram.as_mut() {
                            ec.rebuild(&loaded);
                        }
                        let suffix_len = req.tokens.len() - verify.req_tokens;
                        // Diagnostic: when there's a LARGER snapshot than
                        // the one we just restored AND that snapshot shares
                        // a meaningful byte prefix with the current
                        // request, log where its bytes first diverge —
                        // tells us which turn-boundary re-render is broken.
                        // Suppress when the larger snapshot is clearly
                        // from a different conversation (tiny common
                        // prefix) — that's just LRU index noise.
                        if let Some(diag) = state.snapshot_index.diag_largest_divergence(
                            &req.tokens,
                            loaded_len,
                            state.vocab.as_ref(),
                            &state.byte_decoder,
                        ) {
                            // Threshold: larger snapshot must share at
                            // least half the request's bytes to be
                            // considered "same conversation".
                            if diag.common_byte_len * 2 >= diag.req_byte_len {
                                tracing::warn!(
                                    picked_token_count = loaded_len,
                                    largest_token_count = diag.snap_token_count,
                                    largest_byte_len = diag.snap_byte_len,
                                    req_byte_len = diag.req_byte_len,
                                    common_byte_len = diag.common_byte_len,
                                    before = %diag.before,
                                    snap_after = %diag.snap_after,
                                    req_after = %diag.req_after,
                                    "byte divergence vs largest snapshot"
                                );
                            }
                        }
                        tracing::info!(
                            req_len = req.tokens.len(),
                            restored_live = loaded_len,
                            restored_req = verify.req_tokens,
                            suffix_len,
                            snap_hash = %short_hex(&snap_hash[..4]),
                            // Splits the 12.2% pre-prefill window. Whatever these
                            // two do NOT account for is render/tokenise/LCP, which
                            // is then a subtraction rather than a guess.
                            reset_ms,
                            restore_ms,
                            mode = "restore",
                            fp = %state_fingerprint(state),
                            "prefill"
                        );
                        if suffix_len > 0 {
                            prefill_suffix(
                                state,
                                &req.tokens[verify.req_tokens..],
                                verify.req_tokens,
                                loaded_len,
                                &vl,
                                Some(&cancel),
                            )?;
                            if cancel.load(Ordering::Relaxed) {
                                tracing::info!(
                                    fp = %state_fingerprint(state),
                                    "generate: cancelled mid-prefill (restore path)"
                                );
                                // Restored snapshot is still on-GPU but
                                // we may have partially appended suffix
                                // KV beyond it. Drop live so next request
                                // reset-prefills from a known state.
                                state.live = None;
                                return Ok(());
                            }
                        }
                        let new_pos = loaded_len + suffix_len as u32;
                        // live.tokens after restore + suffix prefill: loaded
                        // (in saved token-id space) + req.tokens[lcp_req..].
                        let mut new_tokens: Vec<i32> =
                            Vec::with_capacity(loaded.len() + suffix_len);
                        new_tokens.extend_from_slice(&loaded);
                        new_tokens.extend_from_slice(&req.tokens[verify.req_tokens..]);
                        // Live spans = the snapshot's own (already in loaded
                        // index space) plus the request's suffix spans,
                        // shifted into it.
                        let mut new_spans = loaded_spans.clone();
                        new_spans.extend(shift_spans(
                            &spans_from(&req.image_spans, verify.req_tokens),
                            loaded.len() as i64 - verify.req_tokens as i64,
                        ));
                        state.live = Some(LiveSession {
                            tokens: new_tokens,
                            image_spans: new_spans,
                            pos: new_pos,
                            dirty: true,
                            session_id: session_id.clone(),
                        });
                        let (pos_after_marker, initial_in_think) =
                            save_and_forward_marker(state, trailing_marker, new_pos)?;
                        return finish_decode(
                            state,
                            req,
                            tx,
                            prompt_tokens,
                            pos_after_marker,
                            session_id,
                            cancel,
                            initial_in_think,
                        );
                }
                }
            }
        }

        if let Some(live_ref) = state.live.as_ref() {
            // Dump a window of bytes + token-ids around the divergence
            // boundary on both sides so we can diagnose what differs.
            // 30 tokens of context before the divergence point, 50
            // tokens after.
            let live_start = lcp_live.saturating_sub(30);
            let live_end = (lcp_live + 50).min(live_ref.tokens.len());
            let req_start = lcp_req.saturating_sub(30);
            let req_end = (lcp_req + 50).min(req.tokens.len());
            let live_slice = &live_ref.tokens[live_start..live_end];
            let req_slice = &req.tokens[req_start..req_end];
            let decode_to_string = |slice: &[i32]| -> String {
                let mut bytes = Vec::new();
                for &t in slice {
                    if let Some(b) = state.vocab.token_text(t) {
                        bytes.extend(gpt2_decode_token(b, &state.byte_decoder));
                    }
                }
                String::from_utf8_lossy(&bytes).into_owned()
            };
            tracing::warn!(
                lcp_live,
                lcp_req,
                live_len,
                req_len = req.tokens.len(),
                live_window_offset = live_start,
                req_window_offset = req_start,
                live_window_tokens = ?live_slice,
                req_window_tokens = ?req_slice,
                live_window_text = ?decode_to_string(live_slice),
                req_window_text = ?decode_to_string(req_slice),
                "live cache divergence (byte-aligned); resetting"
            );
        }
        save_live_if_dirty(state);
        state.state.reset_in_place(state.dgpu, state.igpu)?;
        // The drafter's KV ring is process-lifetime state and is NOT part of
        // `HetModelState`, so resetting the main KV leaves it holding the
        // previous conversation's positions. See `MtpState::reset_ring`.
        reset_drafter_ring(state);
        state.live = None;

        // Split the prefill at the first `<User>` so the system block
        // lands on disk as a standalone, conversation-agnostic
        // snapshot. Reaching the full path means no snapshot covered
        // this request, so the system prefix is genuinely absent from
        // the index (had it been there, `find_longest_prefix` would
        // have matched it at the `<User>` probe and taken the restore
        // path). Every later conversation sharing this system prompt
        // restores it instead of re-prefilling ~13-21K tokens.
        //
        // Boundary is `first_user + 1` — inclusive of `<User>` — to
        // match the probe, which hashes AFTER folding in the token it
        // triggered on.
        let sys_prefix_len = req
            .tokens
            .iter()
            .position(|&t| t == TOK_USER)
            .map(|i| i + 1)
            .filter(|&n| n >= DISK_RESTORE_MIN_TOKENS && n < req.tokens.len())
            // An image block must be prefilled inside one KV-visible
            // unit, so never split the prompt THROUGH one. This is a
            // straddle test, not a "no images on either side" test: a
            // cut with one image before it and another after it is
            // perfectly legal, and rejecting it would silently disable
            // the system-prefix snapshot (the 28.0s -> 1.1s win) for
            // every multi-image conversation. Image blocks live after
            // `<User>` today, so this should never fire at all.
            .filter(|&n| !crate::vision_prompt::spans_straddle(&req.image_spans, n));
        if let Some(n) = sys_prefix_len {
            prefill_suffix(state, &req.tokens[..n], 0, 0, &vl, Some(&cancel))?;
            if cancel.load(Ordering::Relaxed) {
                tracing::info!(
                    fp = %state_fingerprint(state),
                    "generate: cancelled mid-prefill (system-prefix path)"
                );
                return Ok(());
            }
            // Session-agnostic on purpose: this snapshot is shared by
            // every conversation using this system prompt, so it must
            // not join one lineage's R1 retention pool. Global LRU
            // still governs it, and each restore `touch()`es it — the
            // more it's reused, the safer it is from eviction.
            state.live = Some(LiveSession {
                tokens: req.tokens[..n].to_vec(),
                // `?`, not `unwrap_or_default()`: silently dropping the
                // spans would strip the content hashes out of the saved
                // snapshot's byte stream, and two different pictures
                // with the same block layout would then key to the same
                // hash. The filter above makes a straddle impossible.
                image_spans: crate::vision_prompt::spans_in_range(&req.image_spans, 0, n)
                    .wrap_err("system-prefix split cuts through an image block")?,
                pos: n as u32,
                dirty: true,
                session_id: None,
            });
            save_live_if_dirty(state);
            tracing::info!(
                sys_prefix_tokens = n,
                req_len = req.tokens.len(),
                "saved system-prefix snapshot"
            );
        }
        let (resume_from, pos0) = sys_prefix_len.map_or((0, 0u32), |n| (n, n as u32));
        prefill_suffix(
            state,
            &req.tokens[resume_from..],
            resume_from,
            pos0,
            &vl,
            Some(&cancel),
        )?;
        if cancel.load(Ordering::Relaxed) {
            tracing::info!(
                fp = %state_fingerprint(state),
                "generate: cancelled mid-prefill (full path)"
            );
            // KV cache has partial garbage but the next request's
            // reset_in_place will clear it.
            state.live = None;
            return Ok(());
        }
        tracing::info!(
            req_len = req.tokens.len(),
            lcp_live,
            lcp_req,
            live_len,
            mode = "full",
            "prefill"
        );
    } else if lcp_req < req.tokens.len() {
        // Live covers a prefix of req at the byte level. Prefill only
        // the suffix beyond `lcp_req` at the live position `lcp_live`
        // — ROPE positions stay coherent with the existing cache.
        let suffix = &req.tokens[lcp_req..];
        let pos0 = lcp_live as u32;
        tracing::info!(
            req_len = req.tokens.len(),
            lcp_live,
            lcp_req,
            live_len,
            suffix_len = suffix.len(),
            mode = "extend",
            "prefill"
        );
        // The prefill path addresses raw KV slots ABSOLUTELY: it appends at
        // slot `n_raw_before`, computes its per-row SWA windows from slot 0
        // (`causal_end - SWA_WINDOW`), and resets `raw_off = 0` at the
        // post-chunk eviction. Decode does NOT: once `n_raw` reaches
        // SWA_WINDOW it stops growing and advances `raw_off` instead
        // (forward_layer.rs, "ls.raw_off += 1"), so the live window is
        // [raw_off, raw_off + n_raw).
        //
        // Extending a live cache therefore has to reconcile the two models
        // FIRST. Without this, any turn that generated more than SWA_WINDOW
        // (128) tokens leaves raw_off > 0, and the suffix is appended at
        // absolute slot `n_raw` -- BELOW the live window base -- overwriting
        // live KV, while attention reads a window containing stale rows.
        // Wrong attention, no error. DSpark already does this before both of
        // its verifies; the ordinary chat "extend" path did not.
        state.engine.normalize_raw_windows(&mut state.dgpu_scratch, &mut state.state)?;
        prefill_suffix(state, suffix, lcp_req, pos0, &vl, Some(&cancel))?;
        if cancel.load(Ordering::Relaxed) {
            tracing::info!(
                fp = %state_fingerprint(state),
                "generate: cancelled mid-prefill (extend path)"
            );
            // live state's pos no longer reflects the KV cache (we
            // wrote some suffix tokens beyond live.pos). Drop it; the
            // next request will reset_in_place.
            state.live = None;
            return Ok(());
        }
    } else {
        // Exact byte match — no prefill needed; existing logits in
        // dgpu_scratch are for position `lcp_live`.
        tracing::info!(
            req_len = req.tokens.len(),
            lcp_live,
            lcp_req,
            mode = "exact",
            "prefill"
        );
    }

    // Live now reflects the EXTENDED state. The KV cache positions are
    // anchored by live tokens (the original sampled IDs we forwarded);
    // we record req.tokens here so the next request's byte_aligned_lcp
    // can fast-path on the matching prefix. lcp_live + (req-len-lcp_req)
    // is the new pos.
    let suffix_extend = req.tokens.len().saturating_sub(lcp_req);
    let new_pos = (lcp_live + suffix_extend) as u32;
    // The live.tokens we record: take live's prefix [..lcp_live] + req's
    // suffix [lcp_req..]. The prefix is what's actually in the KV cache;
    // the suffix is what we just forwarded.
    let rebuilt: eyre::Result<(Vec<i32>, Vec<ImageSpan>)> = match &state.live {
        Some(l) if lcp_live > 0 => {
            let mut v = Vec::with_capacity(lcp_live + suffix_extend);
            v.extend_from_slice(&l.tokens[..lcp_live]);
            v.extend_from_slice(&req.tokens[lcp_req..]);
            // `?`, not `unwrap_or_default()`. A straddle means the
            // byte-aligned LCP stopped INSIDE an image block; dropping
            // the spans would leave `live.tokens` carrying image ids
            // with no content hashes, and then both the live LCP and the
            // snapshot key would treat two different pictures with the
            // same layout as identical bytes. That is precisely the
            // aliasing the span hashes exist to prevent, so fail loudly.
            crate::vision_prompt::spans_in_range(&l.image_spans, 0, lcp_live).map(|mut sp| {
                sp.extend(shift_spans(
                    &spans_from(&req.image_spans, lcp_req),
                    lcp_live as i64 - lcp_req as i64,
                ));
                (v, sp)
            })
        }
        _ => Ok((req.tokens.clone(), req.image_spans.clone())),
    };
    let (new_live_tokens, new_live_spans) = match rebuilt {
        Ok(v) => v,
        Err(e) => {
            // The KV cache is fine but we cannot describe it honestly.
            // Drop the session so the next request reset-prefills.
            state.live = None;
            return Err(e).wrap_err("live image spans straddle the byte-aligned LCP");
        }
    };
    state.live = Some(LiveSession {
        tokens: new_live_tokens,
        image_spans: new_live_spans,
        pos: new_pos,
        // Force dirty=true so save_and_forward_marker below actually
        // writes. State is canonical (no `<think>` baked in) AND new
        // (we just appended the request's suffix), so it deserves a
        // save.
        dirty: true,
        session_id: session_id.clone(),
    });

    let (pos_after_marker, initial_in_think) =
        save_and_forward_marker(state, trailing_marker, new_pos)?;

    finish_decode(
        state,
        req,
        tx,
        prompt_tokens,
        pos_after_marker,
        session_id,
        cancel,
        initial_in_think,
    )
}

/// Save the start-of-think snapshot, then forward the trailing
/// `<think>`/`</think>` marker (if any) into the KV cache.
///
/// The snapshot's saved tokens (== live.tokens at this point) end at
/// `<｜Assistant｜>`, which is a byte prefix of what the client replays for
/// this turn on every subsequent request, whichever marker follows it. The
/// marker we forward AFTER the save is transient w.r.t. snapshot identity but
/// still required so the model starts sampling in the right "thinking vs
/// responding" mode.
///
/// Returns `(pos_after_marker, initial_in_think)` — the KV
/// position the next sampled token will be written at, and whether
/// we should treat the first sampled token as part of the model's
/// reasoning trace.
fn save_and_forward_marker(
    state: &mut WorkerState,
    trailing_marker: Option<i32>,
    pos_at_save: u32,
) -> eyre::Result<(u32, bool)> {
    save_live_if_dirty(state);

    let mut pos_after_marker = pos_at_save;
    let initial_in_think = if let Some(marker) = trailing_marker {
        let mut residual = vec![0f32; HC_DIM as usize];
        embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, marker, &mut residual);
        forward_one!(state, residual, pos_after_marker, marker)?;
        pos_after_marker += 1;
        if let Some(live) = state.live.as_mut() {
            live.pos = pos_after_marker;
            // TOK_THINK_END is canonical: with no tools in play the
            // replay of a historical assistant turn opens with `</think>`
            // alone (`prompt.rs` `keep_reasoning`), so pushing it keeps
            // live.tokens equal to the next request's bytes.
            //
            // TOK_THINK_BEGIN is NOT pushed. Note that with tools present
            // the replay DOES now carry `<think></think>` on every
            // historical assistant turn (prompt.rs, template line 217), so
            // this is no longer "never in the replay" — but pushing it here
            // would not buy the extend path back. Any turn that actually
            // reasons forwards its reasoning tokens into KV without
            // recording them in live.tokens, so `live.pos >
            // live.tokens.len()` at end of turn and `finish_decode` drops
            // the live session outright (see the transient-token cleanup
            // there). The in-VRAM extend path is structurally unavailable
            // after ANY non-empty thinking turn, tools or not; continuity
            // across turns comes from the disk snapshot saved just above,
            // whose bytes stop at `<｜Assistant｜>` and therefore prefix
            // either replay shape.
            if marker == TOK_THINK_END {
                live.tokens.push(marker);
            }
        }
        marker == TOK_THINK_BEGIN
    } else {
        false
    };
    Ok((pos_after_marker, initial_in_think))
}

// `prompt_tokens` is reported back via OpenAI's usage block (== request
// token count). `start_pos` is the KV-cache position the next sampled
// token will be written at — equals live.pos after byte-aligned
// extend; differs from prompt_tokens when live's token count for the
// matched byte prefix differs from the request's.
/// Replay the drafter over the prompt's last captured positions so its KV ring
/// is warm when generation starts. See the call site in `finish_decode`.
///
/// `tokens` is the CANONICAL live sequence (index == absolute position) and
/// `start_pos` the position the
/// first generated token will occupy, so the captured rows cover absolute
/// positions `[pos0, pos0 + n)` with `pos0` recorded by the capture itself.
/// For each captured position p we need (residual @ p, token @ p+1) -- the same
/// pairing `dspark_draft` and the accept path use -- so the LAST captured row
/// is skipped: the token after it is the one generation is about to produce.
#[cfg(feature = "v41")]
fn seed_mtp_ring(state: &mut WorkerState, tokens: &[i32], start_pos: u32) -> eyre::Result<()> {
    use v4flash_kernels::config::{HC_DIM, N_EMBD};
    let ne = N_EMBD as usize;
    let nsrc = v4flash_kernels::het::mtp::MTP_SRC_LAYERS.len();
    let cap = v4flash_kernels::het::batch_scratch::MTP_CAP_ROWS;

    // Both lanes may have captured; take whichever holds the LATER positions.
    let (n, pos0, from_b) = {
        let (na, pa) = (state.bd_a.mtp_captured, state.bd_a.mtp_captured_pos0);
        let (nb, pb) = (state.bd_b.mtp_captured, state.bd_b.mtp_captured_pos0);
        if nb > 0 && (na == 0 || pb >= pa) { (nb, pb, true) } else { (na, pa, false) }
    };
    if n == 0 {
        return Ok(());
    }
    // The capture must lie inside the COMMITTED context, i.e. end at or before
    // the position the first generated token will take. It need not end exactly
    // there: `save_and_forward_marker` can forward a few more tokens after the
    // prefill, which is the common chat case -- requiring equality there
    // silently disabled seeding entirely. Anything ENDING PAST `start_pos` is a
    // stale buffer from an earlier request and must not be used.
    if pos0 as usize + n > start_pos as usize {
        tracing::info!(n, pos0, start_pos, "dspark: capture ends past start_pos, not seeding");
        return Ok(());
    }

    // `whole` is read at `sl * cap * ne + r * ne` for r in [0, n), so the
    // capture count the driver recorded must fit the buffer it wrote into.
    assert!(
        n <= cap,
        "dspark seed: prefill recorded {n} captured mtp_src rows but the buffer holds {cap}"
    );
    let mut whole = vec![0.0f32; nsrc * cap * ne];
    if from_b {
        state.bd_b.mtp_src.copy_to_host(&mut whole)?;
    } else {
        state.bd_a.mtp_src.copy_to_host(&mut whole)?;
    }

    let mut seeded = 0usize;
    let mut no_token = 0usize;
    // Skip the last row: its (p+1) token has not been generated yet.
    for r in 0..n.saturating_sub(1) {
        let p = pos0 + r as u32;
        // Token AT p+1, from the request's own sequence.
        let idx = (p + 1) as usize;
        let Some(&tok) = tokens.get(idx) else { no_token += 1; break };
        let mut mh = Vec::with_capacity(nsrc * ne);
        for sl in 0..nsrc {
            let o = sl * cap * ne + r * ne;
            mh.extend_from_slice(&whole[o..o + ne]);
        }
        let mut tr = vec![0.0f32; HC_DIM as usize];
        embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, tok, &mut tr);
        let m = state.mtp.as_mut().expect("mtp");
        state.engine.dspark_advance_ring(&mut m.state, &m.w, p, &mh, &tr, &m.noise_row)?;
        seeded += 1;
    }
    tracing::info!(
        seeded, n, pos0, start_pos, no_token,
        seq_len = tokens.len(),
        "dspark: seeded drafter ring from prefill"
    );
    Ok(())
}

fn finish_decode(
    state: &mut WorkerState,
    req: GenerateReq,
    tx: &mpsc::Sender<WorkerEvent>,
    prompt_tokens: u32,
    start_pos: u32,
    _session_id: Option<String>,
    cancel: Arc<AtomicBool>,
    initial_in_think: bool,
) -> eyre::Result<()> {
    let mut pos = start_pos;
    // ---- DSpark PREFILL RING SEEDING -----------------------------------
    // The drafter attends over a MTP_WINDOW(128)-entry KV ring. At the start of
    // generation that ring is EMPTY: the only writes come from accepted
    // positions, so it takes ~128 tokens to fill and the drafter spends most of
    // a normal reply attending over a mostly-empty window. That is a large part
    // of why accept E (~2.5) sits so far under the oracle's seeded 4.382 --
    // NOT, as previously assumed, because E had reached its ceiling.
    //
    // The prompt's positions are real context the drafter should already have.
    // The batched prefill captured the last `mtp_captured` main-model residuals
    // (see `MTP_CAP_ROWS`), so replay the drafter over them here -- same
    // `advance_ring` the accept path uses: full layer forward (ring + carry),
    // skipping only the exit. `V41_DSPARK_SEED_RING=0` disables.
    #[cfg(feature = "v41")]
    if state.mtp.is_some()
        && std::env::var("V41_DSPARK_SEED_RING").as_deref() != Ok("0")
    {
        // `live.tokens` is the CANONICAL sequence backing the KV cache, so its
        // index IS the absolute position. `req.tokens` is not: on the "extend"
        // path the live prefix and the request prefix have different lengths
        // (`lcp_live != lcp_req`) and indexing it by position would seed the
        // ring with the wrong tokens.
        let seq: Vec<i32> = state.live.as_ref().map(|l| l.tokens.clone()).unwrap_or_default();
        if let Err(e) = seed_mtp_ring(state, &seq, start_pos) {
            // Seeding is a pure accept-rate optimisation: the ring is a cache of
            // the drafter's own attention, and a cold one only costs acceptance.
            // Never fail a request over it.
            tracing::warn!(error = %e, "dspark: prefill ring seeding failed, continuing cold");
        }
    }
    // Decode-loop wall, reported as `decode.loop.summary` at the end. Its
    // ms/token is what a harness must compare against `het.token.summary`;
    // curl-wall / completion_tokens ALSO carries the prefill and snapshot save.
    let loop_t0 = std::time::Instant::now();

    // `GenerateReq` is `pub` with no `Default`, so a future caller could hand
    // us the field's natural zero value (or a NaN). `launch_multinomial_topp`
    // rejects those with an `Err` — which would surface as a failed request
    // *after* prefill has already run — so re-clamp here the same way the
    // HTTP layer does, and treat NaN as "no truncation" rather than an error.
    let top_p = if req.top_p.is_nan() {
        1.0
    } else {
        req.top_p.clamp(crate::openai::handler::MIN_TOP_P, 1.0)
    };
    let sample_mode = if req.temperature <= 0.0 {
        SampleMode::Argmax
    } else {
        SampleMode::Multinomial {
            temperature: req.temperature,
            min_p_rel: req.min_p_rel,
            top_p,
        }
    };
    let mut rng = SamplerRng::new(req.seed);
    // Per-request: the first verify after a prefill legitimately starts a fresh
    // stream, so a value left from the PREVIOUS request is not a desync (it
    // fired three times that way -- all at the first verify, same `got`).
    DSPARK_EXPECT_NEXT.store(-1, std::sync::atomic::Ordering::Relaxed);
    // `V41_DUMP_FIRST_LOGITS=<path>`: dump the decode logits for the first token
    // after the prompt (the oracle's `logits_last` reference point) so KL(oracle
    // || engine-decode) can be computed. One-shot correctness probe.
    if let Ok(path) = std::env::var("V41_DUMP_FIRST_LOGITS") {
        let nv = v4flash_kernels::config::N_VOCAB as usize;
        if state.dgpu_scratch.logits.len() >= nv {
            let mut dl = vec![0.0f32; nv];
            state.dgpu_scratch.logits.slice_view(0, nv).copy_to_host(&mut dl)?;
            let bytes: Vec<u8> = dl.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, &bytes).ok();
            // Also dump the exact input token ids, so the oracle can be run with
            // `--prompt-ids` on the identical (chat-templated) sequence and
            // KL(oracle || engine) computed at the same position.
            if let Some(live) = state.live.as_ref() {
                let ids: Vec<String> = live.tokens.iter().map(|t| t.to_string()).collect();
                std::fs::write(format!("{path}.ids"), ids.join(",")).ok();
            }
            tracing::info!(path = %path, nv, "dumped first-token decode logits + input ids");
        }
    }
    let t_sample = std::time::Instant::now();
    let mut next = state
        .engine
        .sample_next(&mut state.dgpu_scratch, sample_mode, rng.next_f32())?;
    v4flash_kernels::het::trace::phase::add(
        &v4flash_kernels::het::trace::phase::CALLER_SAMPLE_NS,
        t_sample.elapsed().as_nanos() as u64,
    );
    let mut completion_tokens: u32 = 1;

    let mut residual = vec![0f32; HC_DIM as usize];
    let max_new = req.max_new as u32;
    // The caller already forwarded the trailing think marker (if any)
    // — see handle_generate_stream — and passes the resulting
    // in_think state here directly. (req.tokens no longer includes
    // the trailing marker now that we strip it for the snapshot.)
    let mut in_think = initial_in_think;
    let _ = start_pos;
    let tok_dsml = state.vocab.dsml_id.unwrap_or(-1);
    let trace_tokens =
        std::env::var("DEEPSTRIX_TRACE_TOKENS").is_ok_and(|v| !v.is_empty() && v != "0");
    // When we sample TOK_DSML, dump the next N tokens too so we can
    // see what bytes are flowing into the scanner's header parse.
    let mut dump_window: usize = 0;
    // Decode-loop heartbeat. The DSML-window trace above goes silent
    // for long stretches of plain-text decode, which makes a
    // genuinely-progressing decode look identical to a hang. Emit a
    // one-line heartbeat every HEARTBEAT_INTERVAL completion tokens
    // with rolling tok/s since the last beat, so the log keeps
    // breathing.
    // `V41_VERIFY_PROBE=K`: every steady-state token, speculatively ingest K extra
    // tokens into KV, then roll them back with `HetModelState::rollback_kv`, and
    // require the generation to come out unchanged. This is DSpark's REJECT path
    // exercised without a drafter — the one piece of speculative decoding that is
    // testable in isolation. Off unless set; it roughly doubles decode time.
    // Accepts a single K or a comma list ("2,4,6,8"), cycled per token, so one
    // run sweeps every verify width instead of one server load per B.
    #[cfg(feature = "v41")]
    let verify_probe_ks: Vec<usize> = std::env::var("V41_VERIFY_PROBE")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    #[cfg(feature = "v41")]
    let verify_probe_k: usize = if verify_probe_ks.is_empty() { 0 } else { 1 };
    // `V41_VERIFY_BATCHED=1`: make the probe do ONE batched forward over K tokens
    // (the real DSpark verify step) instead of K sequential decodes (the reject
    // path). This is the measurement the whole 30 tok/s projection rests on.
    // `V41_DSPARK=accept`: actually ACT on the drafts — one batched verify per
    // step, keep the agreed prefix, roll the rest back. `V41_DSPARK=1` (shadow)
    // drafts and scores without touching the output.
    #[cfg(feature = "v41")]
    let dspark_accept: bool = matches!(std::env::var("V41_DSPARK").as_deref(), Ok("accept"));
    #[cfg(feature = "v41")]
    fn verify_decode_path() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| std::env::var("V41_VERIFY_DECODE_PATH").as_deref() == Ok("1"))
    }
    #[cfg(feature = "v41")]
    let verify_probe_batched: bool =
        matches!(std::env::var("V41_VERIFY_BATCHED").as_deref(), Ok("1") | Ok("on"));
    let heartbeat_interval: u32 = std::env::var("DEEPSTRIX_HEARTBEAT_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64u32)
        .max(1);
    let mut hb_last_count: u32 = 0;
    let mut hb_last_at = std::time::Instant::now();
    #[cfg(feature = "v41")]
    let mut hb_last_pager = state.pager.as_ref().map(|p| p.counters()).unwrap_or_default();
    // Expert-read phase profile, deltaed between heartbeats. Heartbeats only fire
    // inside the decode loop, so a delta is DECODE-only reads (single-threaded, so
    // the ns sums are real wall) — unlike the cumulative figure, which is dominated
    // by prefill's 4-thread dense sweeps.
    #[cfg(feature = "v41")]
    let mut hb_last_read = state.pager.as_ref().map(|p| p.counters()).unwrap_or_default();
    // Tracks whether the decode loop exited because the client cancelled.
    // FinishReason::Stop covers both natural turn-end AND cancel, so we
    // need a separate signal to disambiguate. On cancel we DROP live
    // unconditionally below — leaving live populated with canonical
    // garbage tokens (samples that streamed before cancel) lets a
    // letta-style retry extend into a KV state that includes the
    // model's own half-finished output, producing "grammatically
    // correct but semantically scrambled" garbage on the retry.
    let mut was_cancelled = false;
    #[cfg(feature = "v41")]
    let mut xcheck_pending: Option<i32> = None;
    #[cfg(feature = "v41")]
    let mut xcheck_row0: Vec<f32> = Vec::new();
    // Which arm the in-flight probe ran under: 0 = catch-all off, 1 = on.
    #[cfg(feature = "v41")]
    let mut xcheck_arm: usize = 0;
    #[cfg(feature = "v41")]
    let probe_fprint: bool = std::env::var("V41_PROBE_FPRINT").as_deref() == Ok("1");
    #[cfg(feature = "v41")]
    let mut probe_fp_before: Option<Vec<(String, u64)>> = None;
    let finish: FinishReason = loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::info!("generation cancelled by client");
            was_cancelled = true;
            break FinishReason::Stop;
        }
        if completion_tokens > 0
            && completion_tokens - hb_last_count >= heartbeat_interval
        {
            let elapsed = hb_last_at.elapsed().as_secs_f32();
            let delta = completion_tokens - hb_last_count;
            let tok_per_s = if elapsed > 0.0 {
                delta as f32 / elapsed
            } else {
                0.0
            };
            // Decode's TRUE miss rate: `decode_*` counters only, over the tokens
            // in this beat. The old merged counter could not see this at all.
            #[cfg(feature = "v41")]
            let (miss_per_tok, decode_hit, ms_per_miss) = match state.pager.as_ref() {
                Some(pg) => {
                    let d = pg.counters() - hb_last_pager;
                    hb_last_pager = pg.counters();
                    (
                        d.decode_misses as f64 / delta as f64,
                        if d.decode_requests > 0 {
                            1.0 - d.decode_misses as f64 / d.decode_requests as f64
                        } else { f64::NAN },
                        if d.decode_misses > 0 {
                            (d.decode_read_ns + d.decode_h2d_ns) as f64 / d.decode_misses as f64 / 1e6
                        } else { f64::NAN },
                    )
                }
                None => (f64::NAN, f64::NAN, f64::NAN),
            };
            #[cfg(not(feature = "v41"))]
            let (miss_per_tok, decode_hit, ms_per_miss) = (f64::NAN, f64::NAN, f64::NAN);
            #[cfg(feature = "v41")]
            if let Some(pg) = state.pager.as_ref() {
                let d = pg.counters() - hb_last_read;
                hb_last_read = pg.counters();
                if d.decode_misses > 0 {
                    let m = d.decode_misses as f64;
                    tracing::info!(
                        misses = d.decode_misses,
                        ms_alloc = d.decode_alloc_ns as f64 / m / 1e6,
                        ms_pread = d.decode_pread_ns as f64 / m / 1e6,
                        ms_repack = d.decode_repack_ns as f64 / m / 1e6,
                        ms_h2d = d.decode_h2d_ns as f64 / m / 1e6,
                        ms_total = (d.decode_read_ns + d.decode_h2d_ns) as f64 / m / 1e6,
                        pread_gbps = if d.decode_pread_ns > 0 {
                            d.decode_pread_bytes as f64 / d.decode_pread_ns as f64
                        } else { 0.0 },
                        "decode miss phases (per miss)"
                    );
                }
            }
            tracing::info!(
                completion_tokens,
                pos,
                in_think,
                tok_per_s = format!("{:.1}", tok_per_s),
                miss_per_tok = format!("{miss_per_tok:.2}"),
                decode_hit = format!("{decode_hit:.4}"),
                ms_per_miss = format!("{ms_per_miss:.2}"),
                "decode heartbeat"
            );
            hb_last_count = completion_tokens;
            hb_last_at = std::time::Instant::now();
        }
        // Per-token diagnostic. Logs (token_id, decoded text) for
        // every TOK_DSML sample and the next ~12 tokens after it,
        // plus everything when DEEPSTRIX_TRACE_TOKENS is set.
        let is_dsml = next == tok_dsml;
        if trace_tokens || is_dsml || dump_window > 0 {
            let decoded = state
                .vocab
                .token_text(next)
                .map(|b| gpt2_decode_token(b, &state.byte_decoder))
                .unwrap_or_default();
            tracing::info!(
                token_id = next,
                is_dsml,
                text = ?String::from_utf8_lossy(&decoded),
                "sample"
            );
            if is_dsml {
                dump_window = 12;
            } else {
                dump_window = dump_window.saturating_sub(1);
            }
        }
        if is_turn_end(next) {
            break FinishReason::Stop;
        }
        if next == TOK_THINK_BEGIN {
            in_think = true;
            // Token itself is suppressed.
        } else if next == TOK_THINK_END {
            in_think = false;
            // Token itself is suppressed.
        } else if let Some(bytes) = state.vocab.token_text(next) {
            let t_stream = std::time::Instant::now();
            let raw = gpt2_decode_token(bytes, &state.byte_decoder);
            // Always emit, even for empty raw — TOK_DSML's bytes are
            // routinely the model's primary signal and must be visible
            // to the scanner even though their bytes get suppressed
            // downstream. (For non-DSML empty-decoding tokens this
            // is a no-op for the scanner anyway.)
            //
            // Bounded try_send (not blocking_send) so one slow HTTP
            // client can't stall the single engine thread — which
            // would block every other concurrent request. We retry
            // briefly to absorb transient backpressure (network
            // hiccup, scheduler), then declare the client dead and
            // bail. CHUNK_SEND_RETRY_MS * CHUNK_SEND_RETRIES = ~3s
            // total grace; well past any sane jitter, below any
            // sane HTTP client timeout.
            let chunk = WorkerEvent::Chunk {
                token_id: next,
                bytes: raw,
                reasoning: in_think,
            };
            const CHUNK_SEND_RETRIES: u32 = 30;
            const CHUNK_SEND_RETRY_MS: u64 = 100;
            let mut pending = Some(chunk);
            let mut send_failed = false;
            for attempt in 0..=CHUNK_SEND_RETRIES {
                match tx.try_send(pending.take().unwrap()) {
                    Ok(()) => {
                        // Forward progress: one decoded token shipped.
                        state.progress.pet();
                        break;
                    }
                    Err(mpsc::error::TrySendError::Full(ev)) => {
                        pending = Some(ev);
                        if attempt == CHUNK_SEND_RETRIES {
                            tracing::warn!(
                                completion_tokens,
                                pos,
                                "stream consumer slow ({}ms backpressure); dropping client",
                                CHUNK_SEND_RETRIES as u64 * CHUNK_SEND_RETRY_MS
                            );
                            send_failed = true;
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(
                            CHUNK_SEND_RETRY_MS,
                        ));
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        send_failed = true;
                        break;
                    }
                }
            }
            // Detokenise + chunk hand-off -> `stream_us` on the next summary.
            v4flash_kernels::het::trace::phase::add(
                &v4flash_kernels::het::trace::phase::CALLER_STREAM_NS,
                t_stream.elapsed().as_nanos() as u64,
            );
            if send_failed {
                // Receiver dropped (client disconnected, e.g.) or
                // remained full past our grace window. The KV cache
                // mid-decode is now inconsistent with live.tokens;
                // clear live so the next request reset-prefills.
                tracing::info!(
                    completion_tokens,
                    pos,
                    fp = %state_fingerprint(state),
                    "decode: bailed on send_failed; clearing live"
                );
                state.live = None;
                return Ok(());
            }
        }
        if completion_tokens >= max_new {
            break FinishReason::Length;
        }
        if (next as u32) >= N_VOCAB {
            return Err(eyre!("generate: sampled token id {next} out of vocab"));
        }
        // Speculative ingest + rollback probe. Placed HERE, before the embed, because
        // this is the only point in the loop where clobbering `residual` and the
        // device scratch is harmless: the previous token's logits have already been
        // sampled into `next`, and both are about to be overwritten anyway. After
        // `forward_one` below they hold the logits the next iteration samples.
        #[cfg(feature = "v41")]
        if verify_probe_k > 0 && completion_tokens > 8 {
            let verify_probe_k =
                verify_probe_ks[(completion_tokens as usize) % verify_probe_ks.len()];
            probe_fp_before = if probe_fprint { Some(probe_fingerprint(state)?) } else { None };
            embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, next, &mut residual);
            // Compact any slid window to [0, n_raw) so the prefill-path verify
            // attends to the same keys decode does (see normalize_raw_windows).
            state.engine.normalize_raw_windows(&mut state.dgpu_scratch, &mut state.state)?;
            let mark = state.state.mark_kv();
            // Same as the accept path: hold the raw window in decode addressing
            // so the probe's plain `rollback_kv(&mark)` still addresses it after
            // an SWA eviction. Without this the probe fails on any long-context
            // prompt with "rollback refused ... layer 0 wrapped".
            let _spec = v4flash_kernels::het::forward_prefill::SpeculativeAppend::begin();
            let pc0 = state.pager.as_ref().map(|p| p.counters()).unwrap_or_default();
            // Interleave the small-B catch-all by step parity. Decode drives the
            // text, so both arms verify the SAME token sequence at the same
            // positions — the comparison is of the verify path alone.
            #[cfg(feature = "v41")]
            {
                let ab = small_b_catchall_ab();
                if ab > 0 {
                    xcheck_arm = (completion_tokens as usize) % 2;
                    v4flash_kernels::het::forward_prefill::set_small_b_catchall_max(
                        if xcheck_arm == 1 { ab } else { 0 },
                    );
                } else if single_lane_ab() > 0 {
                    xcheck_arm = (completion_tokens as usize) % 2;
                    v4flash_kernels::het::forward_prefill::set_single_lane_max(
                        if xcheck_arm == 1 { single_lane_ab() } else { 0 },
                    );
                } else if xcheck_poison() {
                    xcheck_arm = (completion_tokens as usize) % 2;
                } else if verify_probe_ks.len() > 1 {
                    // No flag A/B: bucket by probe WIDTH, so one run sweeps B
                    // and shows whether the verify's divergence from decode
                    // degrades smoothly with batch or cliffs at a path switch.
                    xcheck_arm = verify_probe_k.min(XCHECK_ARMS - 1);
                }
            }
            let t_probe = std::time::Instant::now();
            let mut probe_logits: Vec<f32> = Vec::new();
            if verify_probe_batched {
                // The REAL DSpark verify step: ONE batched forward over K tokens
                // appended to the live KV, which is what a draft batch costs.
                // K sequential forwards (the else arm) measure the reject path
                // instead and are ~K x more expensive by construction.
                // ROW-INDEPENDENCE PROBE (`V41_XCHECK_POISON=1`, interleaved by
                // step parity). Rows 1.. are a DIFFERENT token. Row 0's input is
                // untouched, and row 0 is causally first, so its logits must not
                // move. If they do, the batched path leaks later rows into row 0.
                let poison = xcheck_poison() && (completion_tokens as usize) % 2 == 1;
                // Poison BOTH the token and the CARRY of rows 1.. . Tokens alone
                // leave the mHC pre-mix untouched, and that is precisely the
                // stage that could mix across the batch dimension.
                let hcs: Vec<Vec<f32>> = (0..verify_probe_k)
                    .map(|j| {
                        if poison && j > 0 {
                            residual.iter().map(|v| -0.5 * v).collect()
                        } else {
                            residual.clone()
                        }
                    })
                    .collect();
                let toks: Vec<i32> = (0..verify_probe_k)
                    .map(|j| if poison && j > 0 { POISON_TOKEN } else { next })
                    .collect();
                // Engram rows must be gathered and handed to the batched path the
                // same way `prefill_suffix` does it — `forward_prefill` (non-
                // pipelined) has no engram parameter, and the batched MoE fails
                // with "Engram rows not staged" without them.
                let engram_chunk: Option<Vec<Vec<f32>>> =
                    match (state.pager.as_ref(), state.engram.as_mut()) {
                        (Some(pg), Some(ec)) => Some(ec.rows_for_chunk(pg.raw(), &toks, pos)?),
                        _ => None,
                    };
                // `last_only=false`: a REAL verify needs per-token logits, one
                // per speculative position, to compare against the drafts. It
                // also turns CED off (`ced = ced_enabled() && last_only`), and
                // CED is what makes the probe run the layer stack twice — the
                // per-stage profile shows every dGPU stage at 80 calls for 40
                // layers while the iGPU MoE runs 40, which is the signature of
                // a second `CedMode::KvSourceOnly` pass. Measuring with
                // last_only=true therefore priced a pass a real verify does
                // not do.
                probe_logits = state.engine.forward_prefill_pipelined(
                    &mut state.bd_a,
                    &mut state.bi_a,
                    &mut state.bd_b,
                    &mut state.bi_b,
                    &mut state.sd,
                    &mut state.si,
                    &mut state.dgpu_scratch,
                    &mut state.state,
                    &state.weights,
                    &hcs,
                    &toks,
                    pos,
                    false,
                    None,
                    None,
                    None,
                    None,
                    state.pager.as_mut(),
                    engram_chunk.as_deref(),
                )?;
            } else {
                for j in 0..verify_probe_k {
                    forward_one!(state, residual, pos + j as u32, next)?;
                }
            }
            // `V41_XCHECK_ROWS=1`: compare EVERY verify row against decode, not
            // just row 0.
            //
            // Only row 0 has ever been checked, and that gap matters exactly at
            // temperature > 0: when a draft is accepted at temp 0 the emitted
            // token is the DRAFT and row j is only a gate (argmax(row j) == d_j),
            // so a wrong row costs acceptance, not correctness. Above temp 0 the
            // emitted token IS `y_j` sampled FROM row j, so a wrong distribution
            // in a non-first row is emitted directly -- and top_p keeps a long
            // tail for it to be drawn from. This measures those rows.
            //
            // Runs the batched verify, rolls back, then re-runs the SAME tokens
            // through decode capturing per-position logits, and rolls back again,
            // so both sides see the identical prefix.
            if xcheck_rows() && verify_probe_batched && !probe_logits.is_empty() {
                let nv = v4flash_kernels::config::N_VOCAB as usize;
                let rows = probe_logits.len() / nv;
                state.state.rollback_kv(&mark)?;
                let mut truth: Vec<Vec<f32>> = Vec::with_capacity(rows);
                for j in 0..rows {
                    // The probe's batch is K copies of `next` (see `toks`
                    // above), so decode must be fed the same sequence.
                    let t = next;
                    let mut r = vec![0.0f32; v4flash_kernels::config::HC_DIM as usize];
                    embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, t, &mut r);
                    forward_one!(state, r, pos + j as u32, t)?;
                    let mut dl = vec![0.0f32; nv];
                    state.dgpu_scratch.logits.slice_view(0, nv).copy_to_host(&mut dl)?;
                    truth.push(dl);
                }
                state.state.rollback_kv(&mark)?;
                for j in 0..rows {
                    let v = &probe_logits[j * nv..(j + 1) * nv];
                    let d = &truth[j];
                    let am = |x: &[f32]| {
                        let mut bi = 0usize;
                        let mut bv = f32::NEG_INFINITY;
                        for (i, &q) in x.iter().enumerate() {
                            if q > bv { bv = q; bi = i; }
                        }
                        bi
                    };
                    // KL(decode || verify) over the softmaxes, the measure the
                    // tokens are actually drawn from.
                    let (dmax, vmax) = (
                        d.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    );
                    let (mut dz, mut vz) = (0.0f64, 0.0f64);
                    for i in 0..nv {
                        dz += ((d[i] - dmax) as f64).exp();
                        vz += ((v[i] - vmax) as f64).exp();
                    }
                    let (ldz, lvz) = (dz.ln(), vz.ln());
                    let mut kld = 0.0f64;
                    for i in 0..nv {
                        let lp = (d[i] - dmax) as f64 - ldz;
                        let pq = lp.exp();
                        if pq > 1e-12 {
                            kld += pq * (lp - ((v[i] - vmax) as f64 - lvz));
                        }
                    }
                    XROW_N[j.min(XROW_MAX - 1)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if am(v) == am(d) {
                        XROW_OK[j.min(XROW_MAX - 1)]
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    XROW_KLD[j.min(XROW_MAX - 1)]
                        .fetch_add((kld * 1e6) as i64, std::sync::atomic::Ordering::Relaxed);
                }
            }
            // XCHECK: row 0 of the probe sits at `pos` with `next` as its
            // input, exactly like the decode forward that follows, so its
            // argmax must equal the token decode samples next. This is the
            // DIRECT correctness test for the verify path — acceptance is only
            // a proxy, and a bad one, because changing the split changes the
            // generated text and acceptance is content-dependent.
            if verify_probe_batched && !probe_logits.is_empty() {
                let nv = v4flash_kernels::config::N_VOCAB as usize;
                let row = &probe_logits[..nv.min(probe_logits.len())];
                let mut bi = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (i, &v) in row.iter().enumerate() {
                    if v > bv {
                        bv = v;
                        bi = i;
                    }
                }
                xcheck_pending = Some(bi as i32);
                xcheck_row0 = row.to_vec();
            }
            let dt = t_probe.elapsed();
            #[cfg(feature = "v41")]
            {
                // Record for EVERY bucketing mode, not just the catch-all A/B:
                // bucketed by width this is the verify's cost curve, which is
                // what decides the optimal draft depth.
                XCHECK_ARM_US[xcheck_arm]
                    .fetch_add(dt.as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            // Per-stage GPU busy for exactly ONE verify. `forward_prefill_pipelined`
            // feeds the accumulator per chunk but only the real prefill emits, so
            // the probe's breakdown was never visible. NOTE the harvest
            // SYNCHRONIZES (it serialises the two lanes): read the busy times,
            // not the wall.
            v4flash_kernels::het::trace::prefill_profile::emit_and_clear(verify_probe_k);
            if let Some(pg) = state.pager.as_ref() {
                let d = pg.counters() - pc0;
                tracing::info!(
                    k = verify_probe_k,
                    prefill_requests = d.prefill_requests,
                    prefill_misses = d.prefill_misses,
                    prefill_read_ms = d.prefill_read_ns / 1_000_000,
                    prefill_h2d_ms = d.prefill_h2d_ns / 1_000_000,
                    decode_requests = d.decode_requests,
                    decode_misses = d.decode_misses,
                    "verify probe: pager delta"
                );
            }
            v4flash_kernels::het::forward_prefill::emit_layer_miss_hist("verify");
            v4flash_kernels::het::forward_prefill::emit_layer_host_timing(
                "verify",
                v4flash_kernels::config::N_LAYER as usize,
            );
            // A refused rollback means the KV wrapped mid-batch and the mark no
            // longer addresses the same rows. Continuing would silently serve wrong
            // KV, so fail loudly instead.
            state.state.rollback_kv(&mark).map_err(|e| {
                eyre!("verify probe: rollback refused after {verify_probe_k} tokens: {e}")
            })?;
            if let Some(before) = probe_fp_before.take() {
                let after = probe_fingerprint(state)?;
                let mut bad = Vec::new();
                for ((n, a), (_, b)) in before.iter().zip(after.iter()) {
                    if a != b {
                        bad.push(n.clone());
                    }
                }
                if !bad.is_empty() {
                    let n = bad.len();
                    bad.truncate(12);
                    tracing::warn!(
                        k = verify_probe_k, pos, components = n, first = ?bad,
                        "probe rollback did NOT restore these components"
                    );
                } else {
                    tracing::info!(k = verify_probe_k, pos, "probe rollback restored all components");
                }
            }
            tracing::info!(
                k = verify_probe_k,
                total_us = dt.as_micros() as u64,
                per_token_us = (dt.as_micros() as u64) / verify_probe_k.max(1) as u64,
                "verify probe: speculative ingest rolled back"
            );
        }
        // ---- DSpark ACCEPT ------------------------------------------------
        // One batched verify over [next, d0..d_{K-2}] appends K positions, and
        // the longest draft prefix the model agrees with is kept. Everything
        // past the first disagreement is rolled back. The loop below then emits
        // the confirmed tokens one per iteration WITHOUT forwarding them again.
        #[cfg(feature = "v41")]
        if dspark_accept
            && state
                .mtp
                .as_ref()
                .is_some_and(|m| m.ingested == 0 && m.confirmed.is_empty() && m.pending.is_some())
        {
            // `V41_DSPARK_K=<n>`: cap the drafts the verify carries. n=0 makes
            // the verify a ONE-ROW batch -- shape-identical to a decode step --
            // which separates "the accept path diverges" from "a batched verify
            // diverges".
            let k = match std::env::var("V41_DSPARK_K").ok().and_then(|v| v.parse::<usize>().ok()) {
                Some(cap) => cap.min(v4flash_kernels::het::mtp::MTP_BLOCK),
                None => v4flash_kernels::het::mtp::MTP_BLOCK,
            };
            // `V41_DSPARK_CONF_MIN=<f>`: truncate the block at the first draft the
            // drafter itself is unsure of. A rejected draft costs a verify row --
            // its share of the batched GEMV, and (cold) the expert reads that row
            // routes to -- and buys nothing, since everything after a rejection is
            // discarded too. Calibrate with the `conf` table in `dspark.request`
            // before setting this.
            let conf = state.mtp.as_ref().unwrap().pending_conf;
            let k = match std::env::var("V41_DSPARK_CONF_MIN").ok().and_then(|v| v.parse::<f32>().ok()) {
                Some(th) => {
                    let cut = (0..k).position(|j| conf[j] < th).unwrap_or(k);
                    // Always carry at least one draft: at k=0 the verify is a
                    // one-row batch that costs a decode step and can accept
                    // nothing, which is strictly worse than not speculating.
                    cut.max(1)
                }
                None => k,
            };
            // Hold the raw window in decode's monotonic addressing across the
            // whole verify: the prefill path would otherwise compact and reset
            // raw_off, making the `KvMark` below unaddressable and capping accept
            // mode at generations shorter than SWA_WINDOW. The verify appends B
            // rows to the oversized tail; `advanced_by` commits only the accepted
            // prefix by sliding the window pointer, and the rejected rows are
            // overwritten by the next verify (commit-on-accept; rollback = don't
            // commit).
            let _spec = v4flash_kernels::het::forward_prefill::SpeculativeAppend::begin();
            let drafts = state.mtp.as_ref().unwrap().pending.unwrap();
            // INVARIANT: row 0 of this verify is `next`, the token the previous
            // step emitted LAST. If it is not, the KV and the emitted stream
            // have desynchronised -- the cache then holds a prefix the client
            // never saw, which reads exactly like the degeneration we see at
            // temperature > 0.
            {
                use std::sync::atomic::Ordering::Relaxed;
                // Armed only within a request: the first verify after a prefill
                // legitimately starts a fresh stream, and a value left over from
                // the PREVIOUS request is not a desync (it fired three times
                // that way before this guard -- all at the first verify, all
                // with the same `got`).
                let exp = DSPARK_EXPECT_NEXT.load(Relaxed);
                if exp >= 0 && exp != next {
                    tracing::warn!(
                        expected = exp, got = next, pos,
                        "dspark.desync: verify row 0 is not the token the last step emitted"
                    );
                    DSPARK_DESYNC.fetch_add(1, Relaxed);
                }
            }
            // Inputs: the head token, then ALL K drafts — K+1 positions.
            //
            // K inputs would validate K drafts too (logits at pos+j predict
            // pos+j+1), but the last draft would then never be INGESTED, so
            // accepting all K would claim K+1 tokens in KV when only K are
            // there. That off-by-one misaligns the cache and the continuation
            // degrades into garbage. With K+1 inputs every accepted draft is in
            // KV and `keep = n + 1` is exact.
            let mut toks = Vec::with_capacity(k + 1);
            toks.push(next);
            toks.extend_from_slice(&drafts[..k]);
            let hcs: Vec<Vec<f32>> = toks
                .iter()
                .map(|&t| {
                    let mut r = vec![0.0f32; v4flash_kernels::config::HC_DIM as usize];
                    embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, t, &mut r);
                    r
                })
                .collect();
            let engram_chunk: Option<Vec<Vec<f32>>> =
                match (state.pager.as_ref(), state.engram.as_mut()) {
                    (Some(pg), Some(ec)) => Some(ec.rows_for_chunk(pg.raw(), &toks, pos)?),
                    _ => None,
                };
            // `V41_XCHECK_ROWS=2`: per-row verify-vs-decode on the REAL batch.
            //
            // The probe form (=1) feeds K copies of the same token, which is not
            // what a verify sees -- a real batch is [next, d_0..d_k-1], all
            // distinct -- so its magnitudes were not known to transfer. This
            // captures decode truth for the ACTUAL drafts, BEFORE the verify
            // runs, then rolls back so the normal flow is untouched.
            let mut xrow_truth: Vec<Vec<f32>> = Vec::new();
            if xcheck_rows_accept() {
                let nv = v4flash_kernels::config::N_VOCAB as usize;
                let m0 = state.state.mark_kv();
                for (j, &t) in toks.iter().enumerate() {
                    let mut r = vec![0.0f32; v4flash_kernels::config::HC_DIM as usize];
                    embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, t, &mut r);
                    forward_one!(state, r, pos + j as u32, t)?;
                    let mut dl = vec![0.0f32; nv];
                    state.dgpu_scratch.logits.slice_view(0, nv).copy_to_host(&mut dl)?;
                    xrow_truth.push(dl);
                }
                state.state.rollback_kv(&m0)?;
            }
            // Compact any slid window to [0, n_raw) so the prefill-path verify
            // attends to the same keys decode does, and the mark records raw_off=0.
            state.engine.normalize_raw_windows(&mut state.dgpu_scratch, &mut state.state)?;
            let mark = state.state.mark_kv();
            // One phase per step for the alternating slack probe, set before
            // any site in this step can read it.
            v4flash_kernels::het::mtp::slack_probe_step_advance();
            let t_step = std::time::Instant::now();
            // Verify-step boundary: the previous step's logits were read back,
            // so no MoE kernel can be reading the pool. Admit prefetched experts.
            if let Some(pg) = state.pager.as_mut() {
                pg.drain_prefetched()?;
            }
            let spc0 = state.pager.as_ref().map(|p| p.counters()).unwrap_or_default();
            let mut decode_path_logits: Option<Vec<f32>> = None;
            // `V41_VERIFY_DECODE_PATH=1`: run the verify through DECODE's own
            // per-layer function, layer-major over the B rows, instead of the
            // batched prefill driver.
            //
            // WHY. The batched driver computes a different function than decode
            // — different kernel families throughout (measured: argmax
            // agreement 0.49-0.61, cos 0.75-0.79) — and swapping components one
            // at a time does NOT converge, because each swapped component still
            // consumes diverged inputs from the ones ahead of it. Decode's
            // `forward_layer_standalone_graphs_paged` is public and complete, so
            // the faithful verify needs no change to the prefill path at all.
            //
            // Layer-major over rows is numerically IDENTICAL to running the
            // tokens sequentially: each layer sees the same token order, and
            // each row's layer-L input is its own layer-(L-1) output, which is
            // computed first. Only the order of independent work changes.
            //
            // This is the correctness step. It costs B x decode's per-layer work
            // because each row still does its own remote submit — batching those
            // per layer is the follow-up, and it is where speculation's win is.
            if verify_decode_path() {
                use v4flash_kernels::config::{ENGRAM_LAYERS, HC_DIM, N_LAYER, N_VOCAB};
                let bsz = toks.len();
                let mut resid: Vec<Vec<f32>> = hcs.clone();
                let hcm = state.dgpu_scratch.hc_pre_carry.len();
                let mut carry: Vec<Vec<f32>> = vec![vec![0.0f32; hcm]; bsz];
                // Engram rows per row, gathered up front: hashes are token-only.
                let mut erows: Vec<Option<Vec<Vec<f32>>>> = Vec::with_capacity(bsz);
                for (j, &t) in toks.iter().enumerate() {
                    erows.push(match (state.pager.as_ref(), state.engram.as_mut()) {
                        (Some(pg), Some(ec)) => {
                            Some(ec.rows_for(pg.raw(), t, pos + j as u32)?)
                        }
                        _ => None,
                    });
                }
                for layer in 0..N_LAYER as usize {
                    let eidx = ENGRAM_LAYERS.iter().position(|&l| l as usize == layer);
                    for j in 0..bsz {
                        state.dgpu_scratch.residual.copy_from_host(&resid[j])?;
                        if layer > 0 {
                            state.dgpu_scratch.hc_pre_carry.copy_from_host(&carry[j])?;
                        }
                        if let (Some(ei), Some(er)) = (eidx, erows[j].as_ref()) {
                            state.engine.stage_engram_rows(&mut state.dgpu_scratch, &er[ei])?;
                        }
                        {
                            // Layer-major breaks the lockstep the token-major
                            // path assumes, so publish rope pos + KV slot from
                            // THIS layer's counters, per row.
                            let slot = state.state.layers[layer].raw_off
                                + state.state.layers[layer].n_raw;
                            state.engine.publish_pos_slot(
                                &mut state.dgpu_scratch,
                                pos + j as u32,
                                slot,
                            )?;
                        }
                        {
                            // V4.1 reuse layers borrow another layer's
                            // compressor store; `with_kv_source` lends it for
                            // the call and gives it back. Without it the layer
                            // errors with "reuse layer without its source's
                            // store".
                            let pg = state.pager.as_mut().expect("pager");
                            let eng = &state.engine;
                            let dgs = &mut state.dgpu_scratch;
                            let igs = &mut state.igpu_scratch;
                            let dlw = &state.weights.dgpu_layers[layer];
                            let ilw = &state.weights.igpu_layers[layer];
                            let (p, t) = (pos + j as u32, toks[j]);
                            state.state.with_kv_source(layer, |ls| {
                                eng.forward_layer_standalone_graphs_paged(
                                    dgs, igs, ls, dlw, ilw, p, t, pg,
                                )
                            })?;
                        }
                        // A layer READS `residual` and WRITES `residual_next`
                        // (`forward_layer.rs:245`); decode's loop then swaps them
                        // (`engine.rs:1189`). This path never swapped and read back
                        // `residual` -- the layer's own INPUT -- so roughly every
                        // other layer came back bit-unchanged and the verify was
                        // running on a residual that had skipped half the model.
                        // Read the OUTPUT buffer instead; the swap that keeps the
                        // per-layer buffer PARITY in step with decode is done once
                        // per layer, after the row loop, not per row: HIP graphs
                        // capture POINTERS, so layer L must see the same physical
                        // buffer here that it sees in decode (A for even L, B for
                        // odd), and every row of layer L must see the same one.
                        state.dgpu_scratch.residual_next.copy_to_host(&mut resid[j])?;
                        state.dgpu_scratch.hc_pre_carry.copy_to_host(&mut carry[j])?;
                        // V41_VDP_TRACE=1: row-0 residual norm per layer. A blow-up
                        // or NaN localises the first bad layer without needing a
                        // baseline run to diff against.
                        if j == 0 && std::env::var("V41_VDP_TRACE").as_deref() == Ok("1") {
                            let r = &resid[0];
                            let n2: f64 = r.iter().map(|&v| (v as f64) * (v as f64)).sum();
                            let nan = r.iter().filter(|v| !v.is_finite()).count();
                            let c = &carry[0];
                            let cn: f64 = c.iter().map(|&v| (v as f64) * (v as f64)).sum();
                            eprintln!(
                                "VDP L{layer:02} pos={} |resid|={:.4e} nonfinite={} |carry|={:.4e}",
                                pos, n2.sqrt(), nan, cn.sqrt()
                            );
                        }
                    }
                    // One swap per LAYER (not per row), mirroring decode's
                    // `engine.rs:1189`, so layer L+1's rows read the physical
                    // buffer its captured graphs were built against.
                    std::mem::swap(
                        &mut state.dgpu_scratch.residual,
                        &mut state.dgpu_scratch.residual_next,
                    );
                }
                // Head per row, into the same [B * N_VOCAB] layout the batched
                // path returns.
                let nv = N_VOCAB as usize;
                let mut out = vec![0.0f32; bsz * nv];
                for j in 0..bsz {
                    state.dgpu_scratch.residual.copy_from_host(&resid[j])?;
                    state.dgpu_scratch.hc_pre_carry.copy_from_host(&carry[j])?;
                    state.engine.forward_head(&mut state.dgpu_scratch, &state.weights.global)?;
                    state
                        .dgpu_scratch
                        .logits
                        .slice_view(0, nv)
                        .copy_to_host(&mut out[j * nv..(j + 1) * nv])?;
                }
                let _ = HC_DIM;
                decode_path_logits = Some(out);
            }
            // Capture on BOTH lanes, and over the WHOLE verify batch.
            //
            // The verify splits into two lanes (`single_lane_max` defaults to
            // 0, so B=6 becomes 3+3), and `mtp_src` is captured per lane into
            // that lane's own buffer, indexed LANE-LOCALLY. Arming only lane A
            // left rows [b_a, b) -- about 29% of accepted heads at the measured
            // n histogram -- reading STALE residuals out of lane A's buffer,
            // which is what fed the drafter garbage and looked like "the
            // drafter is a quality bug".
            //
            // `k` rather than the full batch was the other half: the capture
            // takes the LAST `min(tokens, rows)` rows of the lane, so a single
            // lane holding all 6 tokens with rows=5 skipped row 0 and shifted
            // every residual by one. That is why forcing single-lane measured
            // WORSE (E 2.26 -> 1.74) instead of better. `toks.len()` makes the
            // skip zero in both configurations.
            state.bd_a.mtp_capture_rows = toks.len();
            state.bd_b.mtp_capture_rows = toks.len();
            // Only when the decode-path verify did not already produce them:
            // running both would ingest B tokens TWICE and the partial rollback
            // would then keep a doubly-ingested cache.
            let logits_batched = if decode_path_logits.is_some() {
                Vec::new()
            } else {
                state.engine.forward_prefill_pipelined(
                &mut state.bd_a, &mut state.bi_a, &mut state.bd_b, &mut state.bi_b,
                &mut state.sd, &mut state.si, &mut state.dgpu_scratch, &mut state.state,
                &state.weights, &hcs, &toks, pos, false, None, None, None, None,
                    state.pager.as_mut(), engram_chunk.as_deref(),
                )?
            };
            state.bd_a.mtp_capture_rows = 0;
            state.bd_b.mtp_capture_rows = 0;
            // `V41_VERIFY_DECODE_PATH=1` skips `forward_prefill_pipelined`
            // entirely, and that call is what captures `mtp_src`. So under this
            // flag `main_hidden` and the dense-ring replay both read the
            // PREVIOUS step's residuals, and any acceptance number measured with
            // it is measuring a drafter fed one-step-stale hidden states. Since
            // the flag exists specifically to adjudicate "does the batched
            // driver diverge", that silently contaminates the arbiter -- so say
            // so loudly rather than let it be quoted as evidence again.
            if decode_path_logits.is_some() {
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(
                        "V41_VERIFY_DECODE_PATH: mtp_src is NOT captured on this path;                          main_hidden and the ring replay use the previous step's residuals.                          Acceptance/E from this flag is NOT a valid drafter measurement."
                    );
                }
            }
            let logits = decode_path_logits.take().unwrap_or(logits_batched);
            let t_fwd = t_step.elapsed();

            // Accept the longest prefix whose argmax matches the draft.
            let nv = v4flash_kernels::config::N_VOCAB as usize;
            // `row_sample(j)` slices `logits[j*nv .. (j+1)*nv]` for every row up
            // to and including row `k`, so the verify must have returned ONE
            // FULL ROW PER INPUT TOKEN. A short buffer (e.g. a last_only driver)
            // would silently read one row's distribution as another's.
            assert_eq!(
                logits.len(),
                toks.len() * nv,
                "dspark accept: verify returned {} logits for {} rows ({nv} per row expected);                  per-row indexing would read the wrong row",
                logits.len(),
                toks.len()
            );
            let row_argmax = |j: usize| -> i32 {
                let r = &logits[j * nv..(j + 1) * nv];
                let mut bi = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (i, &v) in r.iter().enumerate() {
                    if v > bv {
                        bv = v;
                        bi = i;
                    }
                }
                bi as i32
            };
            // DRAW the target token for each verify row, then accept the draft
            // only if it matches.
            //
            // Matching on argmax -- what this did -- is greedy speculative
            // decoding. It is correct ONLY when the request asked for greedy;
            // at the model's own agentic recipe (temperature 1.0, top_p 0.95)
            // it silently threw the sampler away and emitted argmax tokens,
            // which is both wrong and why generations collapsed into
            // repetition loops.
            //
            // Sampling the TARGET and accepting on equality is exactly
            // distribution-preserving: the emitted token is `y_j ~ p_j` either
            // way, so the draft can only change HOW MANY tokens a step yields,
            // never which. That is the whole guarantee speculative decoding
            // needs, and unlike rejection sampling it needs no drafter
            // probabilities.
            //
            // It is also all that rejection sampling would buy us TODAY: our
            // drafts are the drafter's argmax, so `q` is a point mass and
            // `min(1, p/q)` collapses to `p(draft)` -- the same acceptance this
            // gets. Beating it requires sampling the drafts from `q` first;
            // that is the next step, not this one.
            let row_sample = |j: usize, rng: &mut SamplerRng| -> i32 {
                match sample_mode {
                    SampleMode::Argmax => row_argmax(j),
                    SampleMode::Multinomial { temperature, min_p_rel, top_p } => {
                        let r = &logits[j * nv..(j + 1) * nv];
                        let inv_t = 1.0f32 / temperature;
                        let gmax = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
                        // Same weights the device chain forms: exp(logit/T - gmax).
                        let w: Vec<f64> =
                            r.iter().map(|&x| ((x * inv_t - gmax) as f64).exp()).collect();
                        // Same composed top_p/min_p rule as the kernel, so the
                        // truncated distribution here IS the sampler's.
                        let thr = v4flash_kernels::sampler::top_p_min_p_threshold(
                            &w,
                            top_p as f64,
                            min_p_rel as f64,
                        );
                        let z: f64 = w.iter().filter(|&&v| v >= thr).sum();
                        let mut acc = 0.0f64;
                        let target = rng.next_f32() as f64 * z;
                        let mut pick = 0i32;
                        for (i, &v) in w.iter().enumerate() {
                            if v < thr {
                                continue;
                            }
                            acc += v;
                            pick = i as i32;
                            if acc >= target {
                                break;
                            }
                        }
                        pick
                    }
                }
            };
            // One-shot: is row 0 in the space the sampler assumes (raw logits)?
            // If `p_max` comes out tiny the distribution is flat, which means
            // these are not raw logits and exp(x-max) is meaningless.
            if std::env::var("V41_DUMP_VERIFY_DIST").as_deref() == Ok("1")
                && VERIFY_DIST_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3
            {
                let r = &logits[0..nv];
                let mx = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mn = r.iter().copied().fold(f32::INFINITY, f32::min);
                let mean = r.iter().sum::<f32>() / nv as f32;
                let z: f64 = r.iter().map(|&x| ((x - mx) as f64).exp()).sum();
                let mut top: Vec<f32> = r.to_vec();
                top.sort_by(|a, b| b.partial_cmp(a).unwrap());
                tracing::info!(
                    max = mx, min = mn, mean,
                    p_max = format!("{:.4}", 1.0 / z),
                    top5 = format!("{:?}", &top[..5]),
                    "verify.dist"
                );
            }
            let mut n = 0usize;
            let mut corrected = row_sample(0, &mut rng);
            while n < k && corrected == drafts[n] {
                n += 1;
                if n < k {
                    corrected = row_sample(n, &mut rng);
                }
            }
            // `V41_DSPARK_FORCE_N0=1`: run the whole DSpark machinery but keep
            // NOTHING. The verify still runs batched over B rows and still
            // supplies the emitted token from row 0; only the drafted rows are
            // dropped. Isolation probe: if output then matches plain decode,
            // row 0 of a batched verify is faithful and the divergence lives in
            // the KV of the KEPT DRAFT rows; if it still diverges, row 0 itself
            // differs once the verify (not decode) is driving the cache.
            if n > 0 && std::env::var("V41_DSPARK_FORCE_N0").as_deref() == Ok("1") {
                n = 0;
                corrected = row_sample(0, &mut rng);
            }
            if !xrow_truth.is_empty() {
                let nv = v4flash_kernels::config::N_VOCAB as usize;
                for (j, d) in xrow_truth.iter().enumerate() {
                    if (j + 1) * nv > logits.len() {
                        break;
                    }
                    let v = &logits[j * nv..(j + 1) * nv];
                    let am = |x: &[f32]| {
                        let mut bi = 0usize;
                        let mut bv = f32::NEG_INFINITY;
                        for (i, &q) in x.iter().enumerate() {
                            if q > bv { bv = q; bi = i; }
                        }
                        bi
                    };
                    let (dmax, vmax) = (
                        d.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    );
                    let (mut dz, mut vz) = (0.0f64, 0.0f64);
                    for i in 0..nv {
                        dz += ((d[i] - dmax) as f64).exp();
                        vz += ((v[i] - vmax) as f64).exp();
                    }
                    let (ldz, lvz) = (dz.ln(), vz.ln());
                    let mut kld = 0.0f64;
                    for i in 0..nv {
                        let lp = (d[i] - dmax) as f64 - ldz;
                        let pq = lp.exp();
                        if pq > 1e-12 {
                            kld += pq * (lp - ((v[i] - vmax) as f64 - lvz));
                        }
                    }
                    use std::sync::atomic::Ordering::Relaxed;
                    let jj = j.min(XROW_MAX - 1);
                    XROW_N[jj].fetch_add(1, Relaxed);
                    if am(v) == am(d) {
                        XROW_OK[jj].fetch_add(1, Relaxed);
                    }
                    XROW_KLD[jj].fetch_add((kld * 1e6) as i64, Relaxed);
                    // Bucket by whether this row is one we actually USE.
                    //
                    // We emit from row j only when the prefix d_0..d_j-1 was
                    // ACCEPTED, i.e. matched the model's own draw at every
                    // earlier row. Rows past the rejection point are conditioned
                    // on drafts the model rejected, are never emitted from, and
                    // their disagreement costs nothing. If the divergence lives
                    // ONLY there, it is not a correctness bug at all -- if it is
                    // present at j <= n, it is.
                    let used = j <= n;
                    let b = if used { 0 } else { 1 };
                    XUSED_N[b].fetch_add(1, Relaxed);
                    if am(v) == am(d) {
                        XUSED_OK[b].fetch_add(1, Relaxed);
                    }
                    XUSED_KLD[b].fetch_add((kld * 1e6) as i64, Relaxed);
                    // A MEAN is the wrong statistic here: the observed failure is
                    // a cliff (100+ identical tokens), which 0.05 nats averaged
                    // cannot produce. If the corruption is a rare catastrophic
                    // draw, it lives in the TAIL, so bucket it and keep the max.
                    if used {
                        let bucket = if kld < 0.01 {
                            0
                        } else if kld < 0.1 {
                            1
                        } else if kld < 1.0 {
                            2
                        } else if kld < 5.0 {
                            3
                        } else {
                            4
                        };
                        XTAIL[bucket].fetch_add(1, Relaxed);
                        let k6 = (kld * 1e6) as i64;
                        XTAIL_MAX.fetch_max(k6, Relaxed);
                        // Was the token we EMITTED from this row one the true
                        // distribution would essentially never produce? That is
                        // the event that derails a generation.
                        if kld >= 1.0 {
                            tracing::warn!(
                                row = j, n, pos,
                                kld = format!("{kld:.3}"),
                                verify_argmax = am(v) as i64,
                                decode_argmax = am(d) as i64,
                                "dspark.row_blowup: emitted-from row diverges catastrophically"
                            );
                        }
                    }
                }
            }
            dspark_stats::record_accept(n, k);
            dspark_stats::record_conf(&conf, n, k);
            let t_argmax = t_step.elapsed();
            // Attribute the ACCEPT verify's per-layer host time, not just the
            // probe's. The perfetto trace puts ~15.2 ms/layer of host gap
            // between `k.shared_expert.down_matvec` and `k.ffn_combine.vec_add`
            // with no device work in it (both GPUs <13% busy, box 2 0.4%), so
            // this breakdown -- sel_sync / pager / pre_moe / post_moe / remote --
            // is what says WHICH host step owns it. Costs nothing unless
            // `V41_LAYER_HOST_TIMING=1`.
            v4flash_kernels::het::forward_prefill::emit_layer_host_timing(
                "accept",
                v4flash_kernels::config::N_LAYER as usize,
            );
            // KV must keep `next` plus the n accepted drafts and drop the rest.
            // `KvMark` is per-layer `(n_raw, raw_off)`, so a PARTIAL rollback is
            // just the mark advanced by the number of rows kept.
            let keep = (n + 1) as u32; // `next` plus the n accepted drafts
            // The accepted prefix can never be longer than the rows the verify
            // actually appended to KV: the batch is `next` + k drafts, so
            // `keep <= toks.len()`. Claiming more rows than were appended
            // misaligns the cache against the emitted stream permanently.
            assert!(
                n <= k && (keep as usize) <= toks.len(),
                "dspark accept: keeping {keep} rows (n={n} of k={k}) from a {}-row verify",
                toks.len()
            );
            // Row 0 of the verify batch sits at `pos`. `advanced_by` also
            // rewinds the COMPRESSED store to what the accepted prefix earns —
            // without it the rejected drafts' compressor boundaries stayed in
            // the store permanently and decode attended to them.
            let partial = mark.advanced_by(keep, pos);
            state.state.rollback_kv(&partial).map_err(|e| {
                eyre!("dspark accept: partial rollback ({keep} of {k}) refused: {e}")
            })?;

            // `main_hidden` for the next draft is the row that became the head.
            let row = n; // row n is the last position kept: `next` + n drafts
            let ne = v4flash_kernels::config::N_EMBD as usize;
            let nsrc = v4flash_kernels::het::mtp::MTP_SRC_LAYERS.len();
            let cap = v4flash_kernels::het::batch_scratch::MTP_CAP_ROWS;
            // Global batch row -> (lane, lane-local row). See the capture
            // comment above: each lane's `mtp_src` is indexed from 0.
            let cut = state.bd_a.mtp_lane_cut;
            // The cut is the ONLY thing tying a global batch row to the lane
            // that captured its residual, and each lane's `mtp_src` is indexed
            // from 0. If the cut does not cover the rows each lane actually
            // captured, `main_hidden` is read out of the wrong lane's buffer and
            // is silently STALE -- the failure that looked like "the drafter is
            // degenerate" until it was root-caused to this mapping.
            assert!(
                state.bd_a.mtp_captured >= cut.min(toks.len()),
                "dspark accept: lane A captured {} mtp_src rows, but the recorded lane cut claims                  rows [0,{}) of this {}-row verify came from lane A",
                state.bd_a.mtp_captured,
                cut.min(toks.len()),
                toks.len()
            );
            assert!(
                cut >= toks.len() || state.bd_b.mtp_captured >= toks.len() - cut,
                "dspark accept: lane B captured {} mtp_src rows, but the recorded lane cut claims                  rows [{cut},{}) of this verify came from lane B",
                state.bd_b.mtp_captured,
                toks.len()
            );
            let mut whole = vec![0.0f32; nsrc * cap * ne];
            let mut whole_b = vec![0.0f32; nsrc * cap * ne];
            state.bd_a.mtp_src.copy_to_host(&mut whole)?;
            if cut < toks.len() {
                state.bd_b.mtp_src.copy_to_host(&mut whole_b)?;
            }
            let lane_row = |r: usize| -> (&Vec<f32>, usize) {
                let (buf, lr) = if r < cut { (&whole, r) } else { (&whole_b, r - cut) };
                // LANE-LOCAL row, never a global one: `mtp_src` holds at most
                // MTP_CAP_ROWS rows per lane.
                assert!(
                    lr < cap,
                    "dspark accept: lane-local mtp_src row {lr} (global row {r}, lane cut {cut})                      >= MTP_CAP_ROWS {cap}"
                );
                (buf, lr)
            };
            let m = state.mtp.as_mut().expect("mtp");
            m.main_hidden.clear();
            #[allow(clippy::needless_range_loop)]
            for sl in 0..nsrc {
                let (buf, lr) = lane_row(row);
                let o = sl * cap * ne + lr * ne;
                m.main_hidden.extend_from_slice(&buf[o..o + ne]);
            }
            // The drafter's entry projection consumes exactly one N_EMBD row per
            // MTP source layer; a short/long vector means the capture layout and
            // the reader disagree.
            assert_eq!(
                m.main_hidden.len(),
                nsrc * ne,
                "dspark accept: main_hidden has {} floats, expected {} ({nsrc} MTP source layers                  x {ne} N_EMBD)",
                m.main_hidden.len(),
                nsrc * ne
            );
            m.confirmed.clear();
            for d in drafts.iter().take(n) {
                m.confirmed.push_back(*d);
            }
            m.ingested = n + 1;
            // The step yields the n confirmed drafts and then the head.
            // `ingested` counts the rows this verify LEFT IN KV that the decode
            // loop must not forward again: `next` (row 0) plus those n drafts.
            assert_eq!(
                m.confirmed.len() + 1,
                m.ingested,
                "dspark accept: {} confirmed drafts queued but ingested={} (must be                  confirmed + 1, the head row `next`)",
                m.confirmed.len(),
                m.ingested
            );
            let head = if n < k { corrected } else { row_sample(k, &mut rng) };
            m.next_after = Some(head);
            // The LAST token this step yields is `head`; the next verify's row 0
            // must be exactly that.
            DSPARK_EXPECT_NEXT.store(head, std::sync::atomic::Ordering::Relaxed);
            let t_roll = t_step.elapsed();
            m.accept_steps += 1;
            m.accept_tokens += (n as u64) + 1;

            // DENSE RING (`V41_DSPARK_DENSE_RING=1`): the drafter's KV ring must
            // contain every position it will attend over. Accept mode only drafts
            // once per step (at pos+n), so the n intermediate accepted positions
            // pos..pos+n-1 never got a ring write and the window goes ~43% sparse
            // — the reason accept E (~1.8) sits far below shadow E (~3.08). Replay
            // the drafter over each accepted position first (its residual is in
            // mtp_src row r; the token AT p+1 is the accepted draft), so the ring
            // is dense like shadow's. The intermediate drafts are discarded.
            //
            // DEFAULT ON since 2026-09-16. Back-to-back at the current defaults
            // (pool 78 GB, floor 0, sparse verify residency), 120 tokens:
            //     off: 2.22 tok/s  E 2.380  (50 steps)
            //     on:  2.33 tok/s  E 2.553  (47 steps)
            // `V41_DSPARK_DENSE_RING=0` rolls it back.
            // Default ON: MEASURED byte-identical output and E, at 4.2x less
            // per-accepted-token cost. `V41_DSPARK_RING_FAST=0` rolls back.
            let ring_fast = std::env::var("V41_DSPARK_RING_FAST").as_deref() != Ok("0");
            if std::env::var("V41_DSPARK_DENSE_RING").as_deref() != Ok("0") && n > 0 {
                for r in 0..n {
                    let mut mh = Vec::with_capacity(nsrc * ne);
                    for sl in 0..nsrc {
                        let (buf, lr) = lane_row(r);
                        let o = sl * cap * ne + lr * ne;
                        mh.extend_from_slice(&buf[o..o + ne]);
                    }
                    let tok_at_p1 = drafts[r];
                    let mut tr = vec![0.0f32; v4flash_kernels::config::HC_DIM as usize];
                    embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, tok_at_p1, &mut tr);
                    let m = state.mtp.as_mut().expect("mtp");
                    // Correct cheap advance: full layer forward (ring + carry),
                    // skip the exit. Residual at pos+r is mtp_src row r.
                    //
                    // `V41_DSPARK_RING_FAST`: the full forward is only needed
                    // for the CARRY; the ring row itself is a projection of the
                    // main model's residual and needs no draft stream. MEASURED
                    // `draft_ms = 16.8 + 11.00*n`, so paying a full forward for
                    // every accepted token makes acceptance tax itself. Write
                    // the intermediate rings cheaply and let the LAST position
                    // refresh the carry against the now-dense ring.
                    if ring_fast && r + 1 < n {
                        state.engine.dspark_ring_write_only(&mut m.state, &m.w, pos + r as u32, &mh)?;
                    } else {
                        state.engine.dspark_advance_ring(&mut m.state, &m.w, pos + r as u32, &mh, &tr, &m.noise_row)?;
                    }
                }
            }

            // Draft for the NEXT step now, while `main_hidden` is the row that
            // just became the head: the drafter wants (residual @ p, token @
            // p+1), and here p = pos + n and the token at p+1 is `head`.
            let mut token_row = vec![0.0f32; v4flash_kernels::config::HC_DIM as usize];
            embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, head, &mut token_row);
            let m = state.mtp.as_mut().expect("mtp");
            let (d2, _) = state.engine.dspark_draft(
                &mut m.state, &mut m.exit, &m.main_hidden, &m.w, &m.xw, &state.weights,
                &m.markov_embd, m.markov_dtype, pos + n as u32, &token_row, &m.noise_row, head,
            )?;
            m.pending = Some(d2);
            m.pending_conf = m.exit.conf;
            if std::env::var("V41_DSPARK_STEP_TIMING").as_deref() == Ok("1") {
                let d = state
                    .pager
                    .as_ref()
                    .map(|p| p.counters() - spc0)
                    .unwrap_or_default();
                tracing::info!(
                    n,
                    prefill_misses = d.prefill_misses,
                    decode_misses = d.decode_misses,
                    read_ms = (d.prefill_read_ns + d.decode_read_ns) / 1_000_000,
                    fwd_ms = format!("{:.1}", t_fwd.as_secs_f64() * 1e3),
                    argmax_ms = format!("{:.1}", (t_argmax - t_fwd).as_secs_f64() * 1e3),
                    roll_ms = format!("{:.1}", (t_roll - t_argmax).as_secs_f64() * 1e3),
                    draft_ms = format!("{:.1}", (t_step.elapsed() - t_roll).as_secs_f64() * 1e3),
                    step_ms = format!("{:.1}", t_step.elapsed().as_secs_f64() * 1e3),
                    probe_on = v4flash_kernels::het::mtp::slack_probe_phase() as u8,
                    "dspark.step"
                );
            }
        }

        let t_embed = std::time::Instant::now();
        embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, next, &mut residual);
        v4flash_kernels::het::trace::phase::add(
            &v4flash_kernels::het::trace::phase::CALLER_EMBED_NS,
            t_embed.elapsed().as_nanos() as u64,
        );
        // Already in KV from the verify above: advance past it, forward nothing.
        #[cfg(feature = "v41")]
        let spec_ingested = state.mtp.as_mut().is_some_and(|m| {
            let hit = m.ingested > 0;
            if hit {
                m.ingested -= 1;
            }
            hit
        });
        #[cfg(not(feature = "v41"))]
        let spec_ingested = false;
        #[cfg(feature = "v41")]
        let use_mtp = !spec_ingested && state.mtp.is_some() && state.pager.is_some();
        #[cfg(not(feature = "v41"))]
        let use_mtp = false;
        if use_mtp {
            #[cfg(feature = "v41")]
            {
                // Same forward, plus the hc-collapsed residual ENTERING layers
                // 37/38/39 — the drafter's only input from the main model.
                let pg = state.pager.as_mut().expect("pager");
                let engram_rows = match state.engram.as_mut() {
                    Some(ec) => Some(ec.rows_for(pg.raw(), next, pos)?),
                    None => None,
                };
                let m = state.mtp.as_mut().expect("mtp");
                m.capture.begin();
                state.engine.forward_token_paged_mtp(
                    &mut state.dgpu_scratch,
                    &mut state.igpu_scratch,
                    &mut state.state,
                    &state.weights,
                    &residual,
                    pos,
                    next,
                    pg,
                    engram_rows.as_deref(),
                    &mut m.capture,
                )?;
            }
        } else if !spec_ingested {
            forward_one!(state, residual, pos, next)?;
        }
        pos += 1;
        // Successfully ingested `next` into KV at `pos-1`. live.pos
        // always tracks the KV cache position. live.tokens only
        // tracks CANONICAL tokens — what the client will replay as
        // history. Transient tokens (TOK_THINK_BEGIN itself; any
        // token sampled while in_think) go into KV but NOT into
        // live.tokens. TOK_THINK_END IS canonical (every replay of a
        // historical assistant turn renders it). See
        // [[think-cache-design]].
        //
        // Keeping the reasoning tokens out is what makes live.pos run
        // ahead of live.tokens.len() on any turn that reasons, which in
        // turn makes `finish_decode`'s end-of-turn cleanup drop the live
        // session. That is deliberate: the reasoning trace occupies KV
        // positions the replay does not describe, so the cache cannot be
        // extended in place across such a turn at all. If the client ever
        // starts echoing `reasoning_content` back (prompt.rs replays it
        // into the `<think>` block), these tokens would have to become
        // canonical AND TOK_THINK_BEGIN would have to be pushed above,
        // or the two streams keep diverging.
        let canonical =
            next != TOK_THINK_BEGIN && (next == TOK_THINK_END || !in_think);
        if let Some(ref mut live) = state.live {
            live.pos = pos;
            if canonical {
                live.tokens.push(next);
                live.dirty = true;
            }
        }
        if pos >= state.n_kv_max {
            break FinishReason::Length;
        }
        let t_sample = std::time::Instant::now();
        // Speculative tokens the verify already confirmed (and ingested) come
        // from the queue; `next_after` is the model's own correction that ends
        // the accepted run and is NOT yet in KV.
        #[cfg(feature = "v41")]
        let spec_next: Option<i32> = state.mtp.as_mut().and_then(|m| {
            // By this point in the iteration `ingested` has already been
            // decremented for the token just emitted, so the queue of confirmed
            // drafts and the count of verify rows still sitting in KV must be
            // EQUAL. Drift either emits a confirmed token with no KV row behind
            // it, or skips a forward for a row that was never appended -- both
            // desynchronise the cache from the emitted stream silently.
            // Holds trivially (0 == 0) when DSpark accept is off.
            assert_eq!(
                m.confirmed.len(),
                m.ingested,
                "dspark accept: {} confirmed drafts pending but {} verify rows still marked                  ingested in KV",
                m.confirmed.len(),
                m.ingested
            );
            m.confirmed
                .pop_front()
                .or_else(|| if m.ingested == 0 { m.next_after.take() } else { None })
        });
        #[cfg(not(feature = "v41"))]
        let spec_next: Option<i32> = None;
        next = match spec_next {
            Some(t) => t,
            None => state
                .engine
                .sample_next(&mut state.dgpu_scratch, sample_mode, rng.next_f32())?,
        };
        v4flash_kernels::het::trace::phase::add(
            &v4flash_kernels::het::trace::phase::CALLER_SAMPLE_NS,
            t_sample.elapsed().as_nanos() as u64,
        );
        // DSpark SHADOW: draft the next MTP_BLOCK tokens and record them, but
        // do not act on them. The loop's output is untouched, so this measures
        // acceptance against the real model without any chance of changing what
        // the server emits — the drafter is the part that had never been run
        // against real residuals, and acceptance is the only number that says
        // whether it is right.
        #[cfg(feature = "v41")]
        if use_mtp && pos >= 1 {
            let mut token_row = vec![0.0f32; v4flash_kernels::config::HC_DIM as usize];
            embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, next, &mut token_row);
            let m = state.mtp.as_mut().expect("mtp");
            if m.actual.is_empty() {
                m.actual_base = pos;
            }
            m.actual.push(next);
            let mut mh = Vec::new();
            m.capture.read(&mut mh)?;
            m.main_hidden = mh;
            // `capture` is the residual from the forward at `pos - 1`, and
            // `next` is the token sampled from it, which sits at `pos`. That is
            // exactly the reference's (main_hidden @ start_pos, input_ids at
            // start_pos + 1), so the drafts predict `pos + 1 ..= pos + 5`.
            let (drafts, drafts_plain) = state.engine.dspark_draft(
                &mut m.state,
                &mut m.exit,
                &m.main_hidden,
                &m.w,
                &m.xw,
                &state.weights,
                &m.markov_embd,
                m.markov_dtype,
                pos - 1,
                &token_row,
                &m.noise_row,
                next,
            )?;
            m.drafts.push((pos + 1, drafts));
            m.drafts_plain.push((pos + 1, drafts_plain));
            m.pending = Some(drafts);
        }
        #[cfg(feature = "v41")]
        if let Some(vt) = xcheck_pending.take() {
            use std::sync::atomic::Ordering::Relaxed;
            XCHECK_TOT.fetch_add(1, Relaxed);
            XCHECK_ARM_TOT[xcheck_arm].fetch_add(1, Relaxed);
            if vt == next {
                XCHECK_OK.fetch_add(1, Relaxed);
                XCHECK_ARM_OK[xcheck_arm].fetch_add(1, Relaxed);
            }
            // Argmax alone cannot tell a near-tie flip from a real numerical
            // divergence. Compare the whole row: cos ~1 with flips means the
            // two paths agree and the top-2 are close; low cos means they are
            // computing different things.
            if !xcheck_row0.is_empty() {
                let nv = xcheck_row0.len();
                let mut dl = vec![0.0f32; nv];
                if state.dgpu_scratch.logits.len() >= nv {
                    state.dgpu_scratch.logits.slice_view(0, nv).copy_to_host(&mut dl).ok();
                    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                    for (a, bq) in xcheck_row0.iter().zip(&dl) {
                        dot += (*a as f64) * (*bq as f64);
                        na += (*a as f64) * (*a as f64);
                        nb += (*bq as f64) * (*bq as f64);
                    }
                    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
                    XCHECK_COS.fetch_add((cos * 1e6) as i64, Relaxed);
                    XCHECK_COS_N.fetch_add(1, Relaxed);
                    // KL(decode || verify) in nats. decode (`dl`) is ground
                    // truth, verify (`xcheck_row0`) the approximation; softmax
                    // both (max-shifted), then sum q*(log q - log p).
                    {
                        let vmax = xcheck_row0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let dmax = dl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut vz = 0.0f64;
                        let mut dz = 0.0f64;
                        for i in 0..nv {
                            vz += ((xcheck_row0[i] - vmax) as f64).exp();
                            dz += ((dl[i] - dmax) as f64).exp();
                        }
                        let (lvz, ldz) = (vz.ln(), dz.ln());
                        let mut kld = 0.0f64;
                        for i in 0..nv {
                            let lq = (dl[i] - dmax) as f64 - ldz;
                            let q = lq.exp();
                            if q > 1e-12 {
                                let lp = (xcheck_row0[i] - vmax) as f64 - lvz;
                                kld += q * (lq - lp);
                            }
                        }
                        XCHECK_KLD.fetch_add((kld * 1e6) as i64, Relaxed);
                    }
                    XCHECK_ARM_COS[xcheck_arm].fetch_add((cos * 1e6) as i64, Relaxed);
                    XCHECK_ARM_COS_N[xcheck_arm].fetch_add(1, Relaxed);
                }
                xcheck_row0.clear();
            }
        }
        completion_tokens += 1;
    };
    #[cfg(feature = "v41")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        let (ok, tot) = (XCHECK_OK.swap(0, Relaxed), XCHECK_TOT.swap(0, Relaxed));
        if tot > 0 {
            tracing::info!(
                agree = ok,
                total = tot,
                rate = format!("{:.4}", ok as f64 / tot as f64),
                mean_cos = {
                    let n = XCHECK_COS_N.load(Relaxed);
                    let c = XCHECK_COS.swap(0, Relaxed);
                    if n > 0 { format!("{:.6}", c as f64 / 1e6 / n as f64) } else { "n/a".into() }
                },
                mean_kld_nats = {
                    let n = XCHECK_COS_N.swap(0, Relaxed);
                    let k = XCHECK_KLD.swap(0, Relaxed);
                    if n > 0 { format!("{:.5}", k as f64 / 1e6 / n as f64) } else { "n/a".into() }
                },
                "dspark.xcheck: verify-path vs decode-path logits"
            );
        }
        // Per-ROW table. Row 0 is the only row the check above covers, and at
        // temperature > 0 the emitted token is sampled FROM row j, so rows > 0
        // being worse would be emitted directly.
        {
            use std::sync::atomic::Ordering::Relaxed;
            let rows: Vec<String> = (0..XROW_MAX)
                .filter_map(|j| {
                    let n = XROW_N[j].swap(0, Relaxed);
                    if n == 0 {
                        return None;
                    }
                    let ok = XROW_OK[j].swap(0, Relaxed);
                    let kld = XROW_KLD[j].swap(0, Relaxed) as f64 / 1e6 / n as f64;
                    Some(format!("row{j}: n={n} agree={:.4} kld={kld:.6}", ok as f64 / n as f64))
                })
                .collect();
            if !rows.is_empty() {
                tracing::info!(per_row = %rows.join(" | "), "dspark.xcheck.rows");
                let lbl = ["used (j<=n)", "unused (j>n)"];
                let by: Vec<String> = (0..2)
                    .filter_map(|b| {
                        let n = XUSED_N[b].swap(0, Relaxed);
                        if n == 0 {
                            return None;
                        }
                        let ok = XUSED_OK[b].swap(0, Relaxed);
                        let kld = XUSED_KLD[b].swap(0, Relaxed) as f64 / 1e6 / n as f64;
                        Some(format!(
                            "{}: n={n} agree={:.4} kld={kld:.6}",
                            lbl[b], ok as f64 / n as f64
                        ))
                    })
                    .collect();
                tracing::info!(by_use = %by.join(" | "), "dspark.xcheck.used");
                let names = ["<0.01", "<0.1", "<1", "<5", ">=5"];
                let hist: Vec<String> = (0..5)
                    .map(|i| format!("{}: {}", names[i], XTAIL[i].swap(0, Relaxed)))
                    .collect();
                tracing::info!(
                    kld_hist = %hist.join("  "),
                    max_kld = format!("{:.3}", XTAIL_MAX.swap(0, Relaxed) as f64 / 1e6),
                    "dspark.xcheck.tail (rows we emit from)"
                );
            }
        }
        if small_b_catchall_ab() > 0 || single_lane_ab() > 0 || xcheck_poison() || verify_probe_ks.len() > 1 {
            let by_width = small_b_catchall_ab() == 0 && single_lane_ab() == 0;
            for arm in 0..XCHECK_ARMS {
                let t = XCHECK_ARM_TOT[arm].swap(0, Relaxed);
                if t == 0 {
                    continue;
                }
                let o = XCHECK_ARM_OK[arm].swap(0, Relaxed);
                let (c, n) = (XCHECK_ARM_COS[arm].swap(0, Relaxed), XCHECK_ARM_COS_N[arm].swap(0, Relaxed));
                let us = XCHECK_ARM_US[arm].swap(0, Relaxed);
                tracing::info!(
                    arm = if small_b_catchall_ab() > 0 {
                        if arm == 1 { "catchall_on".to_string() } else { "catchall_off".to_string() }
                    } else if single_lane_ab() > 0 {
                        if arm == 1 { "single_lane".to_string() } else { "two_lane".to_string() }
                    } else if xcheck_poison() {
                        if arm == 1 { "rows1+_poisoned".to_string() } else { "clean".to_string() }
                    } else {
                        let _ = by_width;
                        format!("B={arm}")
                    },
                    agree = o,
                    total = t,
                    rate = format!("{:.4}", o as f64 / t as f64),
                    mean_cos = {
                        if n > 0 { format!("{:.6}", c as f64 / 1e6 / n as f64) } else { "n/a".into() }
                    },
                    probe_ms = format!("{:.1}", us as f64 / 1000.0 / t as f64),
                    "dspark.xcheck.ab: small-B catch-all, interleaved by step parity"
                );
            }
        }
    }

    // DSpark shadow scoring: for each draft batch, how long a prefix matched
    // what the model actually generated. `E` counts the ingested token too, so
    // it is directly comparable to the Python oracle's 1.93 / 2.77 / 3.57 / 4.94
    // at K = 1 / 2 / 3 / 5 (a per-token acceptance of ~0.92 reproduces all four).
    #[cfg(feature = "v41")]
    if let Some(m) = state.mtp.as_mut() {
        use v4flash_kernels::het::mtp::MTP_BLOCK;
        let mut hist = [0usize; MTP_BLOCK + 1];
        let (mut batches, mut total) = (0usize, 0usize);
        for (first_pos, d) in &m.drafts {
            let Some(i0) = first_pos.checked_sub(m.actual_base) else { continue };
            let i0 = i0 as usize;
            if i0 + MTP_BLOCK > m.actual.len() {
                continue;
            }
            let mut n = 0usize;
            while n < MTP_BLOCK && d[n] == m.actual[i0 + n] {
                n += 1;
            }
            hist[n] += 1;
            total += n;
            batches += 1;
        }
        // Per-depth acceptance, directly comparable to the oracle's `greedy_acc`
        // in ~/.cache/deepstrix/v41/agentic/gen2/dspark_accept_base.json:
        //   base (with prefill window seeding) [0.843, 0.730, 0.674, 0.607, 0.562]
        //   no seed (what we implement today)  [0.764, 0.618, 0.461, 0.348, 0.213]
        // Those are AGENTIC tool-calling text; freeform prose runs ~2.2x lower
        // (d1 0.562 / E 1.99), so the content of the probe prompt matters as
        // much as the implementation.
        {
            let mut hit = [0usize; MTP_BLOCK];
            let mut tot = 0usize;
            for (first_pos, d) in &m.drafts {
                let Some(i0) = first_pos.checked_sub(m.actual_base) else { continue };
                let i0 = i0 as usize;
                if i0 + MTP_BLOCK > m.actual.len() { continue; }
                for k in 0..MTP_BLOCK {
                    if d[k] == m.actual[i0 + k] { hit[k] += 1; }
                }
                tot += 1;
            }
            if tot > 0 {
                let acc: Vec<String> = hit
                    .iter()
                    .map(|h| format!("{:.3}", *h as f64 / tot as f64))
                    .collect();
                tracing::info!(steps = tot, greedy_acc = ?acc, "dspark.shadow.per_depth");
            }
        }
        // A few concrete batches, so "weak" can be distinguished from "broken":
        // plausible-but-wrong continuations mean a fine-grained problem, junk
        // means something structural.
        for (i, (fp, d)) in m.drafts.iter().enumerate().take(6) {
            let Some(i0) = fp.checked_sub(m.actual_base) else { continue };
            let i0 = i0 as usize;
            if i0 + MTP_BLOCK > m.actual.len() { continue; }
            tracing::info!(
                pos = *fp,
                drafts = ?d,
                plain = ?m.drafts_plain.get(i).map(|x| x.1),
                actual = ?&m.actual[i0..i0 + MTP_BLOCK],
                "dspark.shadow.sample"
            );
        }
        // Markov ablation: same scoring over the transformer-only drafts.
        {
            let (mut b2, mut t2, mut fh) = (0usize, 0usize, 0usize);
            for (first_pos, d) in &m.drafts_plain {
                let Some(i0) = first_pos.checked_sub(m.actual_base) else { continue };
                let i0 = i0 as usize;
                if i0 + MTP_BLOCK > m.actual.len() { continue; }
                let mut n = 0usize;
                while n < MTP_BLOCK && d[n] == m.actual[i0 + n] { n += 1; }
                if n > 0 { fh += 1; }
                t2 += n;
                b2 += 1;
            }
            if b2 > 0 {
                tracing::info!(
                    batches = b2,
                    mean_accepted = format!("{:.3}", t2 as f64 / b2 as f64),
                    first_draft_hit_rate = format!("{:.3}", fh as f64 / b2 as f64),
                    "dspark.shadow.no_markov"
                );
            }
        }
        // Alignment probe: score the same drafts against actual tokens shifted by
        // -1/0/+1. If a neighbouring shift scores better, the drafts are right
        // and the position bookkeeping is off by one — which is invisible in any
        // single-step test and would look exactly like a weak drafter.
        for shift in [-1i32, 0, 1] {
            let (mut b2, mut t2, mut first_hit) = (0usize, 0usize, 0usize);
            for (first_pos, d) in &m.drafts {
                let Some(base) = (*first_pos as i64 - m.actual_base as i64).checked_add(shift as i64)
                else { continue };
                if base < 0 || base as usize + MTP_BLOCK > m.actual.len() {
                    continue;
                }
                let i0 = base as usize;
                let mut n = 0usize;
                while n < MTP_BLOCK && d[n] == m.actual[i0 + n] {
                    n += 1;
                }
                if n > 0 {
                    first_hit += 1;
                }
                t2 += n;
                b2 += 1;
            }
            if b2 > 0 {
                tracing::info!(
                    shift,
                    batches = b2,
                    mean_accepted = format!("{:.3}", t2 as f64 / b2 as f64),
                    first_draft_hit_rate = format!("{:.3}", first_hit as f64 / b2 as f64),
                    "dspark.shadow.align"
                );
            }
        }
        // Cold vs warm ring: the drafter's 128-entry window starts EMPTY (the
        // reference seeds it from the prompt during prefill; we do not yet), so
        // early batches see far less context than late ones.
        {
            let warm: Vec<usize> = m
                .drafts
                .iter()
                .enumerate()
                .filter(|(i, _)| *i >= 128)
                .filter_map(|(_, (fp, d))| {
                    let i0 = fp.checked_sub(m.actual_base)? as usize;
                    if i0 + MTP_BLOCK > m.actual.len() { return None; }
                    let mut n = 0;
                    while n < MTP_BLOCK && d[n] == m.actual[i0 + n] { n += 1; }
                    Some(n)
                })
                .collect();
            if !warm.is_empty() {
                tracing::info!(
                    warm_batches = warm.len(),
                    mean_accepted = format!("{:.3}", warm.iter().sum::<usize>() as f64 / warm.len() as f64),
                    "dspark.shadow.warm_ring"
                );
            }
        }
        if batches > 0 {
            let mean = total as f64 / batches as f64;
            tracing::info!(
                batches,
                mean_accepted = format!("{mean:.3}"),
                e_tokens_per_verify = format!("{:.3}", 1.0 + mean),
                per_token_accept = format!("{:.3}", {
                    // a from E = sum_{k=1..K} a^k, by bisection.
                    let (mut lo, mut hi) = (0.0f64, 1.0f64);
                    for _ in 0..60 {
                        let a = 0.5 * (lo + hi);
                        let mut sum = 0.0;
                        let mut p = 1.0;
                        for _ in 0..MTP_BLOCK { p *= a; sum += p; }
                        if sum < mean { lo = a } else { hi = a }
                    }
                    0.5 * (lo + hi)
                }),
                prefix_hist = ?hist,
                "dspark.shadow: accepted-prefix distribution"
            );
        }
        {
            use std::sync::atomic::Ordering::Relaxed;
            let (ok, tot) = (XCHECK_OK.swap(0, Relaxed), XCHECK_TOT.swap(0, Relaxed));
            if tot > 0 {
                tracing::info!(
                    agree = ok,
                    total = tot,
                    rate = format!("{:.4}", ok as f64 / tot as f64),
                    "dspark.xcheck: verify-path argmax vs decode-path argmax"
                );
            }
        }
        if m.accept_steps > 0 {
            tracing::info!(
                steps = m.accept_steps,
                tokens = m.accept_tokens,
                e_tokens_per_step = format!(
                    "{:.3}",
                    m.accept_tokens as f64 / m.accept_steps as f64
                ),
                "dspark.accept: tokens per verify step"
            );
        }
        m.accept_steps = 0;
        m.accept_tokens = 0;
        m.pending = None;
        m.confirmed.clear();
        m.ingested = 0;
        m.next_after = None;
        m.main_hidden.clear();
        m.drafts.clear();
        m.drafts_plain.clear();
        m.actual.clear();
    }

    // The loop's own wall. Measured 2026-09-14 (l1_1.log): 60.8 ms/token here
    // vs 88 ms/token by curl-wall/256 — the difference was the 6.9 s prefill of
    // a 36-token prompt (CED replay paging ~1500 decoder experts on box 1).
    {
        let wall = loop_t0.elapsed();
        tracing::info!(
            completion_tokens,
            loop_ms = wall.as_millis() as u64,
            ms_per_tok = format!("{:.2}", wall.as_secs_f64() * 1e3 / completion_tokens.max(1) as f64),
            finish = ?finish,
            "decode.loop.summary"
        );
        // One greppable line per request, for reading a long session of real
        // traffic. `n_hist`/`accept_by_pos` say WHERE the drafter fails (E
        // alone cannot); the box-2 counters say whether a slow request was
        // compute or box-2 NVMe, which box 1's own pager counters cannot see.
        let (page_us, miss) =
            v4flash_kernels::het::remote_experts::link_stats::take_paging();
        let (page_us_dec, miss_dec) =
            v4flash_kernels::het::remote_experts::link_stats::take_paging_decode();
        // Per-layer decode misses on box 2, as `L<layer>:<n>` for layers that missed.
        let miss_by_layer = v4flash_kernels::het::expert_pager::take_box2_miss_by_layer()
            .iter()
            .enumerate()
            .filter(|(_, &n)| n > 0)
            .map(|(l, n)| format!("L{l}:{n}"))
            .collect::<Vec<_>>()
            .join(",");
        let link = v4flash_kernels::het::remote_experts::link_stats::take()
            .iter()
            .map(|(b, n, us, by)| format!("b{b}:n={n},link={us:.0}us,bytes={by:.0}"))
            .collect::<Vec<_>>()
            .join(" ");
        {
            let mut acc = REQ_TOUCHED.lock().unwrap();
            if !acc.is_empty() {
                let ne = v4flash_kernels::config::N_EXPERT as usize;
                let nl = v4flash_kernels::config::N_LAYER as usize;
                let mut d: Vec<usize> = (0..nl)
                    .map(|l| acc[l * ne..(l + 1) * ne].iter().filter(|&&v| v > 0).count())
                    .filter(|&c| c > 0)
                    .collect();
                if !d.is_empty() {
                    d.sort_unstable();
                    let mut pf = REQ_TOUCHED_PF.lock().unwrap();
                    let mut dp: Vec<usize> = if pf.is_empty() {
                        Vec::new()
                    } else {
                        (0..nl)
                            .map(|l| pf[l * ne..(l + 1) * ne].iter().filter(|&&v| v > 0).count())
                            .filter(|&c| c > 0)
                            .collect()
                    };
                    dp.sort_unstable();
                    tracing::info!(
                        completion_tokens,
                        decode_distinct_median = d[d.len() / 2],
                        decode_distinct_max = d[d.len() - 1],
                        verify_distinct_median = if dp.is_empty() { 0 } else { dp[dp.len() / 2] },
                        verify_distinct_max = if dp.is_empty() { 0 } else { dp[dp.len() - 1] },
                        verify_total = dp.iter().sum::<usize>(),
                        box2_slots_per_layer = 154,
                        "expert.working_set REQUEST"
                    );
                    pf.clear();
                }
                acc.clear();
            }
        }
        tracing::info!(
            completion_tokens,
            ms_per_tok = format!("{:.2}", wall.as_secs_f64() * 1e3 / completion_tokens.max(1) as f64),
            tok_per_s = format!("{:.2}", completion_tokens as f64 / wall.as_secs_f64().max(1e-9)),
            stats = %dspark_stats::take(),
            conf = %dspark_stats::take_conf(),
            box2_page_ms = page_us / 1000,
            box2_miss = miss,
            box2_miss_decode = miss_dec,
            box2_page_ms_decode = page_us_dec / 1000,
            box2_miss_by_layer = %miss_by_layer,
            b1_prefetch = %state
                .pager
                .as_ref()
                .and_then(|p| p.prefetch_stats())
                .map(|(q, a, d, ms)| format!("queued={q} admitted={a} dropped={d} admit_ms={ms}"))
                .unwrap_or_else(|| "off".into()),
            link = %link,
            "dspark.request"
        );
    }

    // Force EOS into the KV cache at end-of-turn so the next request's
    // history (which always renders EOS after a closed assistant turn,
    // per `prompt.rs`) prefix-matches the live cache. Mirrors
    // `chat.rs:582-600`'s end_of_turn handling.
    //
    // Skipped on cancel: the EOS-forward is for snapshot-prefix
    // continuity, but on cancel we drop live entirely below, so
    // forwarding EOS would just contaminate the in-VRAM KV state for
    // the brief window between here and the next request's
    // reset_in_place.
    if matches!(finish, FinishReason::Stop) && !was_cancelled && pos < state.n_kv_max {
        embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, TOK_EOS, &mut residual);
        forward_one!(state, residual, pos, TOK_EOS)?;
        pos += 1;
        if let Some(ref mut live) = state.live {
            live.tokens.push(TOK_EOS);
            live.pos = pos;
            live.dirty = true;
        }
    }

    let _ = tx.blocking_send(WorkerEvent::Done {
        prompt_tokens,
        completion_tokens,
        finish,
    });

    // Cancel cleanup. live.tokens accumulated canonical samples while
    // the decode loop was running; on cancel those samples are GARBAGE
    // from the user's perspective (they pressed stop because they
    // didn't want it). Letta's retry typically resubmits with the
    // streamed-before-cancel content as a prior assistant turn —
    // byte_aligned_lcp would then match into live, the extend path
    // would prefill the new suffix on top of the contaminated KV
    // cache, and the model would attend back to its own half-finished
    // output. Drop live so the next request reset-prefills cleanly.
    //
    // Symmetric with the send_failed cleanup above (line ~1485).
    if was_cancelled {
        tracing::info!(
            completion_tokens,
            pos,
            fp = %state_fingerprint(state),
            "decode: dropping live on cancel"
        );
        state.live = None;
    }

    // End-of-turn cleanup. We no longer save the snapshot here —
    // the meaningful save happened in handle_generate_stream BEFORE
    // we forwarded the trailing `<think>` marker (so the snapshot
    // bytes match what letta will replay as history).
    //
    // If the KV cache holds transient tokens past the canonical
    // position (i.e. we forwarded `<think>` + thinking content +
    // `</think>` but only pushed `</think>` + content to
    // live.tokens), the in-VRAM state can't be safely extended on
    // the next request — RoPE positions would be off. Drop it; the
    // next request will pick up the start-of-think snapshot from
    // disk and re-forward the now-canonical suffix.
    if let Some(live) = &state.live {
        if (live.pos as usize) > live.tokens.len() {
            tracing::debug!(
                pos = live.pos,
                tokens_len = live.tokens.len(),
                "clearing live cache: transient tokens past canonical \
                 position (cross-turn extension would mis-position the KV)"
            );
            state.live = None;
        }
    }

    // Forensic: end-of-turn fingerprint. The next request's entry
    // fingerprint should match this — divergence between adjacent
    // turns is a smoking gun for state corruption (cancel cleanup
    // missed something, scratch leaked, etc.). `finish` tells us
    // whether we're exiting via natural stop, length cap, or cancel.
    tracing::debug!(
        finish = ?finish,
        completion_tokens,
        fp = %state_fingerprint(state),
        "decode: end"
    );

    Ok(())
}

/// Run the batched-prefill pipeline for `tokens` starting at `pos0`.
/// Used both for full prefill (pos0=0) and for the extension fast-path.
///
/// `cancel` is checked at chunk boundary inside the engine; on trip,
/// the call returns Ok early and the caller is responsible for
/// observing the bool and unwinding (clearing `live`, no events).
/// Spans of `spans` whose START is at or after `from`, unshifted.
/// (Companion to `shift_spans`; the caller applies the delta.)
fn spans_from(spans: &[ImageSpan], from: usize) -> Vec<ImageSpan> {
    spans
        .iter()
        .filter(|s| s.start as usize >= from)
        .cloned()
        .collect()
}

/// Tower output for one request: a 4096-d embedding row per image-block
/// token, keyed by the block's position in `GenerateReq::tokens`.
#[derive(Default)]
struct EncodedImages {
    /// `(block_start, block_len, rows)` where `rows` is
    /// `block_len * N_EMBD` f32 in block order — aligner rows at IMAGE
    /// slots, sentinel vectors at PAD / START / NEWLINE / END.
    blocks: Vec<(usize, usize, Vec<f32>)>,
    /// The request's spans, in `tokens` index space.
    spans: Vec<ImageSpan>,
}

impl EncodedImages {
    /// The 4096-d row for token index `i`, when `i` is an image slot.
    fn row_at(&self, i: usize) -> Option<&[f32]> {
        let n_embd = N_EMBD as usize;
        self.blocks.iter().find_map(|(start, len, rows)| {
            (i >= *start && i < start + len).then(|| {
                let k = i - start;
                &rows[k * n_embd..(k + 1) * n_embd]
            })
        })
    }
}

/// How many `Tower::encode_rows` results `WorkerState::vit_rows` keeps.
/// Four covers a conversation carrying a handful of images without
/// holding more than ~25 MiB of host RAM.
const VIT_ROW_CACHE_MAX: usize = 4;

/// Run the ViT + aligner for every image in `req`. No-op (and no tower
/// required) for text-only requests.
///
/// DEVICE DISCIPLINE: `Tower::encode_rows` sets the iGPU current on this
/// thread and leaves it there. Every text path below assumes the dGPU is
/// current, and `HeterogeneousEngine::set_current_cached` skips the
/// driver call whenever its cached id already says "dGPU" — so returning
/// with the iGPU current would make the NEXT request's decode kernels
/// launch under the wrong context (`forward_token` only calls
/// `set_current_cached`; the batched prefill entry points force a reset).
/// The restore therefore runs on EVERY exit, success or failure, and we
/// invalidate the engine's cache because the tower changed the thread's
/// device behind its back.
fn encode_request_images(
    state: &mut WorkerState,
    req: &GenerateReq,
) -> eyre::Result<EncodedImages> {
    if req.images.is_empty() {
        return Ok(EncodedImages::default());
    }
    let r = encode_request_images_inner(state, req);
    state.engine.invalidate_device_cache();
    match (r, state.dgpu.set_current()) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => {
            Err(eyre!("restoring the dGPU as current after a vision encode: {e:#}"))
        }
    }
}

/// The body of [`encode_request_images`]. May return with the iGPU
/// current — the caller restores unconditionally.
fn encode_request_images_inner(
    state: &mut WorkerState,
    req: &GenerateReq,
) -> eyre::Result<EncodedImages> {
    if req.images.len() != req.image_spans.len() {
        return Err(eyre!(
            "generate: {} images but {} image spans",
            req.images.len(),
            req.image_spans.len()
        ));
    }
    if state.tower.is_none() {
        return Err(eyre!(
            "request carries images but no vision tower is loaded (start with --mmproj)"
        ));
    }
    let mut blocks = Vec::with_capacity(req.images.len());
    for (k, img) in req.images.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let hash = img.image.content_hash;
        // ViT rows depend only on the normalised pixels, so the memo key
        // is the content hash; the block LAYOUT (which varies with the
        // image's position in the prompt) is applied afterwards by the
        // cheap host-side `place_rows` scatter.
        let hit = state.vit_rows.iter().position(|(h, _)| *h == hash);
        let aligner = match hit {
            Some(i) => {
                let e = state.vit_rows.remove(i);
                let rows = std::sync::Arc::clone(&e.1);
                state.vit_rows.insert(0, e);
                rows
            }
            None => {
                let tower = state.tower.as_mut().expect("checked above");
                let rows = std::sync::Arc::new(
                    tower
                        .encode_rows(&img.image)
                        .map_err(|e| eyre!("vision tower encode (image {k}): {e:#}"))?,
                );
                state.vit_rows.insert(0, (hash, std::sync::Arc::clone(&rows)));
                state.vit_rows.truncate(VIT_ROW_CACHE_MAX);
                rows
            }
        };
        let tower = state.tower.as_ref().expect("checked above");
        let rows = tower
            .place_rows(&img.layout, &aligner)
            .map_err(|e| eyre!("vision tower place_rows (image {k}): {e:#}"))?;
        let want = img.block_len() * N_EMBD as usize;
        if rows.len() != want {
            return Err(eyre!(
                "vision tower returned {} floats for image {k}, expected {want}",
                rows.len()
            ));
        }
        tracing::info!(
            image = k,
            cached = hit.is_some(),
            vit = format!("{}x{}", img.image.n_vit_h, img.image.n_vit_w),
            llm = format!("{}x{}", img.layout.n_llm_h, img.layout.n_llm_w),
            block_tokens = img.block_len(),
            ms = t0.elapsed().as_millis() as u64,
            "encoded image"
        );
        blocks.push((img.block_start(), img.block_len(), rows));
    }
    Ok(EncodedImages {
        blocks,
        spans: req.image_spans.clone(),
    })
}

/// Prefill `tokens`, which are `req.tokens[base_idx .. base_idx + len]`,
/// into KV positions starting at `pos0`.
///
/// `vl` supplies the vision embedding rows (image slots) and the image
/// spans; spans overlapping this range are rebased to absolute KV
/// positions and handed to the engine, which keeps every span inside one
/// chunk AND one pipeline lane (`het::image_spans::plan_chunk`). A range
/// boundary that cuts through a block is a hard error here — the callers
/// above choose their boundaries so that cannot happen.
fn prefill_suffix(
    state: &mut WorkerState,
    tokens: &[i32],
    base_idx: usize,
    pos0: u32,
    vl: &EncodedImages,
    cancel: Option<&AtomicBool>,
) -> eyre::Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    // DSpark prefill ring seeding: have the batched path capture the last
    // MTP_CAP_ROWS main-model residuals of this prefill so `seed_mtp_ring` can
    // replay the drafter over them. Costs one `hc_weighted` launch per MTP
    // source layer per chunk and nothing when no drafter is loaded.
    #[cfg(feature = "v41")]
    {
        let rows = if state.mtp.is_some()
            && std::env::var("V41_DSPARK_SEED_RING").as_deref() != Ok("0")
        {
            v4flash_kernels::het::batch_scratch::MTP_CAP_ROWS
        } else {
            0
        };
        state.bd_a.mtp_capture_rows = rows;
        state.bd_b.mtp_capture_rows = rows;
        state.bd_a.mtp_captured = 0;
        state.bd_b.mtp_captured = 0;
    }
    let n_embd = N_EMBD as usize;
    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
    for (i, &tok) in tokens.iter().enumerate() {
        let mut v = vec![0f32; HC_DIM as usize];
        match vl.row_at(base_idx + i) {
            Some(row) => {
                if !crate::vision_prompt::is_image_token(tok) {
                    return Err(eyre!(
                        "prefill: token {tok} at index {} is not an image id but sits inside an \
                         image block",
                        base_idx + i
                    ));
                }
                // Broadcast the vision row into the 4 HC copies — exactly
                // what `embed_lookup` does after dequantising a text row.
                for h in 0..N_HC as usize {
                    v[h * n_embd..(h + 1) * n_embd].copy_from_slice(row);
                }
            }
            None => {
                if crate::vision_prompt::is_image_token(tok) {
                    return Err(eyre!(
                        "prefill: synthetic image token {tok} at index {} has no encoded row \
                         (image block missing from the request)",
                        base_idx + i
                    ));
                }
                embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, tok, &mut v);
            }
        }
        input_hcs.push(v);
    }
    // Rebase the request's spans onto absolute KV positions.
    let spans_abs: Vec<(u32, u32)> =
        crate::vision_prompt::spans_in_range(&vl.spans, base_idx, base_idx + tokens.len())?
            .iter()
            .map(|s| (s.start + pos0, s.len))
            .collect();
    // V4-Flash: the engine widens each image row's raw window to the whole
    // `[START..END]` span (bidirectional inside the image) and keeps every
    // span inside one chunk / lane. V4.1 has no such rule (`model.py` has no
    // `get_image_visible`; image tokens attend causally like text), so it gets
    // NO spans — routing still picks `bias_vl` for image rows off their
    // synthetic ids, and the Engram rows above are already masked.
    let image_spans = if cfg!(feature = "v41") {
        None
    } else {
        (!spans_abs.is_empty()).then_some(spans_abs.as_slice())
    };
    // Clone the progress handle into a local so the per-chunk pet
    // closure doesn't co-borrow `state` with state.engine below.
    // WorkerProgress is two Arc clones — effectively free.
    let progress = state.progress.clone();
    let pet_each_chunk = || progress.pet();
    // Engram rows for the whole prompt: batched prefill stages them per layer
    // per lane. Gathered here because the tables are SSD-resident and the gather
    // needs the same HF source the pager owns.
    #[cfg(feature = "v41")]
    let engram_chunk: Option<Vec<Vec<f32>>> = match (state.pager.as_ref(), state.engram.as_mut()) {
        (Some(pg), Some(ec)) => {
            let raw = pg.raw();
            // Timed: this sits OUTSIDE `prefill_start`, so it never appeared in any
            // prefill stage timing despite being ~0.53 ms/token before the fix.
            let t_eng = std::time::Instant::now();
            let rows = ec.rows_for_chunk(raw, tokens, pos0)?;
            tracing::info!(
                tokens = tokens.len(),
                elapsed_ms = t_eng.elapsed().as_millis() as u64,
                us_per_token = (t_eng.elapsed().as_micros() as f64 / tokens.len().max(1) as f64),
                "engram rows_for_chunk"
            );
            Some(rows)
        }
        _ => None,
    };
    // M7: batched prefill now pages this layer's experts out of the pool
    // (forward_layer_pre_moe_v2), so paged mode takes the SAME batched path as
    // resident mode — no more token-at-a-time fallback.
    let _ = state.engine.forward_prefill_pipelined(
        &mut state.bd_a,
        &mut state.bi_a,
        &mut state.bd_b,
        &mut state.bi_b,
        &mut state.sd,
        &mut state.si,
        &mut state.dgpu_scratch,
        &mut state.state,
        &state.weights,
        &input_hcs,
        tokens,
        pos0,
        true,
        None,
        cancel,
        Some(&pet_each_chunk),
        image_spans,
        // M7: page experts per layer instead of reading resident buffers.
        #[cfg(feature = "v41")]
        state.pager.as_mut(),
        #[cfg(not(feature = "v41"))]
        None,
        #[cfg(feature = "v41")]
        engram_chunk.as_deref(),
        #[cfg(not(feature = "v41"))]
        None,
    )?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Plain text content (post-DSML-scanner, UTF-8-clean).
    pub text: String,
    /// Tool calls parsed out of the DSML markup.
    pub tool_calls: Vec<crate::openai::types::ToolCall>,
    /// True if the scanner saw any tool call or the tool_calls block
    /// closed (used to set finish_reason="tool_calls").
    pub saw_tool: bool,
    /// True if the scanner hit an unknown DSML tag and fell back to
    /// Text mode — the model emitted broken markup (e.g.
    /// `<｜DSML｜command …>` instead of `<｜DSML｜parameter
    /// name="command">`). Callers report `finish_reason: "error"` so
    /// letta treats the turn as failed rather than recording the
    /// corrupted markup as content.
    pub saw_malformed: bool,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: FinishReason,
}

/// Drain a `submit` stream into a single accumulated result, driving
/// the DSML scanner so TOK_DSML bytes never leak into the text field
/// and tool calls are separated out structurally. Used by the
/// non-streaming chat-completions path.
///
/// `tok_dsml` is the vocab's `｜DSML｜` token id — pass `None` to
/// disable DSML scanning (treat all content as plain text).
///
/// The text field gets only NON-reasoning, post-DSML-scanner content.
/// Reasoning tokens are dropped (OpenAI's non-streaming response
/// doesn't have a reasoning_content field). A UTF-8 buffer holds
/// trailing 0–3 bytes of any incomplete multi-byte character across
/// scanner Text events.
pub async fn accumulate(
    mut rx: mpsc::Receiver<WorkerEvent>,
    tok_dsml: Option<i32>,
) -> eyre::Result<GenerateResult> {
    use crate::dsml::{DsmlEvent, DsmlScanner};
    use crate::openai::types::{ToolCall, ToolCallFunction};

    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut saw_tool = false;
    let mut pending: Vec<u8> = Vec::new();
    let mut last: Option<(u32, u32, FinishReason)> = None;
    let mut scanner = DsmlScanner::new(tok_dsml.unwrap_or(-1));

    fn drain_valid_utf8(pending: &mut Vec<u8>, chunk: &[u8]) -> String {
        pending.extend_from_slice(chunk);
        let valid_to = match std::str::from_utf8(pending) {
            Ok(_) => pending.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_to == 0 {
            return String::new();
        }
        let drained: Vec<u8> = pending.drain(..valid_to).collect();
        String::from_utf8(drained).unwrap()
    }

    while let Some(ev) = rx.recv().await {
        match ev {
            WorkerEvent::Chunk {
                token_id,
                bytes,
                reasoning,
            } => {
                if reasoning {
                    continue;
                }
                for de in scanner.push_token(token_id, &bytes) {
                    match de {
                        DsmlEvent::Text(b) => {
                            let s = drain_valid_utf8(&mut pending, &b);
                            if !s.is_empty() {
                                text.push_str(&s);
                            }
                        }
                        DsmlEvent::ToolCall {
                            id, name, arguments, ..
                        } => {
                            saw_tool = true;
                            tool_calls.push(ToolCall {
                                id,
                                kind: "function".into(),
                                function: ToolCallFunction { name, arguments },
                            });
                        }
                        DsmlEvent::ToolCallsEnd => saw_tool = true,
                    }
                }
            }
            WorkerEvent::Done {
                prompt_tokens,
                completion_tokens,
                finish,
            } => {
                last = Some((prompt_tokens, completion_tokens, finish));
            }
            WorkerEvent::Error(e) => return Err(eyre!("engine error: {e}")),
        }
    }
    // Drain scanner state. Since the DSML-repair port, finish() can
    // also emit ToolCall / ToolCallsEnd events (truncated tool-call
    // blocks recovered by appending the missing closing tags) — they
    // must be collected exactly like live-stream ones.
    for de in scanner.finish() {
        match de {
            DsmlEvent::Text(b) => {
                let s = drain_valid_utf8(&mut pending, &b);
                if !s.is_empty() {
                    text.push_str(&s);
                }
            }
            DsmlEvent::ToolCall {
                id, name, arguments, ..
            } => {
                saw_tool = true;
                tool_calls.push(ToolCall {
                    id,
                    kind: "function".into(),
                    function: ToolCallFunction { name, arguments },
                });
            }
            DsmlEvent::ToolCallsEnd => saw_tool = true,
        }
    }
    if !pending.is_empty() {
        text.push_str(&String::from_utf8_lossy(&pending));
    }
    let (p, c, f) = last.ok_or_else(|| eyre!("worker closed without Done"))?;
    let saw_malformed = scanner.saw_malformed();
    Ok(GenerateResult {
        text,
        tool_calls,
        saw_tool,
        saw_malformed,
        prompt_tokens: p,
        completion_tokens: c,
        finish_reason: f,
    })
}

// Compile-time sanity check.
const _: () = {
    let _ = COMPRESS_RATIOS;
    let _: oneshot::Sender<()>;
};

/// Clear the DSpark drafter's KV ring at a conversation boundary.
///
/// Paired with every `state.state.reset_in_place(..)`: the main model's KV lives
/// in `HetModelState` and is reset there, but the drafter's ring lives on
/// `MtpState` for the life of the process and had no reset at all. See
/// `MtpState::reset_ring` for the measured cost of that omission.
#[cfg(feature = "v41")]
/// Per-request DSpark statistics, for reading real traffic rather than a fixed
/// benchmark prompt.
///
/// The acceptance POSITION histogram is the important one. `E` alone says how
/// many tokens a step won; it cannot say whether the drafter dies at the first
/// draft position or the fifth, and those call for completely different work.
/// `accept[j] / reach[j]` is the per-position accept rate, so `d1` is the
/// drafter's first-token quality — the number the oracles are stated in.
/// (layer,expert) touched by the CURRENT request, OR-ed across its harvests.
static REQ_TOUCHED: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());
/// Same, for the PREFILL stream — which is where a DSpark verify's picks land.
static REQ_TOUCHED_PF: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());

static VERIFY_DIST_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Per-ROW verify-vs-decode agreement. Row 0 was the only row ever checked.
const XROW_MAX: usize = 8;
static XROW_N: [std::sync::atomic::AtomicU64; XROW_MAX] =
    [const { std::sync::atomic::AtomicU64::new(0) }; XROW_MAX];
static XROW_OK: [std::sync::atomic::AtomicU64; XROW_MAX] =
    [const { std::sync::atomic::AtomicU64::new(0) }; XROW_MAX];
static XROW_KLD: [std::sync::atomic::AtomicI64; XROW_MAX] =
    [const { std::sync::atomic::AtomicI64::new(0) }; XROW_MAX];

/// `V41_XCHECK_ROWS=2`: the accept-path (real drafts) variant.
/// [0] = rows at or before the acceptance point (rows we EMIT from),
/// [1] = rows past it (conditioned on rejected drafts, never emitted from).
static XUSED_N: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];
static XUSED_OK: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];
/// KLD histogram for rows we EMIT from: <0.01, <0.1, <1, <5, >=5 nats.
static XTAIL: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];
static XTAIL_MAX: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
static XUSED_KLD: [std::sync::atomic::AtomicI64; 2] =
    [const { std::sync::atomic::AtomicI64::new(0) }; 2];

fn xcheck_rows_accept() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("V41_XCHECK_ROWS").as_deref() == Ok("2"))
}

fn xcheck_rows() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("V41_XCHECK_ROWS").as_deref() == Ok("1"))
}
/// Token the NEXT verify's row 0 must carry (-1 = not yet armed).
static DSPARK_EXPECT_NEXT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);
pub static DSPARK_DESYNC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

mod dspark_stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    const MAXB: usize = v4flash_kernels::het::mtp::MTP_BLOCK + 1;
    static HIST: [AtomicU64; MAXB] = [const { AtomicU64::new(0) }; MAXB];
    static REACH: [AtomicU64; MAXB] = [const { AtomicU64::new(0) }; MAXB];
    static ACCEPT: [AtomicU64; MAXB] = [const { AtomicU64::new(0) }; MAXB];

    /// One verify step that accepted `n` of `k` drafts.
    pub fn record_accept(n: usize, k: usize) {
        HIST[n.min(MAXB - 1)].fetch_add(1, Relaxed);
        // Position j was REACHED if the prefix before it was accepted, and
        // ACCEPTED if the draft there matched. Conditioning on reach is what
        // makes the rate comparable across positions.
        for j in 0..k.min(MAXB) {
            REACH[j].fetch_add(1, Relaxed);
            if j < n {
                ACCEPT[j].fetch_add(1, Relaxed);
            }
            if j >= n {
                break;
            }
        }
    }

    /// Confidence calibration: bucket each REACHED draft by its confidence and
    /// count how many were accepted. Answers the only question that matters
    /// before gating on it -- does a low score actually predict rejection.
    const CB: usize = 8;
    static CONF_N: [AtomicU64; CB] = [const { AtomicU64::new(0) }; CB];
    static CONF_OK: [AtomicU64; CB] = [const { AtomicU64::new(0) }; CB];
    /// Bucket edges over the raw head output (a logit, not a probability).
    const CONF_EDGE: [f32; CB] = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, f32::INFINITY];

    fn conf_bucket(c: f32) -> usize {
        CONF_EDGE.iter().position(|&e| c < e).unwrap_or(CB - 1)
    }

    pub fn record_conf(conf: &[f32], n: usize, k: usize) {
        for j in 0..k.min(conf.len()) {
            let b = conf_bucket(conf[j]);
            CONF_N[b].fetch_add(1, Relaxed);
            if j < n {
                CONF_OK[b].fetch_add(1, Relaxed);
            }
            if j >= n {
                break; // positions past the first rejection were never reached
            }
        }
    }

    pub fn take_conf() -> String {
        let n: Vec<u64> = CONF_N.iter().map(|c| c.swap(0, Relaxed)).collect();
        let a: Vec<u64> = CONF_OK.iter().map(|c| c.swap(0, Relaxed)).collect();
        (0..CB)
            .filter(|&i| n[i] > 0)
            .map(|i| {
                let lo = if i == 0 { f32::NEG_INFINITY } else { CONF_EDGE[i - 1] };
                format!("[{lo:.0},{:.0}):{}/{}={:.2}", CONF_EDGE[i], a[i], n[i], a[i] as f64 / n[i] as f64)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// "hist=a/b/c per_pos=x.xx/y.yy" and drains.
    pub fn take() -> String {
        let h: Vec<u64> = HIST.iter().map(|c| c.swap(0, Relaxed)).collect();
        let r: Vec<u64> = REACH.iter().map(|c| c.swap(0, Relaxed)).collect();
        let a: Vec<u64> = ACCEPT.iter().map(|c| c.swap(0, Relaxed)).collect();
        let hist = h.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("/");
        let pos = r
            .iter()
            .zip(&a)
            .take_while(|(rr, _)| **rr > 0)
            .map(|(rr, aa)| format!("{:.3}", *aa as f64 / *rr as f64))
            .collect::<Vec<_>>()
            .join("/");
        format!("n_hist={hist} accept_by_pos={pos}")
    }
}

pub(crate) fn reset_drafter_ring(state: &mut WorkerState) {
    if let Some(m) = state.mtp.as_mut() {
        m.state.reset_ring();
    }
}

#[cfg(not(feature = "v41"))]
pub(crate) fn reset_drafter_ring(_state: &mut WorkerState) {}
