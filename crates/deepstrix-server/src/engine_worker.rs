//! Single OS-thread worker that owns the V4-Flash inference engine and
//! per-session state.
//!
//! Phase 2 model: the worker emits a `WorkerEvent` stream per request.
//! Both streaming and non-streaming HTTP handlers consume the same
//! event stream — the non-streaming handler just accumulates events
//! into a single response before returning.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use color_eyre::eyre::{self, eyre};
use tokio::sync::{mpsc, oneshot};
use v4flash_core::tokenizer::BpeVocab;
use v4flash_core::MappedGguf;
use v4flash_hip::Device;
use v4flash_kernels::config::{COMPRESS_RATIOS, HC_DIM, N_VOCAB};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, DgpuScratch, ExecMode, HetModelState, HetModelWeights,
    HeterogeneousEngine, IgpuScratch, SampleMode,
};
use v4flash_kernels::sampler::SamplerRng;
use v4flash_kernels::RopeParams;

use crate::embed::{build_gpt2_byte_decoder, embed_lookup, gpt2_decode_token};
use crate::rope_for_layer;
use crate::snapshot::{self, ModelFingerprint, SnapshotIndex};
use crate::tokens::{is_turn_end, TOK_EOS, TOK_THINK_BEGIN, TOK_THINK_END};

/// Per-request input.
pub struct GenerateReq {
    pub tokens: Vec<i32>,
    pub max_new: usize,
    pub temperature: f32,
    pub min_p_rel: f32,
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
    /// One token's decoded bytes, with a flag for whether we're inside
    /// a `<think>…</think>` block. The caller routes reasoning vs.
    /// content to different SSE fields based on this flag.
    Chunk { text: String, reasoning: bool },
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

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::UnboundedSender<EngineRequest>,
    pub vocab: Arc<BpeVocab>,
    pub model_name: Arc<String>,
    /// Total KV-cache capacity in tokens — surfaced on `/v1/models` so
    /// clients (notably letta) can size requests to fit. Matches
    /// `WorkerConfig.n_kv_max`.
    pub n_kv_max: u32,
}

impl EngineHandle {
    /// Submit a generation request. Returns the worker-event stream
    /// and a cancellation handle the caller can flip to ask the worker
    /// to stop mid-decode.
    pub fn submit(
        &self,
        req: GenerateReq,
        session_id: Option<String>,
    ) -> eyre::Result<(mpsc::Receiver<WorkerEvent>, Arc<AtomicBool>)> {
        let (tx, rx) = mpsc::channel(64);
        let cancel = Arc::new(AtomicBool::new(false));
        self.tx
            .send(EngineRequest::Generate {
                req,
                tx,
                session_id,
                cancel: cancel.clone(),
            })
            .map_err(|_| eyre!("engine worker channel closed"))?;
        Ok((rx, cancel))
    }

    /// Block until the worker has saved its dirty live state and exited.
    /// Used during graceful shutdown.
    pub async fn shutdown(&self) -> eyre::Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .send(EngineRequest::Shutdown { ack: ack_tx })
            .map_err(|_| eyre!("engine worker channel closed"))?;
        ack_rx
            .await
            .map_err(|_| eyre!("engine worker dropped shutdown ack"))?;
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
}

pub fn spawn(cfg: WorkerConfig) -> eyre::Result<EngineHandle> {
    let (tx, rx) = mpsc::unbounded_channel::<EngineRequest>();
    let (ready_tx, ready_rx) =
        std::sync::mpsc::sync_channel::<eyre::Result<(Arc<BpeVocab>, Arc<String>)>>(1);
    let n_kv_max = cfg.n_kv_max;

    std::thread::Builder::new()
        .name("deepstrix-engine".into())
        .spawn(move || worker_main(cfg, rx, ready_tx))
        .map_err(|e| eyre!("failed to spawn engine thread: {e}"))?;

    let (vocab, model_name) = ready_rx
        .recv()
        .map_err(|_| eyre!("engine thread dropped ready channel"))??;
    Ok(EngineHandle {
        tx,
        vocab,
        model_name,
        n_kv_max,
    })
}

fn worker_main(
    cfg: WorkerConfig,
    mut rx: mpsc::UnboundedReceiver<EngineRequest>,
    ready_tx: std::sync::mpsc::SyncSender<eyre::Result<(Arc<BpeVocab>, Arc<String>)>>,
) {
    let state = match initialize_state(&cfg) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    let vocab = state.vocab.clone();
    let model_name = Arc::new(cfg.model_name.clone());
    let _ = ready_tx.send(Ok((vocab, model_name)));

    worker_loop(state, &mut rx);
}

pub struct WorkerState {
    pub dgpu: Device,
    pub igpu: Device,
    pub engine: HeterogeneousEngine,
    pub weights: HetModelWeights,
    pub vocab: Arc<BpeVocab>,
    pub token_embd_bytes: Vec<u8>,
    pub byte_decoder: std::collections::HashMap<char, u8>,

    pub state: HetModelState,
    pub dgpu_scratch: DgpuScratch,
    pub igpu_scratch: IgpuScratch,
    pub bd_a: BatchDgpuScratch,
    pub bi_a: BatchIgpuScratch,
    pub bd_b: BatchDgpuScratch,
    pub bi_b: BatchIgpuScratch,

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
}

#[derive(Debug, Clone)]
pub struct LiveSession {
    pub tokens: Vec<i32>,
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
    tracing::info!(gguf = %cfg.gguf_path, "loading GGUF");
    let gguf = MappedGguf::open(&cfg.gguf_path)?;
    let vocab = BpeVocab::from_gguf(gguf.gguf())?;
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

    let token_embd_t = gguf
        .gguf()
        .tensor("token_embd.weight")
        .ok_or_else(|| eyre!("missing token_embd.weight"))?;
    if token_embd_t.dtype != v4flash_core::gguf::GgufType::F16 {
        return Err(eyre!("token_embd dtype {:?} != F16", token_embd_t.dtype));
    }
    let token_embd_bytes = gguf.read_tensor(token_embd_t)?.to_vec();

    let rope = |layer: i32| -> eyre::Result<RopeParams> { Ok(rope_for_layer(layer)) };

    tracing::info!("loading het weights (dGPU ~9 GiB + iGPU ~52 GiB)");
    let t0 = std::time::Instant::now();
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope)?;
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "weights loaded");

    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let igpu_scratch = IgpuScratch::alloc(igpu)?;
    let state = HetModelState::alloc(dgpu, igpu, cfg.n_kv_max)?;
    let bd_a = BatchDgpuScratch::alloc(dgpu)?;
    let bi_a = BatchIgpuScratch::alloc(igpu)?;
    let bd_b = BatchDgpuScratch::alloc(dgpu)?;
    let bi_b = BatchIgpuScratch::alloc(igpu)?;
    tracing::info!(n_kv_max = cfg.n_kv_max, "KV cache allocated");

    let byte_decoder = build_gpt2_byte_decoder();

    // Compute model fingerprint and load (or create) the snapshot index.
    let fingerprint = ModelFingerprint::compute(vocab.vocab_size() as u32, &token_embd_bytes);
    if !cfg.snapshot_root.exists() {
        std::fs::create_dir_all(&cfg.snapshot_root)
            .map_err(|e| eyre!("create snapshot root {:?}: {e}", cfg.snapshot_root))?;
    }
    let snapshot_index = SnapshotIndex::load(
        cfg.snapshot_root.clone(),
        fingerprint.clone(),
        cfg.snapshot_cap_bytes,
    )?;

    Ok(WorkerState {
        dgpu,
        igpu,
        engine,
        weights,
        vocab: Arc::new(vocab),
        token_embd_bytes,
        byte_decoder,
        state,
        dgpu_scratch,
        igpu_scratch,
        bd_a,
        bi_a,
        bd_b,
        bi_b,
        n_kv_max: cfg.n_kv_max,
        live: None,
        snapshot_index,
        model_fingerprint: fingerprint,
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

fn byte_aligned_lcp(
    live: &[i32],
    req: &[i32],
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
) -> AlignedLcp {
    let mut out = AlignedLcp {
        live_tokens: 0,
        req_tokens: 0,
        bridged_tokens: 0,
        first_bridge: None,
    };
    let decode = |id: i32| -> Vec<u8> {
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
            if live[li] == req[ri] {
                li += 1;
                ri += 1;
                continue;
            }
            // Different ids — start byte-buffering both sides.
            let (l_id, r_id) = (live[li], req[ri]);
            live_buf = decode(l_id);
            req_buf = decode(r_id);
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
            live_buf.extend(decode(l_id));
            li += 1;
            bridged_in_round += 1;
        } else {
            live_buf.drain(..req_buf.len());
            req_buf.clear();
            if ri >= req.len() {
                break;
            }
            let r_id = req[ri];
            req_buf.extend(decode(r_id));
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
        let path = "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
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

fn worker_loop(mut state: WorkerState, rx: &mut mpsc::UnboundedReceiver<EngineRequest>) {
    while let Some(msg) = rx.blocking_recv() {
        match msg {
            EngineRequest::Generate {
                req,
                tx,
                session_id,
                cancel,
            } => {
                if let Err(e) = handle_generate_stream(&mut state, req, session_id, cancel, &tx) {
                    let _ = tx.blocking_send(WorkerEvent::Error(format!("{e:#}")));
                }
                // tx is dropped here, signaling end of stream.
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

/// If the live session has uncommitted state, persist it to disk and
/// clear its dirty flag. Called before any operation that would evict
/// the live state (conversation switch, shutdown).
fn save_live_if_dirty(state: &mut WorkerState) {
    let Some(live) = &state.live else { return };
    if !live.dirty {
        return;
    }
    let tokens = live.tokens.clone();
    let session_id = live.session_id.clone();
    match snapshot::save(
        &state.state,
        &tokens,
        state.dgpu,
        state.igpu,
        &state.model_fingerprint,
        state.snapshot_index.root(),
        state.vocab.as_ref(),
        &state.byte_decoder,
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
            tracing::info!(
                tokens = n,
                total_disk = state.snapshot_index.total_bytes(),
                "saved live to disk"
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "snapshot.save failed; live not persisted");
        }
    }
}

fn handle_generate_stream(
    state: &mut WorkerState,
    req: GenerateReq,
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

    let prompt_tokens = req.tokens.len() as u32;

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
            let res = byte_aligned_lcp(
                &live.tokens,
                &req.tokens,
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
            TOK_EOS,
            state.vocab.as_ref(),
            &state.byte_decoder,
        );
        let disk_hit = match (disk_hit_session, disk_hit_walk) {
            (Some(a), Some(b)) => Some(if a.0 >= b.0 { a } else { b }),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        // Disk snapshot lookup. The snapshot's IndexEntry.token_count
        // is in the SAVED token-id space (originally-sampled IDs); the
        // walk's match position is the byte boundary in req-token space.
        // We treat the snapshot as "potentially useful" if it offers more
        // req-side coverage than the in-VRAM live cache.
        if let Some((snap_req_tokens, snap_hash, snap_dir)) = disk_hit {
            if snap_req_tokens > lcp_req {
                save_live_if_dirty(state);
                state.state.reset_in_place(state.dgpu, state.igpu)?;
                let loaded = snapshot::restore(
                    &mut state.state,
                    &snap_dir,
                    state.dgpu,
                    state.igpu,
                    &state.model_fingerprint,
                )?;
                let loaded_len = loaded.len() as u32;
                // Verify the snapshot's BYTE stream is actually a prefix
                // of req. With byte-hashed keys (format v2) this should
                // always be true when find_longest_prefix returned this
                // entry; the session-hint path bypasses that check so
                // we re-verify here.
                let verify = byte_aligned_lcp(
                    &loaded,
                    &req.tokens,
                    state.vocab.as_ref(),
                    &state.byte_decoder,
                );
                if verify.live_tokens != loaded.len() {
                    tracing::warn!(
                        loaded_len = loaded.len(),
                        verify_live = verify.live_tokens,
                        verify_req = verify.req_tokens,
                        "restored snapshot bytes are NOT a prefix of the request; falling back"
                    );
                    state.state.reset_in_place(state.dgpu, state.igpu)?;
                    state.live = None;
                } else {
                    let _ = state.snapshot_index.touch(&snap_hash);
                    let suffix_len = req.tokens.len() - verify.req_tokens;
                    tracing::info!(
                        req_len = req.tokens.len(),
                        restored_live = loaded_len,
                        restored_req = verify.req_tokens,
                        suffix_len,
                        mode = "restore",
                        "prefill"
                    );
                    if suffix_len > 0 {
                        prefill_suffix(state, &req.tokens[verify.req_tokens..], loaded_len)?;
                    }
                    let new_pos = loaded_len + suffix_len as u32;
                    // live.tokens after restore + suffix prefill: loaded
                    // (in saved token-id space) + req.tokens[lcp_req..].
                    let mut new_tokens: Vec<i32> =
                        Vec::with_capacity(loaded.len() + suffix_len);
                    new_tokens.extend_from_slice(&loaded);
                    new_tokens.extend_from_slice(&req.tokens[verify.req_tokens..]);
                    state.live = Some(LiveSession {
                        tokens: new_tokens,
                        pos: new_pos,
                        dirty: false,
                        session_id: session_id.clone(),
                    });
                    return finish_decode(
                        state,
                        req,
                        tx,
                        prompt_tokens,
                        new_pos,
                        session_id,
                        cancel,
                    );
                }
            }
        }

        save_live_if_dirty(state);
        state.state.reset_in_place(state.dgpu, state.igpu)?;
        if state.live.is_some() {
            tracing::warn!(
                lcp_live,
                lcp_req,
                live_len,
                req_len = req.tokens.len(),
                "live cache divergence (byte-aligned); resetting"
            );
        }
        state.live = None;
        prefill_suffix(state, &req.tokens, 0)?;
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
        prefill_suffix(state, suffix, pos0)?;
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
    let new_live_tokens: Vec<i32> = match &state.live {
        Some(l) if lcp_live > 0 => {
            let mut v = Vec::with_capacity(lcp_live + suffix_extend);
            v.extend_from_slice(&l.tokens[..lcp_live]);
            v.extend_from_slice(&req.tokens[lcp_req..]);
            v
        }
        _ => req.tokens.clone(),
    };
    state.live = Some(LiveSession {
        tokens: new_live_tokens,
        pos: new_pos,
        dirty: false,
        session_id: session_id.clone(),
    });

    finish_decode(state, req, tx, prompt_tokens, new_pos, session_id, cancel)
}

// `prompt_tokens` is reported back via OpenAI's usage block (== request
// token count). `start_pos` is the KV-cache position the next sampled
// token will be written at — equals live.pos after byte-aligned
// extend; differs from prompt_tokens when live's token count for the
// matched byte prefix differs from the request's.
fn finish_decode(
    state: &mut WorkerState,
    req: GenerateReq,
    tx: &mpsc::Sender<WorkerEvent>,
    prompt_tokens: u32,
    start_pos: u32,
    _session_id: Option<String>,
    cancel: Arc<AtomicBool>,
) -> eyre::Result<()> {
    let mut pos = start_pos;

    let sample_mode = if req.temperature <= 0.0 {
        SampleMode::Argmax
    } else {
        SampleMode::Multinomial {
            temperature: req.temperature,
            min_p_rel: req.min_p_rel,
        }
    };
    let mut rng = SamplerRng::new(req.seed);
    let mut next = state
        .engine
        .sample_next(&mut state.dgpu_scratch, sample_mode, rng.next_f32())?;
    let mut completion_tokens: u32 = 1;

    let mut residual = vec![0f32; HC_DIM as usize];
    let max_new = req.max_new as u32;
    // Track whether we're inside a `<think>…</think>` block so the
    // handler can route reasoning vs. content to different SSE fields.
    // We start in "thinking" iff the last token forwarded into the KV
    // cache was `<think>`. After a byte-aligned-extend, that's
    // `req.tokens.last()` (which we just appended). For the no-extend
    // case it's also `req.tokens.last()`.
    let mut in_think = req
        .tokens
        .last()
        .map(|&t| t == TOK_THINK_BEGIN)
        .unwrap_or(false);
    let _ = start_pos;
    let finish: FinishReason = loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::info!("generation cancelled by client");
            break FinishReason::Stop;
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
            let raw = gpt2_decode_token(bytes, &state.byte_decoder);
            if !raw.is_empty() {
                let s = String::from_utf8_lossy(&raw).into_owned();
                if tx
                    .blocking_send(WorkerEvent::Chunk {
                        text: s,
                        reasoning: in_think,
                    })
                    .is_err()
                {
                    // Receiver dropped (client disconnected, e.g.).
                    // The KV cache mid-decode is now inconsistent
                    // with live.tokens; mark it as such by clearing
                    // live so the next request reset-prefills.
                    state.live = None;
                    return Ok(());
                }
            }
        }
        if completion_tokens >= max_new {
            break FinishReason::Length;
        }
        if (next as u32) >= N_VOCAB {
            return Err(eyre!("generate: sampled token id {next} out of vocab"));
        }
        embed_lookup(&state.token_embd_bytes, next, &mut residual);
        state.engine.forward_token(
            &mut state.dgpu_scratch,
            &mut state.igpu_scratch,
            &mut state.state,
            &state.weights,
            &residual,
            pos,
            next,
        )?;
        pos += 1;
        // Successfully ingested `next` into KV at `pos-1`. Record it.
        if let Some(ref mut live) = state.live {
            live.tokens.push(next);
            live.pos = pos;
            live.dirty = true;
        }
        if pos >= state.n_kv_max {
            break FinishReason::Length;
        }
        next = state
            .engine
            .sample_next(&mut state.dgpu_scratch, sample_mode, rng.next_f32())?;
        completion_tokens += 1;
    };

    // Force EOS into the KV cache at end-of-turn so the next request's
    // history (which always renders EOS after a closed assistant turn,
    // per `prompt.rs`) prefix-matches the live cache. Mirrors
    // `chat.rs:582-600`'s end_of_turn handling.
    if matches!(finish, FinishReason::Stop) && pos < state.n_kv_max {
        embed_lookup(&state.token_embd_bytes, TOK_EOS, &mut residual);
        state.engine.forward_token(
            &mut state.dgpu_scratch,
            &mut state.igpu_scratch,
            &mut state.state,
            &state.weights,
            &residual,
            pos,
            TOK_EOS,
        )?;
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
    Ok(())
}

/// Run the batched-prefill pipeline for `tokens` starting at `pos0`.
/// Used both for full prefill (pos0=0) and for the extension fast-path.
fn prefill_suffix(state: &mut WorkerState, tokens: &[i32], pos0: u32) -> eyre::Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
    for &tok in tokens {
        let mut v = vec![0f32; HC_DIM as usize];
        embed_lookup(&state.token_embd_bytes, tok, &mut v);
        input_hcs.push(v);
    }
    let _ = state.engine.forward_prefill_pipelined(
        &mut state.bd_a,
        &mut state.bi_a,
        &mut state.bd_b,
        &mut state.bi_b,
        &mut state.dgpu_scratch,
        &mut state.state,
        &state.weights,
        &input_hcs,
        tokens,
        pos0,
        true,
        None,
    )?;
    Ok(())
}

/// Convenience helper used by oneshot helpers (kept for backwards
/// compat with the Phase 1 handler; will be removed once everything
/// goes through the stream interface).
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: FinishReason,
}

/// Drain a `submit` stream into a single accumulated result. The
/// `text` field collects only NON-reasoning content (post-`</think>`),
/// which is what OpenAI clients expect for `choices[].message.content`.
/// Reasoning tokens are dropped (clients see them via `reasoning_content`
/// in the streaming path; the non-streaming path doesn't have a
/// `reasoning_content` field today).
pub async fn accumulate(
    mut rx: mpsc::Receiver<WorkerEvent>,
) -> eyre::Result<GenerateResult> {
    let mut text = String::new();
    let mut last: Option<(u32, u32, FinishReason)> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            WorkerEvent::Chunk { text: s, reasoning } => {
                if !reasoning {
                    text.push_str(&s);
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
    let (p, c, f) = last.ok_or_else(|| eyre!("worker closed without Done"))?;
    Ok(GenerateResult {
        text,
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
