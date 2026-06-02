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
use crate::tokens::{is_turn_end, TOK_ASSISTANT, TOK_EOS, TOK_THINK_BEGIN, TOK_THINK_END};

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
            TOK_ASSISTANT,
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
        if let Some((snap_req_tokens, snap_hash, snap_dir)) = disk_hit {
            if snap_req_tokens >= DISK_RESTORE_MIN_TOKENS {
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
/// The snapshot's saved tokens (== live.tokens at this point) match
/// what letta will replay for this turn's history on subsequent
/// requests: canonical bytes through `<Assistant>`, with no
/// `<think>` baked in. The marker we forward AFTER the save is
/// transient w.r.t. snapshot identity but still required so the
/// model starts sampling in the right "thinking vs responding"
/// mode.
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
        embed_lookup(&state.token_embd_bytes, marker, &mut residual);
        state.engine.forward_token(
            &mut state.dgpu_scratch,
            &mut state.igpu_scratch,
            &mut state.state,
            &state.weights,
            &residual,
            pos_after_marker,
            marker,
        )?;
        pos_after_marker += 1;
        if let Some(live) = state.live.as_mut() {
            live.pos = pos_after_marker;
            // TOK_THINK_END is canonical (letta renders it at the
            // start of every historical assistant turn). TOK_THINK_BEGIN
            // is transient — never in letta's replay.
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
    const HEARTBEAT_INTERVAL: u32 = 64;
    let mut hb_last_count: u32 = 0;
    let mut hb_last_at = std::time::Instant::now();
    let finish: FinishReason = loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::info!("generation cancelled by client");
            break FinishReason::Stop;
        }
        if completion_tokens > 0
            && completion_tokens - hb_last_count >= HEARTBEAT_INTERVAL
        {
            let elapsed = hb_last_at.elapsed().as_secs_f32();
            let delta = completion_tokens - hb_last_count;
            let tok_per_s = if elapsed > 0.0 {
                delta as f32 / elapsed
            } else {
                0.0
            };
            tracing::info!(
                completion_tokens,
                pos,
                in_think,
                tok_per_s = format!("{:.1}", tok_per_s),
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
            let raw = gpt2_decode_token(bytes, &state.byte_decoder);
            // Always emit, even for empty raw — TOK_DSML's bytes are
            // routinely the model's primary signal and must be visible
            // to the scanner even though their bytes get suppressed
            // downstream. (For non-DSML empty-decoding tokens this
            // is a no-op for the scanner anyway.)
            if tx
                .blocking_send(WorkerEvent::Chunk {
                    token_id: next,
                    bytes: raw,
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
        // Successfully ingested `next` into KV at `pos-1`. live.pos
        // always tracks the KV cache position. live.tokens only
        // tracks CANONICAL tokens — what letta will replay as
        // history. Transient tokens (TOK_THINK_BEGIN itself; any
        // token sampled while in_think) go into KV but NOT into
        // live.tokens. TOK_THINK_END IS canonical (letta always
        // renders it at the start of each historical assistant
        // turn). See [[think-cache-design]].
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
    // Drain scanner state.
    for de in scanner.finish() {
        if let DsmlEvent::Text(b) = de {
            let s = drain_valid_utf8(&mut pending, &b);
            if !s.is_empty() {
                text.push_str(&s);
            }
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
