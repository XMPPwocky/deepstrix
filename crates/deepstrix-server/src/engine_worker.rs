//! Single OS-thread worker that owns the V4-Flash inference engine and
//! per-session state.
//!
//! Phase 2 model: the worker emits a `WorkerEvent` stream per request.
//! Both streaming and non-streaming HTTP handlers consume the same
//! event stream — the non-streaming handler just accumulates events
//! into a single response before returning.

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
use crate::tokens::{is_think_marker, is_turn_end};

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
    /// One token's decoded bytes (or empty if the token was a structural
    /// marker we suppressed). The caller (handler) feeds these into a
    /// DSML scanner to recover tool-call structure.
    Chunk(String),
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
    },
}

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::UnboundedSender<EngineRequest>,
    pub vocab: Arc<BpeVocab>,
    pub model_name: Arc<String>,
}

impl EngineHandle {
    /// Submit a generation request. Returns a stream of `WorkerEvent`s.
    pub fn submit(
        &self,
        req: GenerateReq,
    ) -> eyre::Result<mpsc::Receiver<WorkerEvent>> {
        let (tx, rx) = mpsc::channel(64);
        self.tx
            .send(EngineRequest::Generate { req, tx })
            .map_err(|_| eyre!("engine worker channel closed"))?;
        Ok(rx)
    }
}

/// Configuration for starting the worker.
pub struct WorkerConfig {
    pub gguf_path: String,
    pub n_kv_max: u32,
    pub model_name: String,
}

pub fn spawn(cfg: WorkerConfig) -> eyre::Result<EngineHandle> {
    let (tx, rx) = mpsc::unbounded_channel::<EngineRequest>();
    let (ready_tx, ready_rx) =
        std::sync::mpsc::sync_channel::<eyre::Result<(Arc<BpeVocab>, Arc<String>)>>(1);

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
    })
}

fn worker_loop(mut state: WorkerState, rx: &mut mpsc::UnboundedReceiver<EngineRequest>) {
    while let Some(msg) = rx.blocking_recv() {
        match msg {
            EngineRequest::Generate { req, tx } => {
                if let Err(e) = handle_generate_stream(&mut state, req, &tx) {
                    let _ = tx.blocking_send(WorkerEvent::Error(format!("{e:#}")));
                }
                // tx is dropped here, signaling end of stream.
            }
        }
    }
    tracing::info!("engine worker channel closed; shutting down");
    let _ = state.engine.shutdown();
}

fn handle_generate_stream(
    state: &mut WorkerState,
    req: GenerateReq,
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

    // Phase 1+2: re-allocate per request. Phase 3 introduces reset_in_place.
    state.state = HetModelState::alloc(state.dgpu, state.igpu, state.n_kv_max)?;

    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(req.tokens.len());
    for &tok in &req.tokens {
        let mut v = vec![0f32; HC_DIM as usize];
        embed_lookup(&state.token_embd_bytes, tok, &mut v);
        input_hcs.push(v);
    }

    let prompt_tokens = req.tokens.len() as u32;
    let _ = state.engine.forward_prefill_pipelined(
        &mut state.bd_a,
        &mut state.bi_a,
        &mut state.bd_b,
        &mut state.bi_b,
        &mut state.dgpu_scratch,
        &mut state.state,
        &state.weights,
        &input_hcs,
        &req.tokens,
        0u32,
        true,
        None,
    )?;
    let mut pos = prompt_tokens;

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
    let finish: FinishReason = loop {
        if is_turn_end(next) {
            break FinishReason::Stop;
        }
        if !is_think_marker(next) {
            if let Some(bytes) = state.vocab.token_text(next) {
                let raw = gpt2_decode_token(bytes, &state.byte_decoder);
                if !raw.is_empty() {
                    let s = String::from_utf8_lossy(&raw).into_owned();
                    if tx.blocking_send(WorkerEvent::Chunk(s)).is_err() {
                        // Receiver dropped (client disconnected, e.g.).
                        return Ok(());
                    }
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
        if pos >= state.n_kv_max {
            break FinishReason::Length;
        }
        next = state
            .engine
            .sample_next(&mut state.dgpu_scratch, sample_mode, rng.next_f32())?;
        completion_tokens += 1;
    };

    let _ = tx.blocking_send(WorkerEvent::Done {
        prompt_tokens,
        completion_tokens,
        finish,
    });
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

/// Drain a `submit` stream into a single accumulated result.
pub async fn accumulate(
    mut rx: mpsc::Receiver<WorkerEvent>,
) -> eyre::Result<GenerateResult> {
    let mut text = String::new();
    let mut last: Option<(u32, u32, FinishReason)> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            WorkerEvent::Chunk(s) => text.push_str(&s),
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
