//! Single OS-thread worker that owns the V4-Flash inference engine and
//! per-session state. HTTP handlers communicate with it through a
//! tokio mpsc channel; the worker uses `blocking_recv` to drain it.
//!
//! Why an OS thread (not `tokio::task::spawn_blocking`): HIP keeps a
//! per-thread current-device context. The engine's `set_current_cached`
//! caches that. A dedicated worker thread keeps the binding stable
//! across the whole process lifetime — there's no thread-pool churn
//! that would invalidate the cache.
//!
//! Phase 1 scope: handle `GenerateBlocking` only. Each request
//! re-allocates `HetModelState` from scratch — no cross-request KV
//! reuse. Phases 3+4 add reset-in-place + on-disk snapshots.

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

/// Per-request output.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: FinishReason,
}

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

/// Messages sent from HTTP handlers to the worker.
pub enum EngineRequest {
    Generate {
        req: GenerateReq,
        resp: oneshot::Sender<eyre::Result<GenerateResult>>,
    },
}

/// Cheaply-cloneable handle held by axum state.
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::UnboundedSender<EngineRequest>,
    pub vocab: Arc<BpeVocab>,
    pub model_name: Arc<String>,
}

impl EngineHandle {
    pub async fn generate(&self, req: GenerateReq) -> eyre::Result<GenerateResult> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(EngineRequest::Generate { req, resp: resp_tx })
            .map_err(|_| eyre!("engine worker channel closed"))?;
        resp_rx
            .await
            .map_err(|_| eyre!("engine worker dropped response sender"))?
    }
}

/// Configuration for starting the worker.
pub struct WorkerConfig {
    pub gguf_path: String,
    pub n_kv_max: u32,
    pub model_name: String,
}

/// Spawn the engine worker thread. Blocks the caller until the model
/// is loaded so the server doesn't start accepting requests before it
/// can answer them.
pub fn spawn(cfg: WorkerConfig) -> eyre::Result<EngineHandle> {
    let (tx, rx) = mpsc::unbounded_channel::<EngineRequest>();
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<eyre::Result<(Arc<BpeVocab>, Arc<String>)>>(1);

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

/// All resources the worker holds across requests. `state` is wiped
/// before each request in Phase 1 (re-allocated). Phases 3+4 reuse.
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
            EngineRequest::Generate { req, resp } => {
                let result = handle_generate(&mut state, req);
                let _ = resp.send(result);
            }
        }
    }
    tracing::info!("engine worker channel closed; shutting down");
    // Drain device queues so HIP teardown doesn't busy-spin.
    let _ = state.engine.shutdown();
}

fn handle_generate(state: &mut WorkerState, req: GenerateReq) -> eyre::Result<GenerateResult> {
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

    // Phase 1: re-allocate HetModelState to wipe per-request. ~50KB of
    // pinned compressor-init writes; cheap relative to the prefill.
    state.state = HetModelState::alloc(state.dgpu, state.igpu, state.n_kv_max)?;

    // Build per-token input HCs for prefill.
    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(req.tokens.len());
    for &tok in &req.tokens {
        let mut v = vec![0f32; HC_DIM as usize];
        embed_lookup(&state.token_embd_bytes, tok, &mut v);
        input_hcs.push(v);
    }

    let prompt_tokens = req.tokens.len() as u32;
    let _last_logits = state.engine.forward_prefill_pipelined(
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
        true, // last_only
        None, // stats
    )?;
    let mut pos = prompt_tokens;

    // Sample first output token from the logits the prefill left on dGPU.
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

    let mut decoded_bytes: Vec<u8> = Vec::new();
    let mut residual = vec![0f32; HC_DIM as usize];
    let max_new = req.max_new as u32;
    let finish: FinishReason = loop {
        // Stop: any role-boundary token ends the turn (incl. hallucinated USER/ASSISTANT).
        if is_turn_end(next) {
            break FinishReason::Stop;
        }

        // Render this token (skip structural <think>/</think>).
        if !is_think_marker(next) {
            if let Some(bytes) = state.vocab.token_text(next) {
                let raw = gpt2_decode_token(bytes, &state.byte_decoder);
                decoded_bytes.extend(raw);
            }
        }

        if completion_tokens >= max_new {
            break FinishReason::Length;
        }

        // Forward this token into KV cache, then sample the next one.
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

    let text = String::from_utf8_lossy(&decoded_bytes).into_owned();
    Ok(GenerateResult {
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason: finish,
    })
}

// Layer-ratio guard: ensure COMPRESS_RATIOS is exposed at compile time.
const _: () = {
    let _ = COMPRESS_RATIOS;
};
