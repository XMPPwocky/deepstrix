//! The hub's EMBED PHASE (docs/v41/EMBED_PHASE_DESIGN.md): Qwen3-Embedding
//! served from the hub process as a third scheduler phase next to prefill and
//! decode. A phase runs between scheduler ticks with no other hub work on any
//! device. It borrows ~0.6 GB of the dGPU IN PLACE from immutable V4.1 weights
//! (the loan), streams the embedding model's layers from its GGUF, and puts
//! the borrowed bytes back from the loan image before the next tick. The
//! embedding model's weights, kernel modules and pinned buffers exist only
//! during a phase: every resident byte would cost expert hit rate. What does
//! stay resident (the tokenizer, the tensor directory) is listed in the design
//! doc §4.5.
//!
//! (Not `embed.rs`: that module is V4.1's token-embedding lookup.)

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use color_eyre::eyre::{self, eyre, WrapErr};
use tokio::sync::oneshot;
use v4flash_core::qwen3_embed::Qwen3EmbedModel;
use v4flash_core::tokenizer::BpeVocab;
use v4flash_core::direct_io::DirectFiles;
use v4flash_core::MappedGguf;
use v4flash_hip::{Device, PinnedBuffer};
use v4flash_kernels::dgpu_loan::{Donor, Loan};
use v4flash_kernels::het::weights::HetModelWeights;
use v4flash_kernels::het::HeterogeneousEngine;
use v4flash_kernels::qwen3_embed::{self as fwd, EmbedBuffers, EmbedSizing, Qwen3EmbedKernels};

use crate::engine_worker::{SubmitError, WorkerProgress};
use crate::knobs;

/// Guard band imaged past each donor's lent bytes (design §4.3).
const GUARD_BYTES: usize = 1 << 20;

/// What a finished request gets: one embedding per input, in order.
pub type EmbedReply = Result<Vec<Vec<f32>>, String>;

/// One `/v1/embeddings` request, queued or part-way through.
pub struct EmbedRequest {
    /// Token ids per input, `<|endoftext|>` appended.
    inputs: Vec<Vec<u32>>,
    tokens: usize,
    dims: Option<usize>,
    results: Vec<Option<Vec<f32>>>,
    /// Inputs `< next` are computed or in the current phase.
    next: usize,
    reply: Option<oneshot::Sender<EmbedReply>>,
    queued: Instant,
}

impl EmbedRequest {
    pub fn new(inputs: Vec<Vec<u32>>, dims: Option<usize>, reply: oneshot::Sender<EmbedReply>) -> Self {
        let n = inputs.len();
        let tokens = inputs.iter().map(Vec::len).sum();
        EmbedRequest { inputs, tokens, dims, results: vec![None; n], next: 0, reply: Some(reply), queued: Instant::now() }
    }

    fn gone(&self) -> bool {
        self.reply.as_ref().is_none_or(|r| r.is_closed())
    }

    fn fail(&mut self, msg: &str) {
        if let Some(tx) = self.reply.take() {
            let _ = tx.send(Err(msg.to_string()));
        }
    }
}

/// Requests waiting for the worker. Shared by the HTTP handlers and the
/// engine thread; bounded by queued tokens (`V41_EMBED_QUEUE_TOKENS`).
pub struct EmbedQueue {
    reqs: Mutex<VecDeque<EmbedRequest>>,
    tokens: AtomicUsize,
    cap_tokens: usize,
    /// A wake (`EngineRequest::EmbedWake`) is in the engine channel or about
    /// to be: at most one at a time, so embedding traffic can never fill the
    /// channel chat submits into.
    wake_pending: AtomicBool,
}

impl EmbedQueue {
    pub fn new(cap_tokens: usize) -> Self {
        EmbedQueue { reqs: Mutex::new(VecDeque::new()), tokens: AtomicUsize::new(0), cap_tokens, wake_pending: AtomicBool::new(false) }
    }

    /// Queue a request, or `Busy` when it would pass the cap. The cap counts
    /// every UNFINISHED request's tokens (queued, or taken by the worker and
    /// not yet replied to: `release`). With nothing unfinished one request is
    /// always taken, so a request larger than the cap is not refused forever.
    pub fn push(&self, r: EmbedRequest) -> Result<(), SubmitError> {
        let mut q = self.reqs.lock().unwrap_or_else(|e| e.into_inner());
        let held = self.tokens.load(Ordering::Relaxed);
        if held > 0 && held + r.tokens > self.cap_tokens {
            return Err(SubmitError::Busy);
        }
        self.tokens.fetch_add(r.tokens, Ordering::Relaxed);
        q.push_back(r);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.reqs.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// The worker takes a request (it still counts against the cap).
    fn pop(&self) -> Option<EmbedRequest> {
        self.reqs.lock().unwrap_or_else(|e| e.into_inner()).pop_front()
    }

    /// A taken request is finished (replied, failed or dropped).
    fn release(&self, tokens: usize) {
        self.tokens.fetch_sub(tokens, Ordering::Relaxed);
    }

    /// Tokens of unfinished requests (tests, telemetry).
    pub fn held_tokens(&self) -> usize {
        self.tokens.load(Ordering::Relaxed)
    }

    /// True when the caller must send a wake (none is pending).
    pub fn claim_wake(&self) -> bool {
        !self.wake_pending.swap(true, Ordering::AcqRel)
    }

    /// The engine got the wake (or is about to block idle, or the send
    /// failed): the next push sends a new one.
    pub fn clear_wake(&self) {
        self.wake_pending.store(false, Ordering::Release);
    }
}

/// What the HTTP layer needs (`EngineHandle::embed`).
pub struct EmbedInfo {
    pub model_name: String,
    pub vocab: Arc<BpeVocab>,
    pub eos_id: u32,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub max_input_tokens: usize,
    pub max_request_tokens: usize,
    pub queue: Arc<EmbedQueue>,
}

/// One phase's numbers (`ms.embed`, evtrace `hub_embed`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseStats {
    pub requests: usize,
    pub inputs: usize,
    pub tokens: usize,
    pub wait_ms: f64,
    pub rows_ms: f64,
    pub read_ms: f64,
    pub wait_read_ms: f64,
    pub fwd_ms: f64,
    pub return_ms: f64,
    pub verify_ms: f64,
    pub pinned_ms: f64,
    pub total_ms: f64,
    pub nonfinite: usize,
    pub guard_violations: u32,
    pub ok: bool,
}

/// The engine thread's embed state (`WorkerState::embed`).
pub struct EmbedCtx {
    info: Arc<EmbedInfo>,
    /// The GGUF for the token-embedding rows (buffered, page cache dropped
    /// after every phase).
    file: MappedGguf,
    /// The GGUF and its replicas on other drives, O_DIRECT: the layer stream.
    direct: DirectFiles,
    /// O_DIRECT reader threads (`V41_EMBED_READERS`).
    readers: usize,
    /// Bytes of each pinned host buffer: a layer's aligned span or a loan
    /// image chunk, whichever is larger.
    host_bytes: usize,
    model: Qwen3EmbedModel,
    arch: String,
    loan: Loan,
    sizing: EmbedSizing,
    queue: Arc<EmbedQueue>,
    /// Requests taken off the queue and not yet replied to (a large request
    /// spans several phases).
    active: VecDeque<EmbedRequest>,
    /// Round-robin start for the next phase's batch.
    rr: usize,
    last_end: Option<Instant>,
    last_dur: Duration,
}

/// The default loan image path.
/// A `:`-separated path list knob's paths (empty entries dropped).
fn split_paths(v: Option<&str>) -> Vec<PathBuf> {
    v.map(|s| s.split(':').filter(|p| !p.is_empty()).map(PathBuf::from).collect()).unwrap_or_default()
}

fn default_image() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".cache/deepstrix/embed-loan.img")
}

/// Donor candidates, best first: the V4.1 LM head (one big contiguous buffer,
/// read only by the head and the drafter exit, both on the engine thread
/// inside ticks), then the per-layer attention projections and shared-expert
/// weights, largest first. All are immutable after load.
fn donor_candidates(w: &HetModelWeights) -> Vec<Donor> {
    let view = |name: String, b: &v4flash_hip::DeviceBuffer<u8>| Donor { name, view: b.slice_view(0, b.len()) };
    let mut out = vec![view("output".into(), &w.global.output.buffer)];
    let mut rest: Vec<Donor> = Vec::new();
    for l in &w.dgpu_layers {
        let i = l.layer_idx;
        rest.push(view(format!("blk.{i}.attn_q_b"), &l.attn_q_b.buffer));
        rest.push(view(format!("blk.{i}.attn_output_a"), &l.attn_output_a.buffer));
        rest.push(view(format!("blk.{i}.attn_output_b"), &l.attn_output_b.buffer));
        rest.push(view(format!("blk.{i}.shared.gate"), &l.shared.gate.buffer));
        rest.push(view(format!("blk.{i}.shared.up"), &l.shared.up.buffer));
        rest.push(view(format!("blk.{i}.shared.down"), &l.shared.down.buffer));
    }
    rest.sort_by_key(|d| std::cmp::Reverse(d.view.byte_len()));
    out.extend(rest);
    out
}

/// Aborts the process if dropped while the loan is out (a panic mid-phase):
/// the V4.1 weights the donors hold would be garbage.
struct LoanOut;

impl Drop for LoanOut {
    fn drop(&mut self) {
        tracing::error!("embed phase: unwound with the dGPU loan out; V4.1 weights are corrupt -- aborting for a supervisor restart");
        std::thread::sleep(Duration::from_millis(50));
        std::process::abort();
    }
}

impl EmbedCtx {
    /// Startup (`--embed-gguf`): validate the GGUF, size and place the loan,
    /// write the loan image (through `engine`'s dGPU transfer stream).
    /// Allocates nothing on any device. Call after the V4.1 weights are
    /// loaded and before the scheduler starts.
    pub fn load(path: &std::path::Path, model_name: String, engine: &HeterogeneousEngine, weights: &HetModelWeights) -> eyre::Result<(EmbedCtx, Arc<EmbedInfo>)> {
        let t0 = Instant::now();
        let dgpu = engine.dgpu.device;
        let mut file = MappedGguf::open(path).wrap_err_with(|| format!("open --embed-gguf {}", path.display()))?;
        let model = Qwen3EmbedModel::from_gguf(&file).wrap_err_with(|| format!("--embed-gguf {}", path.display()))?;
        let vocab = BpeVocab::from_gguf(file.gguf())?;
        if vocab.pre.as_deref() != Some("qwen2") {
            tracing::warn!(pre = ?vocab.pre, "embed GGUF: tokenizer pre-type is not qwen2; encoding with the qwen2 splitter anyway");
        }
        // Every id the tokenizer can emit has a token_embd row.
        if vocab.vocab_size() > model.cfg.n_vocab {
            return Err(eyre!("embed GGUF: the tokenizer has {} tokens, token_embd {} rows", vocab.vocab_size(), model.cfg.n_vocab));
        }
        // The tokenizer arrays now live in `vocab`; free the parsed copy. No
        // readahead around the phase's reads (they drop what they read).
        file.drop_metadata();
        file.advise_random();
        let sizing = EmbedSizing { phase_tokens: knobs::EMBED_PHASE_TOKENS.usize(), sub_rows: knobs::EMBED_SUB_ROWS.usize() };
        let max_input_tokens = knobs::EMBED_MAX_INPUT_TOKENS.usize().min(model.cfg.n_ctx_train);
        if max_input_tokens > sizing.phase_tokens {
            return Err(eyre!(
                "V41_EMBED_MAX_INPUT_TOKENS ({max_input_tokens}) exceeds V41_EMBED_PHASE_TOKENS ({}): such an input could never run",
                sizing.phase_tokens
            ));
        }
        dgpu.set_current()?;
        let arch = dgpu.properties()?.gcn_arch_name;
        // Fail at startup, not on the first request: the kernels load here
        // once and are dropped (they load again per phase).
        drop(Qwen3EmbedKernels::for_arch(&arch)?);
        // The layer stream: the GGUF plus its replicas on other drives,
        // O_DIRECT, pieces spread over all of them (design §5.3).
        let mut gguf_paths = vec![path.to_path_buf()];
        gguf_paths.extend(split_paths(knobs::EMBED_GGUF_REPLICAS.str()));
        let direct = DirectFiles::open(&gguf_paths).wrap_err("the embed GGUF's O_DIRECT readers (V41_EMBED_GGUF_REPLICAS)")?;
        if direct.size() != file.gguf().file_size {
            return Err(eyre!("V41_EMBED_GGUF_REPLICAS: a replica is not the same file as --embed-gguf"));
        }
        let readers = knobs::EMBED_READERS.usize();
        let sizes: Vec<usize> = sizing.buffer_sizes(&model.cfg, &model.layout).iter().map(|(_, b)| *b).collect();
        let mut images = split_paths(knobs::EMBED_LOAN_IMAGE.str());
        if images.is_empty() {
            images.push(default_image());
        }
        // Image chunks are O_DIRECT units: a multiple of the alignment.
        let chunk = v4flash_core::direct_io::align_up(model.layout.bytes as u64) as usize;
        let host_bytes = fwd::host_bytes(&model)?.max(chunk);
        let mut loan = Loan::new(donor_candidates(weights), &sizes, images, chunk, GUARD_BYTES)?;
        let ti = Instant::now();
        loan.write_image(&engine.dgpu.xfer)?;
        tracing::info!(
            gguf = %path.display(),
            layers = model.cfg.n_layer,
            n_embd = model.cfg.n_embd,
            layer_mib = model.layout.bytes as f64 / (1 << 20) as f64,
            phase_tokens = sizing.phase_tokens,
            sub_rows = sizing.sub_rows,
            lent_mib = loan.lent_bytes() as f64 / (1 << 20) as f64,
            imaged_mib = loan.total_bytes() as f64 / (1 << 20) as f64,
            donors = ?loan.donor_summary(),
            gguf_readers = ?direct.describe(),
            image_readers = ?loan.image_readers(),
            readers,
            image_ms = ti.elapsed().as_millis() as u64,
            load_ms = t0.elapsed().as_millis() as u64,
            "embed phase ready (weights stream per phase; nothing on the devices between phases)"
        );
        // Opening and validating read the GGUF's header (the tokenizer arrays,
        // ~5 MB) through the page cache before FADV_RANDOM was set: drop it,
        // so nothing of the file is cached from startup on (gate E5).
        if let Err(e) = file.drop_page_cache() {
            tracing::warn!(error = %e, "embed GGUF: dropping its page cache failed");
        }
        let queue = Arc::new(EmbedQueue::new(knobs::EMBED_QUEUE_TOKENS.usize()));
        let info = Arc::new(EmbedInfo {
            model_name,
            eos_id: model.eos_id,
            n_embd: model.cfg.n_embd,
            n_vocab: model.cfg.n_vocab,
            max_input_tokens,
            max_request_tokens: knobs::EMBED_MAX_REQUEST_TOKENS.usize(),
            vocab: Arc::new(vocab),
            queue: queue.clone(),
        });
        Ok((
            EmbedCtx {
                info: info.clone(), file, direct, readers, host_bytes, model, arch, loan, sizing, queue,
                active: VecDeque::new(), rr: 0, last_end: None, last_dur: Duration::ZERO,
            },
            info,
        ))
    }

    /// The HTTP layer's view (`EngineHandle::embed`).
    pub fn info(&self) -> Arc<EmbedInfo> {
        self.info.clone()
    }

    /// Work is waiting (queued, or a request part-way through).
    pub fn has_work(&self) -> bool {
        !self.active.is_empty() || !self.queue.is_empty()
    }

    pub fn queue(&self) -> &EmbedQueue {
        &self.queue
    }

    /// May a phase start now? With LLM work pending, phases may take at most
    /// `V41_EMBED_MAX_SHARE` % of wall time: the next starts no earlier than
    /// `last_dur * (100 / share - 1)` after the last ended.
    pub fn due(&self, llm_busy: bool) -> bool {
        if !self.has_work() {
            return false;
        }
        if !llm_busy {
            return true;
        }
        let share = knobs::EMBED_MAX_SHARE.get().clamp(1, 100) as f64;
        let gap = self.last_dur.mul_f64(100.0 / share - 1.0);
        self.last_end.is_none_or(|e| e.elapsed() >= gap)
    }

    /// Reply an error to every queued and active request (shutdown).
    pub fn fail_all(&mut self, msg: &str) {
        while let Some(mut r) = self.queue.pop() {
            r.fail(msg);
            self.queue.release(r.tokens);
        }
        for r in self.active.iter_mut() {
            r.fail(msg);
            self.queue.release(r.tokens);
        }
        self.active.clear();
    }

    /// Reply to finished requests and drop failed or abandoned ones; each
    /// leaving request releases its tokens from the queue's cap.
    fn settle_requests(&mut self) {
        let queue = &self.queue;
        self.active.retain_mut(|r| {
            if r.reply.is_some() && r.results.iter().all(Option::is_some) {
                let out: Vec<Vec<f32>> = r.results.iter_mut().map(|x| x.take().expect("checked")).collect();
                let _ = r.reply.take().expect("checked").send(Ok(out));
            }
            if r.gone() {
                queue.release(r.tokens);
                return false;
            }
            true
        });
    }

    /// The next phase's inputs: (index in `active`, input index), round-robin
    /// across requests (one input from each in turn, starting after the
    /// last phase's first request) within the token budget. Every input fits
    /// alone (the handler caps inputs at `max_input_tokens <= phase_tokens`).
    fn take_batch(&mut self) -> Vec<(usize, usize)> {
        while let Some(r) = self.queue.pop() {
            self.active.push_back(r);
        }
        self.settle_requests();
        let n = self.active.len();
        let mut batch = Vec::new();
        if n == 0 {
            return batch;
        }
        let start = self.rr % n;
        self.rr = self.rr.wrapping_add(1);
        let mut tokens = 0usize;
        loop {
            let mut took = false;
            for k in 0..n {
                let ri = (start + k) % n;
                let r = &mut self.active[ri];
                if r.next < r.inputs.len() {
                    let len = r.inputs[r.next].len();
                    if tokens + len > self.sizing.phase_tokens {
                        continue;
                    }
                    tokens += len;
                    batch.push((ri, r.next));
                    r.next += 1;
                    took = true;
                }
            }
            if !took {
                break;
            }
        }
        batch
    }

    /// One embed phase. On return the loan is back (a failed return aborts
    /// the process: a hub whose V4.1 weights differ from what it loaded must
    /// not serve), each request in the phase has its inputs' results or its
    /// error, finished requests are replied to. Never fails: embedding errors
    /// go to the requests, never to the LLM streams. `live` / `prefills` are
    /// for the log only.
    pub fn run_phase(&mut self, engine: &HeterogeneousEngine, progress: &WorkerProgress, live: usize, prefills: usize) -> PhaseStats {
        let t0 = Instant::now();
        let mut st = PhaseStats::default();
        let batch = self.take_batch();
        if batch.is_empty() {
            return st;
        }
        st.inputs = batch.len();
        st.tokens = batch.iter().map(|(ri, ii)| self.active[*ri].inputs[*ii].len()).sum();
        st.wait_ms = self.active.iter().map(|r| r.queued.elapsed().as_secs_f64() * 1e3).fold(0.0, f64::max);
        // The pinned buffers BEFORE the loan is out: failing here (host memory
        // pressure) touches nothing on the device, so it fails the batch's
        // requests, not the hub.
        let tp = Instant::now();
        let mut host = match PinnedBuffer::<u8>::new(self.host_bytes)
            .and_then(|a| Ok([a, PinnedBuffer::<u8>::new(self.host_bytes)?]))
        {
            Ok(h) => h,
            Err(e) => {
                let msg = format!("embedding failed: pinned host buffers: {e:#}");
                tracing::error!(error = %msg, inputs = st.inputs, "embed phase: not started");
                for (ri, _) in &batch {
                    self.active[*ri].fail(&msg);
                }
                self.settle_requests();
                return st;
            }
        };
        st.pinned_ms = tp.elapsed().as_secs_f64() * 1e3;
        let out = LoanOut;
        let r = self.forward(engine, progress, &batch, &mut host, &mut st);
        // The loan goes back whatever the forward did.
        let tr = Instant::now();
        match self.loan.give_back(&engine.dgpu.xfer, &mut host, knobs::EMBED_VERIFY.on(), self.readers) {
            Ok(rs) => {
                std::mem::forget(out);
                st.return_ms = tr.elapsed().as_secs_f64() * 1e3;
                st.verify_ms = rs.verify_ms;
                st.guard_violations = rs.guard_violations;
                if rs.retried > 0 {
                    tracing::warn!(retried = rs.retried, "embed phase: loan chunks needed a second return");
                }
                if rs.guard_violations > 0 {
                    // The canaries are repaired, but a kernel that wrote
                    // outside its buffer may also have written outside the
                    // loan, into V4.1 state no image covers.
                    tracing::error!(canaries = rs.guard_violations, "embed phase: a kernel wrote outside its loan buffer");
                    drop(LoanOut); // aborts
                }
            }
            Err(e) => {
                tracing::error!(error = %format!("{e:#}"), "embed phase: the dGPU loan could not be returned");
                drop(out); // aborts
                unreachable!();
            }
        }
        drop(host);
        // The engine's cached current device is stale after the phase (both
        // the serial and the multistream loop).
        engine.invalidate_device_cache();
        if let Err(e) = engine.dgpu.device.set_current() {
            tracing::warn!(error = %e, "embed phase: restoring the dGPU as current failed");
        }
        match r {
            Ok(rows) => {
                st.ok = true;
                for ((ri, ii), row) in batch.iter().zip(rows) {
                    let req = &mut self.active[*ri];
                    let e = self.model.finish(&row, req.dims);
                    if e.iter().all(|x| x.is_finite()) {
                        req.results[*ii] = Some(e);
                    } else {
                        st.nonfinite += 1;
                        req.fail(&format!("embedding of input {ii} is not finite"));
                    }
                }
            }
            Err(e) => {
                let msg = format!("embedding failed: {e:#}");
                tracing::error!(error = %msg, inputs = st.inputs, "embed phase: forward failed; its requests get the error");
                for (ri, _) in &batch {
                    self.active[*ri].fail(&msg);
                }
            }
        }
        let mut reqs: Vec<usize> = batch.iter().map(|(ri, _)| *ri).collect();
        reqs.sort_unstable();
        reqs.dedup();
        st.requests = reqs.len();
        self.settle_requests();
        // Constraint 2: no page of the GGUF stays cached. Per-range DONTNEED
        // keeps partial edge pages (every 2.7 KB token row is one).
        if let Err(e) = self.file.drop_page_cache() {
            tracing::warn!(error = %e, "embed phase: dropping the GGUF's page cache failed");
        }
        st.total_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.last_end = Some(Instant::now());
        self.last_dur = t0.elapsed();
        tracing::info!(
            requests = st.requests, inputs = st.inputs, tokens = st.tokens, ok = st.ok, live, prefills,
            wait_ms = st.wait_ms as u64, rows_ms = st.rows_ms as u64, read_ms = st.read_ms as u64,
            wait_read_ms = st.wait_read_ms as u64, fwd_ms = st.fwd_ms as u64, return_ms = st.return_ms as u64,
            verify_ms = st.verify_ms as u64, pinned_ms = st.pinned_ms as u64, total_ms = st.total_ms as u64,
            nonfinite = st.nonfinite, guard_violations = st.guard_violations,
            lent_mib = self.loan.lent_bytes() >> 20, "ms.embed"
        );
        use v4flash_kernels::het::{evtrace, evtrace_kinds};
        evtrace::emit(&evtrace_kinds::HUB_EMBED, &[
            evtrace::now(), st.requests as f64, st.inputs as f64, st.tokens as f64, self.loan.lent_bytes() as f64,
            st.wait_ms, st.rows_ms, st.read_ms, st.wait_read_ms, st.fwd_ms, st.return_ms, st.verify_ms,
            st.pinned_ms, st.total_ms, live as f64, prefills as f64, f64::from(u8::from(st.ok)),
            st.nonfinite as f64, f64::from(st.guard_violations),
        ]);
        crate::engine_worker::trim_heap_and_log("host heap after embed phase");
        st
    }

    /// The forward over `batch`: quiesce the dGPU, load the kernels, carve the
    /// loan, run. Returns each batch entry's last hidden row. `h` = the
    /// phase's pinned buffers (the return reuses them).
    fn forward(
        &mut self,
        engine: &HeterogeneousEngine,
        progress: &WorkerProgress,
        batch: &[(usize, usize)],
        h: &mut [PinnedBuffer<u8>; 2],
        st: &mut PhaseStats,
    ) -> eyre::Result<Vec<Vec<f32>>> {
        let dgpu: Device = engine.dgpu.device;
        dgpu.set_current()?;
        // L1: nothing queued on the dGPU may still read a donor.
        dgpu.synchronize()?;
        let k = Qwen3EmbedKernels::for_arch(&self.arch)?;
        let mut alloc = self.loan.allocator()?;
        let mut bufs = EmbedBuffers::carve(&mut alloc, self.sizing, &self.model.cfg, &self.model.layout)?;
        let inputs: Vec<&[u32]> = batch.iter().map(|(ri, ii)| self.active[*ri].inputs[*ii].as_slice()).collect();
        let tf = Instant::now();
        let (rows, tm) = fwd::run(
            &k, &engine.dgpu.q8_wmma, &self.model, &self.file, &self.direct, self.readers, &mut bufs, h,
            &engine.dgpu.compute, &engine.dgpu.xfer, &inputs,
            &mut |l| {
                progress.pet();
                // Gate E6's fault injection: fail the forward at this layer.
                if knobs::EMBED_FAULT_LAYER.get() == l as u64 {
                    return Err(eyre!("injected fault at layer {l} (V41_EMBED_FAULT_LAYER)"));
                }
                Ok(())
            },
        )?;
        st.fwd_ms = tf.elapsed().as_secs_f64() * 1e3;
        st.rows_ms = tm.rows_ms;
        st.read_ms = tm.read_ms;
        st.wait_read_ms = tm.wait_read_ms;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(lens: &[usize]) -> (EmbedRequest, oneshot::Receiver<EmbedReply>) {
        let (tx, rx) = oneshot::channel();
        (EmbedRequest::new(lens.iter().map(|&n| vec![1u32; n]).collect(), None, tx), rx)
    }

    #[test]
    fn queue_caps_tokens_but_takes_one_into_an_empty_queue() {
        let q = EmbedQueue::new(10);
        let (big, _r1) = req(&[50]);
        assert!(q.push(big).is_ok(), "an empty queue takes anything");
        let (small, _r2) = req(&[2]);
        assert!(matches!(q.push(small), Err(SubmitError::Busy)));
        // Taken by the worker, the request still counts until it finishes.
        let taken = q.pop().expect("queued");
        let (small, _r3) = req(&[2]);
        assert!(matches!(q.push(small), Err(SubmitError::Busy)));
        q.release(taken.tokens);
        assert_eq!(q.held_tokens(), 0);
        let (small, _r4) = req(&[2]);
        assert!(q.push(small).is_ok());
    }

    #[test]
    fn wake_is_edge_triggered() {
        let q = EmbedQueue::new(10);
        assert!(q.claim_wake());
        assert!(!q.claim_wake());
        q.clear_wake();
        assert!(q.claim_wake());
    }
}
