//! Multi-stream decode for the V4.1 server (docs/v41/MULTISTREAM_DECODE_PLAN.md,
//! M1b). `V41_MULTISTREAM=1` replaces the serial `worker_loop` with a scheduler
//! that keeps up to `V41_MS_SLOTS` live streams in one `KvArena` and runs ONE
//! batched decode step per tick (`forward_step_arena`), interleaved with chunks
//! of at most one prompt prefill in flight (`PrefillJob`, plan 5.3: chunks and
//! steps are separate forwards).
//!
//! v1 scope (deliberately): text-only requests on the batched path (image and
//! DSpark requests take the legacy serial handler while the arena is empty);
//! FIFO admission; the whole-prompt snapshot is probed on entry (restore into
//! the scratch single-sequence state, prefill the suffix, admit into the arena)
//! and saved right after the prefill as the legacy path does; no RESIDENT
//! streams / extend path yet; per-row sampling on the host with the kernel's
//! composed top_p/min_p rule; per-step watchdog pet; non-blocking emits.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use tokio::sync::mpsc;
use v4flash_kernels::config::{ENGRAM_IN, HC_DIM, N_VOCAB};
use v4flash_kernels::het::forward_prefill::PrefillJob;
use v4flash_kernels::het::kv_arena::{KvArena, RowTablesDev};
use v4flash_kernels::het::SampleMode;
use v4flash_kernels::sampler::SamplerRng;

use crate::embed::{embed_lookup, gpt2_decode_token};
use crate::engine_worker::{
    flush_expert_stats, handle_generate_stream, save_live_if_dirty, EngineRequest, FinishReason,
    GenerateReq, WorkerEvent, WorkerState,
};
use crate::snapshot;
use crate::tokens::{is_turn_end, TOK_ASSISTANT, TOK_EOS, TOK_THINK_BEGIN, TOK_THINK_END, TOK_USER};

pub fn enabled() -> bool {
    matches!(std::env::var("V41_MULTISTREAM").as_deref(), Ok("1") | Ok("on"))
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// One live decode stream (a request that has been prefilled and admitted).
struct Stream {
    slot: u32,
    tx: mpsc::Sender<WorkerEvent>,
    cancel: Arc<AtomicBool>,
    /// The token this stream feeds into the NEXT step.
    next: i32,
    /// Every token forwarded so far (prompt, marker, generated): the Engram
    /// hasher reads the last four, the snapshot the whole thing.
    seq: Vec<i32>,
    /// `hasher.compress` of `seq`, in step.
    compressed: Vec<i32>,
    prompt_tokens: u32,
    completion_tokens: u32,
    max_new: usize,
    sample_mode: SampleMode,
    rng: SamplerRng,
    in_think: bool,
    send_failures: u32,
    started: Instant,
    session_id: Option<String>,
}

/// A request waiting for its prefill.
struct Pending {
    req: GenerateReq,
    tx: mpsc::Sender<WorkerEvent>,
    session_id: Option<String>,
    cancel: Arc<AtomicBool>,
    trailing_marker: Option<i32>,
    prompt_tokens: u32,
    queued: Instant,
}

/// A prefill in flight. Each job owns a scratch single-sequence state (from
/// `Sched::spare_states`), so `V41_MS_PREFILL_JOBS` prompts can be prefilled
/// round-robin, chunk by chunk — a short prompt is not stuck behind a 100K one.
struct Prefill {
    p: Pending,
    job: PrefillJob,
    kv: v4flash_kernels::het::HetModelState,
    /// Tokens already in the scratch state (restored prefix), then the suffix.
    prefix: Vec<i32>,
    compressed: Vec<i32>,
    started: Instant,
}

/// `Prefill` after its scratch state has been recycled.
struct PrefillDone {
    p: Pending,
    job: PrefillJob,
    prefix: Vec<i32>,
    compressed: Vec<i32>,
    started: Instant,
}

const CHUNK_SEND_FAILURES_MAX: u32 = 30;

pub fn worker_loop_ms(mut state: WorkerState, rx: &mut mpsc::Receiver<EngineRequest>) {
    let n_slots = env_usize("V41_MS_SLOTS", 8) as u32;
    // Context budget across all live streams (positions); each store gets
    // budget / ratio rows. Default 2 x --ctx: two 240K agents, or eight 75K
    // ones. ~1 KB per row at ratio 1 plus 0.5 KB per ratio-2 store; MEASURED
    // 2026-09-20 on the 16 GB dGPU with two prefill states: 3 x --ctx left
    // 190 MiB free (unsafe), 2 x --ctx 1.0 GiB.
    let ctx_rows = env_usize("V41_MS_CTX_ROWS", 2 * state.n_kv_max as usize) as u32;
    let chunk_rows = env_usize("V41_MS_CHUNK_ROWS", 1024);
    let arena = match KvArena::alloc_ctx(state.dgpu, n_slots, ctx_rows) {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(error = %e, "multistream: arena alloc failed; falling back to the serial loop is not possible here — aborting");
            return;
        }
    };
    let dev = match RowTablesDev::alloc(state.dgpu, n_slots, arena.stores.len()) {
        Ok(d) => d,
        Err(e) => {
            tracing::error!(error = %e, "multistream: row tables alloc failed");
            return;
        }
    };
    tracing::info!(n_slots, ctx_rows, chunk_rows, prefill_burst_ms = env_usize("V41_MS_PREFILL_BURST_MS", 120_000), decode_burst_ms = env_usize("V41_MS_DECODE_BURST_MS", 30_000), "multistream scheduler ON");
    let n_jobs = env_usize("V41_MS_PREFILL_JOBS", 2).max(1);
    let mut spare_states = Vec::with_capacity(n_jobs);
    for _ in 0..n_jobs {
        match v4flash_kernels::het::HetModelState::alloc(state.dgpu, state.igpu, state.n_kv_max) {
            Ok(st) => spare_states.push(st),
            Err(e) => { tracing::error!(error = %e, "multistream: prefill scratch state alloc failed"); return; }
        }
    }
    tracing::info!(n_jobs, "multistream: prefill scratch states allocated");
    let (bounce_f16, bounce_u8) = match (|| -> eyre::Result<_> {
        state.dgpu.set_current()?;
        Ok((v4flash_hip::DeviceBuffer::<u16>::new(state.dgpu.id, 4096 * 512)?, v4flash_hip::DeviceBuffer::<u8>::new(state.dgpu.id, 4096 * 80)?))
    })() {
        Ok(b) => b,
        Err(e) => { tracing::error!(error = %e, "multistream: bounce alloc failed"); return; }
    };
    let mut sched = Sched { profile_acc: ProfileAcc::default(), parked: Vec::new(), bounce_f16, bounce_u8, phase: Phase::Decode, phase_since: Instant::now(), arena, dev, streams: Vec::new(), queue: VecDeque::new(), prefills: Vec::new(), spare_states, rr: 0, tick: 0 };

    loop {
        // 1. Intake: never block while there is work; block when idle.
        let idle = sched.streams.is_empty() && sched.prefills.is_empty() && sched.queue.is_empty();
        let msg = if idle { rx.blocking_recv() } else { match rx.try_recv() { Ok(m) => Some(m), Err(_) => None } };
        match msg {
            Some(EngineRequest::Generate { req, tx, session_id, cancel }) => {
                sched.enqueue(req, tx, session_id, cancel);
            }
            Some(EngineRequest::Shutdown { ack }) => {
                sched.abort_all("server shutting down");
                save_live_if_dirty(&mut state);
                let _ = ack.send(());
                break;
            }
            None if idle => break, // channel closed
            None => {}
        }
        if sched.streams.is_empty() && sched.prefills.is_empty() && sched.queue.is_empty() {
            state.progress.end();
            continue;
        }
        state.progress.begin();
        if let Err(e) = sched.tick(&mut state) {
            tracing::error!(error = %e, "multistream: step failed; aborting every live stream");
            sched.abort_all(&format!("{e:#}"));
            let _ = state.engine.remote_drain_in_flight();
            let _ = state.engine.remote_reconnect_if_dead();
        }
        state.progress.pet();
    }
    let _ = state.engine.shutdown();
}

/// Scheduler phase with hysteresis (plan 5.3 + locality): prefill chunks and
/// decode steps run in BURSTS, not alternately — one 256-row chunk drags ~100
/// experts per layer through the pool and evicts the decode working set, so
/// alternating chunk/step made every step pay the misses back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase { Decode, Prefill }

#[derive(Default)]
struct ProfileAcc {
    pf_last: (u64, u64, u64, u64),
    last_misses: u64,
    last_read_ns: u64,
    steps: u64,
    rows: u64,
    wall_ms: f64,
    stages: std::collections::HashMap<(&'static str, &'static str), (f64, u64)>,
}

fn ms_profile() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| matches!(std::env::var("V41_MS_PROFILE").as_deref(), Ok("1")));
    *ON
}

struct Sched {
    profile_acc: ProfileAcc,
    /// Prefilled requests waiting for arena room (their scratch state stays
    /// parked with them; admission is retried every tick).
    parked: Vec<(Prefill, Vec<f32>)>,
    bounce_f16: v4flash_hip::DeviceBuffer<u16>,
    bounce_u8: v4flash_hip::DeviceBuffer<u8>,
    phase: Phase,
    phase_since: Instant,
    arena: KvArena,
    dev: RowTablesDev,
    streams: Vec<Stream>,
    queue: VecDeque<Pending>,
    prefills: Vec<Prefill>,
    spare_states: Vec<v4flash_kernels::het::HetModelState>,
    /// Round-robin cursor over `prefills` for chunk ticks.
    rr: usize,
    tick: u64,
}

impl Sched {
    fn enqueue(&mut self, mut req: GenerateReq, tx: mpsc::Sender<WorkerEvent>, session_id: Option<String>, cancel: Arc<AtomicBool>) {
        let prompt_tokens = req.tokens.len() as u32;
        let trailing_marker = req.tokens.last().copied().filter(|&t| t == TOK_THINK_BEGIN || t == TOK_THINK_END);
        if trailing_marker.is_some() {
            req.tokens.truncate(req.tokens.len() - 1);
        }
        self.queue.push_back(Pending { req, tx, session_id, cancel, trailing_marker, prompt_tokens, queued: Instant::now() });
    }

    fn abort_all(&mut self, why: &str) {
        for s in self.streams.drain(..) {
            let _ = s.tx.try_send(WorkerEvent::Error(why.to_string()));
            let _ = self.arena.release(s.slot);
        }
        for p in self.prefills.drain(..) {
            let _ = p.p.tx.try_send(WorkerEvent::Error(why.to_string()));
            self.spare_states.push(p.kv);
        }
        for (p, _) in self.parked.drain(..) {
            let _ = p.p.tx.try_send(WorkerEvent::Error(why.to_string()));
            self.spare_states.push(p.kv);
        }
        for p in self.queue.drain(..) {
            let _ = p.tx.try_send(WorkerEvent::Error(why.to_string()));
        }
    }

    /// One scheduler tick: a prefill chunk (or its start/finish) or a decode step.
    fn tick(&mut self, state: &mut WorkerState) -> eyre::Result<()> {
        self.tick += 1;
        // Cancelled / dead streams leave before the step.
        let mut i = 0;
        while i < self.streams.len() {
            let s = &self.streams[i];
            if s.cancel.load(Ordering::Relaxed) || s.tx.is_closed() {
                let s = self.streams.remove(i);
                tracing::info!(slot = s.slot, completion_tokens = s.completion_tokens, "multistream: stream cancelled");
                self.arena.release(s.slot)?;
            } else {
                i += 1;
            }
        }
        // Parked (prefilled, no room yet): retry admission now that streams may
        // have finished. Oldest first.
        if !self.parked.is_empty() {
            let mut i = 0;
            while i < self.parked.len() {
                let (pf, _) = &self.parked[i];
                if pf.p.cancel.load(Ordering::Relaxed) || pf.p.tx.is_closed() {
                    let (pf, _) = self.parked.remove(i);
                    self.spare_states.push(pf.kv);
                    continue;
                }
                let ctx_cap = pf.prefix.len() as u32 + pf.p.req.max_new as u32 + 2;
                if self.arena.live() < self.arena.n_slots as usize && self.arena.fits_after_compaction(ctx_cap) {
                    let (pf, logits) = self.parked.remove(i);
                    let tx = pf.p.tx.clone();
                    if let Err((kv, e)) = self.try_admit(state, pf, logits) {
                        tracing::error!(error = %e, "multistream: parked admission failed");
                        let _ = tx.try_send(WorkerEvent::Error(format!("{e:#}")));
                        if let Some(kv) = kv { self.spare_states.push(kv); }
                    }
                    continue;
                }
                i += 1;
            }
        }
        // Start prefills while scratch states are spare. Shortest prompt first
        // (plan 5.3: SJF on the suffix; the prompt length is the proxy we have
        // before the snapshot probe), with aging: a request that has waited
        // longer than `V41_MS_AGING_S` (default 60 s) goes first regardless.
        while !self.spare_states.is_empty() && !self.queue.is_empty() {
            let aging = std::time::Duration::from_secs(env_usize("V41_MS_AGING_S", 60) as u64);
            if let Some(i) = self.queue.iter().position(|p| p.queued.elapsed() >= aging) {
                let p = self.queue.remove(i).unwrap();
                self.queue.push_front(p);
            } else if let Some((i, _)) = self.queue.iter().enumerate().min_by_key(|(_, p)| p.req.tokens.len()) {
                let p = self.queue.remove(i).unwrap();
                self.queue.push_front(p);
            }
            let mut started = false;
            while let Some(p) = self.queue.pop_front() {
                if p.cancel.load(Ordering::Relaxed) || p.tx.is_closed() {
                    continue;
                }
                if !p.req.images.is_empty() || state.mtp.is_some() {
                    // Legacy serial path (vision / DSpark): only with an empty arena
                    // and no prefill in flight.
                    if self.streams.is_empty() && self.prefills.is_empty() {
                        let Pending { req, tx, session_id, cancel, trailing_marker, .. } = p;
                        let mut req = req;
                        if let Some(m) = trailing_marker { req.tokens.push(m); }
                        if let Err(e) = handle_generate_stream(state, req, session_id, cancel, &tx) {
                            let _ = tx.blocking_send(WorkerEvent::Error(format!("{e:#}")));
                            let _ = state.engine.remote_drain_in_flight();
                            let _ = state.engine.remote_reconnect_if_dead();
                        }
                        save_live_if_dirty(state);
                        state.live = None;
                        state.state.reset_in_place(state.dgpu, state.igpu)?;
                        return Ok(());
                    }
                    self.queue.push_front(p);
                    break;
                }
                if self.arena.live() as u32 >= self.arena.n_slots {
                    self.queue.push_front(p);
                    break;
                }
                let kv = self.spare_states.pop().expect("checked");
                match self.start_prefill(state, p, kv) {
                    Ok(pf) => { self.prefills.push(pf); started = true; break; }
                    Err((p, kv, e)) => {
                        self.spare_states.push(kv);
                        let _ = p.tx.try_send(WorkerEvent::Error(format!("{e:#}")));
                    }
                }
            }
            if !started { break; }
        }
        // Chunk or step? Bursts with hysteresis: stay in a phase until its
        // budget elapses (V41_MS_PREFILL_BURST_MS / V41_MS_DECODE_BURST_MS,
        // default 4000 each) or it runs out of work.
        let have_pf = !self.prefills.is_empty();
        let have_dec = !self.streams.is_empty();
        let budget = |ph: Phase| std::time::Duration::from_millis(match ph {
            Phase::Prefill => env_usize("V41_MS_PREFILL_BURST_MS", 120_000) as u64,
            Phase::Decode => env_usize("V41_MS_DECODE_BURST_MS", 30_000) as u64,
        });
        let next = match (have_pf, have_dec) {
            (true, false) => Phase::Prefill,
            (false, true) => Phase::Decode,
            (false, false) => return Ok(()),
            (true, true) => {
                if self.phase_since.elapsed() >= budget(self.phase) {
                    match self.phase { Phase::Prefill => Phase::Decode, Phase::Decode => Phase::Prefill }
                } else {
                    self.phase
                }
            }
        };
        if next != self.phase {
            tracing::info!(from = ?self.phase, to = ?next, live = self.streams.len(), prefills = self.prefills.len(), queued = self.queue.len(), "ms.phase");
            self.phase = next;
            self.phase_since = Instant::now();
        }
        match self.phase {
            Phase::Prefill => self.prefill_tick(state)?,
            Phase::Decode => self.decode_step(state)?,
        }
        Ok(())
    }

    /// Probe the snapshot index, restore the longest prefix into the scratch
    /// state, and build the suffix job.
    fn start_prefill(&mut self, state: &mut WorkerState, p: Pending, mut kv: v4flash_kernels::het::HetModelState) -> Result<Prefill, (Pending, v4flash_kernels::het::HetModelState, eyre::Report)> {
        let t0 = Instant::now();
        save_live_if_dirty(state);
        state.live = None;
        if let Err(e) = kv.reset_in_place(state.dgpu, state.igpu) { return Err((p, kv, e)); }
        kv.restore_compressor_lending();
        let tokens = &p.req.tokens;
        if tokens.is_empty() && p.trailing_marker.is_none() {
            return Err((p, kv, eyre!("empty prompt")));
        }
        // Snapshot probe (session hint, then longest byte prefix), as the legacy path.
        let mut prefix: Vec<i32> = Vec::new();
        if std::env::var("DEEPSTRIX_SNAPSHOT_REUSE").as_deref() != Ok("0") {
            let hit_session = p.session_id.as_deref().and_then(|sid| state.snapshot_index.lookup_session(sid, tokens));
            let hit_walk = state.snapshot_index.find_longest_prefix(tokens, &p.req.image_spans, TOK_EOS, TOK_ASSISTANT, TOK_USER, state.vocab.as_ref(), &state.byte_decoder);
            let hit = match (hit_session, hit_walk) {
                (Some(a), Some(b)) => Some(if a.0 >= b.0 { a } else { b }),
                (a, b) => a.or(b),
            };
            if let Some((snap_req_tokens, snap_hash, snap_dir)) = hit {
                // Keep >= 1 suffix token to prefill (or a marker to forward).
                let usable = snap_req_tokens >= 64 && (snap_req_tokens < tokens.len() || p.trailing_marker.is_some());
                if usable {
                    match snapshot::restore_vl(&mut kv, &snap_dir, state.dgpu, state.igpu, &state.model_fingerprint,
                        snapshot::RestoreKernels { fp8: &state.engine.dgpu.comp_kv_fp8, stream: &state.engine.dgpu.compute }) {
                        Ok(r) => {
                            if r.tokens.len() <= tokens.len() && tokens[..r.tokens.len()] == r.tokens[..] {
                                let _ = state.snapshot_index.touch(&snap_hash);
                                prefix = r.tokens;
                                tracing::info!(restored = prefix.len(), total = tokens.len(), ms = t0.elapsed().as_millis() as u64, "multistream: snapshot restored");
                            } else {
                                tracing::warn!("multistream: restored snapshot is not a token prefix of the request; full prefill");
                                if let Err(e) = kv.reset_in_place(state.dgpu, state.igpu) { return Err((p, kv, e)); }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "multistream: snapshot restore failed; evicting, full prefill");
                            state.snapshot_index.evict(&snap_hash, "multistream: restore failed");
                            if let Err(e) = kv.reset_in_place(state.dgpu, state.igpu) { return Err((p, kv, e)); }
                        }
                    }
                }
            }
        }
        kv.restore_compressor_lending();
        let mut suffix: Vec<i32> = tokens[prefix.len()..].to_vec();
        let mut marker_in_prefill = false;
        if suffix.is_empty() {
            // The snapshot covers the whole prompt: forward the marker through
            // the prefill instead (same computation as the legacy decode-side
            // forward of it, modulo kernel family).
            match p.trailing_marker {
                Some(m) => { suffix.push(m); marker_in_prefill = true; }
                None => return Err((p, kv, eyre!("multistream: snapshot covered the whole prompt with no marker (unreachable by construction)"))),
            }
        }
        let pos0 = prefix.len() as u32;
        // Engram: compress the whole sequence now (cheap); embeddings and
        // Engram rows are produced PER CHUNK in `prefill_job_tick` (lazy job
        // inputs) so a long prompt neither blocks the scheduler for a bulk
        // gather nor holds its whole prompt's inputs in host RAM.
        let mut compressed: Vec<i32> = Vec::with_capacity(tokens.len() + 1);
        if let Some(ec) = state.engram.as_ref() {
            for &t in prefix.iter().chain(suffix.iter()) { compressed.push(ec.hasher.compress(t)); }
        }
        let job = match PrefillJob::new(suffix.clone(), Vec::new(), None, None, pos0, if self.streams.is_empty() { chunk_rows_idle() } else { chunk_rows_busy() }) {
            Ok(j) => j,
            Err(e) => return Err((p, kv, e)),
        };
        let mut pf = Prefill { p, job, kv, prefix, compressed, started: t0 };
        if marker_in_prefill {
            pf.p.trailing_marker = None; // consumed
            pf.prefix.push(suffix[0]);
        } else {
            pf.prefix.extend_from_slice(&suffix);
        }
        Ok(pf)
    }

    /// Run one prefill chunk; on the last one, finish (replay + head), snapshot
    /// the prompt, admit the stream.
    fn prefill_tick(&mut self, state: &mut WorkerState) -> eyre::Result<()> {
        if self.prefills.is_empty() { return Ok(()); }
        let i = self.rr % self.prefills.len();
        self.rr = self.rr.wrapping_add(1);
        let mut pf = self.prefills.remove(i);
        if pf.p.cancel.load(Ordering::Relaxed) || pf.p.tx.is_closed() {
            // CHECKPOINT the partial prefill: a client that times out (the
            // agent's HTTP limit is ~15 min) re-sends the same prompt, and
            // without this the retry started from zero (2026-09-20: a 135K
            // prompt lost 115K prefilled tokens). The encoder state at a chunk
            // boundary is exactly what a resumed prefill restores (the decoder
            // rings are rebuilt by the replay at finish either way), so the
            // snapshot key is the prefix plus the suffix rows done so far.
            let done = pf.job.done_rows();
            if done >= checkpoint_min_rows() && !pf.job.chunks_done() {
                let t = Instant::now();
                pf.kv.restore_compressor_lending();
                let mut tokens_saved: Vec<i32> = pf.prefix.clone();
                tokens_saved.extend_from_slice(&pf.job.tokens()[..done]);
                match snapshot::save(&pf.kv, &tokens_saved, &[], state.dgpu, state.igpu, &state.model_fingerprint,
                    state.snapshot_index.root(), state.vocab.as_ref(), &state.byte_decoder, None) {
                    Ok(entry) => {
                        let hash = entry.hash;
                        state.snapshot_index.insert(entry);
                        if let Some(sid) = pf.p.session_id.clone() { state.snapshot_index.session_to_hash.insert(sid, hash); }
                        tracing::info!(tokens = tokens_saved.len(), done, total = pf.job.total(), ms = t.elapsed().as_millis() as u64, "multistream: prefill cancelled; partial snapshot saved");
                    }
                    Err(e) => tracing::warn!(error = %e, "multistream: prefill cancelled; partial snapshot FAILED"),
                }
            } else {
                tracing::info!(done, total = pf.job.total(), "multistream: prefill cancelled");
            }
            self.spare_states.push(pf.kv);
            return Ok(());
        }
        // A failure in ONE job (paging, admission, box 2) fails that request
        // only; the live streams keep going. The engine-level drain/redial is
        // still done, since a box-2 fault leaves tickets in flight.
        let tx = pf.p.tx.clone();
        match self.prefill_job_tick(state, pf, i) {
            Ok(()) => Ok(()),
            Err((kv, e)) => {
                tracing::error!(error = %e, "multistream: prefill failed; failing that request only");
                let _ = tx.try_send(WorkerEvent::Error(format!("{e:#}")));
                if let Some(kv) = kv { self.spare_states.push(kv); }
                let _ = state.engine.remote_drain_in_flight();
                let _ = state.engine.remote_reconnect_if_dead();
                Ok(())
            }
        }
    }

    /// One chunk (or the finish + admit) of `pf`. On error returns the scratch
    /// state (if still owned) for recycling.
    fn prefill_job_tick(&mut self, state: &mut WorkerState, mut pf: Prefill, i: usize) -> Result<(), (Option<v4flash_kernels::het::HetModelState>, eyre::Report)> {
        if !pf.job.chunks_done() {
            let t = Instant::now();
            if let Some(pg) = state.pager.as_mut() {
                if let Err(e) = pg.drain_prefetched() { return Err((Some(pf.kv), e)); }
            }
            if let Err(e) = chunk_inputs(&mut pf, state) { return Err((Some(pf.kv), e)); }
            let inputs_ms = t.elapsed().as_millis() as u64;
            let WorkerState { engine, bd_a, bi_a, bd_b, bi_b, sd, si, dgpu_scratch, weights, pager, .. } = state;
            let rows = match engine.prefill_job_chunk(&mut pf.job, bd_a, bi_a, bd_b, bi_b, sd, si, dgpu_scratch, &mut pf.kv, weights, pager.as_mut()) {
                Ok(r) => r,
                Err(e) => return Err((Some(pf.kv), e)),
            };
            tracing::debug!(rows, done = pf.job.done_rows(), total = pf.job.total(), inputs_ms, ms = t.elapsed().as_millis() as u64, "multistream: prefill chunk");
            if !pf.job.chunks_done() {
                self.prefills.insert(i.min(self.prefills.len()), pf);
                return Ok(());
            }
        }
        let WorkerState { engine, bd_a, bi_a, bd_b, bi_b, sd, si, dgpu_scratch, weights, pager, .. } = state;
        let kv = &mut pf.kv;
        let logits = match engine.prefill_job_finish(&mut pf.job, bd_a, bi_a, bd_b, bi_b, sd, si, dgpu_scratch, kv, weights, pager.as_mut()) {
            Ok(l) => l,
            Err(e) => return Err((Some(pf.kv), e)),
        };
        kv.restore_compressor_lending();
        // Snapshot the prompt (the legacy path saves here too, before the marker).
        let tokens_saved: Vec<i32> = pf.prefix.clone();
        flush_expert_stats(state);
        match snapshot::save(&pf.kv, &tokens_saved, &[], state.dgpu, state.igpu, &state.model_fingerprint,
            state.snapshot_index.root(), state.vocab.as_ref(), &state.byte_decoder, pf.p.session_id.as_deref()) {
            Ok(entry) => {
                let hash = entry.hash;
                state.snapshot_index.insert(entry);
                if let Some(sid) = pf.p.session_id.clone() { state.snapshot_index.session_to_hash.insert(sid, hash); }
            }
            Err(e) => tracing::error!(error = %e, "multistream: snapshot.save failed"),
        }
        self.try_admit(state, pf, logits)
    }

    /// Admit a prefilled request: `ctx_cap` = what this turn can grow to; the
    /// arena carves that many comp rows per store (first fit). Fragmented =>
    /// compact the stores first. No room at all => park it (its scratch state
    /// stays with it) and retry as streams finish.
    fn try_admit(&mut self, state: &mut WorkerState, pf: Prefill, logits: Vec<f32>) -> Result<(), (Option<v4flash_kernels::het::HetModelState>, eyre::Report)> {
        let pos = pf.prefix.len() as u32;
        let ctx_cap = pos + pf.p.req.max_new as u32 + 2;
        if ctx_cap > state.n_kv_max {
            return Err((Some(pf.kv), eyre!("prompt {pos} + max_tokens {} exceeds the context {}", pf.p.req.max_new, state.n_kv_max)));
        }
        let mut slot = self.arena.admit_from_state(&pf.kv, ctx_cap, pos, &state.engine.dgpu.compute);
        if slot.is_err() && self.arena.live() < self.arena.n_slots as usize && self.arena.fits_after_compaction(ctx_cap) {
            let t = Instant::now();
            if let Err(e) = self.arena.compact_stores(&state.engine.dgpu.compute, &mut self.bounce_f16, &mut self.bounce_u8) {
                return Err((Some(pf.kv), e));
            }
            tracing::info!(ms = t.elapsed().as_millis() as u64, live = self.streams.len(), "multistream: stores compacted");
            slot = self.arena.admit_from_state(&pf.kv, ctx_cap, pos, &state.engine.dgpu.compute);
        }
        let slot = match slot {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(ctx_cap, live = self.streams.len(), parked = self.parked.len() + 1, error = %e, "multistream: no room; parking the request until a stream finishes");
                self.parked.push((pf, logits));
                return Ok(());
            }
        };
        if let Err(e) = state.engine.dgpu.compute.synchronize() { return Err((Some(pf.kv), e)); }
        let Prefill { p: pp, job, kv: kv_done, prefix, compressed, started } = pf;
        self.spare_states.push(kv_done);
        let pf = PrefillDone { p: pp, job, prefix, compressed, started };
        self.admit_stream(state, pf, slot, logits).map_err(|e| (None, e))
    }

    fn admit_stream(&mut self, state: &mut WorkerState, pf: PrefillDone, slot: u32, logits: Vec<f32>) -> eyre::Result<()> {
        let top_p = if pf.p.req.top_p.is_finite() && pf.p.req.top_p > 0.0 { pf.p.req.top_p.min(1.0) } else { 1.0 };
        let sample_mode = if pf.p.req.temperature <= 0.0 { SampleMode::Argmax } else {
            SampleMode::Multinomial { temperature: pf.p.req.temperature, min_p_rel: pf.p.req.min_p_rel, top_p }
        };
        let mut rng = SamplerRng::new(pf.p.req.seed);
        let mut s = Stream {
            slot, tx: pf.p.tx.clone(), cancel: pf.p.cancel.clone(), next: 0, seq: pf.prefix.clone(), compressed: pf.compressed.clone(),
            prompt_tokens: pf.p.prompt_tokens, completion_tokens: 0, max_new: pf.p.req.max_new, sample_mode, rng,
            in_think: false, send_failures: 0, started: pf.started, session_id: pf.p.session_id.clone(),
        };
        // Ensure `compressed` covers `seq` (a marker forwarded in the prefill was hashed above).
        if let Some(ec) = state.engram.as_ref() {
            while s.compressed.len() < s.seq.len() { let t = s.seq[s.compressed.len()]; s.compressed.push(ec.hasher.compress(t)); }
        }
        match pf.p.trailing_marker {
            Some(m) => {
                // Legacy: forward the marker, THEN sample. The marker is this
                // stream's first step input (so it is part of `seq`, at the
                // position the step runs); nothing is emitted yet.
                s.next = m;
                s.seq.push(m);
                if let Some(ec) = state.engram.as_ref() { s.compressed.push(ec.hasher.compress(m)); }
                s.in_think = m == TOK_THINK_BEGIN;
            }
            None => {
                let tok = sample_row(&logits, &s.sample_mode, &mut s.rng);
                s.next = tok;
                if !emit(state, &mut s, tok) { self.arena.release(slot)?; return Ok(()); }
                if let Some(f) = stop_reason(&s, tok) {
                    finish(state, &mut self.arena, s, f)?;
                    return Ok(());
                }
            }
        }
        tracing::info!(slot, prompt = pf.prefix.len(), restored = pf.prefix.len() - pf.job.total(), prefill_ms = pf.started.elapsed().as_millis() as u64,
            live = self.streams.len() + 1, "multistream: stream admitted");
        self.streams.push(s);
        Ok(())
    }

    /// One batched decode step over every live stream.
    fn decode_step(&mut self, state: &mut WorkerState) -> eyre::Result<()> {
        // Token boundary: nothing is reading the pool (the previous step and
        // chunk both synchronized), so admit box-1's background-read experts
        // (`V41_B1_PREFETCH`, catch-all mode: misses are computed on box 2 and
        // read from box 1's disk off the critical path). Same as decode's
        // `forward_one!`; without this the multistream path never warmed box 1.
        if let Some(pg) = state.pager.as_mut() { pg.drain_prefetched()?; }
        let t0 = Instant::now();
        let b = self.streams.len();
        let slots: Vec<u32> = self.streams.iter().map(|s| s.slot).collect();
        let toks: Vec<i32> = self.streams.iter().map(|s| s.next).collect();
        let mut hcs: Vec<Vec<f32>> = Vec::with_capacity(b);
        for &t in &toks {
            let mut v = vec![0f32; HC_DIM as usize];
            embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, t, &mut v);
            hcs.push(v);
        }
        // Engram rows: one hash per row from its own stream's sequence.
        let ein = ENGRAM_IN as usize;
        let t_eng = Instant::now();
        let engram_rows: Option<Vec<Vec<f32>>> = match (state.pager.as_ref(), state.engram.as_ref()) {
            (Some(pg), Some(ec)) => {
                let mut rows = vec![vec![0f32; b * ein]; ec.tables.len()];
                for (r, s) in self.streams.iter_mut().enumerate() {
                    // `seq` already ends with `next` (pushed when it was chosen).
                    let pos = s.seq.len() - 1;
                    let hashes = ec.hasher.hash_ids(&s.compressed, pos);
                    if s.compressed[pos] == v4flash_core::engram_hash::DEAD { continue; }
                    for (li, tbl) in ec.tables.iter().enumerate() {
                        tbl.gather_position(pg.raw(), &hashes[li], &mut rows[li][r * ein..(r + 1) * ein])?;
                    }
                }
                Some(rows)
            }
            _ => None,
        };
        let engram_ms = t_eng.elapsed().as_secs_f64() * 1e3;
        let WorkerState { engine, bd_a, bi_a, sd, si, dgpu_scratch, weights, pager, .. } = state;
        // V41_MS_PROFILE=1: per-stage GPU busy time of the batched step (HIP
        // events per stage, ~100 us/layer), rolled up over V41_MS_PROFILE_EVERY
        // steps and logged as "ms.stage". Wall - busy = host / link / sync.
        let profile = ms_profile();
        if profile {
            v4flash_kernels::het::forward_prefill::LH_FORCE.store(true, Ordering::Relaxed);
            let _ = v4flash_kernels::het::forward_prefill::take_layer_host_timing();
            engine.dgpu.events.set_enabled(true);
            engine.igpu.events.set_enabled(true);
            engine.dgpu.events.reset();
            engine.igpu.events.reset();
            v4flash_kernels::het::trace::phase::reset();
        }
        let t_fwd = Instant::now();
        engine.forward_step_arena(bd_a, bi_a, sd, si, &mut self.arena, &mut self.dev, &slots, weights, &hcs, &toks, engram_rows.as_deref(), pager.as_mut())?;
        let fwd_only_ms = t_fwd.elapsed().as_secs_f64() * 1e3;
        let logits = engine.head_rows(dgpu_scratch, bd_a, b, weights)?;
        let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1e3;
        if profile {
            use v4flash_kernels::het::trace::{phase as counters, rollup_by_name};
            let dg = rollup_by_name(&engine.dgpu.events.harvest()?);
            let ig = rollup_by_name(&engine.igpu.events.harvest()?);
            let host = [
                ("host.sel_sync", counters::get(&counters::SEL_SYNC_NS)),
                ("host.ensure", counters::get(&counters::ENSURE_NS)),
                ("host.engram_stage", counters::get(&counters::ENGRAM_STAGE_NS)),
                ("host.remote_rtt", counters::get(&counters::REMOTE_RTT_NS)),
                ("host.remote_srv", counters::get(&counters::REMOTE_SRV_NS)),
                ("box2.page_ms", counters::get(&counters::REMOTE_PAGE_NS)),
                ("box2.compute_ms", counters::get(&counters::REMOTE_COMPUTE_NS)),
                ("box2.misses_x1e6", counters::get(&counters::REMOTE_MISSES) * 1_000_000),
            ];
            let acc = &mut self.profile_acc;
            acc.steps += 1;
            acc.rows += b as u64;
            acc.wall_ms += fwd_only_ms;
            for (dev, r) in [("dgpu", &dg), ("igpu", &ig)] {
                for &(name, ms, calls) in r {
                    let e = acc.stages.entry((dev, name)).or_insert((0.0, 0));
                    e.0 += ms as f64;
                    e.1 += calls as u64;
                }
            }
            for (name, ns) in host {
                let e = acc.stages.entry(("host", name)).or_insert((0.0, 0));
                e.0 += ns as f64 / 1e6;
                e.1 += 1;
            }
            if let Some(pg) = pager.as_ref() {
                if let Some((q, a, df, ams)) = pg.prefetch_stats() {
                    let (dq, da, ddf, dms) = (q.saturating_sub(acc.pf_last.0), a.saturating_sub(acc.pf_last.1), df.saturating_sub(acc.pf_last.2), ams.saturating_sub(acc.pf_last.3));
                    acc.pf_last = (q, a, df, ams);
                    for (name, v) in [("prefetch.queued", dq as f64), ("prefetch.admitted", da as f64), ("prefetch.dropped_full", ddf as f64), ("prefetch.admit_ms", dms as f64)] {
                        let e = acc.stages.entry(("host", name)).or_insert((0.0, 0));
                        e.0 += v; e.1 += 1;
                    }
                }
                let c = pg.counters();
                let (dm, dr) = (c.prefill_misses.saturating_sub(acc.last_misses), c.prefill_read_ns.saturating_sub(acc.last_read_ns));
                acc.last_misses = c.prefill_misses;
                acc.last_read_ns = c.prefill_read_ns;
                let e = acc.stages.entry(("host", "pager.misses_per_step")).or_insert((0.0, 0));
                e.0 += dm as f64; e.1 += 1;
                let e = acc.stages.entry(("host", "pager.read_ms")).or_insert((0.0, 0));
                e.0 += dr as f64 / 1e6; e.1 += 1;
            }
            for (name, us) in v4flash_kernels::het::forward_prefill::take_layer_host_timing() {
                let e = acc.stages.entry(("host", name)).or_insert((0.0, 0));
                e.0 += us as f64 / 1e3;
                e.1 += 1;
            }
            {
                let e = acc.stages.entry(("host", "step.fwd_wall")).or_insert((0.0, 0));
                e.0 += fwd_only_ms;
                e.1 += 1;
                let e = acc.stages.entry(("host", "step.head")).or_insert((0.0, 0));
                e.0 += fwd_ms - fwd_only_ms;
                e.1 += 1;
            }
            let every = env_usize("V41_MS_PROFILE_EVERY", 20) as u64;
            if acc.steps >= every {
                let mut v: Vec<_> = acc.stages.iter().map(|(&(d, n), &(ms, c))| (d, n, ms / acc.steps as f64, c)).collect();
                v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                // Parent stages only ("dgpu.*" / "igpu.*"): the "k.*" kernel
                // sub-stages nest inside them and would double count.
                let dgpu_busy: f64 = v.iter().filter(|e| e.0 == "dgpu" && e.1.starts_with("dgpu.")).map(|e| e.2).sum();
                let igpu_busy: f64 = v.iter().filter(|e| e.0 == "igpu" && e.1.starts_with("igpu.")).map(|e| e.2).sum();
                tracing::info!(steps = acc.steps, rows_avg = format!("{:.1}", acc.rows as f64 / acc.steps as f64),
                    wall_ms = format!("{:.1}", acc.wall_ms / acc.steps as f64), dgpu_busy_ms = format!("{dgpu_busy:.1}"),
                    igpu_busy_ms = format!("{igpu_busy:.1}"), "ms.stage.total (per step)");
                for (d, n, ms, c) in v.iter().take(96) {
                    tracing::info!(device = *d, stage = *n, ms_per_step = format!("{ms:.2}"), calls = *c, "ms.stage");
                }
                acc.stages.clear();
                acc.steps = 0;
                acc.rows = 0;
                acc.wall_ms = 0.0;
            }
        }
        let nv = N_VOCAB as usize;
        // Sample, emit, retire.
        let t_s = Instant::now();
        let mut done: Vec<(usize, FinishReason)> = Vec::new();
        for (r, s) in self.streams.iter_mut().enumerate() {
            let tok = sample_row(&logits[r * nv..(r + 1) * nv], &s.sample_mode, &mut s.rng);
            s.next = tok;
            if !emit(state, s, tok) { done.push((r, FinishReason::Stop)); continue; }
            if let Some(f) = stop_reason(s, tok) { done.push((r, f)); }
        }
        let sample_ms = t_s.elapsed().as_secs_f64() * 1e3;
        for (r, f) in done.into_iter().rev() {
            let s = self.streams.remove(r);
            finish(state, &mut self.arena, s, f)?;
        }
        tracing::info!(rows = b, step_ms = format!("{:.1}", t0.elapsed().as_secs_f64() * 1e3), fwd_ms = format!("{fwd_ms:.1}"),
            engram_ms = format!("{engram_ms:.1}"), sample_ms = format!("{sample_ms:.1}"), live = self.streams.len(), "ms.step");
        Ok(())
    }
}

/// Cancelled prefills with at least this many rows done are checkpointed
/// (`V41_MS_CHECKPOINT_MIN_ROWS`, default 4096).
fn checkpoint_min_rows() -> usize { env_usize("V41_MS_CHECKPOINT_MIN_ROWS", 4096) }
fn chunk_rows_idle() -> usize { env_usize("V41_MS_CHUNK_ROWS_IDLE", 1024) }
fn chunk_rows_busy() -> usize { env_usize("V41_MS_CHUNK_ROWS", 1024) }

/// Same rule as `HeterogeneousEngine::sample_next` / the DSpark host twin.
fn sample_row(r: &[f32], mode: &SampleMode, rng: &mut SamplerRng) -> i32 {
    match *mode {
        SampleMode::Argmax => {
            let mut best = 0usize;
            for (i, &x) in r.iter().enumerate() { if x > r[best] { best = i; } }
            best as i32
        }
        SampleMode::Multinomial { temperature, min_p_rel, top_p } => {
            // Same chain as `top_p_min_p_threshold` (temperature, top-p over
            // the tempered weights, then min-p), but without a 129K-entry f64
            // exp + full sort per row (~3 ms/row, 10 ms/step at 4 rows). The
            // weight is exp(logit/T - gmax) in (0, 1]; entries below FLOOR
            // cannot move a top-p cutoff by more than N_VOCAB * FLOOR of the
            // mass (< 1e-5 of the total, which is >= 1), so only the survivors
            // are sorted -- typically a few hundred.
            const FLOOR: f32 = 1e-10;
            let inv_t = 1.0f32 / temperature;
            let gmax = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
            let lo = (min_p_rel.max(FLOOR)).ln(); // survivors: x*inv_t - gmax >= lo
            let mut cand: Vec<(f32, u32)> = Vec::with_capacity(512);
            for (i, &x) in r.iter().enumerate() {
                let l = x * inv_t - gmax;
                if l >= lo { cand.push((l.exp(), i as u32)); }
            }
            // top-p cutoff over the survivors (sorted descending).
            let thr = if top_p >= 1.0 { 0.0f32 } else {
                cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let z: f32 = cand.iter().map(|c| c.0).sum();
                let target = top_p * z;
                let mut cum = 0.0f32;
                let mut t = cand.last().map(|c| c.0).unwrap_or(0.0);
                for c in &cand { cum += c.0; if cum >= target { t = c.0; break; } }
                t
            }.max(min_p_rel);
            let z: f32 = cand.iter().filter(|c| c.0 >= thr).map(|c| c.0).sum();
            let target = rng.next_f32() * z;
            let mut acc = 0.0f32;
            let mut pick = cand.first().map(|c| c.1).unwrap_or(0);
            // Cumulative pick in VOCAB order (as before), over the survivors.
            if top_p < 1.0 { cand.sort_unstable_by_key(|c| c.1); }
            for c in &cand {
                if c.0 < thr { continue; }
                acc += c.0;
                pick = c.1;
                if acc >= target { break; }
            }
            pick as i32
        }
    }
}

/// Record `tok` as the stream's next input and emit it (unless suppressed).
/// Returns false when the client is gone.
fn emit(state: &WorkerState, s: &mut Stream, tok: i32) -> bool {
    s.seq.push(tok);
    if let Some(ec) = state.engram.as_ref() { s.compressed.push(ec.hasher.compress(tok)); }
    s.completion_tokens += 1;
    if is_turn_end(tok) {
        return true; // the stop is decided by the caller; nothing to emit
    }
    if tok == TOK_THINK_BEGIN { s.in_think = true; return true; }
    if tok == TOK_THINK_END { s.in_think = false; return true; }
    let Some(bytes) = state.vocab.token_text(tok) else { return true };
    let raw = gpt2_decode_token(bytes, &state.byte_decoder);
    match s.tx.try_send(WorkerEvent::Chunk { token_id: tok, bytes: raw, reasoning: s.in_think }) {
        Ok(()) => { s.send_failures = 0; true }
        Err(mpsc::error::TrySendError::Full(_)) => {
            s.send_failures += 1;
            if s.send_failures > CHUNK_SEND_FAILURES_MAX {
                tracing::warn!(slot = s.slot, "multistream: stream consumer slow; dropping client");
                false
            } else { true }
        }
        Err(mpsc::error::TrySendError::Closed(_)) => false,
    }
}

fn stop_reason(s: &Stream, tok: i32) -> Option<FinishReason> {
    if is_turn_end(tok) { return Some(FinishReason::Stop); }
    if s.completion_tokens as usize >= s.max_new { return Some(FinishReason::Length); }
    None
}

/// Lazy job inputs for the next chunk: layer-0 embeddings and Engram rows
/// (batched gather over runs of live positions, as `EngramCtx::rows_for_chunk`).
fn chunk_inputs(pf: &mut Prefill, state: &mut WorkerState) -> eyre::Result<()> {
    let (a, z) = pf.job.next_chunk_range((state.bd_a.rows, state.bd_b.rows))?;
    let toks = &pf.job.tokens()[a..z];
    let mut hcs: Vec<Vec<f32>> = Vec::with_capacity(toks.len());
    for &tok in toks {
        let mut v = vec![0f32; HC_DIM as usize];
        embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, tok, &mut v);
        hcs.push(v);
    }
    let engram = match (state.pager.as_ref(), state.engram.as_ref()) {
        (Some(pg), Some(ec)) => {
            const GATHER_THREADS: usize = 32;
            let ein = ENGRAM_IN as usize;
            let p0 = pf.job.pos0() as usize;
            let n = z - a;
            let mut rows = vec![vec![0f32; n * ein]; ec.tables.len()];
            let mut k = 0usize;
            while k < n {
                if pf.compressed[p0 + a + k] == v4flash_core::engram_hash::DEAD { k += 1; continue; }
                let start = k;
                while k < n && pf.compressed[p0 + a + k] != v4flash_core::engram_hash::DEAD { k += 1; }
                let hashes: Vec<_> = (start..k).map(|q| ec.hasher.hash_ids(&pf.compressed, p0 + a + q)).collect();
                for (li, tbl) in ec.tables.iter().enumerate() {
                    let flat: Vec<i64> = hashes.iter().flat_map(|h| h[li]).collect();
                    tbl.gather(pg.raw(), &flat, &mut rows[li][start * ein..k * ein], GATHER_THREADS)?;
                }
            }
            Some(rows)
        }
        _ => None,
    };
    pf.job.set_chunk_inputs(hcs, engram);
    Ok(())
}

fn finish(state: &mut WorkerState, arena: &mut KvArena, s: Stream, f: FinishReason) -> eyre::Result<()> {
    let elapsed = s.started.elapsed().as_secs_f64();
    tracing::info!(slot = s.slot, prompt_tokens = s.prompt_tokens, completion_tokens = s.completion_tokens,
        tok_per_s = format!("{:.2}", s.completion_tokens as f64 / elapsed.max(1e-3)), finish = ?f, "multistream: stream done");
    let _ = s.tx.try_send(WorkerEvent::Done { prompt_tokens: s.prompt_tokens, completion_tokens: s.completion_tokens, finish: f });
    arena.release(s.slot)?;
    let _ = state; // (RESIDENT streams / turn-end snapshots: M2)
    Ok(())
}
