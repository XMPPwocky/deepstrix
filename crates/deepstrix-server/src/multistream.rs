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
use v4flash_kernels::het::forward_prefill::{lane_rows, LazyEngramRows, PrefillJob};
use v4flash_kernels::het::kv_arena::{KvArena, RowTablesDev, ARENA_ROWS_PER_STREAM};
use v4flash_kernels::het::step_rows::StepRows;
use v4flash_kernels::het::scratch::{HEAD_BATCH_MAX, HEAD_CAND_BAND, HEAD_CAND_CAP, HEAD_CAND_STRIDE};
use v4flash_kernels::het::SampleMode;
use v4flash_kernels::sampler::SamplerRng;

use crate::embed::{embed_lookup, gpt2_decode_token};
use crate::engine_worker::{
    byte_aligned_lcp_vl, encode_request_images, flush_expert_stats, handle_generate_stream, save_live_if_dirty, trim_heap_and_log, EncodedImages,
    EngineRequest, FinishReason, GenerateReq, WorkerEvent, WorkerState,
};
use crate::knobs;
use crate::ms_dspark::{self, LaneRule, MsDspark, PlainLanes};
use v4flash_kernels::knobs::Source;
use crate::snapshot;
use crate::spec_sample::{verify_block, Draft, DraftDist, TargetDist};
use crate::tokens::{is_turn_end, TOK_ASSISTANT, TOK_EOS, TOK_THINK_BEGIN, TOK_THINK_END, TOK_USER};

pub fn enabled() -> bool {
    knobs::MULTISTREAM.on()
}

/// Positions reserved beyond the prompt when a request is admitted
/// (`V41_MS_KV_HEADROOM`, default 16384; 0 = reserve the whole `max_new`, the
/// pre-2026-09-27 rule). Every client sends no `max_tokens`, so the whole
/// `max_new` was the 64K default: a ~170K agent turn reserved ~235K positions
/// and the 844,800-row arena held ~3.6 of them. 09-27 (942 streams):
/// completions p50 542, p99 5016, max 14,343, none hit the limit. A stream
/// that runs past its reservation grows (`KvArena::grow`).
fn kv_headroom() -> u32 { knobs::MS_KV_HEADROOM.get() as u32 }
/// Positions a growing stream adds per growth (`V41_MS_KV_GROW`).
fn kv_grow_step() -> u32 { knobs::MS_KV_GROW.get() as u32 }
/// Grow when this few positions are left in the reservation (`V41_MS_KV_GROW_AT`).
fn kv_grow_at() -> u32 { knobs::MS_KV_GROW_AT.get() as u32 }
/// Positions of rows a new reservation must leave free while streams are live
/// (`V41_MS_KV_SPARE`, default one grow step): the room they grow into.
fn kv_spare() -> u32 {
    if knobs::MS_KV_SPARE.source() == Source::Default { kv_grow_step() } else { knobs::MS_KV_SPARE.get() as u32 }
}

/// `max_new` for a request whose context will be `pos`: a defaulted one is
/// shrunk to what the context leaves (never fail a long prompt for the
/// server's own default); `Err` if an explicit one does not fit.
fn effective_max_new(req: &GenerateReq, pos: u32, n_kv_max: u32) -> eyre::Result<usize> {
    let mut max_new = req.max_new;
    if req.max_new_defaulted {
        let room = n_kv_max.saturating_sub(pos + 2) as usize;
        max_new = max_new.min(room).max(1);
    }
    if pos as usize + max_new + 2 > n_kv_max as usize {
        return Err(eyre!("prompt {pos} + max_tokens {max_new} exceeds the context {n_kv_max}"));
    }
    Ok(max_new)
}

/// Positions to reserve for a stream at `pos` that may generate `max_new`.
fn reservation(pos: u32, max_new: usize) -> u32 {
    let h = kv_headroom();
    let tail = if h == 0 { max_new as u32 } else { (max_new as u32).min(h) };
    pos + tail + 2
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
    /// DSpark's draft draws (sampled drafts): its own stream, independent of
    /// the verifier's `rng` (plan 2.4).
    draft_rng: SamplerRng,
    in_think: bool,
    send_failures: u32,
    started: Instant,
    session_id: Option<String>,
    /// Context positions this turn may reach (`prompt + max_new + 2`): the
    /// most its reservation ever grows to.
    ctx_full: u32,
    /// Set while the stream sits steps out because its reservation is full and
    /// the arena has no rows to grow it (`Sched::take_stalled`).
    stalled_since: Option<Instant>,
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
    /// First time it stepped aside because its KV reservation did not fit.
    room_wait: Option<Instant>,
}

/// A prefill in flight. Each job owns a scratch single-sequence state (from
/// `Sched::spare_states`), so `V41_MS_PREFILL_JOBS` prompts can be prefilled
/// round-robin, chunk by chunk — a short prompt is not stuck behind a 100K one.
struct Prefill {
    p: Pending,
    /// Arena slot RESERVED for this request before its prefill started
    /// (`KvArena::reserve`); filled at admission. Every path that drops a
    /// `Prefill` without admitting it must release it.
    slot: u32,
    job: PrefillJob,
    kv: v4flash_kernels::het::HetModelState,
    /// Tokens already in the scratch state (restored prefix), then the suffix.
    prefix: Vec<i32>,
    compressed: Vec<i32>,
    started: Instant,
    /// Tower output for the request's images (empty when none): rows for
    /// the synthetic image ids, spliced into the chunk inputs.
    vl: EncodedImages,
    /// The trailing think marker when the restored snapshot already covers
    /// the whole prompt and the marker itself is the prefill (`plan_suffix`;
    /// `p.trailing_marker` is then None). No prompt snapshot is saved at
    /// finish then: prompt + marker is a key the prefix walk
    /// (EOS/Assistant/User boundaries) can never match, and it took one of
    /// the lineage's slots.
    prefilled_marker: Option<i32>,
    /// Engram rows gathered AHEAD of the chunks that need them
    /// (`engram_lookahead`), contiguous blocks in token order.
    engram_ahead: EngramAhead,
}

/// A prefill job's Engram rows gathered ahead (2026-10-01): a long prefill
/// spent 9-20% of its wall gathering each sub-chunk's Engram rows on the
/// scheduler thread before the sub-chunk could start (0.15-0.25 s cached,
/// 0.4-1.1 s from box-1 NVMe), while box-1 disk sat otherwise idle. Now each
/// prefill tick with room gathers the next block on a scoped thread while its
/// unit runs on the GPUs, up to one layer-major window ahead
/// (`PrefillJob::input_lookahead_rows`): the 16 non-input units of a window
/// (groups 1..) gather the next window's sub-chunks, so its group-0 units find
/// their rows ready. Positions are absolute (`p0 + index`), so a block serves
/// any later chunk cut.
/// `V41_MS_ENGRAM_AHEAD=0` turns it off (each sub-chunk gathers its own rows,
/// as before). Memory: at most one window + one block of rows cached (~250
/// MB at 4096 + 1024 rows x 2 tables x 6144 f32), beside the window's own copy.
#[derive(Default)]
struct EngramAhead {
    /// `(start, end, rows per Engram table [(end - start) * ENGRAM_IN])`.
    blocks: VecDeque<(usize, usize, Vec<Vec<f32>>)>,
    /// First job token index whose rows are neither consumed nor cached.
    next: usize,
    /// Rows before this index have been served to a chunk (or gathered for it).
    consumed: usize,
}

/// `V41_MS_ENGRAM_AHEAD` (default on; `0` off): `EngramAhead`.
fn engram_ahead_on() -> bool {
    knobs::MS_ENGRAM_AHEAD.on()
}

impl EngramAhead {
    /// Cached rows not yet consumed (a partly consumed block counts its rest).
    fn cached_rows(&self) -> usize {
        self.blocks.iter().map(|b| b.1.saturating_sub(b.0.max(self.consumed))).sum()
    }

    /// Rows `[a, z)` (non-empty) from the cache if it covers them, else `None`
    /// (the caller gathers them). Either way everything before `z` counts as
    /// consumed: blocks wholly before `z` are dropped, later ones are KEPT (a
    /// miss must not throw away the look-ahead -- 10-01 review: dropping it on
    /// layer-major's empty-input units discarded every window's look-ahead).
    fn take(&mut self, a: usize, z: usize, n_tables: usize) -> Option<Vec<Vec<f32>>> {
        let out = self.assemble(a, z, n_tables);
        self.consumed = self.consumed.max(z);
        self.next = self.next.max(z);
        while self.blocks.front().is_some_and(|b| b.1 <= z) {
            self.blocks.pop_front();
        }
        out
    }

    fn assemble(&mut self, a: usize, z: usize, n_tables: usize) -> Option<Vec<Vec<f32>>> {
        let ein = ENGRAM_IN as usize;
        while self.blocks.front().is_some_and(|b| b.1 <= a) {
            self.blocks.pop_front();
        }
        let covered = self.blocks.front().is_some_and(|b| b.0 <= a) && {
            let mut end = self.blocks.front().map_or(a, |b| b.0);
            for b in &self.blocks {
                if b.0 != end {
                    break;
                }
                end = b.1;
            }
            end >= z
        };
        if !covered {
            return None;
        }
        let mut out = vec![vec![0f32; (z - a) * ein]; n_tables];
        for (s, e, rows) in &self.blocks {
            let (lo, hi) = (a.max(*s), z.min(*e));
            if lo >= hi {
                continue;
            }
            for (t, r) in rows.iter().enumerate() {
                out[t][(lo - a) * ein..(hi - a) * ein].copy_from_slice(&r[(lo - s) * ein..(hi - s) * ein]);
            }
        }
        Some(out)
    }

    /// The next block worth gathering ahead (`engram_lookahead`): from the
    /// first row neither consumed nor cached (and not before `done_rows`),
    /// `block` rows, while fewer than `span` cached rows are unconsumed.
    fn next_block(&self, done_rows: usize, block: usize, total: usize, span: usize) -> Option<(usize, usize)> {
        let start = self.next.max(done_rows);
        let end = (start + block).min(total);
        (start < end && self.cached_rows() < span).then_some((start, end))
    }
}

/// `Prefill` after its scratch state has been recycled.
struct PrefillDone {
    p: Pending,
    job: PrefillJob,
    prefix: Vec<i32>,
    compressed: Vec<i32>,
    started: Instant,
    prefilled_marker: Option<i32>,
}


pub fn worker_loop_ms(mut state: WorkerState, rx: &mut mpsc::Receiver<EngineRequest>) {
    let n_slots = knobs::MS_SLOTS.get() as u32;
    // Context budget across all live streams (positions); each store gets
    // budget / ratio rows. Default 2 x --ctx: two 240K agents, or eight 75K
    // ones. ~1 KB per row at ratio 1 plus 0.5 KB per ratio-2 store; MEASURED
    // 2026-09-20 on the 16 GB dGPU with two prefill states: 3 x --ctx left
    // 190 MiB free (unsafe), 2 x --ctx 1.0 GiB.
    let ctx_rows = if knobs::MS_CTX_ROWS.source() == Source::Default { 2 * state.n_kv_max } else { knobs::MS_CTX_ROWS.get() as u32 };
    let chunk_rows = knobs::MS_CHUNK_ROWS.usize();
    let arena = match KvArena::alloc_ctx(state.dgpu, n_slots, ctx_rows) {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(error = %e, "multistream: arena alloc failed; falling back to the serial loop is not possible here — aborting");
            return;
        }
    };
    // Rows per step: one per stream, plus a speculating stream's draft rows.
    let rows_cap = n_slots * ARENA_ROWS_PER_STREAM;
    let dev = match RowTablesDev::alloc(state.dgpu, rows_cap, arena.stores.len()) {
        Ok(d) => d,
        Err(e) => {
            tracing::error!(error = %e, "multistream: row tables alloc failed");
            return;
        }
    };
    let dev_b = match RowTablesDev::alloc(state.dgpu, rows_cap, arena.stores.len()) {
        Ok(d) => d,
        Err(e) => {
            tracing::error!(error = %e, "multistream: row tables (lane B) alloc failed");
            return;
        }
    };
    let dev_c = match RowTablesDev::alloc(state.dgpu, rows_cap, arena.stores.len()) {
        Ok(d) => d,
        Err(e) => {
            tracing::error!(error = %e, "multistream: row tables (lane C) alloc failed");
            return;
        }
    };
    let head_out = match v4flash_hip::PinnedBuffer::<u32>::new(HEAD_BATCH_MAX * HEAD_CAND_STRIDE) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "multistream: head candidate buffer alloc failed");
            return;
        }
    };
    tracing::info!(n_slots, ctx_rows, chunk_rows, prefill_burst_ms = knobs::MS_PREFILL_BURST_MS.get(), decode_burst_ms = knobs::MS_DECODE_BURST_MS.get(), head_cands = ?head_cands_mode(), "multistream scheduler ON");
    // Logs its UNTESTED-fidelity warning when on: at every start, not only on first use.
    let _ = v4flash_kernels::het::forward_prefill::prefill_f16_replies();
    // Event trace (`V41_EVTRACE_DIR`): every V41_* env var goes in the header,
    // and every registered knob's effective value (`v4flash_kernels::knobs`).
    {
        let env: serde_json::Map<String, serde_json::Value> = std::env::vars()
            .filter(|(k, _)| k.starts_with("V41_") || k.starts_with("GPU_") || k.starts_with("HIP_"))
            .map(|(k, v)| (k, serde_json::Value::String(v)))
            .collect();
        let eff: serde_json::Map<String, serde_json::Value> = v4flash_kernels::knobs::snapshot()
            .into_iter()
            .map(|(k, v, src, _)| (k.to_string(), serde_json::Value::String(format!("{v} ({src:?})"))))
            .collect();
        v4flash_kernels::het::evtrace::init("hub", serde_json::json!({ "n_slots": n_slots, "ctx_rows": ctx_rows, "chunk_rows": chunk_rows, "env": env, "knobs": eff }));
    }
    let n_jobs = knobs::MS_PREFILL_JOBS.usize();
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
    // DSpark on the arena (`V41_MS_DSPARK=accept`): one drafter ring per slot,
    // and every step (and the prefill replay that seeds a ring) captures the
    // drafter's input residuals on its rows.
    let dsp = match (ms_dspark::enabled(), state.mtp.as_ref()) {
        (true, Some(m)) => match MsDspark::alloc(m, state.igpu.id, n_slots) {
            Ok(d) => {
                let cap = v4flash_kernels::het::batch_scratch::MTP_CAP_ROWS;
                state.bd_a.mtp_capture_rows = cap;
                state.bd_b.mtp_capture_rows = cap;
                state.bd_c.mtp_capture_rows = cap;
                tracing::info!(n_slots, ring_all = d.ring_all(), "multistream: DSpark ON (lone stream verifies its drafts in the arena step)");
                Some(d)
            }
            Err(e) => { tracing::error!(error = %e, "multistream: DSpark rings alloc failed; DSpark off"); None }
        },
        (true, None) => { tracing::error!("multistream: V41_MS_DSPARK set but no drafter loaded; DSpark off"); None }
        _ => None,
    };
    let plain_lanes = PlainLanes::from_env(n_slots as usize, pipeline_min_rows());
    let mut sched = Sched { dsp, plain_lanes, trace: LiveTrace::default(), profile_acc: ProfileAcc::default(), legacy_wait_logged: None, dev_b, dev_c, head_out, parked: Vec::new(), bounce_f16, bounce_u8, phase: Phase::Decode, phase_since: Instant::now(), group_hold: None, arena, dev, streams: Vec::new(), queue: VecDeque::new(), prefills: Vec::new(), spare_states, rr: 0, tick: 0 };

    // Set by a tick, cleared once the idle-transition housekeeping has run.
    let mut worked = false;
    loop {
        // 1. Intake: never block while there is work; block when idle.
        // Parked requests are work too: blocking here with one parked left it
        // waiting for the NEXT request to arrive before it was retried.
        // Embedding work (the embed queue) is work too.
        let embed_work = state.embed.as_ref().is_some_and(|e| e.has_work());
        let idle = !sched.has_llm_work() && !embed_work;
        if idle && worked {
            // The serial loop trims after every request; here streams overlap,
            // so trim only when the last one has drained and nothing waits.
            worked = false;
            trim_heap_and_log("host heap at idle");
        }
        if idle {
            // Clear `inflight` BEFORE blocking: the tick that finished the last
            // stream left it set, and the hang watchdog then aborted an IDLE
            // engine 30 min later (DEEPSTRIX_HANG_DEADLINE_MS) -- twice on
            // 2026-09-23, each exactly 30:00 after the last "stream done".
            state.progress.end();
            // Release the embed wake BEFORE blocking, then re-check: a push
            // after this sends a new wake (`EmbedQueue::claim_wake`).
            if let Some(e) = state.embed.as_ref() {
                e.queue().clear_wake();
                if e.has_work() {
                    continue;
                }
            }
        }
        let msg = if idle { rx.blocking_recv() } else { match rx.try_recv() { Ok(m) => Some(m), Err(_) => None } };
        match msg {
            Some(EngineRequest::Generate { req, tx, session_id, cancel }) => {
                sched.enqueue(req, tx, session_id, cancel);
            }
            Some(EngineRequest::EmbedWake) => {
                if let Some(e) = state.embed.as_ref() {
                    e.queue().clear_wake();
                }
            }
            Some(EngineRequest::Shutdown { ack }) => {
                sched.abort_all("server shutting down");
                if let Some(e) = state.embed.as_mut() {
                    e.fail_all("server shutting down");
                }
                save_live_if_dirty(&mut state);
                let _ = ack.send(());
                break;
            }
            None if idle => break, // channel closed
            None => {}
        }
        let llm = sched.has_llm_work();
        // 2. The embed phase (docs/v41/EMBED_PHASE_DESIGN.md §6.2), BETWEEN
        // ticks: an embedding failure never reaches the step-failure path
        // below (it fails only its own requests), and a failed loan return
        // aborts the process inside `run_phase`. Scheduling policy: not in the
        // middle of a layer-major group (the group's box-2 pages, as for
        // decode), and at most V41_EMBED_MAX_SHARE % of wall time while the
        // LLM has work.
        let mid_group = finish_group() && sched.prefills.iter().any(|p| p.job.lm_mid_group());
        if let Some(e) = state.embed.as_mut() {
            if !mid_group && e.due(llm) {
                state.progress.begin();
                worked = true;
                let t = Instant::now();
                e.run_phase(&state.engine, &state.progress, sched.streams.len(), sched.prefills.len());
                // The interrupted burst is not charged for the phase.
                sched.phase_since += t.elapsed();
                state.progress.pet();
                continue;
            }
        }
        if !llm {
            state.progress.end();
            continue;
        }
        state.progress.begin();
        worked = true;
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
    /// Decode steps the device rows cover, dGPU / iGPU: with Tier B on
    /// (`evtrace_dev`) their sums come back 1-2 steps late (plan N3).
    steps_dev: [u64; 2],
    rows: u64,
    wall_ms: f64,
    stages: std::collections::HashMap<(&'static str, &'static str), (f64, u64)>,
}

fn ms_pipeline() -> bool {
    knobs::MS_PIPELINE.on()
}

fn ms_profile() -> bool {
    knobs::MS_PROFILE.on()
}

/// How a decode step's head reaches the sampler (`V41_MS_HEAD_CANDS`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum HeadCands {
    /// `0`: the full logit rows (`head_rows` per lane, 517 KB a row).
    Off,
    /// Default: one head for every lane, then per row only the sampler's
    /// candidates (`HeterogeneousEngine::head_cands`), the full row only for a
    /// row they cannot serve.
    On,
    /// `check`: both, every row compared (`ms.head_cands mismatch` warns); the
    /// full rows are what is sampled.
    Check,
}



/// `V41_MS_HEAD_CANDS` (live knob; `check` -> `1` -> `0` without a restart).
fn head_cands_mode() -> HeadCands {
    match knobs::MS_HEAD_CANDS.pick() {
        1 => HeadCands::Off,
        2 => HeadCands::Check,
        _ => HeadCands::On,
    }
}

/// `V41_MS_SPEC_LANES` (default on; `0` off; live knob).
fn spec_lanes_on() -> bool {
    knobs::MS_SPEC_LANES.on()
}

/// `V41_MS_FINISH_GROUP` (default on; `0` off): a prefill burst whose budget
/// runs out in the middle of a layer-major group keeps the prefill until the
/// group's remaining sub-chunks have run (`PrefillJob::lm_mid_group`). Cutting
/// there let the decode burst evict the group's box-2 pages (decode claims, the
/// box-2 delta restore and the hub's pin restore all take the prefill class
/// first) and the next unit page them all again (SF-B, owner's call 10-01).
fn finish_group() -> bool {
    knobs::MS_FINISH_GROUP.on()
}



/// `V41_MS_PIPELINE_MIN_ROWS` (live knob, default 6; production 4): a step of at least
/// this many rows runs two lanes -- a plain multi-stream step, and a
/// speculating lone stream's verify -- under `LaneRule::Threshold`, i.e.
/// unless the lane choice is learned (`lanes_learned`). Never below 2 (one
/// row cannot be split). The ONE reader of the knob (it was parsed in three
/// places with different clamps); a step reads it once. Why live: at 1-3 rows one lane waits out box 2
/// (14 / 27 / 39 ms per step exposed, MEASURED 2026-10-01 17:03-19:05) where
/// two lanes at 4-5 rows hide most of it (13 / 3 ms).
pub(crate) fn pipeline_min_rows() -> usize {
    knobs::MS_PIPELINE_MIN_ROWS.usize()
}

/// `V41_MS_LANES_LEARNED` (default off; `1` on; live knob). On: every step's
/// lane count is learned per row count
/// (`LaneRule::Learned`, `ms_dspark::LaneTables`) and `pipeline_min_rows` is
/// only the cold start; off: the fixed threshold (owner, 10-01: "if we have
/// cells1 and cells2, why have a fixed min_rows at all?").
fn lanes_learned() -> bool {
    knobs::MS_LANES_LEARNED.on()
}

/// How a speculating lone stream's verify picks its lanes (two = an ordered
/// cut, `forward_step_arena_ready_first`): `Off` when it cannot -- spec lanes
/// off, lanes off, or not the ready-first driver (`V41_MS_STAGGER=2`). Also
/// what the K policy prices a block's rows by. `min_rows` / `learned` /
/// `stagger_mode` = this step's snapshot (one read per step: a second read of
/// `MS_STAGGER` for the driver choice could see a live flip and send an ordered
/// cut to a driver that refuses it).
pub(crate) fn spec_lane_rule(min_rows: usize, learned: bool, stagger_mode: usize) -> LaneRule {
    let ready_first = stagger_mode == 2;
    match (ready_first && ms_pipeline() && spec_lanes_on(), learned) {
        (false, _) => LaneRule::Off,
        (true, true) => LaneRule::Learned,
        (true, false) => LaneRule::Threshold(min_rows.max(2)),
    }
}

/// How a plain multi-stream step picks its lanes: `Off` with lanes off
/// (`V41_MS_PIPELINE=0`).
fn plain_lane_rule(min_rows: usize, learned: bool) -> LaneRule {
    match (ms_pipeline(), learned) {
        (false, _) => LaneRule::Off,
        (true, true) => LaneRule::Learned,
        (true, false) => LaneRule::Threshold(min_rows.max(2)),
    }
}

/// `logits_nucleus_cands` params of a row sampled under `mode`: `[inv_t, lo,
/// band]`, as `TargetDist::from_cands` reads them back.
fn head_cand_params(mode: &SampleMode) -> [f32; 3] {
    match *mode {
        // The maxima only (l >= 0); no survivor total.
        SampleMode::Argmax => [1.0, f32::INFINITY, 0.0],
        SampleMode::Multinomial { temperature, min_p_rel, .. } => {
            [1.0 / temperature, min_p_rel.max(crate::spec_sample::FLOOR).ln(), HEAD_CAND_BAND]
        }
    }
}

/// The sampler's distribution for row `r` of the last `head_cands` (`out`),
/// or `None` when its candidates cannot serve it (the caller reads the row).
fn row_from_cands(out: &[u32], r: usize, mode: &SampleMode) -> Option<TargetDist> {
    let o = &out[r * HEAD_CAND_STRIDE..(r + 1) * HEAD_CAND_STRIDE];
    let n = o[0] as usize;
    if n > HEAD_CAND_CAP {
        return None;
    }
    let cands: Vec<(f32, u32)> = (0..n).map(|k| (f32::from_bits(o[4 + 2 * k]), o[5 + 2 * k])).collect();
    let band = match *mode {
        SampleMode::Argmax => 0.0,
        SampleMode::Multinomial { .. } => HEAD_CAND_BAND,
    };
    TargetDist::from_cands(f32::from_bits(o[1]), f32::from_bits(o[2]), &cands, band, mode)
}

/// What a step's head did (`ms.step` `head_full` / `head_mismatch`).
#[derive(Default, Clone, Copy)]
struct HeadStats {
    /// Rows sampled from their full logit row.
    full: usize,
    /// `V41_MS_HEAD_CANDS=check`: rows whose candidates gave another
    /// distribution than the full row.
    mismatch: usize,
    /// `check`: rows whose merged-head logits differ from the per-lane head's.
    head_diff: usize,
}

/// The step's head: per row (lanes' rows concatenated in `srcs` order) the
/// distribution its stream samples from. Candidates first (`head_cands`,
/// `V41_MS_HEAD_CANDS`), the full row for a row they cannot serve; the full
/// rows of every lane when the candidates are off or the batched head does
/// not apply.
#[allow(clippy::too_many_arguments)]
fn head_targets(
    engine: &v4flash_kernels::het::HeterogeneousEngine,
    head: &mut v4flash_kernels::het::scratch::DgpuScratch,
    srcs: &[(&v4flash_kernels::het::batch_scratch::BatchDgpuScratch, usize)],
    weights: &v4flash_kernels::het::HetModelWeights,
    modes: &[SampleMode],
    params: &[f32],
    out: &mut v4flash_hip::PinnedBuffer<u32>,
    mode: HeadCands,
) -> eyre::Result<(Vec<TargetDist>, HeadStats)> {
    let mut st = HeadStats::default();
    let nv = N_VOCAB as usize;
    // `check`: the per-lane rows first (`head_rows`, today's head), to compare
    // the merged head against them.
    let per_lane: Option<Vec<f32>> = if mode == HeadCands::Check {
        let mut l = Vec::with_capacity(modes.len() * nv);
        for &(bd, n) in srcs {
            if n > 0 {
                l.extend(engine.head_rows(head, bd, n, weights)?);
            }
        }
        Some(l)
    } else {
        None
    };
    if mode != HeadCands::Off && engine.head_cands(head, srcs, weights, params, out)? {
        let mut ts = Vec::with_capacity(modes.len());
        for (r, m) in modes.iter().enumerate() {
            let c = row_from_cands(out.as_slice(), r, m);
            let t = match (mode, c) {
                (HeadCands::On, Some(t)) => t,
                (HeadCands::Check, c) => {
                    let old = &per_lane.as_ref().expect("check mode")[r * nv..(r + 1) * nv];
                    let merged = engine.head_logits_row(head, r)?;
                    if merged.iter().zip(old).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        st.head_diff += 1;
                        tracing::warn!(row = r, "ms.head_cands merged head differs from the per-lane head");
                    }
                    let full = TargetDist::from_logits(old, m);
                    st.full += 1;
                    if let Some(c) = c {
                        if !same_target(&c, &full) {
                            st.mismatch += 1;
                            tracing::warn!(row = r, mode = ?m, cands = ?summary(&c), full = ?summary(&full), "ms.head_cands mismatch");
                        }
                    }
                    full
                }
                (_, _) => {
                    st.full += 1;
                    TargetDist::from_logits(&engine.head_logits_row(head, r)?, m)
                }
            };
            ts.push(t);
        }
        return Ok((ts, st));
    }
    let mut logits = Vec::with_capacity(modes.len() * nv);
    for &(bd, n) in srcs {
        if n > 0 {
            logits.extend(engine.head_rows(head, bd, n, weights)?);
        }
    }
    st.full = modes.len();
    Ok((modes.iter().enumerate().map(|(r, m)| TargetDist::from_logits(&logits[r * nv..(r + 1) * nv], m)).collect(), st))
}

/// The same distribution: support, weights and normaliser bit for bit (the
/// unused `fallback` pick may differ between tied maxima).
fn same_target(a: &TargetDist, b: &TargetDist) -> bool {
    match (a, b) {
        (TargetDist::Argmax(x), TargetDist::Argmax(y)) => x == y,
        (TargetDist::Weighted { ids: i1, w: w1, z: z1, .. }, TargetDist::Weighted { ids: i2, w: w2, z: z2, .. }) => {
            i1 == i2 && z1.to_bits() == z2.to_bits() && w1.len() == w2.len() && w1.iter().zip(w2).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        _ => false,
    }
}

/// `(support size, z)` for a mismatch log line.
fn summary(t: &TargetDist) -> (usize, f32) {
    match t {
        TargetDist::Argmax(a) => (1, *a as f32),
        TargetDist::Weighted { ids, z, .. } => (ids.len(), *z),
    }
}

struct Sched {
    /// DSpark on the arena (`ms_dspark`); `None` = off.
    dsp: Option<MsDspark>,
    /// The plain multi-stream steps' lane choice (`LaneRule::Learned`).
    plain_lanes: PlainLanes,
    /// A live perfetto trace (`V41_PERFETTO_STEPS`).
    trace: LiveTrace,
    profile_acc: ProfileAcc,
    legacy_wait_logged: Option<Instant>,
    /// Lane-B tables for the two-lane step (`V41_MS_PIPELINE`).
    dev_b: RowTablesDev,
    /// Lane-C tables for the three-lane step (`V41_MS_LANES=3`).
    dev_c: RowTablesDev,
    /// Host side of `head_cands` (pinned; `V41_MS_HEAD_CANDS`).
    head_out: v4flash_hip::PinnedBuffer<u32>,
    /// Prefilled requests waiting for arena room (their scratch state stays
    /// parked with them; admission is retried every tick).
    parked: Vec<(Prefill, Vec<f32>)>,
    bounce_f16: v4flash_hip::DeviceBuffer<u16>,
    bounce_u8: v4flash_hip::DeviceBuffer<u8>,
    phase: Phase,
    phase_since: Instant,
    /// This prefill burst ran past its budget to finish a layer-major group
    /// (`finish_group`): since when, and units held. Logged (`hold_ms`,
    /// `hold_units`) on the `ms.phase` line that ends the burst.
    group_hold: Option<(Instant, u32)>,
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

/// A live perfetto trace of N decode steps (`knobs::PERFETTO_STEPS`):
/// attached between steps, detached after the N-th.
#[derive(Default)]
struct LiveTrace {
    /// The knob value last acted on (a new non-zero value starts a trace).
    seen: u64,
    /// Decode steps still to trace (0 = none of ours attached).
    left: u64,
    path: Option<std::path::PathBuf>,
}

/// `YYYYmmdd-HHMMSS` (UTC) of a unix time: trace file names (no date crate).
fn utc_stamp(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

impl Sched {
    /// Start a live trace when `V41_PERFETTO_STEPS` changed to N > 0 (between
    /// decode steps).
    fn live_trace_begin(&mut self, engine: &mut v4flash_kernels::het::HeterogeneousEngine) {
        let want = knobs::PERFETTO_STEPS.get();
        if want == self.trace.seen {
            return;
        }
        self.trace.seen = want;
        if want == 0 {
            return; // re-armed: the same N may be written again
        }
        if self.trace.left > 0 {
            tracing::warn!(left = self.trace.left, "perfetto: a live trace is running; the new request is ignored");
            return;
        }
        if engine.perfetto_attached() {
            tracing::warn!("perfetto: a trace is already attached (V41_PERFETTO_OUT); the live request is ignored");
            return;
        }
        let dir = knobs::PERFETTO_DIR.str().map(std::path::PathBuf::from).unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())).join("traces")
        });
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let path = dir.join(format!("hub-{}-{want}steps.pftrace", utc_stamp(now)));
        let kernels = knobs::PERFETTO_KERNELS.on();
        match std::fs::create_dir_all(&dir).map_err(eyre::Report::from).and_then(|_| engine.attach_perfetto_with(&path, kernels)) {
            Ok(()) => {
                tracing::info!(path = %path.display(), steps = want, kernels, "perfetto: live trace started");
                self.trace.left = want;
                self.trace.path = Some(path);
            }
            Err(e) => tracing::warn!(path = %path.display(), error = %e, "perfetto: live trace not started"),
        }
    }

    /// One traced decode step done; detach after the last.
    fn live_trace_step(&mut self, engine: &mut v4flash_kernels::het::HeterogeneousEngine) {
        if self.trace.left == 0 {
            return;
        }
        self.trace.left -= 1;
        if self.trace.left == 0 {
            let path = self.trace.path.take().map(|p| p.display().to_string()).unwrap_or_default();
            match engine.detach_perfetto() {
                Ok(()) => tracing::info!(path, "perfetto: live trace written"),
                Err(e) => tracing::warn!(path, error = %e, "perfetto: live trace detached with an error"),
            }
        }
    }

    fn enqueue(&mut self, mut req: GenerateReq, tx: mpsc::Sender<WorkerEvent>, session_id: Option<String>, cancel: Arc<AtomicBool>) {
        let prompt_tokens = req.tokens.len() as u32;
        let trailing_marker = req.tokens.last().copied().filter(|&t| t == TOK_THINK_BEGIN || t == TOK_THINK_END);
        if trailing_marker.is_some() {
            req.tokens.truncate(req.tokens.len() - 1);
        }
        self.queue.push_back(Pending { req, tx, session_id, cancel, trailing_marker, prompt_tokens, queued: Instant::now(), room_wait: None });
    }

    /// LLM work waits: live streams, prefills, queued or parked requests.
    fn has_llm_work(&self) -> bool {
        !(self.streams.is_empty() && self.prefills.is_empty() && self.queue.is_empty() && self.parked.is_empty())
    }

    fn abort_all(&mut self, why: &str) {
        for s in self.streams.drain(..) {
            let _ = s.tx.try_send(WorkerEvent::Error(why.to_string()));
            let _ = self.arena.release(s.slot);
        }
        for p in self.prefills.drain(..) {
            let _ = p.p.tx.try_send(WorkerEvent::Error(why.to_string()));
            let _ = self.arena.release(p.slot);
            self.spare_states.push(p.kv);
        }
        for (p, _) in self.parked.drain(..) {
            let _ = p.p.tx.try_send(WorkerEvent::Error(why.to_string()));
            let _ = self.arena.release(p.slot);
            self.spare_states.push(p.kv);
        }
        for p in self.queue.drain(..) {
            let _ = p.tx.try_send(WorkerEvent::Error(why.to_string()));
        }
        // A failed step may have left a grow/compaction half done.
        self.arena.rebuild_free_lists();
    }

    /// One scheduler tick: a prefill chunk (or its start/finish) or a decode step.
    fn tick(&mut self, state: &mut WorkerState) -> eyre::Result<()> {
        // The drafter's async ring writes (`MsDspark::settle_writes`): before any
        // prefill / admission work blocks on the iGPU, so a write's fault is
        // charged to its slot, not to an unrelated request.
        if let Some(dsp) = self.dsp.as_mut() { dsp.settle_writes(); }
        // Under a perfetto trace: the last decode step's leftovers (its ring
        // writes) go out before a prefill unit resets the pools.
        if state.engine.perfetto_attached() {
            if let Err(e) = state.engine.export_pending_perfetto() {
                tracing::warn!(error = %e, "perfetto: export failed");
            }
        }
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
        // Parked (prefilled, but its reservation cannot hold the prompt's first
        // step -- only if a prefill ran past its prompt-length reservation and
        // the arena had no rows to grow it): retry admission (which retries the
        // growth) now that streams may have finished. Oldest first; each is
        // tried once per tick (`try_admit` re-parks it).
        for (pf, logits) in std::mem::take(&mut self.parked) {
            if pf.p.cancel.load(Ordering::Relaxed) || pf.p.tx.is_closed() {
                let _ = self.arena.release(pf.slot);
                self.spare_states.push(pf.kv);
                continue;
            }
            let (tx, slot) = (pf.p.tx.clone(), pf.slot);
            if let Err((kv, e)) = self.try_admit(state, pf, logits) {
                tracing::error!(error = %e, "multistream: parked admission failed");
                let _ = tx.try_send(WorkerEvent::Error(format!("{e:#}")));
                // `Some(kv)`: the error came before the stream owned the slot.
                if let Some(kv) = kv { let _ = self.arena.release(slot); self.spare_states.push(kv); }
            }
        }
        // Start prefills while scratch states are spare. Shortest prompt first
        // (plan 5.3: SJF on the suffix; the prompt length is the proxy we have
        // before the snapshot probe), with aging: a request that has waited
        // longer than `V41_MS_AGING_S` (default 60 s) goes first regardless.
        let aging = std::time::Duration::from_secs(knobs::MS_AGING_S.get());
        let starve = std::time::Duration::from_secs(knobs::MS_STARVE_S.get());
        // Images ride the multistream path since 2026-09-21 (tower rows spliced
        // into the chunk inputs); only legacy DSpark (`V41_DSPARK` without
        // `V41_MS_DSPARK`) still needs the serial driver.
        let mtp_on = state.mtp.is_some() && self.dsp.is_none();
        let is_legacy = move |_p: &Pending| mtp_on;
        // A legacy (vision / DSpark) request can only run on an empty arena. It
        // must NOT block the requests behind it (2026-09-21: one screenshot
        // request held three normal ones for 20+ min while a single stream kept
        // decoding). It steps aside; once it has aged, admissions stop so the
        // arena drains and it gets its turn.
        let legacy_aged = self.queue.iter().any(|p| is_legacy(p) && p.queued.elapsed() >= aging);
        if legacy_aged && !(self.streams.is_empty() && self.prefills.is_empty()) {
            tracing::debug!(queued = self.queue.len(), live = self.streams.len(), "multistream: draining the arena for an aged legacy request");
        }
        let mut deferred: Vec<Pending> = Vec::new();
        while !self.spare_states.is_empty() && !self.queue.is_empty() {
            // Aged requests go first, OLDEST first (ties in the starvation rule
            // below break by request age too); otherwise shortest prompt first.
            if let Some((i, _)) = self.queue.iter().enumerate().filter(|(_, p)| p.queued.elapsed() >= aging).max_by_key(|(_, p)| p.queued.elapsed()) {
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
                if is_legacy(&p) {
                    // Legacy serial path (vision / DSpark): only with an empty arena
                    // and no prefill in flight.
                    if self.streams.is_empty() && self.prefills.is_empty() {
                        if state.state.n_kv_max < state.n_kv_max {
                            // `initialize_state` only stubs the state when DSpark is
                            // off, which is when nothing is legacy: unreachable.
                            let _ = p.tx.try_send(WorkerEvent::Error(format!(
                                "legacy serial path needs the full single-sequence state ({} < {} positions)", state.state.n_kv_max, state.n_kv_max)));
                            continue;
                        }
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
                        for d in deferred.drain(..) { self.queue.push_front(d); }
                        return Ok(());
                    }
                    if !deferred.iter().any(is_legacy) && self.legacy_wait_logged.is_none_or(|t| t.elapsed().as_secs() >= 60) {
                        self.legacy_wait_logged = Some(Instant::now());
                        tracing::info!(queued = self.queue.len(), live = self.streams.len(), waited_s = p.queued.elapsed().as_secs(),
                            "multistream: legacy (DSpark) request waits for an empty arena; others proceed");
                    }
                    deferred.push(p);
                    continue;
                }
                if legacy_aged {
                    // Drain: no new streams until the aged legacy request has run.
                    self.queue.push_front(p);
                    break;
                }
                if self.arena.live() as u32 >= self.arena.n_slots {
                    self.queue.push_front(p);
                    break;
                }
                // Reserve the stream's KV BEFORE its prefill: a request that
                // does not fit waits here holding nothing, instead of being
                // prefilled and then parked with the scratch state (09-27: the
                // one scratch state sat in a parked request while the queue
                // behind it waited, 19-38% of all queue time). The final
                // position is the prompt plus at most the trailing marker.
                // Admission lands at the prompt length, or one past it when a
                // restored snapshot covers the whole prompt and the trailing
                // marker is prefilled: check the context at the LOWER bound
                // (as admission does -- never refuse what it would accept),
                // size the reservation at the upper one.
                let pos_lo = p.req.tokens.len() as u32;
                let pos = pos_lo + u32::from(p.trailing_marker.is_some());
                let max_new = match effective_max_new(&p.req, pos_lo, state.n_kv_max) {
                    Ok(m) => m,
                    Err(e) => { let _ = p.tx.try_send(WorkerEvent::Error(format!("{e:#}"))); continue; }
                };
                let cap = reservation(pos, max_new);
                if !self.arena.could_fit(cap) {
                    // Bigger than the whole arena (V41_MS_CTX_ROWS below the
                    // context): it would wait for room forever.
                    let _ = p.tx.try_send(WorkerEvent::Error(format!(
                        "prompt needs {cap} KV positions, more than the arena holds (V41_MS_CTX_ROWS)")));
                    continue;
                }
                // Stalled streams get freed rows first: no new reservation
                // while one waits for room to grow (a finishing stream's rows
                // would otherwise be re-reserved before it could take them).
                let stalled = self.streams.iter().any(|s| s.stalled_since.is_some());
                let spare = if self.streams.is_empty() || kv_headroom() == 0 { 0 } else { kv_spare() };
                if stalled || !self.arena.fits_with_spare(cap, spare) {
                    let mut p = p;
                    if p.room_wait.is_none() {
                        p.room_wait = Some(Instant::now());
                        tracing::info!(prompt = p.req.tokens.len(), reserve = cap, live = self.streams.len(), prefills = self.prefills.len(),
                            free_rows = self.arena.stores.iter().map(|st| st.free.free_rows() * st.ratio).min().unwrap_or(0),
                            "multistream: request waits for KV room");
                    }
                    if p.queued.elapsed() >= starve {
                        // Starved: stop admitting others until it fits, or
                        // short requests could pass it forever.
                        self.queue.push_front(p);
                        break;
                    }
                    deferred.push(p);
                    continue;
                }
                let reserved = (|| -> eyre::Result<u32> {
                    if !self.arena.fits_now(cap) {
                        let t = Instant::now();
                        self.arena.compact_stores(&state.engine.dgpu.compute, &mut self.bounce_f16, &mut self.bounce_u8)?;
                        tracing::info!(ms = t.elapsed().as_millis() as u64, live = self.streams.len(), "multistream: stores compacted");
                    }
                    self.arena.reserve(cap)
                })();
                let slot = match reserved {
                    Ok(s) => s,
                    Err(e) => {
                        // Device error: the step-failure path aborts everything
                        // in the queue, so put this request (and the ones that
                        // stepped aside) back where it can be told.
                        self.queue.push_front(p);
                        for d in deferred.drain(..).rev() { self.queue.push_front(d); }
                        return Err(e);
                    }
                };
                if let Some(t) = p.room_wait {
                    tracing::info!(slot, reserve = cap, waited_ms = t.elapsed().as_millis() as u64, "multistream: KV room found");
                }
                let kv = self.spare_states.pop().expect("checked");
                match self.start_prefill(state, p, kv, slot) {
                    Ok(pf) => { self.prefills.push(pf); started = true; break; }
                    Err((p, kv, e)) => {
                        let _ = self.arena.release(slot);
                        self.spare_states.push(kv);
                        let _ = p.tx.try_send(WorkerEvent::Error(format!("{e:#}")));
                    }
                }
            }
            if !started { break; }
        }
        for d in deferred.drain(..).rev() { self.queue.push_front(d); }
        // Chunk or step? Bursts with hysteresis: stay in a phase until its
        // budget elapses (V41_MS_PREFILL_BURST_MS / V41_MS_DECODE_BURST_MS,
        // default 4000 each) or it runs out of work.
        let have_pf = !self.prefills.is_empty();
        let have_dec = !self.streams.is_empty();
        // STARVATION RULE (2026-09-21): a request that has not started its
        // prefill within `V41_MS_STARVE_S` (default 600) is prefilled before
        // anything else -- the scheduler stays in the prefill phase, budget or
        // not, until it is admitted; among starved requests the OLDEST goes
        // first (admission order above). Everything else keeps the plain
        // alternating bursts. `V41_MS_BURST_SCALE=1` additionally scales the
        // bursts with the other side's backlog (decode / (1 + waiting
        // prefills), prefill / (1 + live streams), floored).
        // A request waiting for KV ROOM is not helped by prefilling (only
        // decoding frees rows): it holds new admissions in the queue loop
        // instead of forcing the prefill phase.
        let starved = self.queue.iter().any(|p| p.room_wait.is_none() && p.queued.elapsed() >= starve)
            || self.prefills.iter().any(|pf| !pf.job.chunks_done() && pf.p.queued.elapsed() >= starve);
        let scale = knobs::MS_BURST_SCALE.on();
        let waiting_pf = if scale { self.prefills.len() + self.queue.len() } else { 0 };
        let live = if scale { self.streams.len() } else { 0 };
        let budget = |ph: Phase| std::time::Duration::from_millis(match ph {
            Phase::Prefill => (knobs::MS_PREFILL_BURST_MS.usize() / (1 + live))
                .max(knobs::MS_PREFILL_BURST_MIN_MS.usize()) as u64,
            Phase::Decode => (knobs::MS_DECODE_BURST_MS.usize() / (1 + waiting_pf))
                .max(knobs::MS_DECODE_BURST_MIN_MS.usize()) as u64,
        });
        let next = match (have_pf, have_dec) {
            (true, false) => Phase::Prefill,
            (false, true) => Phase::Decode,
            (false, false) => return Ok(()),
            (true, true) => {
                if starved {
                    Phase::Prefill
                } else if self.phase_since.elapsed() >= budget(self.phase) {
                    match self.phase {
                        // Not in the middle of a layer-major group: its next
                        // unit needs the box-2 experts the group's earlier units
                        // just paged (`PrefillJob::lm_mid_group`). At most the
                        // rest of one group's sub-chunks (under one window's
                        // rows through <= 6 layers; the sticky job selection in
                        // `prefill_tick` runs exactly that job's next unit).
                        Phase::Prefill if finish_group() && self.prefills.iter().any(|p| p.job.lm_mid_group()) => {
                            let h = self.group_hold.get_or_insert((Instant::now(), 0));
                            h.1 += 1;
                            Phase::Prefill
                        }
                        Phase::Prefill => Phase::Decode,
                        Phase::Decode => Phase::Prefill,
                    }
                } else {
                    self.phase
                }
            }
        };
        if next != self.phase {
            // Box-2 pinning (`V41_B2_PIN`): a prefill phase gets a band of
            // unpinned slots, and the next decode phase restores what it released.
            // `pin_released`: the band's opening release (-> Prefill), or the
            // phase's per-chunk reopens (-> Decode).
            let (pin_released, pin_restore) = match next {
                Phase::Prefill => {
                    // Predicted-miss hint words (`het::lookahead`, design 2.3)
                    // never ride a prefill chunk: the queue empties here, not in
                    // the per-chunk `expire_incoming` (review round 2, finding 6).
                    v4flash_kernels::het::remote_experts::clear_hint_words();
                    // The SOFT-HELD rows describe the pre-prefill pool from here.
                    v4flash_kernels::het::b2_mirror::clear_soft();
                    (v4flash_kernels::het::b2_mirror::pin_enter_prefill(), 0)
                }
                Phase::Decode => (
                    v4flash_kernels::het::b2_mirror::take_band_reopened(),
                    v4flash_kernels::het::b2_mirror::pin_enter_decode(),
                ),
            };
            tracing::info!(from = ?self.phase, to = ?next, live = self.streams.len(), prefills = self.prefills.len(), queued = self.queue.len(),
                burst_ms = self.phase_since.elapsed().as_millis() as u64, next_budget_ms = budget(next).as_millis() as u64, starved,
                pin_released, pin_restore, hold_units = self.group_hold.map_or(0, |h| h.1),
                hold_ms = self.group_hold.map_or(0, |h| h.0.elapsed().as_millis() as u64), "ms.phase");
            let code = |p: &Phase| match p { Phase::Decode => 0.0, Phase::Prefill => 1.0 };
            v4flash_kernels::het::evtrace::emit(&v4flash_kernels::het::evtrace_kinds::HUB_PHASE, &[
                v4flash_kernels::het::evtrace::now(), code(&self.phase), code(&next), self.streams.len() as f64,
                self.prefills.len() as f64, self.queue.len() as f64, self.phase_since.elapsed().as_secs_f64() * 1e3,
                budget(next).as_secs_f64() * 1e3, f64::from(u8::from(starved)), pin_released as f64, pin_restore as f64,
            ]);
            self.phase = next;
            self.phase_since = Instant::now();
            self.group_hold = None;
        }
        match self.phase {
            Phase::Prefill => {
                // Keep the box-2 prefill band open (`b2_mirror::pin_prefill_tick`).
                v4flash_kernels::het::b2_mirror::pin_prefill_tick();
                self.prefill_tick(state)?
            }
            Phase::Decode => self.decode_step(state)?,
        }
        Ok(())
    }

    /// Probe the snapshot index, restore the longest prefix into the scratch
    /// state, and build the suffix job.
    /// `slot` = the request's arena reservation (the caller releases it if
    /// this fails).
    fn start_prefill(&mut self, state: &mut WorkerState, p: Pending, mut kv: v4flash_kernels::het::HetModelState, slot: u32) -> Result<Prefill, (Pending, v4flash_kernels::het::HetModelState, eyre::Report)> {
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
            // Candidates, longest first (the session hint wins a tie). Each one
            // that fails -- load error or not a token prefix -- falls through
            // to the NEXT instead of to a full prefill: at 180K tokens a failed
            // session hint used to throw away a perfectly good walk match and
            // cost minutes of prefill (enough to trip the client's timeout).
            for (snap_req_tokens, snap_hash, snap_dir) in restore_candidates(hit_session, hit_walk) {
                // Keep >= 1 suffix token to prefill (or a marker to forward).
                let usable = snap_req_tokens >= 64 && (snap_req_tokens < tokens.len() || p.trailing_marker.is_some());
                if !usable {
                    continue;
                }
                // A mid-prefill checkpoint this suffix cannot resume is refused
                // from its meta.json, before the restore reads its blobs; the
                // session hint stops pointing at it (it stays for a longer one).
                let ced = v4flash_kernels::het::forward_prefill::ced_enabled();
                if !snapshot::resume_ok_from_meta(&snap_dir, tokens.len().saturating_sub(snap_req_tokens), ced) {
                    tracing::warn!(covered = snap_req_tokens, total = tokens.len(),
                        "multistream: snapshot is a mid-prefill checkpoint (decoder rings empty) and the suffix is too short to resume it; not restored, trying the next candidate");
                    if let Some(sid) = p.session_id.as_deref() { state.snapshot_index.drop_session_hint(sid, &snap_hash); }
                    continue;
                }
                match snapshot::restore_vl(&mut kv, &snap_dir, state.dgpu, state.igpu, &state.model_fingerprint,
                    snapshot::RestoreKernels { fp8: &state.engine.dgpu.comp_kv_fp8, stream: &state.engine.dgpu.compute }) {
                    Ok(r) => {
                        // Token ids alone are not enough: synthetic image ids
                        // encode only the block layout, so a same-session
                        // request with a different picture of the same size
                        // would match. Verify the byte stream with the image
                        // content hashes folded in, as the serial path does.
                        let is_prefix = r.tokens.len() <= tokens.len()
                            && tokens[..r.tokens.len()] == r.tokens[..]
                            && byte_aligned_lcp_vl(&r.tokens, &r.image_spans, tokens, &p.req.image_spans,
                                state.vocab.as_ref(), &state.byte_decoder).live_tokens == r.tokens.len();
                        // A mid-prefill checkpoint (decoder rings saved empty)
                        // resumes only a suffix longer than the CED replay
                        // (decided above from the meta; this is the backstop).
                        let resumable = snapshot::resume_ok(r.decoder_rings_empty, tokens.len().saturating_sub(r.tokens.len()), ced);
                        if is_prefix && resumable {
                            let _ = state.snapshot_index.touch(&snap_hash);
                            prefix = r.tokens;
                            tracing::info!(restored = prefix.len(), total = tokens.len(), ms = t0.elapsed().as_millis() as u64, "multistream: snapshot restored");
                            break;
                        }
                        if is_prefix {
                            tracing::warn!(restored = r.tokens.len(), total = tokens.len(),
                                "multistream: snapshot is a mid-prefill checkpoint (decoder rings empty) and the suffix is too short to resume it; trying the next candidate");
                            if let Some(sid) = p.session_id.as_deref() { state.snapshot_index.drop_session_hint(sid, &snap_hash); }
                        } else {
                            tracing::warn!("multistream: restored snapshot is not a prefix of the request (tokens or image content); trying the next candidate");
                        }
                        if let Err(e) = kv.reset_in_place(state.dgpu, state.igpu) { return Err((p, kv, e)); }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "multistream: snapshot restore failed; evicting, trying the next candidate");
                        state.snapshot_index.evict(&snap_hash, "multistream: restore failed");
                        if let Err(e) = kv.reset_in_place(state.dgpu, state.igpu) { return Err((p, kv, e)); }
                    }
                }
            }
        }
        kv.restore_compressor_lending();
        let Some((suffix, prefilled_marker)) = plan_suffix(tokens, prefix.len(), p.trailing_marker) else {
            return Err((p, kv, eyre!("multistream: snapshot covered the whole prompt with no marker (unreachable by construction)")));
        };
        let pos0 = prefix.len() as u32;
        // Engram: compress the whole sequence now (cheap); embeddings and
        // Engram rows are produced PER CHUNK in `prefill_job_tick` (lazy job
        // inputs) so a long prompt neither blocks the scheduler for a bulk
        // gather nor holds its whole prompt's inputs in host RAM.
        let mut compressed: Vec<i32> = Vec::with_capacity(tokens.len() + 1);
        if let Some(ec) = state.engram.as_ref() {
            for &t in prefix.iter().chain(suffix.iter()) { compressed.push(ec.hasher.compress(t)); }
        }
        // No image spans, as `prefill_suffix`: V4.1 image tokens attend causally
        // (`model.py` has no `get_image_visible`), so the engine's span rules
        // (bidirectional raw window, cut planning) do not apply. Image rows still
        // get the tower rows (`chunk_inputs`), DEAD Engram ids and `bias_vl`
        // routing off their synthetic ids.
        let mut job = match PrefillJob::new(suffix.clone(), Vec::new(), None, None, pos0, if self.streams.is_empty() { chunk_rows_idle() } else { chunk_rows_busy() }) {
            Ok(j) => j,
            Err(e) => return Err((p, kv, e)),
        };
        // Layer-major or chunked: `PrefillJob::new` reads the live knob
        // `V41_LM_PREFILL` (the 10-01 `V41_MS_LM_FILE` still sets it); a job
        // keeps the mode it started with.
        // Vision: run the tower now (dGPU, between steps) so the chunk inputs
        // can splice the aligned rows in at the image positions.
        let vl = match encode_request_images(state, &p.req) {
            Ok(v) => v,
            Err(e) => return Err((p, kv, e)),
        };
        let mut pf = Prefill { p, slot, job, kv, prefix, compressed, started: t0, vl, prefilled_marker, engram_ahead: EngramAhead::default() };
        if prefilled_marker.is_some() {
            pf.p.trailing_marker = None; // consumed (`admit_stream` reads `prefilled_marker`)
        }
        pf.prefix.extend_from_slice(&suffix);
        Ok(pf)
    }

    /// Run one prefill chunk; on the last one, finish (replay + head), snapshot
    /// the prompt, admit the stream.
    fn prefill_tick(&mut self, state: &mut WorkerState) -> eyre::Result<()> {
        if self.prefills.is_empty() { return Ok(()); }
        // Prefill requests carry no decode step (also after a failed step,
        // whose own clear never ran).
        v4flash_kernels::het::evtrace_kinds::clear_step();
        // A job with an open layer-major window keeps the prefill until the
        // window closes: alternating units between two jobs would make box 2's
        // pool hold both jobs' group unions at once, which is the paging the
        // window exists to avoid. Otherwise round-robin as before.
        let i = match self.prefills.iter().position(|p| !p.job.checkpoint_ok()) {
            Some(j) => j,
            None => {
                let i = self.rr % self.prefills.len();
                self.rr = self.rr.wrapping_add(1);
                i
            }
        };
        let mut pf = self.prefills.remove(i);
        if pf.p.cancel.load(Ordering::Relaxed) || pf.p.tx.is_closed() {
            // CHECKPOINT the partial prefill: a client that times out (the
            // agent's HTTP limit is ~15 min) re-sends the same prompt, and
            // without this the retry started from zero (2026-09-20: a 135K
            // prompt lost 115K prefilled tokens). The encoder state at a chunk
            // boundary is exactly what a resumed prefill restores, so the
            // snapshot key is the request's first `pos0 + done` tokens
            // (`checkpoint_tokens`). The DECODER rings are not: under CED the
            // chunks never touch them, so they are saved EMPTY rather than
            // stale by `done` positions, and a resume must replay past them
            // (`snapshot::resume_ok`).
            let done = pf.job.done_rows();
            // Not mid layer-major window: its early groups hold more rows than
            // its late ones, so no prefix length describes the state (the work
            // since the last closed window is lost, <= V41_LM_ROWS rows).
            let key = (done >= checkpoint_min_rows() && !pf.job.chunks_done() && pf.job.checkpoint_ok())
                .then(|| checkpoint_tokens(&pf.p.req.tokens, pf.job.pos0(), done))
                .flatten();
            if let Some(tokens_saved) = key {
                let t = Instant::now();
                pf.kv.restore_compressor_lending();
                pf.job.clear_decoder_rings_for_checkpoint(&mut pf.kv);
                match checkpoint_spans(&pf.p.req.image_spans, tokens_saved.len()).and_then(|spans_saved| snapshot::save(&pf.kv, &tokens_saved, &spans_saved, state.dgpu, state.igpu, &state.model_fingerprint,
                    state.snapshot_index.root(), state.vocab.as_ref(), &state.byte_decoder, None)) {
                    Ok(entry) => {
                        let hash = entry.hash;
                        state.snapshot_index.insert(entry);
                        if let Some(sid) = pf.p.session_id.clone() { state.snapshot_index.session_to_hash.insert(sid, hash); }
                        tracing::info!(tokens = tokens_saved.len(), done, total = pf.job.total(), ms = t.elapsed().as_millis() as u64, "multistream: prefill cancelled; partial snapshot saved");
                    }
                    Err(e) => tracing::warn!(error = %e, "multistream: prefill cancelled; partial snapshot FAILED"),
                }
            } else {
                tracing::info!(done, total = pf.job.total(), lm_open = !pf.job.checkpoint_ok(), "multistream: prefill cancelled");
            }
            let _ = self.arena.release(pf.slot);
            self.spare_states.push(pf.kv);
            return Ok(());
        }
        // A failure in ONE job (paging, admission, box 2) fails that request
        // only; the live streams keep going. The engine-level drain/redial is
        // still done, since a box-2 fault leaves tickets in flight.
        let (tx, slot) = (pf.p.tx.clone(), pf.slot);
        match self.prefill_job_tick(state, pf, i) {
            Ok(()) => Ok(()),
            Err((kv, e)) => {
                tracing::error!(error = %e, "multistream: prefill failed; failing that request only");
                let _ = tx.try_send(WorkerEvent::Error(format!("{e:#}")));
                // `Some(kv)`: the request still owned its reservation (errors
                // after `admit_stream` took the slot carry `None`).
                if let Some(kv) = kv { let _ = self.arena.release(slot); self.spare_states.push(kv); }
                let _ = state.engine.remote_drain_in_flight();
                let _ = state.engine.remote_reconnect_if_dead();
                Ok(())
            }
        }
    }

    /// One chunk (or the finish + admit) of `pf`. On error returns the scratch
    /// state (if still owned) for recycling.
    fn prefill_job_tick(&mut self, state: &mut WorkerState, mut pf: Prefill, i: usize) -> Result<(), (Option<v4flash_kernels::het::HetModelState>, eyre::Report)> {
        // Every unit of the job (chunks, finish, replay) says whether it is long.
        let long_min = crate::knobs::LONG_PREFILL_TOKENS.get() as usize;
        state.engine.remote_set_long_job(long_min > 0 && pf.job.total() > long_min);
        if !pf.job.chunks_done() {
            let t = Instant::now();
            if let Some(pg) = state.pager.as_mut() {
                if let Err(e) = pg.drain_prefetched() { return Err((Some(pf.kv), e)); }
            }
            if let Err(e) = chunk_inputs(&mut pf, state) { return Err((Some(pf.kv), e)); }
            let inputs_ms = t.elapsed().as_millis() as u64;
            let ahead = engram_lookahead(&pf, state.pager.is_some() && state.engram.is_some());
            let WorkerState { engine, bd_a, bi_a, bd_b, bi_b, sd, si, dgpu_scratch, weights, pager, engram, .. } = state;
            // The next block of Engram rows gathers BESIDE this unit (`EngramAhead`).
            let p0 = pf.job.pos0() as usize;
            let Prefill { job, kv, compressed, .. } = &mut pf;
            let (chunk, gathered) = std::thread::scope(|sc| {
                let g = match (ahead, engram.as_ref()) {
                    (Some((a, z)), Some(ec)) => {
                        let compressed = &*compressed;
                        Some((a, z, sc.spawn(move || gather_prefill_engram(ec, compressed, p0, a, z))))
                    }
                    _ => None,
                };
                let chunk = engine.prefill_job_chunk(job, bd_a, bi_a, bd_b, bi_b, sd, si, dgpu_scratch, kv, weights, pager.as_mut());
                let gathered = g.map(|(a, z, h)| (a, z, h.join().unwrap_or_else(|_| Err(eyre!("engram look-ahead gather panicked")))));
                (chunk, gathered)
            });
            match gathered {
                // A failed look-ahead only costs the overlap: the chunk gathers it.
                Some((a, z, Ok(rows))) => {
                    pf.engram_ahead.blocks.push_back((a, z, rows));
                    pf.engram_ahead.next = z;
                }
                Some((a, z, Err(e))) => tracing::warn!(a, z, error = %e, "multistream: Engram look-ahead gather failed; the chunk gathers it"),
                None => {}
            }
            let rows = match chunk {
                Ok(r) => r,
                Err(e) => return Err((Some(pf.kv), e)),
            };
            tracing::debug!(rows, done = pf.job.done_rows(), total = pf.job.total(), inputs_ms, ms = t.elapsed().as_millis() as u64, "multistream: prefill chunk");
            if !pf.job.chunks_done() {
                // PERIODIC CHECKPOINT (every V41_MS_CHECKPOINT_EVERY rows, default
                // 32768): a server restart does not run the cancel path, so a
                // long prefill used to restart from zero (2026-09-22: a 262K
                // prompt lost 98K rows). The encoder state at a chunk boundary is
                // what a resumed prefill restores; the decoder rings are saved
                // empty (see the cancel checkpoint above).
                let every = knobs::MS_CHECKPOINT_EVERY.usize();
                let done = pf.job.done_rows();
                let key = (every > 0 && done >= every && (done - rows) / every != done / every && pf.job.checkpoint_ok())
                    .then(|| checkpoint_tokens(&pf.p.req.tokens, pf.job.pos0(), done))
                    .flatten();
                if let Some(tokens_saved) = key {
                    let t = Instant::now();
                    pf.kv.restore_compressor_lending();
                    pf.job.clear_decoder_rings_for_checkpoint(&mut pf.kv);
                    match checkpoint_spans(&pf.p.req.image_spans, tokens_saved.len()).and_then(|spans_saved| snapshot::save(&pf.kv, &tokens_saved, &spans_saved, state.dgpu, state.igpu, &state.model_fingerprint,
                        state.snapshot_index.root(), state.vocab.as_ref(), &state.byte_decoder, None)) {
                        Ok(entry) => {
                            state.snapshot_index.insert(entry);
                            tracing::info!(tokens = tokens_saved.len(), done, total = pf.job.total(), ms = t.elapsed().as_millis() as u64, "multistream: prefill checkpoint saved");
                        }
                        Err(e) => tracing::warn!(error = %e, "multistream: prefill checkpoint FAILED"),
                    }
                }
                self.prefills.insert(i.min(self.prefills.len()), pf);
                return Ok(());
            }
        }
        let WorkerState { engine, bd_a, bi_a, bd_b, bi_b, sd, si, dgpu_scratch, weights, pager, mtp, .. } = state;
        let kv = &mut pf.kv;
        // The CED replay runs layers 37-39 over the prompt's last window: it
        // captures exactly the rows a drafter ring holds.
        bd_a.mtp_captured = 0;
        bd_b.mtp_captured = 0;
        let logits = match engine.prefill_job_finish(&mut pf.job, bd_a, bi_a, bd_b, bi_b, sd, si, dgpu_scratch, kv, weights, pager.as_mut()) {
            Ok(l) => l,
            Err(e) => return Err((Some(pf.kv), e)),
        };
        if let (Some(dsp), Some(m)) = (self.dsp.as_mut(), mtp.as_mut()) {
            let t = Instant::now();
            let last = pf.prefix.len() as u32 - 1;
            let seeded = dsp.reset(pf.slot).and_then(|()| {
                let rows = ms_dspark::prefill_captures(&[&*bd_a, &*bd_b])?;
                dsp.seed(engine, m, pf.slot, rows, last)
            });
            match seeded {
                Ok(n) => tracing::info!(slot = pf.slot, seeded = n, last, ms = t.elapsed().as_millis() as u64, "ms dspark: ring seeded"),
                Err(e) => {
                    let _ = dsp.reset(pf.slot);
                    tracing::warn!(slot = pf.slot, error = %e, "ms dspark: ring seeding failed; the stream drafts once it has decoded a row");
                }
            }
        }
        kv.restore_compressor_lending();
        // Snapshot the prompt (the legacy path saves here too, before the marker).
        flush_expert_stats(state);
        if pf.prefilled_marker.is_none() {
            let tokens_saved: Vec<i32> = pf.prefix.clone();
            match checkpoint_spans(&pf.p.req.image_spans, tokens_saved.len()).and_then(|spans_saved| snapshot::save(&pf.kv, &tokens_saved, &spans_saved, state.dgpu, state.igpu, &state.model_fingerprint,
                state.snapshot_index.root(), state.vocab.as_ref(), &state.byte_decoder, pf.p.session_id.as_deref())) {
                Ok(entry) => {
                    let hash = entry.hash;
                    state.snapshot_index.insert(entry);
                    if let Some(sid) = pf.p.session_id.clone() { state.snapshot_index.session_to_hash.insert(sid, hash); }
                }
                Err(e) => tracing::error!(error = %e, "multistream: snapshot.save failed"),
            }
        }
        self.try_admit(state, pf, logits)
    }

    /// Admit a prefilled request into the slot it reserved before its prefill
    /// (`reservation`, grown here if the prompt ended past it). A reservation
    /// that cannot hold even the first step and cannot grow parks the request
    /// (its scratch state stays with it) until a stream finishes.
    fn try_admit(&mut self, state: &mut WorkerState, mut pf: Prefill, logits: Vec<f32>) -> Result<(), (Option<v4flash_kernels::het::HetModelState>, eyre::Report)> {
        let pos = pf.prefix.len() as u32;
        let max_new = match effective_max_new(&pf.p.req, pos, state.n_kv_max) {
            Ok(m) => m,
            Err(e) => return Err((Some(pf.kv), e)),
        };
        pf.p.req.max_new = max_new;
        let slot = pf.slot;
        let want = reservation(pos, max_new);
        let have = self.arena.reserved_positions(slot);
        if have < want {
            match self.arena.grow(slot, want, &state.engine.dgpu.compute, &mut self.bounce_f16, &mut self.bounce_u8) {
                Ok(Some(how)) => tracing::info!(slot, from = have, to = want, how = ?how, "multistream: reservation grown at admission"),
                Ok(None) => {}
                Err(e) => return Err((Some(pf.kv), e)),
            }
        }
        if self.arena.reserved_positions(slot) < pos + 1 {
            tracing::warn!(slot, pos, reserved = self.arena.reserved_positions(slot), live = self.streams.len(), parked = self.parked.len() + 1,
                "multistream: no room; parking the request until a stream finishes");
            self.parked.push((pf, logits));
            return Ok(());
        }
        if let Err(e) = self.arena.fill_reserved(slot, &pf.kv, pos, &state.engine.dgpu.compute) { return Err((Some(pf.kv), e)); }
        if let Err(e) = state.engine.dgpu.compute.synchronize() { return Err((Some(pf.kv), e)); }
        let Prefill { p: pp, slot: _, job, kv: kv_done, prefix, compressed, started, vl: _, prefilled_marker, engram_ahead: _ } = pf;
        self.spare_states.push(kv_done);
        let pf = PrefillDone { p: pp, job, prefix, compressed, started, prefilled_marker };
        self.admit_stream(state, pf, slot, logits).map_err(|e| (None, e))
    }

    fn admit_stream(&mut self, state: &mut WorkerState, pf: PrefillDone, slot: u32, logits: Vec<f32>) -> eyre::Result<()> {
        let top_p = if pf.p.req.top_p.is_finite() && pf.p.req.top_p > 0.0 { pf.p.req.top_p.min(1.0) } else { 1.0 };
        let sample_mode = if pf.p.req.temperature <= 0.0 { SampleMode::Argmax } else {
            SampleMode::Multinomial { temperature: pf.p.req.temperature, min_p_rel: pf.p.req.min_p_rel, top_p }
        };
        let mut rng = SamplerRng::new(pf.p.req.seed);
        let draft_rng = SamplerRng::new(pf.p.req.seed ^ 0xD5A9_C3E1_7B24_6F01);
        let mut s = Stream {
            slot, tx: pf.p.tx.clone(), cancel: pf.p.cancel.clone(), next: 0, seq: pf.prefix.clone(), compressed: pf.compressed.clone(),
            prompt_tokens: pf.p.prompt_tokens, completion_tokens: 0, max_new: pf.p.req.max_new, sample_mode, rng, draft_rng,
            in_think: initial_in_think(pf.p.trailing_marker, pf.prefilled_marker), send_failures: 0, started: pf.started, session_id: pf.p.session_id.clone(),
            ctx_full: pf.prefix.len() as u32 + pf.p.req.max_new as u32 + 2, stalled_since: None,
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
            lm_windows = pf.job.lm_windows_run(), reserved = self.arena.reserved_positions(slot), live = self.streams.len() + 1, "multistream: stream admitted");
        self.streams.push(s);
        Ok(())
    }

    /// One batched decode step over every live stream that has KV room for
    /// its next position (`take_stalled`).
    fn decode_step(&mut self, state: &mut WorkerState) -> eyre::Result<()> {
        let stalled = self.take_stalled(state)?;
        let r = if self.streams.is_empty() {
            // Every stream is stalled behind an in-flight prefill's
            // reservation: run the prefill (it becomes a runnable stream)
            // instead of spinning here until the decode burst ends.
            if !self.prefills.is_empty() && self.phase == Phase::Decode {
                let pin_released = v4flash_kernels::het::b2_mirror::pin_enter_prefill();
                tracing::info!(stalled = stalled.len(), prefills = self.prefills.len(), pin_released, "multistream: every stream stalled; prefill phase");
                self.phase = Phase::Prefill;
                self.phase_since = Instant::now();
            }
            Ok(())
        } else {
            self.decode_rows(state)
        };
        // Back into the live set whatever happened: a failed step's
        // `abort_all` must reach them too.
        self.streams.extend(stalled);
        r
    }

    /// Grow every stream whose reservation is nearly used up (`KvArena::grow`,
    /// `V41_MS_KV_GROW` positions at a time, never past `ctx_full`). A stream
    /// that cannot run its next position and could not grow sits the step out:
    /// it is returned, and `decode_step` puts it back afterwards, so it resumes
    /// once a finishing stream frees rows. If NO stream can run, nothing would
    /// ever finish: the stalled stream with the most completion tokens is
    /// ended with `Length` (logged as an error) and the rest retried.
    fn take_stalled(&mut self, state: &mut WorkerState) -> eyre::Result<Vec<Stream>> {
        let (step, at) = (kv_grow_step(), kv_grow_at());
        loop {
            for s in &mut self.streams {
                let pos = self.arena.stream(s.slot).map(|k| k.pos).unwrap_or(0);
                let have = self.arena.reserved_positions(s.slot);
                if have >= s.ctx_full || have > pos.saturating_add(at) {
                    continue;
                }
                let want = have.saturating_add(step).min(s.ctx_full);
                let t = Instant::now();
                if let Some(how) = self.arena.grow(s.slot, want, &state.engine.dgpu.compute, &mut self.bounce_f16, &mut self.bounce_u8)? {
                    tracing::info!(slot = s.slot, pos, from = have, to = want, how = ?how, ms = t.elapsed().as_millis() as u64,
                        completion_tokens = s.completion_tokens, "multistream: stream reservation grown");
                    if let Some(t0) = s.stalled_since.take() {
                        tracing::info!(slot = s.slot, stalled_ms = t0.elapsed().as_millis() as u64, "multistream: stalled stream resumes");
                    }
                }
            }
            let mut stalled = Vec::new();
            let mut i = 0;
            while i < self.streams.len() {
                if self.arena.can_step(self.streams[i].slot) {
                    i += 1;
                    continue;
                }
                let mut s = self.streams.remove(i);
                if s.stalled_since.is_none() {
                    s.stalled_since = Some(Instant::now());
                    tracing::warn!(slot = s.slot, reserved = self.arena.reserved_positions(s.slot), completion_tokens = s.completion_tokens,
                        live = self.streams.len(), "multistream: stream out of KV room; it waits for a stream to finish");
                }
                stalled.push(s);
            }
            // A prefill in flight holds a reservation that fits its first
            // step: it becomes a stream that runs, finishes and frees rows,
            // so that is not a deadlock. (A PARKED request cannot run.)
            if !self.streams.is_empty() || stalled.is_empty() || !self.prefills.is_empty() {
                return Ok(stalled);
            }
            let k = (0..stalled.len()).max_by_key(|&k| stalled[k].completion_tokens).expect("non-empty");
            let s = stalled.remove(k);
            tracing::error!(slot = s.slot, completion_tokens = s.completion_tokens, stalled = stalled.len() + 1,
                "multistream: EVERY stream is out of KV room; ending the longest with finish=length");
            self.streams.append(&mut stalled);
            finish(state, &mut self.arena, s, FinishReason::Length)?;
        }
    }

    /// The batched step over `self.streams` (all of them can step).
    fn decode_rows(&mut self, state: &mut WorkerState) -> eyre::Result<()> {
        // This step's id FIRST: the drafter below records device stages with
        // it, and Tier B sums every pair by its own step (`hub_step.step`;
        // docs/v41/EVTRACE_REBUILD_PLAN.md section 6 S1).
        static EV_STEP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let step_id = EV_STEP.fetch_add(1, Ordering::Relaxed);
        let _step_ctx = v4flash_kernels::het::trace::ctx_step(step_id);
        // Hot-set ownership refresh (see expert_pager::hot_set); `tick` is
        // advanced once per scheduler tick.
        if self.tick % knobs::B1_HOT_REFRESH.get() == 0 {
            use v4flash_kernels::het::expert_pager::hot_set;
            if let Some((owned, mass, changed)) = hot_set::refresh() {
                if hot_set::interleave_active() || hot_set::returning() {
                    // HOT SPLIT (docs/v41/HOT_SPLIT_DESIGN.md): the placement's
                    // own numbers, and the holder / pre-warm / KEEP totals.
                    let (share1, swaps, b1_new, b2_new, keep) = hot_set::il_stats();
                    let (holder_b1, holder_b2) = hot_set::holder_totals();
                    tracing::info!(owned, per_layer = owned / v4flash_kernels::config::N_LAYER as usize, mass = format!("{mass:.3}"), changed,
                        picks = hot_set::picks_seen(), interleave = hot_set::interleave_active(), returning = hot_set::returning(),
                        share1 = format!("{share1:.3}"), swaps, b1_new, b2_new, keep, holder_b1, holder_b2,
                        prewarm_queued = hot_set::prewarm_queued(), keep_queued = v4flash_kernels::het::b2_mirror::keep_queued_total(),
                        "multistream: box-1 hot set refreshed");
                } else {
                    tracing::info!(owned, per_layer = owned / v4flash_kernels::config::N_LAYER as usize, mass = format!("{mass:.3}"), changed,
                        picks = hot_set::picks_seen(), "multistream: box-1 hot set refreshed");
                }
            }
        }
        // Token boundary: nothing is reading the pool (the previous step and
        // chunk both synchronized), so admit box-1's background-read experts
        // (`V41_B1_PREFETCH`, catch-all mode: misses are computed on box 2 and
        // read from box 1's disk off the critical path). Same as decode's
        // `forward_one!`; without this the multistream path never warmed box 1.
        // The drafter's async ring writes first: the pager's blocking copies
        // would otherwise return a write's fault as their own (`settle_writes`).
        // (The wait is what the step's host tail did not hide of the last write.)
        let ring_settle_ms = self.dsp.as_mut().map(|d| d.settle_writes()).unwrap_or(0.0);
        if let Some(pg) = state.pager.as_mut() {
            pg.drain_prefetched()?;
            // HOT SPLIT pre-warm (design 5): box-1 newcomers of the last
            // refreshes, a few per step (empty outside an interleave / return).
            let n = v4flash_kernels::knobs::B1_HOT_PREWARM_STEP.usize();
            if n > 0 {
                let batch = v4flash_kernels::het::expert_pager::hot_set::take_prewarm(n);
                for (i, &(l, e)) in batch.iter().enumerate() {
                    if pg.prefetch_now(l, e).is_none() {
                        // The in-flight cap is full: this one and the rest back to
                        // the front (in order), retry next step.
                        for &(l2, e2) in batch[i..].iter().rev() {
                            v4flash_kernels::het::expert_pager::hot_set::requeue_prewarm(l2, e2);
                        }
                        break;
                    }
                }
            }
        }
        // Perfetto (`V41_PERFETTO_STEPS` live, or `V41_PERFETTO_OUT`): the
        // previous step's leftovers (the drafter's ring writes, recorded after
        // its export) go out before this step's pool reset.
        self.live_trace_begin(&mut state.engine);
        let pf_on = state.engine.perfetto_attached();
        if pf_on {
            if let Err(e) = state.engine.export_pending_perfetto() {
                tracing::warn!(error = %e, "perfetto: export failed");
            }
        }
        // DSpark (`V41_MS_DSPARK`): a LONE stream drafts from its last row in KV
        // and verifies K of the drafts in this step's rows, K from the drafter's
        // confidence (`ms_dspark::choose_k`). Several live streams step plainly.
        // A drafter failure only costs the drafts (plan 5.8), never the step.
        let mut drafts: Vec<Vec<i32>> = vec![Vec::new(); self.streams.len()];
        // Sampled drafts: each draft's q (plan 2.2); `None` = point mass.
        let mut draft_q: Vec<Option<Vec<Vec<(i32, f64)>>>> = vec![None; self.streams.len()];
        // The lone stream's confidence logits (calibration, `MsDspark::record`).
        let mut draft_conf = [0f32; v4flash_kernels::het::mtp::MTP_BLOCK];
        // The lane rules, ONE snapshot per step (the run-time files can change
        // between reads): the K decision prices blocks by the spec rule and
        // returns the block's lanes (docs/v41/DSPARK_SINGLE_STREAM_PERF.md
        // section 2); the plain lane choice below uses the plain rule.
        let min_rows = pipeline_min_rows();
        let learned = lanes_learned();
        let stagger_mode = knobs::MS_STAGGER.pick();
        let spec_rule = spec_lane_rule(min_rows, learned, stagger_mode);
        // The drafted block's lane count (`MsDspark::k_for`).
        let mut spec_two = false;
        // The lone stream drafted and `k_for` gave K = 0 (`MsDspark::record_k0`).
        let mut drafted_k0 = false;
        // Two or more live streams drafted: (stream index, confidence logits)
        // of each that did, whatever its K (`MsDspark::record_multi`).
        let mut multi_drafted: Vec<(usize, [f32; v4flash_kernels::het::mtp::MTP_BLOCK])> = Vec::new();
        // How many live streams may all draft (`V41_MS_DSPARK_STREAMS`, one read per step).
        let spec_max = ms_dspark::spec_streams();
        // Every draft attempt of this step (failed, empty and K = 0 ones
        // included): `step_ms` starts after it, so `ms.step` logs it apart
        // (`draft_ms`) and a step's full cost is their sum (review round 1).
        let t_draft = Instant::now();
        if self.streams.len() == 1 {
            if let (Some(dsp), Some(m)) = (self.dsp.as_mut(), state.mtp.as_mut()) {
                let u: [f32; v4flash_kernels::het::mtp::MTP_BLOCK] = std::array::from_fn(|_| self.streams[0].draft_rng.next_f32());
                let s = &self.streams[0];
                let sampling = ms_dspark::draft_sampling(&s.sample_mode, u);
                let pos = self.arena.stream(s.slot).map(|k| k.pos).unwrap_or(0);
                let remaining = s.max_new.saturating_sub(s.completion_tokens as usize);
                let mut cap = remaining.saturating_sub(1).min(ms_dspark::k_max());
                while cap > 0 && !self.arena.can_step_rows(s.slot, 1 + cap as u32) {
                    cap -= 1;
                }
                if cap > 0 && pos > 0 && dsp.should_draft(s.slot, false) {
                    let mut row = vec![0f32; HC_DIM as usize];
                    embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, s.next, &mut row);
                    match dsp.draft(&state.engine, &state.weights, m, s.slot, pos - 1, s.next, &row, sampling.as_ref()) {
                        Ok(Some(d)) => {
                            // Sampled drafts need the stopping rule; point-mass
                            // tests allow the global search (plan 2.4).
                            let (k, lanes) = dsp.k_for(&d.conf, cap, d.q.is_some(), spec_rule);
                            spec_two = lanes >= 2;
                            drafted_k0 = k == 0;
                            draft_conf = d.conf;
                            drafts[0] = d.ids[..k].to_vec();
                            draft_q[0] = d.q.map(|mut q| { q.truncate(k); q });
                        }
                        Ok(None) => {}
                        Err(e) => {
                            // The ring may be part-written: restart it (the next
                            // kept rows reseed it) rather than draft from it.
                            tracing::warn!(slot = s.slot, error = %e, "ms dspark: draft failed; plain step, ring restarts");
                            let _ = dsp.reset(s.slot);
                        }
                    }
                }
            }
        } else if self.streams.len() <= spec_max {
            // Two or more live streams, all of them may draft
            // (`V41_MS_DSPARK_STREAMS`, docs/v41/MS_DSPARK_STREAMS_DESIGN.md 2):
            // each drafts as the lone stream does (its own gate, KV cap and
            // draft RNG; drafts run one after the other), then ONE joint policy
            // (`MsDspark::ks_for`) picks every stream's K and the lanes.
            if let (Some(dsp), Some(m)) = (self.dsp.as_mut(), state.mtp.as_mut()) {
                let mut blocks: Vec<ms_dspark::BlockConf> = Vec::new();
                let mut got: Vec<(usize, ms_dspark::Drafted)> = Vec::new();
                for (i, s) in self.streams.iter_mut().enumerate() {
                    let u: [f32; v4flash_kernels::het::mtp::MTP_BLOCK] = std::array::from_fn(|_| s.draft_rng.next_f32());
                    let sampling = ms_dspark::draft_sampling(&s.sample_mode, u);
                    let pos = self.arena.stream(s.slot).map(|k| k.pos).unwrap_or(0);
                    let remaining = s.max_new.saturating_sub(s.completion_tokens as usize);
                    let mut cap = remaining.saturating_sub(1).min(ms_dspark::k_max());
                    while cap > 0 && !self.arena.can_step_rows(s.slot, 1 + cap as u32) {
                        cap -= 1;
                    }
                    if cap == 0 || pos == 0 || !dsp.should_draft(s.slot, true) {
                        continue;
                    }
                    let mut row = vec![0f32; HC_DIM as usize];
                    embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, s.next, &mut row);
                    match dsp.draft(&state.engine, &state.weights, m, s.slot, pos - 1, s.next, &row, sampling.as_ref()) {
                        Ok(Some(d)) => {
                            blocks.push(ms_dspark::BlockConf { conf: d.conf, cap });
                            got.push((i, d));
                        }
                        Ok(None) => {}
                        Err(e) => {
                            tracing::warn!(slot = s.slot, error = %e, "ms dspark: draft failed; plain row, ring restarts");
                            let _ = dsp.reset(s.slot);
                        }
                    }
                }
                if !blocks.is_empty() {
                    let sampled = got.iter().any(|(_, d)| d.q.is_some());
                    let (ks, lanes) = dsp.ks_for(&blocks, self.streams.len(), sampled, spec_rule);
                    spec_two = lanes >= 2;
                    for ((i, d), k) in got.into_iter().zip(ks) {
                        drafts[i] = d.ids[..k].to_vec();
                        draft_q[i] = d.q.map(|mut q| { q.truncate(k); q });
                        multi_drafted.push((i, d.conf));
                    }
                }
            }
        }
        let draft_wall_ms = t_draft.elapsed().as_secs_f64() * 1e3;
        let spec = drafts.iter().any(|d| !d.is_empty());
        // `V41_SUB_DEFER_ACCEPTED`: a lone stream's speculative block records its
        // cache-prior admissions / pin wants / hot-set picks by row position and
        // applies the KEPT rows' after the accept (`b2_mirror::defer_flush`
        // below). The guard turns recording off however this step ends, so an
        // error can never leave a later step (or a prefill) recording.
        let defer = spec && self.streams.len() == 1 && v4flash_kernels::het::b2_mirror::defer_accepted();
        v4flash_kernels::het::b2_mirror::defer_step(defer);
        struct DeferGuard;
        impl Drop for DeferGuard {
            fn drop(&mut self) {
                v4flash_kernels::het::b2_mirror::defer_step(false);
            }
        }
        let _defer_guard = DeferGuard;
        // Rows: per stream its next token (a root), then its draft rows, each
        // continuing the one before it (`StepRows::chains`: positions pos+1..).
        // `step_rows` is the step's ONE description of its rows: the drivers'
        // tables and lane ordering read it, and every per-row array below is
        // indexed through it (`row0` = each stream's root). Blocks go in
        // DESCENDING row count (stable: plain and lone steps keep the stream
        // order), so the balanced lane cut falls between two streams' blocks
        // whenever their K differ by at most one (MS_DSPARK_STREAMS_DESIGN 2.4).
        let order: Vec<usize> = {
            let mut o: Vec<usize> = (0..self.streams.len()).collect();
            o.sort_by_key(|&i| std::cmp::Reverse(drafts[i].len()));
            o
        };
        let step_rows = StepRows::chains(&order.iter().map(|&i| (self.streams[i].slot, drafts[i].len())).collect::<Vec<_>>())?;
        // The per-row arrays below are filled root + offset: a chain layout.
        if !step_rows.is_chain_layout() {
            return Err(eyre!("ms.step: the step's rows are not a chain layout (a StepRows bug)"));
        }
        let row0: Vec<usize> = self
            .streams
            .iter()
            .map(|s| step_rows.root_of(s.slot).ok_or_else(|| eyre!("ms.step: stream slot {} has no root row", s.slot)))
            .collect::<eyre::Result<_>>()?;
        let mut toks: Vec<i32> = vec![0; step_rows.len()];
        for ((s, d), &r) in self.streams.iter().zip(&drafts).zip(&row0) {
            toks[r] = s.next;
            toks[r + 1..=r + d.len()].copy_from_slice(d);
        }
        let t0 = Instant::now();
        let b = step_rows.len();
        // Event trace: one `hub_step` per step, fields filled BY NAME (see
        // `evtrace_kinds::HUB_STEP`); `hub_req` records carry this step number.
        let ev_on = v4flash_kernels::het::evtrace::enabled();
        let mut ev: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        if ev_on {
            let step = step_id;
            v4flash_kernels::het::evtrace_kinds::set_step(step, b as u64);
            ev.insert("t_start".into(), v4flash_kernels::het::evtrace::now());
            ev.insert("step".into(), step as f64);
            ev.insert("rows".into(), b as f64);
            let pos = self.streams.iter().map(|s| s.seq.len() as f64);
            ev.insert("pos_min".into(), pos.clone().fold(f64::INFINITY, f64::min));
            ev.insert("pos_max".into(), pos.fold(f64::NEG_INFINITY, f64::max));
        }
        let mut hcs: Vec<Vec<f32>> = Vec::with_capacity(b);
        for &t in &toks {
            let mut v = vec![0f32; HC_DIM as usize];
            embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, t, &mut v);
            hcs.push(v);
        }
        // Engram rows: one hash per row from its own stream's sequence. The
        // hashing stays here (cheap); the SSD gather (13.8 ms/step at 8 rows,
        // 5-26, profile audit 2026-09-21) runs on a helper thread under layer 0
        // and is joined at the first Engram layer (`LazyEngramRows`); the
        // exposed remainder shows as `lh.engram_join`.
        let ein = ENGRAM_IN as usize;
        let t_eng = Instant::now();
        let engram_on = state.pager.is_some() && state.engram.is_some();
        let mut live: Vec<(usize, [[i64; v4flash_core::engram_hash::ENGRAM_COLS]; v4flash_core::engram_hash::ENGRAM_LAYERS])> = Vec::with_capacity(b);
        if let Some(ec) = state.engram.as_ref().filter(|_| engram_on) {
            for ((s, d), &r) in self.streams.iter().zip(&drafts).zip(&row0) {
                // `seq` already ends with `next` (pushed when it was chosen).
                let pos = s.seq.len() - 1;
                if d.is_empty() {
                    if s.compressed[pos] == v4flash_core::engram_hash::DEAD { continue; }
                    live.push((r, ec.hasher.hash_ids(&s.compressed, pos)));
                    continue;
                }
                // Draft rows hash the sequence as the drafts extend it.
                let mut ext = s.compressed.clone();
                ext.extend(d.iter().map(|&t| ec.hasher.compress(t)));
                for j in 0..=d.len() {
                    if ext[pos + j] == v4flash_core::engram_hash::DEAD { continue; }
                    live.push((r + j, ec.hasher.hash_ids(&ext, pos + j)));
                }
            }
        }
        let engram_ms = t_eng.elapsed().as_secs_f64() * 1e3;
        let WorkerState { engine, bd_a, bi_a, bd_b, bi_b, bd_c, bi_c, sd, si, dgpu_scratch, weights, pager, engram, .. } = state;
        // V41_MS_PROFILE=1: per-stage GPU busy time of the batched step (HIP
        // events per stage, ~100 us/layer), rolled up over V41_MS_PROFILE_EVERY
        // steps and logged as "ms.stage". Wall - busy = host / link / sync.
        // `V41_MS_PROFILE_SAMPLE`: one decode step in N (every step under Tier B, whose sums come
        // back for the steps they cover). An unprofiled step records no timing events and runs no
        // `lh.*` timers: back to the construction-time flags unless a perfetto trace owns the pools.
        let profile = ms_profile()
            && (step_id % knobs::MS_PROFILE_SAMPLE.get().max(1) == 0 || engine.dgpu.events.offloading());
        if profile {
            v4flash_kernels::het::forward_prefill::LH_FORCE.store(true, Ordering::Relaxed);
            let _ = v4flash_kernels::het::forward_prefill::take_layer_host_timing();
            engine.dgpu.events.set_enabled(true);
            engine.igpu.events.set_enabled(true);
        } else if v4flash_kernels::het::forward_prefill::LH_FORCE.swap(false, Ordering::Relaxed) && !pf_on {
            engine.dgpu.events.restore_defaults();
            engine.igpu.events.restore_defaults();
        }
        // Under a perfetto trace too: the pools hold 16384 events and a full
        // one fails the step (everything before this point is exported).
        if profile || pf_on {
            engine.dgpu.events.reset();
            engine.igpu.events.reset();
        }
        // Tier B held every buffer at a reset since the last step: those
        // epochs (this step's included) recorded no device stages (plan N4).
        let dev_skipped = engine.dgpu.events.take_skipped() + engine.igpu.events.take_skipped();
        if ev_on && engine.dgpu.events.offloading() {
            ev.insert("dev_skipped".into(), dev_skipped as f64);
        }
        if profile {
            v4flash_kernels::het::trace::phase::reset();
        }
        let t_fwd = Instant::now();
        // Two-lane pipelined step (default for >= 2 rows, `V41_MS_PIPELINE=0`
        // disables): lane B's layer runs under lane A's box-2 wait.
        // MEASURED 2026-09-22 with partitioned paging: at 2 rows the two-lane
        // step's doubled dGPU chain (+55 ms) exceeds the box-2 wait it hides
        // (~30 ms), at 4 it is a wash; default to lanes from 6 rows.
        // A speculating stream (only a lone stream speculates) runs two lanes
        // only as an ORDERED cut through its rows: the ready-first driver, whose
        // later lane enters each layer after the earlier one (`spec_lane_rule`,
        // docs/v41/DSPARK_SINGLE_STREAM_PERF.md); the other drivers refuse it.
        // One lane takes turns between the dGPU attention and the MoE legs; two
        // overlap them (09-30: 2->3 rows +19.5 ms on one lane, 3->4 +5.7 on two).
        // Both lane counts come from a rule (`LaneRule`): the fixed threshold, or
        // learned per row count (`lanes_learned`; a plain step's from its own
        // tables, `PlainLanes`, a block's from `k_for`).
        let spec_lanes = spec && spec_two;
        // Rows that would run THREE lanes keep the threshold even when learned:
        // a three-lane step feeds no table, so a learned choice there would
        // compare a live one-lane cell with a frozen two-lane one (review 12).
        let lanes3_rows = knobs::MS_LANES.get() >= 3 && b >= knobs::MS_LANES3_MIN_ROWS.usize() && b >= 3;
        let plain_rule = plain_lane_rule(min_rows, learned && !lanes3_rows);
        let plain_two = !spec && b >= 2 && self.plain_lanes.pick(b, plain_rule) >= 2;
        let pipelined = plain_two || spec_lanes;
        // Three lanes (`V41_MS_LANES=3`, DEFAULT 2) from `V41_MS_LANES3_MIN_ROWS`
        // rows (default 6). MEASURED 2026-09-21 at 8 rows, box-1 hot set warm:
        // 2 lanes 272 ms/step (27.2 tok/s), 3 lanes 324 (23.3). The third lane
        // does keep a request queued on box 2 (its exposed wait fell 131 -> 24
        // ms while box 1 was still paging its hash share), but every lane
        // splits the rows further, so each request shares fewer expert reads:
        // box-2 compute 94 -> 135 ms, dGPU 91 -> 126, box-1 iGPU 72 -> 93 per
        // step, and box 2 -- the saturated resource -- ends up busier, not
        // idler. Lanes cost bytes; only worth it when the pole has slack.
        let lanes3 = pipelined && !spec && lanes3_rows;
        // STAGGERED two lanes (`V41_MS_STAGGER=1`, default OFF until A/B'd on real
        // traffic; gated bit-exact by `multistream_step` G5d). The lockstep driver
        // above fuses both lanes into ONE box-2 pass per layer, so box 1 and box 2
        // take turns: the daemon logs `queued 0%` and the step costs box1 + box2
        // (2026-09-22: box 2 idle ~37%, box 1 in `lh.remote_wait` ~45%). The
        // N-lane driver at n=2 offsets the lanes by half a layer instead -- lane
        // A's next request is out while box 1 works on lane B -- with the SAME
        // row split, so unlike three lanes it costs no extra expert reads; it
        // gives up the daemon's same-layer merge in exchange for overlap.
        // `V41_MS_STAGGER=2`: the READY-FIRST variant of the same driver --
        // the host runs whichever lane's next step is ready instead of a fixed
        // round robin (it was blocking 67 ms/step on its own dGPU router while
        // 52 of 80 box-2 replies sat ready). Gated by multistream_step G5e.
        // `stagger_mode`: the snapshot read with the spec rule above.
        let stagger2 = pipelined && !lanes3 && stagger_mode >= 1;
        let ready_first = stagger_mode == 2;
        let mut fwd_only_ms = 0.0f64;
        let n_tables = engram.as_ref().map(|ec| ec.tables.len()).unwrap_or(0);
        if self.dsp.is_some() {
            bd_a.mtp_captured = 0;
            bd_b.mtp_captured = 0;
            bd_c.mtp_captured = 0;
        }
        // Each row's sampling mode (a speculating stream's draft rows share
        // its stream's) and its `head_cands` params.
        let mut row_modes: Vec<SampleMode> = vec![SampleMode::Argmax; b];
        for ((s, d), &r) in self.streams.iter().zip(&drafts).zip(&row0) {
            row_modes[r..=r + d.len()].fill(s.sample_mode);
        }
        if row_modes.len() != b {
            return Err(eyre!("ms.step: {} row modes for {b} rows", row_modes.len()));
        }
        let cand_params: Vec<f32> = row_modes.iter().flat_map(head_cand_params).collect();
        let hc_mode = head_cands_mode();
        let (targets, head_stats) = std::thread::scope(|sc| {
        let mut engram_rows = if engram_on && !live.is_empty() {
            let ec: &crate::engine_worker::EngramCtx = engram.as_ref().expect("engram_on");
            let live = &live;
            // One gather per table, each joined at its own layer (dGPU bundle slice 1).
            ENGRAM_GATHER_US.store(0, Ordering::Relaxed);
            LazyEngramRows::pending_tables((0..n_tables).map(|li| sc.spawn(move || gather_engram_table(ec, live, b, li))).collect())
        } else if engram_on {
            // Every live row is DEAD (image rows): zero rows, no reads.
            LazyEngramRows::ready(Some(vec![vec![0f32; b * ein]; n_tables]))
        } else {
            LazyEngramRows::ready(None)
        };
        let targets = if lanes3 {
            {
                let mut lanes: [(&mut v4flash_kernels::het::batch_scratch::BatchDgpuScratch, &mut v4flash_kernels::het::batch_scratch::BatchIgpuScratch, &mut RowTablesDev); 3] =
                    [(&mut *bd_a, &mut *bi_a, &mut self.dev), (&mut *bd_b, &mut *bi_b, &mut self.dev_b), (&mut *bd_c, &mut *bi_c, &mut self.dev_c)];
                engine.forward_step_arena_lanes(&mut lanes, sd, si, &mut self.arena, &step_rows, weights, &hcs, &toks, &mut engram_rows, pager.as_mut())?;
            }
            fwd_only_ms = t_fwd.elapsed().as_secs_f64() * 1e3;
            let sz = |i: usize| lane_rows(b, 3)[i];
            head_targets(engine, dgpu_scratch, &[(&*bd_a, sz(0)), (&*bd_b, sz(1)), (&*bd_c, sz(2))], weights, &row_modes, &cand_params, &mut self.head_out, hc_mode)?
        } else if stagger2 {
            {
                let mut lanes: [(&mut v4flash_kernels::het::batch_scratch::BatchDgpuScratch, &mut v4flash_kernels::het::batch_scratch::BatchIgpuScratch, &mut RowTablesDev); 2] =
                    [(&mut *bd_a, &mut *bi_a, &mut self.dev), (&mut *bd_b, &mut *bi_b, &mut self.dev_b)];
                if ready_first {
                    engine.forward_step_arena_ready_first(&mut lanes, sd, si, &mut self.arena, &step_rows, weights, &hcs, &toks, &mut engram_rows, pager.as_mut())?;
                } else {
                    engine.forward_step_arena_lanes(&mut lanes, sd, si, &mut self.arena, &step_rows, weights, &hcs, &toks, &mut engram_rows, pager.as_mut())?;
                }
            }
            fwd_only_ms = t_fwd.elapsed().as_secs_f64() * 1e3;
            // Same split as the lanes driver: the first lane takes the odd row.
            let b_a = lane_rows(b, 2)[0];
            head_targets(engine, dgpu_scratch, &[(&*bd_a, b_a), (&*bd_b, b - b_a)], weights, &row_modes, &cand_params, &mut self.head_out, hc_mode)?
        } else if pipelined {
            engine.forward_step_arena_pipelined(bd_a, bi_a, bd_b, bi_b, sd, si, &mut self.arena, &mut self.dev, &mut self.dev_b, &step_rows, weights, &hcs, &toks, &mut engram_rows, pager.as_mut())?;
            fwd_only_ms = t_fwd.elapsed().as_secs_f64() * 1e3;
            let b_a = lane_rows(b, 2)[0];
            head_targets(engine, dgpu_scratch, &[(&*bd_a, b_a), (&*bd_b, b - b_a)], weights, &row_modes, &cand_params, &mut self.head_out, hc_mode)?
        } else {
            engine.forward_step_arena(bd_a, bi_a, sd, si, &mut self.arena, &mut self.dev, &step_rows, weights, &hcs, &toks, &mut engram_rows, pager.as_mut())?;
            fwd_only_ms = t_fwd.elapsed().as_secs_f64() * 1e3;
            head_targets(engine, dgpu_scratch, &[(&*bd_a, b)], weights, &row_modes, &cand_params, &mut self.head_out, hc_mode)?
        };
        Ok::<_, eyre::Report>(targets)
        })?;
        let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1e3;
        // The forward's final dGPU sync returned (causality rule (b), N1).
        let t_fwd_sync = v4flash_kernels::het::forward_prefill::take_fwd_sync();
        if ev_on && t_fwd_sync.is_finite() {
            ev.insert("t_fwd_sync".into(), t_fwd_sync);
        }
        // This step's device stages (perfetto), then the live trace's count.
        if pf_on {
            if let Err(e) = engine.export_step_perfetto() {
                tracing::warn!(error = %e, "perfetto: step export failed");
            }
            self.live_trace_step(engine);
        }
        // Ordered two-lane verify: how often / how long the later lane waited
        // to enter a layer behind the earlier one (`Ph::Chain`).
        let (chain_waits, chain_wait_us) = v4flash_kernels::het::forward_prefill::take_chain_waits();
        // The drafter's input for every row (lane-local captures, in row order).
        // A failed readback costs the drafts (no ring rows, no next draft), never the step.
        let caps: Option<Vec<Vec<f32>>> = if self.dsp.is_some() {
            let sizes: Vec<usize> = if lanes3 {
                lane_rows(b, 3)
            } else if stagger2 || pipelined {
                lane_rows(b, 2)
            } else {
                vec![b]
            };
            let lanes: [&v4flash_kernels::het::batch_scratch::BatchDgpuScratch; 3] = [&*bd_a, &*bd_b, &*bd_c];
            let mut v = Vec::with_capacity(b);
            let mut r = Ok(());
            for (bd, n) in lanes.iter().zip(sizes) {
                match ms_dspark::lane_captures(bd, n) {
                    Ok(c) => v.extend(c),
                    Err(e) => { r = Err(e); break; }
                }
            }
            match r {
                Ok(()) if v.len() == b => Some(v),
                Ok(()) => { tracing::warn!(rows = b, captured = v.len(), "ms dspark: captures do not cover the step"); None }
                Err(e) => { tracing::warn!(error = %e, "ms dspark: capture readback failed"); None }
            }
        } else {
            None
        };
        // Box-2 pinning (`V41_B2_PIN`), drained EVERY step so `hub_step`
        // carries them with or without the profile: `b2_surprises` must stay
        // 0 -- a held pick box 2 paged anyway (ERROR-logged where it happens).
        let pin_stats = v4flash_kernels::het::b2_mirror::take_pin_stats();
        if let (true, Some([sur, held, rel, pinned, budget, unused])) = (ev_on, pin_stats) {
            for (k, v) in [("b2_surprises", sur), ("b2_held_picks", held), ("b2_pin_released", rel), ("b2_pinned", pinned), ("b2_pin_budget", budget), ("b2_pin_released_unused", unused)] {
                ev.insert(k.into(), v);
            }
        }
        if profile {
            use v4flash_kernels::het::trace::{phase as counters, rollup_by_name, EventPool};
            // Tier B on (`evtrace_dev`): the step's buffers are timed off this
            // thread; their sums come back 1-2 steps late with the decode
            // steps they cover, and this step's `hub_step` device fields come
            // in `step_dev` records instead. Otherwise the synchronous harvest.
            let take = |p: &EventPool| -> eyre::Result<(Vec<(&'static str, f32, u32)>, u64, bool)> {
                Ok(if p.offloading() { let (s, n) = p.take_dev_sums(); (s, n, true) } else { (rollup_by_name(&p.harvest()?), 1, false) })
            };
            let (dg, n_dg, off_dg) = take(&engine.dgpu.events)?;
            let (ig, n_ig, off_ig) = take(&engine.igpu.events)?;
            // Box-2 leg, all per step, all sums over both lanes x 40 layers.
            // (`host.sel_sync/ensure/engram_stage` were the LEGACY path's
            // counters and read 0.00 here for a week; the live ones are `lh.*`.)
            let wait = counters::get(&counters::REMOTE_WAIT_NS);
            let rtt = counters::get(&counters::REMOTE_RTT_NS);
            let srv = counters::get(&counters::REMOTE_SRV_NS);
            let page = counters::get(&counters::REMOTE_PAGE_NS);
            let service = counters::get(&counters::REMOTE_COMPUTE_NS);
            let host = [
                // hub thread BLOCKED in wait(): the exposed part of the round trip
                ("host.remote_wait", wait),
                // submit -> reply landed, hidden or not
                ("host.remote_rtt", rtt),
                // box 2: frame complete -> reply handed to its writer (INCLUDES its queue wait)
                ("host.remote_srv", srv),
                // wire + wake-ups + hub reader/writer handoffs
                ("box2.link_ms", rtt.saturating_sub(srv)),
                // box 2 run_path wall INCLUDING its own paging (was misnamed box2.compute_ms)
                ("box2.service_ms", service),
                ("box2.page_ms", page),
                ("box2.compute_ms", service.saturating_sub(page)),
                ("box2.misses_x1e6", counters::get(&counters::REMOTE_MISSES) * 1_000_000),
                // replies that paged or PARKED (count; page_ms / this = ms each)
                ("box2.paged_x1e6", counters::get(&counters::REMOTE_PAGED) * 1_000_000),
            ];
            let acc = &mut self.profile_acc;
            acc.steps += 1;
            acc.steps_dev[0] += n_dg;
            acc.steps_dev[1] += n_ig;
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
            if ev_on {
                for (name, key) in [("host.remote_wait", "remote_wait_ms"), ("host.remote_rtt", "remote_rtt_ms"), ("host.remote_srv", "remote_srv_ms"),
                    ("box2.page_ms", "b2_page_ms"), ("box2.service_ms", "b2_service_ms"), ("box2.misses_x1e6", "b2_misses"),
                    ("box2.paged_x1e6", "b2_paged_replies")] {
                    if let Some(&(_, ns)) = host.iter().find(|h| h.0 == name) {
                        ev.insert(key.into(), ns as f64 / 1e6);
                    }
                }
                ev.insert("profiled".into(), 1.0);
                let mut busy = (0.0f64, 0.0f64);
                let mut other = (0.0f64, 0.0f64);
                let named = &v4flash_kernels::het::evtrace_kinds::HUB_STEP.fields;
                for (pre, short, r, off) in [("dgpu.", "d_", &dg, off_dg), ("igpu.", "i_", &ig, off_ig)] {
                    if off {
                        continue; // `step_dev` carries them (other steps' sums are in `r`)
                    }
                    for &(name, ms, _) in r.iter() {
                        if let Some(rest) = name.strip_prefix(pre) {
                            let key = format!("{short}{}", rest.replace('.', "_"));
                            let d = pre == "dgpu.";
                            if d { busy.0 += ms as f64 } else { busy.1 += ms as f64 }
                            if named.contains(&key.as_str()) {
                                *ev.entry(key).or_insert(0.0) += ms as f64;
                            } else {
                                if d { other.0 += ms as f64 } else { other.1 += ms as f64 }
                                // Name the unlisted stages once, so the kind can grow.
                                static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
                                if let Ok(mut seen) = SEEN.lock() {
                                    if !seen.contains(&key) {
                                        tracing::info!(stage = name, "evtrace: hub_step stages not named (in d_other / i_other)");
                                        seen.push(key);
                                    }
                                }
                            }
                        }
                    }
                }
                if !off_dg {
                    ev.insert("dgpu_busy_ms".into(), busy.0);
                    ev.insert("d_other".into(), other.0);
                }
                if !off_ig {
                    ev.insert("igpu_busy_ms".into(), busy.1);
                    ev.insert("i_other".into(), other.1);
                }
            }
            // Per-wait phase split (remote_experts::take_hop_stats): of the
            // box-2 waits this step, how many found the reply already in the
            // channel (slack: box 2 finished under local work) vs. blocked
            // (wake: the leg was exposed). `hop.wake_us` / `hop.slack_us` are
            // per-call means, so they are folded as ms-per-step means too.
            {
                let (a, wake, slack, nb, n) = v4flash_kernels::het::remote_experts::take_hop_stats();
                if ev_on {
                    for (k, v) in [("hop_submit_to_write_us", a), ("hop_wake_us", wake), ("hop_slack_us", slack), ("hop_blocked", nb as f64), ("hop_waits", n as f64)] {
                        ev.insert(k.into(), v);
                    }
                }
                for (name, v) in [("hop.waits_per_step", n as f64), ("hop.blocked_per_step", nb as f64), ("hop.wake_us_mean", wake), ("hop.slack_us_mean", slack), ("hop.submit_to_write_us_mean", a)] {
                    let e = acc.stages.entry(("host", name)).or_insert((0.0, 0));
                    e.0 += v; e.1 += 1;
                }
            }
            // Box-2 miss substitution (`V41_SUB`), counts per step. Compare
            // `sub.predicted_miss` with `box2.misses_x1e6` for the mirror's
            // accuracy (dry run: `V41_SUB=1`).
            if v4flash_kernels::het::b2_mirror::wanted() {
                let (p, av, sw, bl, fl, ad, inc) = v4flash_kernels::het::b2_mirror::take_sub_stats();
                let gated = v4flash_kernels::het::b2_mirror::take_admit_gated();
                if ev_on {
                    for (k, v) in [("sub_predicted_miss", p), ("sub_reads_avoided", av), ("sub_picks_swapped", sw), ("sub_blocked", bl),
                        ("sub_plan_failed", fl), ("sub_admits_queued", ad), ("sub_incoming_covered", inc), ("sub_admits_gated", gated)] {
                        ev.insert(k.into(), v as f64);
                    }
                }
                for (name, v) in [("sub.predicted_miss", p as f64), ("sub.reads_avoided", av as f64), ("sub.picks_swapped", sw as f64), ("sub.blocked", bl as f64), ("sub.plan_failed", fl as f64), ("sub.admits_queued", ad as f64), ("sub.incoming_covered", inc as f64), ("sub.admits_gated", gated as f64)] {
                    let e = acc.stages.entry(("host", name)).or_insert((0.0, 0));
                    e.0 += v; e.1 += 1;
                }
            }
            // Predicted-miss look-ahead prefetch (`V41_B2_MISS_PREFETCH`) and
            // the per-step speculative budget (`V41_B2_SPEC_BUDGET`), counts per
            // step (`het::lookahead::STATS`): their own `hub_lh2` record keyed
            // by the step (`hub_step` is at its field limit), and `lh2.*` host
            // stages in the `ms.stage` rollup. The dry run reads
            // `lh2.dry_hits_r1 / lh2.nonres_r1` (rank-1 recall; design 0.71)
            // and `lh2.dry_words` per step against the section-5 abort bar.
            if v4flash_kernels::het::lookahead::active() {
                use v4flash_kernels::het::lookahead::{take_stats, STATS};
                let st = take_stats();
                if ev_on {
                    let mut pairs: Vec<(&str, f64)> = vec![("step", step_id as f64), ("rows", b as f64)];
                    pairs.extend(STATS.iter().zip(&st).map(|((field, _), v)| (*field, *v as f64)));
                    v4flash_kernels::het::evtrace::emit_named(&v4flash_kernels::het::evtrace_kinds::HUB_LH2, &pairs);
                }
                for ((_, stage), v) in STATS.iter().zip(&st) {
                    let e = acc.stages.entry(("host", *stage)).or_insert((0.0, 0));
                    e.0 += *v as f64; e.1 += 1;
                }
            }
            // Box-2 pinning, per step (drained above).
            if let Some([sur, held, rel, pinned, _budget, unused]) = pin_stats {
                for (name, v) in [("pin.surprises", sur), ("pin.held_picks", held), ("pin.released", rel), ("pin.pinned", pinned), ("pin.released_unused", unused)] {
                    if v.is_finite() {
                        let e = acc.stages.entry(("host", name)).or_insert((0.0, 0));
                        e.0 += v; e.1 += 1;
                    }
                }
            }
            if let Some(pg) = pager.as_ref() {
                if let Some((q, a, df, ams)) = pg.prefetch_stats() {
                    let (dq, da, ddf, dms) = (q.saturating_sub(acc.pf_last.0), a.saturating_sub(acc.pf_last.1), df.saturating_sub(acc.pf_last.2), ams.saturating_sub(acc.pf_last.3));
                    acc.pf_last = (q, a, df, ams);
                    if ev_on {
                        for (k, v) in [("b1_pf_queued", dq), ("b1_pf_admitted", da), ("b1_pf_dropped_full", ddf), ("b1_pf_admit_ms", dms)] {
                            ev.insert(k.into(), v as f64);
                        }
                    }
                    for (name, v) in [("prefetch.queued", dq as f64), ("prefetch.admitted", da as f64), ("prefetch.dropped_full", ddf as f64), ("prefetch.admit_ms", dms as f64)] {
                        let e = acc.stages.entry(("host", name)).or_insert((0.0, 0));
                        e.0 += v; e.1 += 1;
                    }
                }
                // CONTAMINATED BY PREFILL (2026-09-22 audit A2). These delta a
                // LIFETIME counter across consecutive DECODE steps, but the counter
                // is also advanced by prefill chunks (`count_as_prefill`), and a
                // prefill burst runs up to 120 s between two decode steps. So the
                // first decode step after a burst is charged with the whole burst:
                // one observed rollup read 381 misses and 1.66 s of read inside a
                // 295 ms forward. Snapshot at the top and bottom of THIS step to
                // fix. Until then, treat a large value as "a prefill happened".
                let c = pg.counters();
                let (dm, dr) = (c.prefill_misses.saturating_sub(acc.last_misses), c.prefill_read_ns.saturating_sub(acc.last_read_ns));
                acc.last_misses = c.prefill_misses;
                acc.last_read_ns = c.prefill_read_ns;
                if ev_on {
                    ev.insert("b1_misses".into(), dm as f64);
                    ev.insert("b1_read_ms".into(), dr as f64 / 1e6);
                }
                let e = acc.stages.entry(("host", "pager.misses_per_step")).or_insert((0.0, 0));
                e.0 += dm as f64; e.1 += 1;
                let e = acc.stages.entry(("host", "pager.read_ms")).or_insert((0.0, 0));
                e.0 += dr as f64 / 1e6; e.1 += 1;
            }
            let lh = v4flash_kernels::het::forward_prefill::take_layer_host_timing();
            for &(name, us) in &lh {
                let e = acc.stages.entry(("host", name)).or_insert((0.0, 0));
                e.0 += us as f64 / 1e3;
                e.1 += 1;
                if ev_on {
                    ev.insert(name.replacen("lh.", "lh_", 1), us as f64 / 1e3);
                }
            }
            // The predicted-miss look-ahead's host cost PER LANE-LAYER (us), the
            // number the design's 2.6 compares with the 0.03 ms turnaround slack
            // (`lh.look_*` above are per step; `_n_x1e3` counts the lane-layers).
            for (us_name, n_name, stage) in [("lh.look_launch", "lh.look_launch_n_x1e3", "lh2.look_launch_us_per_ll"), ("lh.look_filter", "lh.look_filter_n_x1e3", "lh2.look_filter_us_per_ll")] {
                let get = |k: &str| lh.iter().find(|h| h.0 == k).map(|h| h.1).unwrap_or(0);
                let (us, n) = (get(us_name), get(n_name) / 1000);
                if n > 0 {
                    let e = acc.stages.entry(("host", stage)).or_insert((0.0, 0));
                    e.0 += us as f64 / n as f64; e.1 += 1;
                }
            }
            {
                let e = acc.stages.entry(("host", "step.fwd_wall")).or_insert((0.0, 0));
                e.0 += fwd_only_ms;
                e.1 += 1;
                let e = acc.stages.entry(("host", "step.head")).or_insert((0.0, 0));
                e.0 += fwd_ms - fwd_only_ms;
                e.1 += 1;
            }
            let every = knobs::MS_PROFILE_EVERY.get();
            if acc.steps >= every {
                // Device rows per step THEY cover (`steps_dev`, plan N3).
                let per = |d: &str| match d {
                    "dgpu" => acc.steps_dev[0],
                    "igpu" => acc.steps_dev[1],
                    _ => acc.steps,
                }
                .max(1) as f64;
                let mut v: Vec<_> = acc.stages.iter().map(|(&(d, n), &(ms, c))| (d, n, ms / per(d), c)).collect();
                v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                // Parent stages only ("dgpu.*" / "igpu.*"): the "k.*" kernel
                // sub-stages nest inside them and would double count.
                let dgpu_busy: f64 = v.iter().filter(|e| e.0 == "dgpu" && e.1.starts_with("dgpu.")).map(|e| e.2).sum();
                let igpu_busy: f64 = v.iter().filter(|e| e.0 == "igpu" && e.1.starts_with("igpu.")).map(|e| e.2).sum();
                tracing::info!(steps = acc.steps, steps_dev = format!("{}/{}", acc.steps_dev[0], acc.steps_dev[1]), rows_avg = format!("{:.1}", acc.rows as f64 / acc.steps as f64),
                    wall_ms = format!("{:.1}", acc.wall_ms / acc.steps as f64), dgpu_busy_ms = format!("{dgpu_busy:.1}"),
                    igpu_busy_ms = format!("{igpu_busy:.1}"), "ms.stage.total (per step)");
                // Zero rows are counters this path never feeds; drop them.
                for (d, n, ms, c) in v.iter().filter(|e| e.2 >= 0.005).take(96) {
                    tracing::info!(device = *d, stage = *n, ms_per_step = format!("{ms:.2}"), calls = *c, "ms.stage");
                }
                acc.stages.clear();
                acc.steps = 0;
                acc.steps_dev = [0; 2];
                acc.rows = 0;
                acc.wall_ms = 0.0;
            }
        }
        // Sample, emit, retire. A stream with drafts runs the block procedure
        // (plan 2.3) over its rows and emits token by token, stopping at the
        // first stop; it keeps one KV row per token emitted.
        let t_s = Instant::now();
        let mut done: Vec<(usize, FinishReason)> = Vec::new();
        let mut keeps: Vec<u32> = vec![1; self.streams.len()];
        let mut spec_out: Option<(usize, usize, usize)> = None;
        // Per stream: (K, accepted, emitted) of its verified block.
        let mut spec_outs: Vec<Option<(usize, usize, usize)>> = vec![None; self.streams.len()];
        for (i, s) in self.streams.iter_mut().enumerate() {
            let r = row0[i];
            if drafts[i].is_empty() {
                let tok = targets[r].sample(&mut s.rng);
                s.next = tok;
                if !emit(state, s, tok) { done.push((i, FinishReason::Stop)); continue; }
                if let Some(f) = stop_reason(s, tok) { done.push((i, f)); }
                continue;
            }
            let k = drafts[i].len();
            let rows: Vec<TargetDist> = targets[r..=r + k].to_vec();
            let ds: Vec<Draft> = drafts[i]
                .iter()
                .enumerate()
                .map(|(j, &t)| Draft {
                    token: t,
                    q: match draft_q[i].as_ref() {
                        Some(q) => DraftDist::Sampled(q[j].clone()),
                        None => DraftDist::PointMass,
                    },
                })
                .collect();
            let out = verify_block(&rows, &ds, &mut s.rng);
            let mut n = 0u32;
            for &tok in &out.tokens {
                n += 1;
                s.next = tok;
                if !emit(state, s, tok) { done.push((i, FinishReason::Stop)); break; }
                if let Some(f) = stop_reason(s, tok) { done.push((i, f)); break; }
            }
            keeps[i] = n;
            spec_outs[i] = Some((k, out.accepted, n as usize));
            // The `ms.step` summary: the lone block, or the sums over blocks.
            let (k0, a0, e0) = spec_out.unwrap_or((0, 0, 0));
            spec_out = Some((k0 + k, a0 + out.accepted, e0 + n as usize));
        }
        let sample_ms = t_s.elapsed().as_secs_f64() * 1e3;
        // Keep each stream's emitted rows (rollback = the counters stop there):
        // a prefix of the rows it ran (`StepRows`; a chain keeps a prefix).
        for (s, &keep) in self.streams.iter().zip(&keeps) {
            let ran = step_rows.rows_of(s.slot) as u32;
            if keep > ran {
                return Err(eyre!("ms.step: slot {} keeps {keep} rows of the {ran} it ran", s.slot));
            }
            self.arena.accept(s.slot, keep, &state.engine.dgpu.compute)?;
        }
        if defer {
            // The lone stream's kept rows are positions [pos_end - keep, pos_end).
            let pos_end = self.arena.stream(self.streams[0].slot).map(|k| k.pos).unwrap_or(0) as i32;
            v4flash_kernels::het::b2_mirror::defer_flush(pos_end - keeps[0] as i32, pos_end);
        }
        // Kept rows into the drafter rings; the last one feeds the next draft.
        if let (Some(dsp), Some(m), Some(caps)) = (self.dsp.as_mut(), state.mtp.as_mut(), caps.as_ref()) {
            // A LONE stream's ring is written every step, drafted or not: it is
            // the stream that drafts. Writing only on speculating steps
            // (2026-10-01 09:54-10:05) left a gap of every plain step the
            // draft-or-not gate took while backing off; the next probe drafted
            // from that stale window, accepted less, and extended the back-off
            // (73% of lone steps plain). `solo` only skips the steps of more
            // streams than may draft together -- once the knob has ever been
            // above 1, more than may EVER (`ms_dspark::ring_streams`): a live
            // flip of `V41_MS_DSPARK_STREAMS` (the per-turn A/B) must find every
            // stream's ring dense, or the knob-2 arm would draft from rings with
            // the knob-1 steps missing (the 10-01 gap bug; review rounds 2-3).
            let write = dsp.ring_all() || self.streams.len() <= ms_dspark::ring_streams(spec_max);
            for (i, s) in self.streams.iter().enumerate() {
                if done.iter().any(|&(d, _)| d == i) {
                    continue;
                }
                let pos_end = self.arena.stream(s.slot).map(|k| k.pos).unwrap_or(0);
                let pos0 = pos_end - keeps[i];
                let rows: Vec<Vec<f32>> = (0..keeps[i] as usize).map(|j| caps[row0[i] + j].clone()).collect();
                if let Err(e) = dsp.keep_rows(&state.engine, m, s.slot, pos0, rows, write) {
                    tracing::warn!(slot = s.slot, pos0, error = %e, "ms dspark: ring write failed; the stream's ring restarts");
                    let _ = dsp.reset(s.slot);
                }
            }
            if self.streams.len() == 1 {
                if let Some((k, accepted, emitted)) = spec_out {
                    let lanes = if spec_lanes { 2 } else { 1 };
                    dsp.record(self.streams[0].slot, &draft_conf, k, accepted, emitted, t0.elapsed().as_secs_f64() * 1e3, lanes);
                } else if b == 1 {
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    if drafted_k0 {
                        dsp.record_k0(self.streams[0].slot, ms);
                    } else {
                        dsp.note_plain_step(ms);
                    }
                }
            } else if !multi_drafted.is_empty() {
                // Two or more live streams drafted (design 2.5): per drafted
                // block calibration, stats and the stage-1 gate (baseline: the
                // plain step of these streams), and ONE cost sample -- into
                // the multi tables if any draft was verified, else this was a
                // plain step and `PlainLanes` takes it below.
                let step_ms = t0.elapsed().as_secs_f64() * 1e3;
                let plain_ms = self.plain_lanes.cost(self.streams.len(), plain_rule);
                if spec {
                    let blocks: Vec<(u32, [f32; v4flash_kernels::het::mtp::MTP_BLOCK], usize, usize, usize)> = multi_drafted
                        .iter()
                        .map(|&(i, conf)| {
                            let (k, acc, em) = spec_outs[i].unwrap_or((0, 0, keeps[i] as usize));
                            (self.streams[i].slot, conf, k, acc, em)
                        })
                        .collect();
                    dsp.record_multi(&blocks, b, if pipelined { 2 } else { 1 }, spec_rule, step_ms, plain_ms);
                } else {
                    let slots: Vec<u32> = multi_drafted.iter().map(|&(i, _)| self.streams[i].slot).collect();
                    dsp.record_k0_multi(&slots, step_ms);
                }
            }
        }
        // A plain multi-stream step feeds its lane tables (a three-lane step
        // feeds nothing: no table prices it).
        if !spec && b >= 2 && !lanes3 {
            self.plain_lanes.observe(b, if pipelined { 2 } else { 1 }, t0.elapsed().as_secs_f64() * 1e3);
        }
        // Per speculating stream `K/accepted/emitted` (fairness, MS_DSPARK_STREAMS_DESIGN
        // 2.6); `spec` above is their sum.
        let spec_streams: String = spec_outs.iter().flatten().map(|(k, a, e)| format!("{k}/{a}/{e}")).collect::<Vec<_>>().join(",");
        // Every stream's `slot:emitted` this step, riders and finishing streams
        // included (before they are removed): the fairness metric is the slower
        // stream's rate over steps with two live streams (design 2.6, review
        // round 3).
        let live_emitted: String = self.streams.iter().zip(&keeps).map(|(s, k)| format!("{}:{k}", s.slot)).collect::<Vec<_>>().join(",");
        for (r, f) in done.into_iter().rev() {
            let s = self.streams.remove(r);
            finish(state, &mut self.arena, s, f)?;
        }
        tracing::info!(rows = b, spec = ?spec_out, lanes = if lanes3 { 3 } else if pipelined { 2 } else { 1 },
            step_ms = format!("{:.1}", t0.elapsed().as_secs_f64() * 1e3), fwd_ms = format!("{fwd_ms:.1}"),
            engram_ms = format!("{engram_ms:.1}"), sample_ms = format!("{sample_ms:.1}"), live = self.streams.len(),
            head_full = head_stats.full, head_mismatch = head_stats.mismatch, head_diff = head_stats.head_diff,
            chain_waits, chain_wait_us, ring_settle_ms = format!("{ring_settle_ms:.2}"),
            engram_gather_ms = format!("{:.2}", ENGRAM_GATHER_US.swap(0, Ordering::Relaxed) as f64 / 1e3), spec_streams = %spec_streams, live_emitted = %live_emitted, draft_ms = format!("{draft_wall_ms:.1}"), "ms.step");
        if ev_on {
            let lanes = if lanes3 { 3.0 } else if stagger2 || pipelined { 2.0 } else { 1.0 };
            for (k, v) in [("t_end", v4flash_kernels::het::evtrace::now()), ("live", self.streams.len() as f64), ("lanes", lanes),
                ("fwd_ms", fwd_only_ms), ("fwd_all_ms", fwd_ms), ("engram_ms", engram_ms), ("sample_ms", sample_ms),
                ("step_ms", t0.elapsed().as_secs_f64() * 1e3), ("rf_chain_waits", chain_waits as f64), ("rf_chain_wait_us", chain_wait_us as f64)] {
                ev.insert(k.into(), v);
            }
            ev.entry("profiled".into()).or_insert(0.0);
            let k = &v4flash_kernels::het::evtrace_kinds::HUB_STEP;
            let v: Vec<f64> = k.fields.iter().map(|f| ev.get(*f).copied().unwrap_or(f64::NAN)).collect();
            v4flash_kernels::het::evtrace::emit(k, &v);
            v4flash_kernels::het::evtrace_kinds::clear_step();
        }
        Ok(())
    }
}

/// The image spans of the saved prefix `[0, n)`. A cut that straddles an image
/// block is an error, never an empty span list: dropping the spans strips the
/// content hashes from the snapshot's byte stream, and two different pictures
/// with the same block layout would then key to the same hash (same rule as
/// the serial path's system-prefix save).
fn checkpoint_spans(spans: &[crate::vision_prompt::ImageSpan], n: usize) -> eyre::Result<Vec<crate::vision_prompt::ImageSpan>> {
    use color_eyre::eyre::WrapErr;
    crate::vision_prompt::spans_in_range(spans, 0, n).wrap_err("snapshot cut splits an image block; not saved")
}

/// The rows a prefill runs after a restored `prefix_len`-token prefix of the
/// (marker-stripped) prompt `tokens`, and the trailing marker it consumes: the
/// rest of the prompt (the marker is forwarded at admission), or, when the
/// snapshot covers the whole prompt, the marker itself (same computation as
/// the legacy decode-side forward of it, modulo kernel family). None: nothing
/// to forward at all.
fn plan_suffix(tokens: &[i32], prefix_len: usize, trailing_marker: Option<i32>) -> Option<(Vec<i32>, Option<i32>)> {
    match (&tokens[prefix_len..], trailing_marker) {
        ([], Some(m)) => Some((vec![m], Some(m))),
        ([], None) => None,
        (rest, _) => Some((rest.to_vec(), None)),
    }
}

/// Does the stream start inside the reasoning block? Exactly when the prompt's
/// trailing marker is `<think>`, whether it is still to be forwarded at
/// admission (`pending`) or the prefill already consumed it (`prefilled`, a
/// whole-prompt snapshot hit). `emit` flips the flag only on SAMPLED markers,
/// so missing the prefilled case streamed the whole chain of thought as
/// answer content (the legacy twin: `save_and_forward_marker`'s
/// `initial_in_think`).
fn initial_in_think(pending: Option<i32>, prefilled: Option<i32>) -> bool {
    pending.or(prefilled) == Some(TOK_THINK_BEGIN)
}

/// Tokens a prefill's state holds when checkpointed after `done` of its rows
/// (the job starting at `pos0`): the request's own first `pos0 + done`
/// tokens. (`Prefill::prefix` already holds the whole prompt; appending the
/// job's done rows to it once more made a key no request could ever match.)
/// None past the marker-stripped prompt, i.e. into a marker forwarded in the
/// prefill: requests are matched without their marker.
fn checkpoint_tokens(req_tokens: &[i32], pos0: u32, done: usize) -> Option<Vec<i32>> {
    req_tokens.get(..pos0 as usize + done).map(<[i32]>::to_vec)
}

/// Snapshot restore candidates in the order to try them: longest first, the
/// session hint winning a tie, the same snapshot never twice. `start_prefill`
/// falls through to the next one when a restore fails.
fn restore_candidates<H: PartialEq, D>(session: Option<(usize, H, D)>, walk: Option<(usize, H, D)>) -> Vec<(usize, H, D)> {
    match (session, walk) {
        (Some(a), Some(b)) if a.1 == b.1 => vec![a],
        (Some(a), Some(b)) if a.0 >= b.0 => vec![a, b],
        (Some(a), Some(b)) => vec![b, a],
        (a, b) => a.into_iter().chain(b).collect(),
    }
}

#[cfg(test)]
mod restore_candidate_tests {
    use super::restore_candidates;

    #[test]
    fn longest_first_session_wins_ties_no_duplicates() {
        // Regression (2026-09-23): a failed session-hint restore used to fall
        // straight to a FULL prefill; now every candidate is tried in order.
        let order = |s, w| restore_candidates::<u8, &str>(s, w).into_iter().map(|c| c.2).collect::<Vec<_>>();
        assert_eq!(order(Some((100, 1, "sess")), Some((90, 2, "walk"))), vec!["sess", "walk"]);
        assert_eq!(order(Some((80, 1, "sess")), Some((90, 2, "walk"))), vec!["walk", "sess"]);
        assert_eq!(order(Some((90, 1, "sess")), Some((90, 2, "walk"))), vec!["sess", "walk"], "tie: session hint first");
        assert_eq!(order(Some((90, 7, "sess")), Some((90, 7, "walk"))), vec!["sess"], "same snapshot only once");
        assert_eq!(order(None, Some((90, 2, "walk"))), vec!["walk"]);
        assert_eq!(order(Some((90, 1, "sess")), None), vec!["sess"]);
        assert!(order(None, None).is_empty());
    }
}

#[cfg(test)]
mod prefill_plan_tests {
    use super::{checkpoint_tokens, initial_in_think, plan_suffix};
    use crate::tokens::{TOK_THINK_BEGIN, TOK_THINK_END};

    #[test]
    fn whole_prompt_snapshot_hit_still_starts_in_think() {
        // Regression: an exact re-send of a thinking-mode prompt restores a
        // snapshot covering the whole stripped prompt; the `<think>` marker is
        // then the prefill, and the stream used to start with in_think=false.
        let tokens = [1, 2, 3, 4];
        let (suffix, prefilled) = plan_suffix(&tokens, 4, Some(TOK_THINK_BEGIN)).unwrap();
        assert_eq!((suffix, prefilled), (vec![TOK_THINK_BEGIN], Some(TOK_THINK_BEGIN)));
        assert!(initial_in_think(None, prefilled), "marker consumed by the prefill");
        // Chat mode is answer content either way.
        let (_, prefilled) = plan_suffix(&tokens, 4, Some(TOK_THINK_END)).unwrap();
        assert!(!initial_in_think(None, prefilled));
        // A partial hit forwards the marker at admission.
        let (suffix, prefilled) = plan_suffix(&tokens, 2, Some(TOK_THINK_BEGIN)).unwrap();
        assert_eq!((suffix, prefilled), (vec![3, 4], None));
        assert!(initial_in_think(Some(TOK_THINK_BEGIN), prefilled));
        assert!(!initial_in_think(None, None));
        assert!(plan_suffix(&tokens, 4, None).is_none(), "nothing to forward");
    }

    #[test]
    fn checkpoint_key_is_the_requests_own_prefix() {
        // Restored 3 tokens, the job prefills the other 5; checkpoint after 4.
        // The key used to be the whole prompt plus the 4 done rows again.
        let req: Vec<i32> = (10..18).collect();
        assert_eq!(checkpoint_tokens(&req, 3, 4).unwrap(), vec![10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(checkpoint_tokens(&req, 0, 8).unwrap(), req);
        // Rows past the stripped prompt (a marker forwarded in the prefill).
        assert!(checkpoint_tokens(&req, 8, 1).is_none());
    }
}

/// Cancelled prefills with at least this many rows done are checkpointed
/// (`V41_MS_CHECKPOINT_MIN_ROWS`, default 4096).
fn checkpoint_min_rows() -> usize { knobs::MS_CHECKPOINT_MIN_ROWS.usize() }
fn chunk_rows_idle() -> usize { knobs::MS_CHUNK_ROWS_IDLE.usize() }
fn chunk_rows_busy() -> usize { knobs::MS_CHUNK_ROWS.usize() }

/// Same rule as `HeterogeneousEngine::sample_next` / the DSpark host twin.
///
/// The distribution is `spec_sample::TargetDist`, the ONE definition of `p`
/// that plain sampling and DSpark's rejection sampling share; this draws from
/// it exactly as the pre-refactor code did (bit-identical tokens and RNG
/// consumption, test `g_rs2_refactor_is_bit_identical_to_the_old_sampler`).
fn sample_row(r: &[f32], mode: &SampleMode, rng: &mut SamplerRng) -> i32 {
    crate::spec_sample::TargetDist::from_logits(r, mode).sample(rng)
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
            // NEVER skip a token and carry on: that hands the client a response
            // with a silent hole in it. The buffer is `STREAM_CHUNK_BUFFER`
            // tokens deep, so being full means the client has not read for
            // minutes -- drop it cleanly and say so.
            s.send_failures += 1;
            tracing::warn!(slot = s.slot, completion_tokens = s.completion_tokens,
                buffered = crate::engine_worker::STREAM_CHUNK_BUFFER,
                "multistream: stream consumer stalled (token buffer full); dropping client");
            false
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
    let p0 = pf.job.pos0() as usize;
    let mut hcs: Vec<Vec<f32>> = Vec::with_capacity(toks.len());
    for (k, &tok) in toks.iter().enumerate() {
        let mut v = vec![0f32; HC_DIM as usize];
        // Absolute prompt index = prefix + suffix offset (the tower rows are
        // indexed over the whole request's tokens).
        match pf.vl.row_at(p0 + a + k) {
            Some(row) => {
                if !crate::vision_prompt::is_image_token(tok) {
                    return Err(eyre!("prefill: token {tok} at index {} is not an image id but sits inside an image block", p0 + a + k));
                }
                let n_embd = v4flash_kernels::config::N_EMBD as usize;
                for h in 0..v4flash_kernels::config::N_HC as usize {
                    v[h * n_embd..(h + 1) * n_embd].copy_from_slice(row);
                }
            }
            None => {
                if crate::vision_prompt::is_image_token(tok) {
                    return Err(eyre!("prefill: synthetic image token {tok} at index {} has no encoded row", p0 + a + k));
                }
                embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, tok, &mut v);
            }
        }
        hcs.push(v);
    }
    let engram = match (state.pager.as_ref(), state.engram.as_ref()) {
        // A unit that takes no inputs (layer-major groups after the first):
        // empty rows, the look-ahead untouched.
        (Some(_), Some(ec)) if a == z => Some(vec![Vec::new(); ec.tables.len()]),
        (Some(_), Some(ec)) => Some(match pf.engram_ahead.take(a, z, ec.tables.len()) {
            Some(rows) => rows,
            None => gather_prefill_engram(ec, &pf.compressed, p0, a, z)?,
        }),
        _ => None,
    };
    pf.job.set_chunk_inputs(hcs, engram);
    Ok(())
}

/// Engram rows of job tokens `[a, z)` (absolute positions `p0 + index`):
/// batched gathers over runs of live positions, as `EngramCtx::rows_for_chunk`.
/// Reads through the Engram context's OWN checkpoint handle (`EngramCtx::st`),
/// so it can run beside a forward that holds the pager (`engram_lookahead`).
fn gather_prefill_engram(ec: &crate::engine_worker::EngramCtx, compressed: &[i32], p0: usize, a: usize, z: usize) -> eyre::Result<Vec<Vec<f32>>> {
    const GATHER_THREADS: usize = 32;
    let ein = ENGRAM_IN as usize;
    let n = z - a;
    let mut rows = vec![vec![0f32; n * ein]; ec.tables.len()];
    let mut k = 0usize;
    while k < n {
        if compressed[p0 + a + k] == v4flash_core::engram_hash::DEAD { k += 1; continue; }
        let start = k;
        while k < n && compressed[p0 + a + k] != v4flash_core::engram_hash::DEAD { k += 1; }
        let hashes: Vec<_> = (start..k).map(|q| ec.hasher.hash_ids(compressed, p0 + a + q)).collect();
        for (li, tbl) in ec.tables.iter().enumerate() {
            let flat: Vec<i64> = hashes.iter().flat_map(|h| h[li]).collect();
            tbl.gather(&ec.st, &flat, &mut rows[li][start * ein..k * ein], GATHER_THREADS)?;
        }
    }
    Ok(rows)
}

/// The next block of Engram rows worth gathering ahead for `pf` (block comment
/// at `EngramAhead`): from the first row neither consumed nor cached, one chunk
/// long, while the cache holds less than the job's look-ahead span. `None` when
/// Engram is off, the job is done, or the cache is full.
fn engram_lookahead(pf: &Prefill, engram_on: bool) -> Option<(usize, usize)> {
    if !engram_on || !engram_ahead_on() || !pf.job.lazy_inputs() || pf.job.chunks_done() {
        return None;
    }
    pf.engram_ahead.next_block(pf.job.done_rows(), pf.job.chunk_rows(), pf.job.total(), pf.job.input_lookahead_rows())
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


/// The Engram SSD gather for one step: one thread per table (the two tables
/// used to be read back to back), each a batched `EngramTable::gather` over
/// the live rows. Runs on the scoped helper thread `decode_step` spawns.
/// Read threads per Engram table of a decode step's gather
/// (`V41_MS_ENGRAM_THREADS`, default 32; live knob). Cold reads cost ~0.8-1.7
/// ms each, so 32 threads over a 5-row step's 120 ids per table is ~4 rounds;
/// the spawns are serial (~15 us each), so more threads start the last read
/// later -- measure (`ms.step` `engram_gather_ms` vs `lh_engram_join`).
fn engram_threads() -> usize {
    knobs::MS_ENGRAM_THREADS.usize()
}

/// Wall time (us) of the last decode-step Engram gather (`ms.step`).
static ENGRAM_GATHER_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One Engram table's rows for a decode step, on its own thread so the step joins each table at
/// its own layer (dGPU bundle slice 1; was one gather of every table joined at layer 1). `ENGRAM_GATHER_US` = the slowest table.
fn gather_engram_table(
    ec: &crate::engine_worker::EngramCtx,
    live: &[(usize, [[i64; v4flash_core::engram_hash::ENGRAM_COLS]; v4flash_core::engram_hash::ENGRAM_LAYERS])],
    b: usize,
    li: usize,
) -> eyre::Result<Vec<f32>> {
    let t0 = Instant::now();
    let ein = ENGRAM_IN as usize;
    let mut out = vec![0f32; b * ein];
    let flat: Vec<i64> = live.iter().flat_map(|(_, h)| h[li]).collect();
    let mut tmp = vec![0f32; live.len() * ein];
    ec.tables[li].gather(&ec.st, &flat, &mut tmp, engram_threads())?;
    for (k, (r, _)) in live.iter().enumerate() {
        out[r * ein..(r + 1) * ein].copy_from_slice(&tmp[k * ein..(k + 1) * ein]);
    }
    ENGRAM_GATHER_US.fetch_max(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
    Ok(out)
}

#[cfg(test)]
mod reservation_tests {
    use super::{effective_max_new, reservation};
    use crate::engine_worker::GenerateReq;

    fn req(max_new: usize, defaulted: bool) -> GenerateReq {
        GenerateReq { tokens: Vec::new(), images: Vec::new(), image_spans: Vec::new(), max_new, max_new_defaulted: defaulted,
            temperature: 0.0, min_p_rel: 0.0, top_p: 1.0, seed: 0 }
    }

    #[test]
    fn defaulted_max_new_shrinks_to_the_context_explicit_overflow_fails() {
        assert_eq!(effective_max_new(&req(65536, true), 100_000, 368_640).unwrap(), 65536);
        assert_eq!(effective_max_new(&req(65536, true), 330_000, 368_640).unwrap(), 38_638);
        assert!(effective_max_new(&req(65536, false), 330_000, 368_640).is_err());
        assert!(effective_max_new(&req(65536, true), 368_640, 368_640).is_err(), "no room at all");
    }

    #[test]
    fn reservation_caps_the_tail_at_the_headroom() {
        if std::env::var_os("V41_MS_KV_HEADROOM").is_some() {
            return; // the default is what is under test
        }
        assert_eq!(reservation(170_000, 65536), 170_000 + 16384 + 2);
        assert_eq!(reservation(170_000, 500), 170_502);
    }
}

#[cfg(test)]
mod live_trace_tests {
    use super::utc_stamp;

    #[test]
    fn utc_stamps() {
        assert_eq!(utc_stamp(0), "19700101-000000");
        assert_eq!(utc_stamp(951_782_400), "20000229-000000");
        // The hub's 10-01 21:41:44 UTC restart (evt header realtime).
        assert_eq!(utc_stamp(1_790_890_904), "20261001-214144");
        assert_eq!(utc_stamp(1_790_899_199), "20261001-235959");
    }
}

#[cfg(test)]
mod engram_ahead_tests {
    use super::*;

    /// A block of 2 tables whose row r holds r (+0.5 in table 1) in every column.
    fn block(a: usize, z: usize) -> (usize, usize, Vec<Vec<f32>>) {
        let ein = ENGRAM_IN as usize;
        let rows = (0..2).map(|t| (a..z).flat_map(|r| std::iter::repeat_n(r as f32 + 0.5 * t as f32, ein)).collect()).collect();
        (a, z, rows)
    }

    fn check(rows: &[Vec<f32>], a: usize, z: usize) {
        let ein = ENGRAM_IN as usize;
        for (t, r) in rows.iter().enumerate() {
            assert_eq!(r.len(), (z - a) * ein);
            for (i, &v) in r.iter().enumerate() {
                assert_eq!(v, (a + i / ein) as f32 + 0.5 * t as f32);
            }
        }
    }

    #[test]
    fn layer_major_windows_find_their_rows_gathered_ahead() {
        // Two 4096-row windows of four 1024-row sub-chunks; groups 1.. take no
        // inputs (16 empty units per window). Drive the scheduler's sequence:
        // inputs for the unit (a hit or a gather), then one look-ahead block.
        let (total, chunk, span) = (8192usize, 1024usize, 4096usize);
        let mut ahead = EngramAhead::default();
        let mut gathered_sync = Vec::new();
        let mut done_rows = 0usize;
        for w in 0..2 {
            for unit in 0..20 {
                let (a, z) = if unit < 4 { (w * span + unit * chunk, w * span + (unit + 1) * chunk) } else { (done_rows, done_rows) };
                if a < z {
                    match ahead.take(a, z, 2) {
                        Some(rows) => check(&rows, a, z),
                        None => gathered_sync.push((a, z)),
                    }
                }
                if let Some((s, e)) = ahead.next_block(done_rows, chunk, total, span) {
                    ahead.blocks.push_back(block(s, e));
                    ahead.next = e;
                }
            }
            done_rows += span;
        }
        // Only the job's very first sub-chunk was gathered on the spot.
        assert_eq!(gathered_sync, vec![(0, 1024)]);
        assert!(ahead.blocks.is_empty() && ahead.next == total);
    }

    #[test]
    fn a_miss_keeps_later_blocks_and_partial_blocks_count_their_rest() {
        let mut ahead = EngramAhead::default();
        ahead.blocks.push_back(block(2000, 3000));
        ahead.next = 3000;
        // [1000, 1500) is not cached: a miss that keeps the later block.
        assert!(ahead.take(1000, 1500, 2).is_none());
        assert_eq!((ahead.blocks.len(), ahead.consumed), (1, 1500));
        // A chunk cut across blocks (an image-shortened chunk) assembles.
        ahead.blocks.push_back(block(3000, 4000));
        let rows = ahead.take(2500, 3300, 2).expect("covered by two blocks");
        check(&rows, 2500, 3300);
        // The partly consumed block counts only its unconsumed rest.
        assert_eq!(ahead.cached_rows(), 700);
    }
}
