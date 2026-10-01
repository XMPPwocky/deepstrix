//! DSpark on the multistream arena (docs/v41/DSPARK_ARENA_PLAN.md), the
//! single-stream build: when exactly one stream is live it verifies the
//! drafter's block in the SAME arena step as its next token (rows `[next, d_0
//! .. d_{K-1}]`, `KvArena::tables`), rejection-samples the rows against the
//! drafts (`spec_sample::verify_block`, point-mass tests) and keeps the KV of
//! the rows it emitted (`KvArena::accept`). With several live streams every
//! step is a plain multistream step (K = 0), and the rings are kept current
//! so a stream that becomes the lone one drafts from a dense window.
//!
//! What lives here: one drafter ring per arena slot (`MtpState::rings` is the
//! drafter's only state that persists across drafts; the rest of `MtpState`
//! is per-draft scratch, so ONE `MtpState` serves every slot by swapping the
//! ring in), the main-model residual of each stream's last row in KV (the
//! input of its next draft), seeding from the prefill's capture, and the
//! confidence-gated K policy (plan section 6).
//!
//! Who writes which ring row: every row a step KEEPS is ring-written after
//! the step (`ring_write_only`), in position order, once. A draft from
//! position `pos` reads the ring INCLUDING row `pos` and its forward writes row
//! `pos` itself; when that row is already the ring's latest write the counter
//! is wound back by one first, so the forward rewrites it with the same values
//! (same residual, same position) instead of appending a duplicate key.
use std::collections::BTreeMap;
use std::sync::LazyLock;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::DeviceBuffer;
use v4flash_kernels::config::N_EMBD;
use v4flash_kernels::het::batch_scratch::{BatchDgpuScratch, MTP_CAP_ROWS};
use v4flash_kernels::het::mtp::{DraftSampling, MTP_BLOCK, MTP_DRAFT_TOP_M, MTP_SRC_LAYERS, MTP_WINDOW, RING_ROWS_MAX};
use v4flash_kernels::het::SampleMode;
use v4flash_kernels::het::{HetModelWeights, HeterogeneousEngine};

use crate::engine_worker::MtpCtx;

/// `V41_MS_DSPARK=accept`: DSpark on the arena path. Loads the drafter
/// (as `V41_DSPARK` does) and keeps multistream on for every request.
pub fn enabled() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| matches!(std::env::var("V41_MS_DSPARK").as_deref(), Ok("accept") | Ok("1") | Ok("on")));
    *ON
}

/// `V41_MS_DSPARK_RING=solo`: ring-write only the speculating stream's rows.
/// Default `all`: every live stream's kept rows go into its ring every step
/// (about one `main_proj` + 3 `attn_kv` matvecs on the iGPU per row), so a
/// stream that becomes the lone stream drafts from a dense window instead of
/// one with a gap where it ran beside others.
fn ring_all() -> bool {
    static ALL: LazyLock<bool> = LazyLock::new(|| std::env::var("V41_MS_DSPARK_RING").as_deref() != Ok("solo"));
    *ALL
}

/// `V41_MS_DSPARK_DRAFTS=argmax`: point-mass drafts (the drafter's argmax,
/// accepted with probability p(d)) for every request. Default `sampled`: a
/// request that samples (temperature > 0) gets SAMPLED drafts, each drawn from
/// the drafter's own tempered top-M distribution q and accepted with
/// min(1, p/q) (plan 2.2, M6) -- the drafter as designed; at temperature 0 a
/// draft is the argmax either way.
fn sampled_drafts() -> bool {
    static S: LazyLock<bool> = LazyLock::new(|| std::env::var("V41_MS_DSPARK_DRAFTS").as_deref() != Ok("argmax"));
    *S
}

/// How a stream's drafts are drawn: `Some` (sampled, plan 2.2) when its
/// requests sample and sampled drafts are on, else `None` (point mass). `u`
/// are the drafter RNG's uniforms for the block's positions.
pub fn draft_sampling(mode: &SampleMode, u: [f32; MTP_BLOCK]) -> Option<DraftSampling> {
    match *mode {
        SampleMode::Multinomial { temperature, top_p, .. } if temperature > 0.0 && sampled_drafts() => {
            Some(DraftSampling { temperature, top_p, m: MTP_DRAFT_TOP_M, u })
        }
        _ => None,
    }
}

/// One drafted block.
pub struct Drafted {
    pub ids: [i32; MTP_BLOCK],
    /// Confidence logits (`MtpExit::conf`).
    pub conf: [f32; MTP_BLOCK],
    /// Sampled drafts: each position's q, `(token, prob)`; `None` for argmax.
    pub q: Option<Vec<Vec<(i32, f64)>>>,
}

/// Floats of one row's drafter input: `cat(mean_hc(resid@37), @38, @39)`.
pub const HIDDEN: usize = MTP_SRC_LAYERS.len() * N_EMBD as usize;

/// One arena slot's drafter memory.
struct SlotDraft {
    /// `MtpState::rings` for this slot's stream.
    rings: Vec<DeviceBuffer<u16>>,
    /// Its `ring_writes`.
    writes: usize,
    /// Position of the latest ring row (rows go in, in position order).
    last_ring_pos: Option<u32>,
    /// `(pos, mtp_src of row pos)` for the stream's LAST row in KV: what its
    /// next draft reads.
    hidden: Option<(u32, Vec<f32>)>,
    /// Draft-or-not (plan section 6, stage 1): EWMA of the realized speed-up
    /// of a drafted step over a plain one, `emitted * cost(1) / (step +
    /// draft)`; below `V41_MS_DSPARK_MIN_GAIN` the stream stops drafting for
    /// `backoff` steps (4, doubling to 64), then probes with one block.
    gain: f64,
    skip_left: u32,
    backoff: u32,
    /// The last draft's wall time (ms), for its block's gain.
    last_draft_ms: f64,
}

/// Initial `gain` of a stream: optimistic, so it drafts until measured.
const GAIN0: f64 = 1.5;

fn min_gain() -> f64 {
    static G: LazyLock<f64> = LazyLock::new(|| env_f64("V41_MS_DSPARK_MIN_GAIN", 1.0));
    *G
}

#[derive(Default)]
struct Stats {
    blocks: u64,
    drafts_verified: u64,
    accepted: u64,
    emitted: u64,
    k_hist: [u64; MTP_BLOCK + 1],
    draft_ms: f64,
    step_ms: f64,
    no_hidden: u64,
    /// Steps a stream sat out drafting (stage-1 back-off).
    skipped: u64,
}

pub struct MsDspark {
    slots: Vec<SlotDraft>,
    /// EWMA of a plain one-row step's wall (ms), the stage-1 baseline: a
    /// drafted block pays when it emits more per ms than a plain step NOW
    /// (a cold pool slows both; the static ladder made cold blocks look like
    /// losses and backed off 324 steps of a code stream, 2026-10-01).
    plain_ms: f64,
    stats: Stats,
    since: Instant,
}

impl MsDspark {
    /// One ring per arena slot, shaped like `mtp.state`'s.
    pub fn alloc(mtp: &MtpCtx, igpu_id: i32, n_slots: u32) -> eyre::Result<Self> {
        let mut slots = Vec::with_capacity(n_slots as usize);
        for _ in 0..n_slots {
            let rings = mtp.state.rings.iter().map(|r| DeviceBuffer::<u16>::new(igpu_id, r.len())).collect::<eyre::Result<Vec<_>>>()?;
            slots.push(SlotDraft { rings, writes: 0, last_ring_pos: None, hidden: None, gain: GAIN0, skip_left: 0, backoff: 0, last_draft_ms: 0.0 });
        }
        Ok(Self { slots, plain_ms: step_cost(1), stats: Stats::default(), since: Instant::now() })
    }

    fn slot(&mut self, slot: u32) -> eyre::Result<&mut SlotDraft> {
        self.slots.get_mut(slot as usize).ok_or_else(|| eyre!("ms dspark: slot {slot} out of range"))
    }

    /// Run `f` with `slot`'s ring swapped into `mtp.state` (pointer swap, no
    /// copy), and swap it back whatever `f` returns.
    fn with_ring<T>(
        &mut self,
        mtp: &mut MtpCtx,
        slot: u32,
        rewind: bool,
        f: impl FnOnce(&mut MtpCtx) -> eyre::Result<T>,
    ) -> eyre::Result<T> {
        let sd = self.slot(slot)?;
        std::mem::swap(&mut mtp.state.rings, &mut sd.rings);
        mtp.state.set_ring_writes(sd.writes - usize::from(rewind && sd.writes > 0));
        let r = f(mtp);
        sd.writes = mtp.state.ring_writes();
        std::mem::swap(&mut mtp.state.rings, &mut sd.rings);
        r
    }

    /// Forget `slot`'s stream (a new request took the slot).
    pub fn reset(&mut self, slot: u32) -> eyre::Result<()> {
        let sd = self.slot(slot)?;
        sd.writes = 0;
        sd.last_ring_pos = None;
        sd.hidden = None;
        sd.gain = GAIN0;
        sd.skip_left = 0;
        sd.backoff = 0;
        Ok(())
    }

    /// Stage 1: should `slot`'s stream draft this step? False while it backs
    /// off after its drafted blocks stopped paying (each call counts a step).
    pub fn should_draft(&mut self, slot: u32) -> bool {
        let Ok(sd) = self.slot(slot) else { return false };
        if sd.skip_left > 0 {
            sd.skip_left -= 1;
            self.stats.skipped += 1;
            return false;
        }
        true
    }

    /// Record row `pos` of `slot`'s stream as kept: its residual becomes the
    /// next draft's input, and (if `write_ring` and not already there) it goes
    /// into the ring.
    pub fn keep_row(
        &mut self,
        engine: &HeterogeneousEngine,
        mtp: &mut MtpCtx,
        slot: u32,
        pos: u32,
        hidden: Vec<f32>,
        write_ring: bool,
    ) -> eyre::Result<()> {
        if hidden.len() != HIDDEN {
            return Err(eyre!("ms dspark: residual of {} floats, want {HIDDEN}", hidden.len()));
        }
        let fresh = self.slot(slot)?.last_ring_pos.is_none_or(|p| pos > p);
        if write_ring && fresh {
            self.with_ring(mtp, slot, false, |m| engine.dspark_ring_write_only(&mut m.state, &m.w, pos, &hidden))?;
            self.slot(slot)?.last_ring_pos = Some(pos);
        }
        self.slot(slot)?.hidden = Some((pos, hidden));
        Ok(())
    }

    /// `keep_row` for the consecutive rows `pos0..pos0 + rows.len()` a step
    /// kept: rows not yet in the ring go in with ONE batched write per
    /// `RING_ROWS_MAX` (`MtpState::ring_write_rows`: one upload, one read of
    /// each projection, one sync), and the last row's residual feeds the next
    /// draft.
    pub fn keep_rows(
        &mut self,
        engine: &HeterogeneousEngine,
        mtp: &mut MtpCtx,
        slot: u32,
        pos0: u32,
        mut rows: Vec<Vec<f32>>,
        write_ring: bool,
    ) -> eyre::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        if rows.iter().any(|h| h.len() != HIDDEN) {
            return Err(eyre!("ms dspark: a kept row's residual is not {HIDDEN} floats"));
        }
        let last = pos0 + rows.len() as u32 - 1;
        if write_ring {
            let skip = match self.slot(slot)?.last_ring_pos {
                Some(p) if p >= pos0 => ((p - pos0) as usize + 1).min(rows.len()),
                _ => 0,
            };
            for (c, chunk) in rows[skip..].chunks(RING_ROWS_MAX).enumerate() {
                let p = pos0 + (skip + c * RING_ROWS_MAX) as u32;
                let flat: Vec<f32> = chunk.concat();
                self.with_ring(mtp, slot, false, |m| engine.dspark_ring_write_rows(&mut m.state, &m.w, p, &flat))?;
            }
            if skip < rows.len() {
                self.slot(slot)?.last_ring_pos = Some(last);
            }
        }
        let h = rows.pop().expect("non-empty");
        self.slot(slot)?.hidden = Some((last, h));
        Ok(())
    }

    /// Seed `slot`'s ring from a prefill's captured residuals (absolute
    /// position -> row), `last` = the prompt's last position: the contiguous
    /// run ending at `last`, at most a window, every row including the last
    /// (the first draft reads the ring through `last`).
    pub fn seed(
        &mut self,
        engine: &HeterogeneousEngine,
        mtp: &mut MtpCtx,
        slot: u32,
        mut rows: BTreeMap<u32, Vec<f32>>,
        last: u32,
    ) -> eyre::Result<usize> {
        self.reset(slot)?;
        let mut first = last;
        while first > 0 && rows.contains_key(&(first - 1)) && (last - first + 1) < MTP_WINDOW as u32 {
            first -= 1;
        }
        if !rows.contains_key(&last) {
            return Ok(0);
        }
        let run: Vec<Vec<f32>> = (first..=last).map(|p| rows.remove(&p).expect("contiguous run")).collect();
        let n = run.len();
        self.keep_rows(engine, mtp, slot, first, run, true)?;
        Ok(n)
    }

    /// Draft a block for `slot`'s stream: its last row in KV is `pos`, its
    /// next input `next` (at `pos + 1`, embedded in `token_row`). Returns the
    /// drafts for `pos + 2 ..= pos + 1 + MTP_BLOCK`, their confidence logits
    /// and (sampled drafts) their q, or `None` when the stream has no residual
    /// for `pos` (then it steps without drafts).
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &mut self,
        engine: &HeterogeneousEngine,
        weights: &HetModelWeights,
        mtp: &mut MtpCtx,
        slot: u32,
        pos: u32,
        next: i32,
        token_row: &[f32],
        sampling: Option<&DraftSampling>,
    ) -> eyre::Result<Option<Drafted>> {
        let t = Instant::now();
        let (hidden, rewind) = {
            let sd = self.slot(slot)?;
            let h = match sd.hidden.as_ref() {
                Some((p, h)) if *p == pos && pos > 0 => Some(h.clone()),
                _ => None,
            };
            (h, sd.last_ring_pos == Some(pos))
        };
        let Some(hidden) = hidden else {
            self.stats.no_hidden += 1;
            return Ok(None);
        };
        let ids = self.with_ring(mtp, slot, rewind, |m| {
            let MtpCtx { state, exit, w, xw, markov_embd, markov_dtype, noise_row, .. } = m;
            engine
                .dspark_draft(state, exit, &hidden, w, xw, weights, markov_embd, *markov_dtype, pos, token_row, noise_row, next, false, sampling)
                .map(|(ids, _plain)| ids)
        })?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let sd = self.slot(slot)?;
        sd.last_ring_pos = Some(pos);
        sd.last_draft_ms = ms;
        self.stats.draft_ms += ms;
        let q = sampling.map(|_| std::mem::take(&mut mtp.exit.q));
        Ok(Some(Drafted { ids, conf: mtp.exit.conf, q }))
    }

    /// A plain one-row decode step took `ms` (no drafts): the stage-1 baseline.
    pub fn note_plain_step(&mut self, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            self.plain_ms = 0.9 * self.plain_ms + 0.1 * ms;
        }
    }

    /// Account one verified block of `slot` (stage-1 gain, stats) and log a
    /// rollup every 50 blocks.
    pub fn record(&mut self, slot: u32, k: usize, accepted: usize, emitted: usize, step_ms: f64) {
        let plain_ms = self.plain_ms;
        if let Ok(sd) = self.slot(slot) {
            let g = emitted as f64 * plain_ms / (step_ms + sd.last_draft_ms).max(1.0);
            // A probe after a back-off moves the estimate half way at once.
            let a = if sd.backoff > 0 { 0.5 } else { 0.25 };
            sd.gain = (1.0 - a) * sd.gain + a * g;
            if sd.gain < min_gain() {
                sd.backoff = (sd.backoff * 2).clamp(4, 64);
                sd.skip_left = sd.backoff;
            } else {
                sd.backoff = 0;
            }
        }
        let s = &mut self.stats;
        s.blocks += 1;
        s.drafts_verified += k as u64;
        s.accepted += accepted as u64;
        s.emitted += emitted as u64;
        s.k_hist[k.min(MTP_BLOCK)] += 1;
        s.step_ms += step_ms;
        if s.blocks >= 50 {
            let n = s.blocks as f64;
            tracing::info!(
                blocks = s.blocks,
                tokens_per_block = format!("{:.2}", s.emitted as f64 / n),
                k_mean = format!("{:.2}", s.drafts_verified as f64 / n),
                accept_rate = format!("{:.3}", s.accepted as f64 / (s.drafts_verified as f64).max(1.0)),
                draft_ms = format!("{:.1}", s.draft_ms / n),
                step_ms = format!("{:.1}", s.step_ms / n),
                tok_per_s = format!("{:.2}", s.emitted as f64 / ((s.draft_ms + s.step_ms) / 1e3).max(1e-9)),
                k_hist = ?s.k_hist,
                no_hidden = s.no_hidden,
                skipped_steps = s.skipped,
                plain_ms = format!("{:.1}", self.plain_ms),
                window_s = self.since.elapsed().as_secs(),
                "ms dspark: blocks"
            );
            self.stats = Stats::default();
            self.since = Instant::now();
        }
    }

    pub fn ring_all(&self) -> bool {
        ring_all()
    }
}

/// The residuals a lane's step captured, one `[HIDDEN]` row per lane row
/// (`mtp_src` is slot-major `[3][MTP_CAP_ROWS][N_EMBD]`). The step must have
/// synchronized its stream.
pub fn lane_captures(bd: &BatchDgpuScratch, n: usize) -> eyre::Result<Vec<Vec<f32>>> {
    if n > MTP_CAP_ROWS || bd.mtp_captured != n {
        return Err(eyre!("ms dspark: lane captured {} rows, the step ran {n}", bd.mtp_captured));
    }
    let ne = N_EMBD as usize;
    let mut out = vec![vec![0f32; HIDDEN]; n];
    let mut tmp = vec![0f32; n * ne];
    for s in 0..MTP_SRC_LAYERS.len() {
        bd.mtp_src.slice_view(s * MTP_CAP_ROWS * ne, n * ne).copy_to_host(&mut tmp)?;
        for (k, row) in out.iter_mut().enumerate() {
            row[s * ne..(s + 1) * ne].copy_from_slice(&tmp[k * ne..(k + 1) * ne]);
        }
    }
    Ok(out)
}

/// A prefill's captures by absolute position (both lanes; `mtp_captured_pos0`
/// is each lane's first captured position).
pub fn prefill_captures(lanes: &[&BatchDgpuScratch]) -> eyre::Result<BTreeMap<u32, Vec<f32>>> {
    let mut m = BTreeMap::new();
    for bd in lanes {
        let n = bd.mtp_captured;
        if n == 0 {
            continue;
        }
        for (k, row) in lane_captures(bd, n)?.into_iter().enumerate() {
            m.insert(bd.mtp_captured_pos0 + k as u32, row);
        }
    }
    Ok(m)
}

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Step cost (ms) of `rows` rows of one stream: `V41_MS_DSPARK_COST`, a comma
/// list for 1, 2, ... rows (default: the 2026-09-30..10-01 production p50
/// ladder), extrapolated linearly past its end.
fn step_cost(rows: usize) -> f64 {
    static C: LazyLock<Vec<f64>> = LazyLock::new(|| {
        std::env::var("V41_MS_DSPARK_COST")
            .ok()
            .and_then(|v| v.split(',').map(|x| x.trim().parse().ok()).collect::<Option<Vec<f64>>>())
            .filter(|v| v.len() >= 2)
            .unwrap_or_else(|| vec![60.0, 81.0, 101.0, 107.0, 125.0, 139.0, 156.0, 172.0])
    });
    let c = &*C;
    if rows == 0 {
        return 0.0;
    }
    if rows <= c.len() {
        return c[rows - 1];
    }
    let n = c.len();
    c[n - 1] + (c[n - 1] - c[n - 2]) * (rows - n) as f64
}

/// `V41_MS_DSPARK_K`: verify exactly this many drafts (capped like the policy).
fn fixed_k() -> Option<usize> {
    static FIXED: LazyLock<Option<usize>> =
        LazyLock::new(|| std::env::var("V41_MS_DSPARK_K").ok().and_then(|v| v.parse().ok()));
    *FIXED
}

/// Most drafts a block may verify: `V41_MS_DSPARK_K` if set (0 = never draft,
/// the plain-decode control on the same binary), else `V41_MS_DSPARK_KMAX`
/// (default the block size).
pub fn k_max() -> usize {
    static KMAX: LazyLock<usize> =
        LazyLock::new(|| std::env::var("V41_MS_DSPARK_KMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(MTP_BLOCK));
    fixed_k().unwrap_or(*KMAX).min(MTP_BLOCK)
}

/// K for SAMPLED drafts (plan 2.4 / section 6): a STOPPING rule. Whether draft
/// k is verified is decided from `conf[..=k]` only -- conf_k reads the markov
/// row of draft k-1, never draft k -- with the depths beyond k forecast at
/// sigmoid(conf_k). So no decision reads the value it would test. The global
/// search of `choose_k` lets conf_{k+1}, which depends on d_k, decide whether
/// d_k is verified: exact for point-mass tests only (review N1).
pub fn choose_k_stopping(conf: &[f32; MTP_BLOCK], cap: usize) -> usize {
    let cap = cap.min(MTP_BLOCK);
    if let Some(k) = fixed_k() {
        return k.min(cap);
    }
    let draft = env_f64("V41_MS_DSPARK_DRAFT_MS", 20.0);
    let sig = |c: f32| 1.0 / (1.0 + (-(c as f64)).exp());
    let (mut k, mut run, mut e) = (0usize, 1.0f64, 1.0f64);
    while k < cap {
        let p = sig(conf[k]);
        let stop = e / (step_cost(1 + k) + draft);
        let (mut r, mut ee, mut go) = (run, e, f64::MIN);
        for kk in k + 1..=cap {
            r *= p;
            ee += r;
            go = go.max(ee / (step_cost(1 + kk) + draft));
        }
        if go <= stop {
            break;
        }
        run *= p;
        e += run;
        k += 1;
    }
    k
}

/// How many drafts to verify (plan section 6, point-mass tests: any K rule is
/// exact, 2.4): maximize expected tokens per ms, `E(K) = 1 + sum_{k<=K}
/// prod_{j<k} sigmoid(conf_j)` over `cost(1 + K) + draft`, the draft being
/// paid either way. `V41_MS_DSPARK_K` fixes K (capped like the policy).
pub fn choose_k(conf: &[f32; MTP_BLOCK], cap: usize) -> usize {
    let cap = cap.min(MTP_BLOCK);
    if let Some(k) = fixed_k() {
        return k.min(cap);
    }
    let draft = env_f64("V41_MS_DSPARK_DRAFT_MS", 20.0);
    let (mut best_k, mut best) = (0usize, 1.0 / (step_cost(1) + draft));
    let (mut run, mut e) = (1.0f64, 1.0f64);
    for k in 1..=cap {
        run *= 1.0 / (1.0 + (-(conf[k - 1] as f64)).exp());
        e += run;
        let r = e / (step_cost(1 + k) + draft);
        if r > best {
            best = r;
            best_k = k;
        }
    }
    best_k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_k_follows_confidence() {
        // Confident blocks go deep, unconfident ones verify nothing.
        assert_eq!(choose_k(&[6.0; MTP_BLOCK], MTP_BLOCK), MTP_BLOCK);
        assert_eq!(choose_k(&[-6.0; MTP_BLOCK], MTP_BLOCK), 0);
        // The cap binds.
        assert_eq!(choose_k(&[6.0; MTP_BLOCK], 2), 2);
        assert_eq!(choose_k(&[6.0; MTP_BLOCK], 0), 0);
        // One confident draft then noise: stop after it.
        assert_eq!(choose_k(&[6.0, -6.0, -6.0, -6.0, -6.0], MTP_BLOCK), 1);
    }

    #[test]
    fn stopping_rule_never_reads_past_its_stop() {
        // Changing any confidence AFTER the chosen K must not change K: the
        // decision on draft k reads conf[..=k] only (exactness for sampled
        // drafts, plan 2.4).
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) * 12.0 - 6.0
        };
        for _ in 0..2000 {
            let conf: [f32; MTP_BLOCK] = std::array::from_fn(|_| next());
            let k = choose_k_stopping(&conf, MTP_BLOCK);
            for j in (k + 1)..MTP_BLOCK {
                let mut c2 = conf;
                c2[j] = next();
                assert_eq!(choose_k_stopping(&c2, MTP_BLOCK), k, "conf {conf:?}: K moved when conf[{j}] changed");
            }
        }
        assert_eq!(choose_k_stopping(&[6.0; MTP_BLOCK], MTP_BLOCK), MTP_BLOCK);
        assert_eq!(choose_k_stopping(&[-6.0; MTP_BLOCK], MTP_BLOCK), 0);
    }

    #[test]
    fn step_cost_extrapolates() {
        assert_eq!(step_cost(1), 60.0);
        assert_eq!(step_cost(8), 172.0);
        assert_eq!(step_cost(9), 188.0);
    }
}
