//! DSpark on the multistream arena (docs/v41/DSPARK_ARENA_PLAN.md), the
//! single-stream build: when exactly one stream is live it verifies the
//! drafter's block in the SAME arena step as its next token (rows `[next, d_0
//! .. d_{K-1}]`, `KvArena::tables`), rejection-samples the rows against the
//! drafts (`spec_sample::verify_block`, point-mass tests) and keeps the KV of
//! the rows it emitted (`KvArena::commit`). With several live streams every
//! step is a plain multistream step (K = 0), and the rings are kept current
//! so a stream that becomes the lone one drafts from a dense window.
//!
//! What lives here: one drafter ring per arena slot (`DrafterState::rings` is the
//! drafter's only state that persists across drafts; the rest of `DrafterState`
//! is per-draft scratch, so ONE `DrafterState` serves every slot by swapping the
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
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use v4flash_hip::DeviceBuffer;
use v4flash_kernels::config::N_EMBD;
use v4flash_kernels::het::batch_scratch::{BatchDgpuScratch, DRAFT_CAP_ROWS};
use v4flash_kernels::het::drafter::{DraftSampling, DRAFT_BLOCK, DRAFT_TOP_M, DRAFT_SRC_LAYERS, DRAFT_WINDOW, RING_ROWS_MAX};
use v4flash_kernels::het::SampleMode;
use v4flash_kernels::het::{HetModelWeights, HeterogeneousEngine};

use crate::engine_worker::DrafterCtx;

/// `V41_MS_DSPARK=accept`: DSpark on the arena path. Loads the drafter
/// (as `V41_DSPARK` does) and keeps multistream on for every request.
pub fn enabled() -> bool {
    crate::knobs::MS_DSPARK.pick() == 1
}

/// `V41_MS_DSPARK_RING=solo`: ring-write only a LONE stream's rows (every step,
/// drafted or not); steps with 2+ live streams write none.
/// Default `all`: every live stream's kept rows go into its ring every step
/// (about one `main_proj` + 3 `attn_kv` matvecs on the iGPU per row), so a
/// stream that becomes the lone stream drafts from a dense window instead of
/// one with a gap where it ran beside others.
fn ring_all() -> bool {
    crate::knobs::MS_DSPARK_RING.pick() == 0
}

/// `V41_MS_DSPARK_DRAFTS=argmax`: point-mass drafts (the drafter's argmax,
/// accepted with probability p(d)) for every request. Default `sampled`: a
/// request that samples (temperature > 0) gets SAMPLED drafts, each drawn from
/// the drafter's own tempered top-M distribution q and accepted with
/// min(1, p/q) (plan 2.2, M6) -- the drafter as designed; at temperature 0 a
/// draft is the argmax either way.
fn sampled_drafts() -> bool {
    crate::knobs::MS_DSPARK_DRAFTS.pick() == 0
}

/// How a stream's drafts are drawn: `Some` (sampled, plan 2.2) when its
/// requests sample and sampled drafts are on, else `None` (point mass). `u`
/// are the drafter RNG's uniforms for the block's positions.
pub fn draft_sampling(mode: &SampleMode, u: [f32; DRAFT_BLOCK]) -> Option<DraftSampling> {
    match *mode {
        SampleMode::Multinomial { temperature, top_p, min_p_rel } if temperature > 0.0 && sampled_drafts() => {
            Some(DraftSampling { temperature, top_p, min_p_rel, m: DRAFT_TOP_M, u })
        }
        _ => None,
    }
}

/// One drafted block.
pub struct Drafted {
    pub ids: [i32; DRAFT_BLOCK],
    /// Confidence logits (`DrafterExit::conf`).
    pub conf: [f32; DRAFT_BLOCK],
    /// Sampled drafts: each position's q, `(token, prob)`; `None` for argmax.
    pub q: Option<Vec<Vec<(i32, f64)>>>,
}

/// Floats of one row's drafter input: `cat(mean_hc(resid@37), @38, @39)`.
pub const HIDDEN: usize = DRAFT_SRC_LAYERS.len() * N_EMBD as usize;

/// One arena slot's drafter memory.
struct SlotDraft {
    /// `DrafterState::rings` for this slot's stream.
    rings: Vec<DeviceBuffer<u16>>,
    /// Its `ring_writes`.
    writes: usize,
    /// Position of the latest ring row (rows go in, in position order).
    last_ring_pos: Option<u32>,
    /// `(pos, drafter_src of row pos)` for the stream's LAST row in KV: what its
    /// next draft reads.
    hidden: Option<(u32, Vec<f32>)>,
    /// Draft-or-not (plan section 6, stage 1), one gate per regime:
    /// `[LONE, MULTI]` -- drafting alone and drafting beside other live
    /// streams pay differently (a K = 0 beside another stream's block can be
    /// a lost competition for rows), so a back-off earned in one says nothing
    /// about the other (MS_DSPARK_STREAMS_DESIGN 2.5).
    gates: [Gate; 2],
    /// The last draft's wall time (ms), for its block's gain.
    last_draft_ms: f64,
    /// Recorded on `igpu.compute` after this slot's last ASYNC ring write
    /// (`keep_rows`); `write_pending` until `settle_writes` has waited on it.
    write_done: v4flash_hip::Event,
    write_pending: bool,
}

/// `V41_MS_DSPARK_RING_ASYNC=0`: kept-row ring writes synchronize as before
/// (default: enqueued without a sync, overlapping the step's host tail).
fn ring_async() -> bool {
    crate::knobs::MS_DSPARK_RING_ASYNC.on()
}

/// Initial `gain` of a stream: optimistic, so it drafts until measured.
const GAIN0: f64 = 1.5;

fn min_gain() -> f64 {
    crate::knobs::MS_DSPARK_MIN_GAIN.f64()
}

#[derive(Default)]
struct Stats {
    blocks: u64,
    drafts_verified: u64,
    accepted: u64,
    emitted: u64,
    k_hist: [u64; DRAFT_BLOCK + 1],
    draft_ms: f64,
    step_ms: f64,
    no_hidden: u64,
    /// Steps a stream sat out drafting (stage-1 back-off).
    skipped: u64,
    /// Blocks verified as an ordered two-lane cut.
    two_lane: u64,
    /// `keep_rows` calls and their host wall time (ms).
    keeps: u64,
    keep_ms: f64,
    /// Async ring writes that failed (the slot was reset).
    ring_errors: u64,
    /// Blocks whose K (and lanes) was an exploration draw (`explore`), K >= 1.
    explored: u64,
    /// Exploration draws of K = 0: drafted, then a plain step (`record_k0`:
    /// in `blocks` and `k_hist[0]`).
    explored_k0: u64,
    /// The POLICY's K = 0 (`choose_k*` priced no draft worth a row): drafted,
    /// then a plain step, like `explored_k0`.
    policy_k0: u64,
    /// Exploration draws by K (K = 0 included).
    explored_hist: [u64; DRAFT_BLOCK + 1],
    /// Exploration draws that verified on two lanes.
    explored_two: u64,
    /// Blocks by rows (index rows - 1): on one lane, on two.
    lanes_by_rows: [[u64; 2]; CELLS],
    /// Steps where two or more streams were offered drafts
    /// (`MsDspark::ks_for`; docs/v41/MS_DSPARK_STREAMS_DESIGN.md 2), of them
    /// steps that verified any draft, and the blocks verified in them.
    multi_steps: u64,
    multi_spec_steps: u64,
    multi_blocks: u64,
    /// Exploration draws among those steps.
    multi_explored: u64,
}

impl Stats {
    /// A drafted step that verified nothing (K = 0) took `step_ms`: one token,
    /// counted like a block (`k_hist[0]`), so `tokens_per_block`, `k_mean`
    /// and `tok_per_s` see the draft it paid for (`draft_ms` already has it).
    fn note_k0(&mut self, step_ms: f64) {
        self.blocks += 1;
        self.emitted += 1;
        self.k_hist[0] += 1;
        self.lanes_by_rows[0][0] += 1;
        self.step_ms += step_ms;
    }
}

/// `SlotDraft::gates` index: one live stream.
const LONE: usize = 0;
/// `SlotDraft::gates` index: two or more live streams.
const MULTI: usize = 1;

/// One slot's stage-1 gate in one regime: EWMA of the realized speed-up of a
/// drafted step over the step it would otherwise have run (`gain`); below
/// `V41_MS_DSPARK_MIN_GAIN` the stream stops drafting for `backoff` steps (4,
/// doubling to 64), then probes with one block.
#[derive(Clone, Copy, Debug)]
struct Gate {
    gain: f64,
    skip_left: u32,
    backoff: u32,
}

impl Default for Gate {
    fn default() -> Self {
        Self { gain: GAIN0, skip_left: 0, backoff: 0 }
    }
}

impl Gate {
    fn update(&mut self, g: f64) {
        gate_update(&mut self.gain, &mut self.backoff, &mut self.skip_left, g, min_gain());
    }
}

/// Stage-1 gate (`Gate::gain`): fold one drafted step's realized speed-up
/// `g` into the EWMA and back off below `min_gain` (4 steps, doubling to 64).
/// A probe after a back-off moves the estimate half way at once.
fn gate_update(gain: &mut f64, backoff: &mut u32, skip_left: &mut u32, g: f64, min_gain: f64) {
    let a = if *backoff > 0 { 0.5 } else { 0.25 };
    *gain = (1.0 - a) * *gain + a * g;
    if *gain < min_gain {
        *backoff = (*backoff * 2).clamp(4, 64);
        *skip_left = *backoff;
    } else {
        *backoff = 0;
    }
}

pub struct MsDspark {
    slots: Vec<SlotDraft>,
    /// EWMA of a plain one-row step's wall (ms), the stage-1 baseline when the
    /// cost model is static: a drafted block pays when it emits more per ms
    /// than a plain step NOW (a cold pool slows both; the static ladder made
    /// cold blocks look like losses and backed off 324 steps of a code
    /// stream, 2026-10-01). With the live model, `cost.cost(1)` is that
    /// baseline (fitted from every lone step, not only the rare plain ones).
    plain_ms: f64,
    /// A block's step cost by rows on one lane and on two (an ordered verify
    /// cut): each table observed only from steps that ran its lane count, so
    /// the step change where the second lane switches on bends neither.
    lanes: LaneTables,
    /// The policy's lane count changes by rows (`Switches`; blocks line).
    switches: Switches,
    calib: Calib,
    stats: Stats,
    since: Instant,
    /// Exploration draws (`explore`); `V41_MS_DSPARK_EXPLORE_SEED` for a
    /// reproducible replay, else entropy.
    rng: StdRng,
    /// The last `k_for` returning K = 0 was an exploration draw (`record_k0`
    /// then feeds no gain sample).
    k0_explored: bool,
    /// Step cost of steps where two or more streams speculate, by the step's
    /// TOTAL rows, on one lane and on two (design 2.3): their own tables --
    /// neither the lone blocks' (one KV stream, an ordered cut at every split)
    /// nor the plain steps' (one row per stream) -- aged by those steps.
    multi: LaneTables,
    /// The last `ks_for` result was an exploration draw: its K = 0 blocks
    /// were not the policy's choice and feed no stage-1 sample.
    multi_explored_last: bool,
}

/// Most rows of a speculating step per lane: kernel families switch regime
/// above 8 rows per lane (plan 3.7: `mhc_pre_scaled_for`, the dp4a arms), a
/// regime production decode never runs; it also keeps a lane under the pin
/// (16) and hot-set (8) bookkeeping limits.
pub const SPEC_ROWS_PER_LANE: usize = 8;

/// The most rows a multi-stream speculating step prices: two streams' blocks.
const MULTI_ROWS: usize = 2 * (1 + DRAFT_BLOCK);

/// The multi-stream tables' starting cells (design 2.3), MEASURED 2026-10-04
/// 00:50-08:33 UTC (hub 9c6f8ea5, `ms.step` p50): one lane, rows 1..=6 =
/// the plain lone row and the lone one-lane verify blocks, then +9.7 ms/row;
/// two lanes, rows 2..=6 = 2 plain streams (62.5) and the lone two-lane
/// verify blocks, then +9.4 ms/row (row 1 never runs two lanes).
const MULTI_LADDER_ONE: [f64; 8] = [54.7, 71.1, 84.4, 97.9, 108.3, 117.1, 126.8, 136.5];
const MULTI_LADDER_TWO: [f64; 12] = [54.7, 62.5, 74.4, 83.1, 93.9, 102.8, 112.2, 121.6, 131.0, 140.4, 149.8, 159.2];

/// The multi-stream cost tables (design 2.3), starting from the 10-04 ladders;
/// `m` = the startup two-lane threshold (`LaneTables::new`).
fn multi_tables(m: usize) -> LaneTables {
    let memory = crate::knobs::MS_DSPARK_COST_MEMORY.f64();
    let live = crate::knobs::MS_DSPARK_COST_LIVE.on();
    LaneTables::new(
        StepCost::new(MULTI_LADDER_ONE.to_vec(), 0.0, live, memory).with_shape(CostShape::Cells).with_rows(MULTI_ROWS),
        StepCost::with_first_row(MULTI_LADDER_TWO.to_vec(), m, 0.0, live, memory).with_shape(CostShape::Cells).with_rows(MULTI_ROWS),
        m,
    )
}

/// The most live streams that may ever draft together (the knob's ceiling;
/// owner: more than two is not worth it).
pub const MAX_SPEC_STREAMS: usize = 2;

/// Set the first time `spec_streams` returned more than 1 in this process.
static MULTI_EVER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `V41_MS_DSPARK_STREAMS`: the most live streams that may all draft in one
/// step (1 = a lone stream only, today's rule).
pub fn spec_streams() -> usize {
    let n = crate::knobs::MS_DSPARK_STREAMS.usize().clamp(1, MAX_SPEC_STREAMS);
    if n > 1 {
        MULTI_EVER.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    n
}

/// The most live streams whose drafter rings are kept written under
/// `V41_MS_DSPARK_RING=solo`: the knob's value until it has ever been above
/// 1 in this process (a deploy at 1 writes exactly what it did before), then
/// its ceiling, so a live flip back and forth (the per-turn A/B) always finds
/// every stream's ring dense (review rounds 2-3).
pub fn ring_streams(spec_max: usize) -> usize {
    if MULTI_EVER.load(std::sync::atomic::Ordering::Relaxed) { MAX_SPEC_STREAMS } else { spec_max }
}

impl MsDspark {
    /// One ring per arena slot, shaped like `mtp.state`'s.
    pub fn alloc(drafter: &DrafterCtx, igpu_id: i32, n_slots: u32) -> eyre::Result<Self> {
        let mut slots = Vec::with_capacity(n_slots as usize);
        // The ring-write events live on the iGPU, where the writes run. A
        // SCOPED switch: a bare set_current would leave the engine's cached
        // current-device mirror stale (KNOWN_BUGS #10).
        let _dev = v4flash_hip::Device::new(igpu_id).scoped_current()?;
        for _ in 0..n_slots {
            let rings = drafter.state.rings.iter().map(|r| DeviceBuffer::<u16>::new(igpu_id, r.len())).collect::<eyre::Result<Vec<_>>>()?;
            let write_done = v4flash_hip::Event::new_no_timing()?;
            slots.push(SlotDraft {
                rings, writes: 0, last_ring_pos: None, hidden: None, gates: [Gate::default(); 2], last_draft_ms: 0.0,
                write_done, write_pending: false,
            });
        }
        // The threshold at startup shapes the two-lane line's prior and the
        // cells' starts only (`LaneTables::new`); both tables cover every row
        // count, so a live change of the threshold or the rule just changes
        // which cells the steps feed.
        let m = crate::multistream::pipeline_min_rows();
        let lanes = LaneTables::new(StepCost::from_env(), StepCost::from_env_two_lane(m), m);
        Ok(Self::with_slots(slots, lanes, m))
    }

    fn with_slots(slots: Vec<SlotDraft>, lanes: LaneTables, m: usize) -> Self {
        Self {
            slots, plain_ms: lanes.one.cost(1), lanes, switches: Switches::default(), calib: Calib::default(), stats: Stats::default(),
            since: Instant::now(), rng: explore_rng(), k0_explored: false, multi: multi_tables(m), multi_explored_last: false,
        }
    }

    /// No slots (no device rings or events): the policy and accounting alone.
    #[cfg(test)]
    fn bare() -> Self {
        let lanes = LaneTables::new(StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0), StepCost::with_first_row(DEFAULT_LADDER_TWO_LANE.to_vec(), 4, 12.0, true, 500.0), 4);
        Self::with_slots(Vec::new(), lanes, 4)
    }

    /// K per drafting stream and the lane count for a step where two or more
    /// streams speculate (design 2.2). `blocks` = the drafting streams' blocks,
    /// `base_rows` = the step's rows before drafts (one per live stream),
    /// `sampled` = any stream's drafts are sampled (then the joint stopping
    /// rule, else the global search), `rule` = this step's lane snapshot. The
    /// draft time charged is one draft per drafting stream (serial drafts).
    /// Exploration first, drawn before any confidence is read. Rows are
    /// capped at `SPEC_ROWS_PER_LANE` per lane.
    pub fn ks_for(&mut self, blocks: &[BlockConf], base_rows: usize, sampled: bool, rule: LaneRule) -> (Vec<usize>, usize) {
        self.stats.multi_steps += 1;
        let draft = self.lanes.one.draft_ms() * blocks.len() as f64;
        if fixed_k().is_none() {
            // Draw a feasible (rows, lanes) CELL by staleness, then one of the
            // K vectors that land there uniformly: drawing over K vectors
            // would favour the cells many vectors map to.
            let cands = multi_choices(&self.multi, blocks, base_rows, rule);
            let mut cells: Vec<(usize, usize)> = cands.iter().map(|c| (c.1, c.2)).collect();
            cells.sort_unstable();
            cells.dedup();
            let t = &self.multi;
            if let Some(i) = explore(&mut self.rng, cells.len(), explore_p(), |i| t.weight(cells[i].0, cells[i].1)) {
                self.stats.multi_explored += 1;
                let mut these: Vec<&(Vec<usize>, usize, usize)> = cands.iter().filter(|c| (c.1, c.2) == cells[i]).collect();
                let pick = these.swap_remove(self.rng.gen_range(0..these.len()));
                self.k0_explored = pick.0.iter().all(|&k| k == 0);
                self.multi_explored_last = true;
                return (pick.0.clone(), pick.2);
            }
        }
        self.k0_explored = false;
        self.multi_explored_last = false;
        let cost = MultiPriced { t: &self.multi, rule, draft };
        let mut ks = if sampled { choose_ks_stopping(blocks, base_rows, &cost) } else { choose_ks(blocks, base_rows, &cost) };
        if fixed_k().is_some() {
            fit_rows(&mut ks, base_rows, &self.multi, rule);
        } else {
            // The policies price an unfit step at infinity and never pick it;
            // trimming a stopping-rule result after it read `conf_s[K_s]` would
            // let a draft's inclusion read its own value (review round 2).
            debug_assert!(
                ks.iter().all(|&k| k == 0) || multi_best(&self.multi, base_rows + ks.iter().sum::<usize>(), rule).is_some(),
                "ks_for: the policy chose an unfit step {ks:?}"
            );
        }
        let rows = base_rows + ks.iter().sum::<usize>();
        let lanes = multi_best(&self.multi, rows, rule).map(|b| b.1).unwrap_or(1);
        (ks, lanes)
    }


    /// One step where two or more live streams were offered drafts and some
    /// verified any (`ks_for` gave a K > 0): every drafting stream's block
    /// (`(slot, conf, k, accepted, emitted)`; K = 0 included: it drafted and
    /// paid for it), the step's rows / lanes / lane rule / wall time, and
    /// `plain_ms` = the plain step of the same streams (`PlainLanes`).
    ///
    /// The stage-1 sample of stream s compares it with the step it would have
    /// RIDDEN without drafting (review round 2): the same step less its rows
    /// and its draft, `g_s = emitted_s x (c(R - K_s) + D - D_s) / (step + D)`,
    /// `c` = `plain_ms` when the rest is plain, else the multi tables. A
    /// stream whose draft bought no rows (K_s = 0) gets `< 1` and backs off; a
    /// weaker stream beside a deep block is judged on what its own rows bought.
    #[allow(clippy::too_many_arguments)]
    pub fn record_multi(&mut self, blocks: &[(u32, [f32; DRAFT_BLOCK], usize, usize, usize)], rows: usize, lanes: usize, rule: LaneRule, step_ms: f64, plain_ms: f64) {
        let draft_of = |d: &Self, slot: u32| d.slots.get(slot as usize).map(|sd| sd.last_draft_ms).unwrap_or(0.0);
        let drafts_ms: f64 = blocks.iter().map(|b| draft_of(self, b.0)).sum();
        let base_rows = rows - blocks.iter().map(|b| b.2).sum::<usize>();
        // Each block's ridden step, priced BEFORE this step's own sample lands.
        let rode: Vec<f64> = blocks
            .iter()
            .map(|b| if rows - b.2 == base_rows { plain_ms } else { multi_best(&self.multi, rows - b.2, rule).map(|c| c.0).unwrap_or(plain_ms) })
            .collect();
        self.multi.observe(rows, lanes, step_ms);
        self.stats.multi_spec_steps += 1;
        let explored = self.multi_explored_last;
        for (&(slot, conf, k, accepted, emitted), &rode) in blocks.iter().zip(&rode) {
            if k > 0 {
                self.calib.observe(&conf, k, accepted);
                if self.calib.blocks % CALIB_EVERY == 0 {
                    self.calib.log();
                }
            }
            // An exploration draw's K = 0 was not the policy's choice: no
            // sample (as `record_k0`).
            let d_s = draft_of(self, slot);
            if !(explored && k == 0) {
                if let Ok(sd) = self.slot(slot) {
                    sd.gates[MULTI].update(emitted as f64 * (rode + drafts_ms - d_s) / (step_ms + drafts_ms).max(1.0));
                }
            }
            let s = &mut self.stats;
            s.blocks += 1;
            s.multi_blocks += 1;
            s.drafts_verified += k as u64;
            s.accepted += accepted as u64;
            s.emitted += emitted as u64;
            s.k_hist[k.min(DRAFT_BLOCK)] += 1;
            s.step_ms += step_ms / blocks.len() as f64;
        }
        self.maybe_log();
    }

    /// A step where two or more live streams drafted and `ks_for` verified
    /// none: a plain step that paid for its drafts. Each drafting stream's
    /// sample is the step without its own draft over the step with it,
    /// `(step + D - D_s) / (step + D)` (< 1: it backs off, as `record_k0`);
    /// the step's time goes to the PLAIN tables (the caller's
    /// `PlainLanes::observe`), not these.
    pub fn record_k0_multi(&mut self, slots: &[u32], step_ms: f64) {
        let draft_of = |d: &Self, slot: u32| d.slots.get(slot as usize).map(|sd| sd.last_draft_ms).unwrap_or(0.0);
        let drafts_ms: f64 = slots.iter().map(|&s| draft_of(self, s)).sum();
        for &slot in slots {
            // An exploration draw measures a choice the policy did not make: no gain.
            if !self.k0_explored {
                let d_s = draft_of(self, slot);
                if let Ok(sd) = self.slot(slot) {
                    sd.gates[MULTI].update((step_ms + drafts_ms - d_s) / (step_ms + drafts_ms).max(1.0));
                }
            }
            self.stats.blocks += 1;
            self.stats.emitted += 1;
            self.stats.k_hist[0] += 1;
            self.stats.step_ms += step_ms / slots.len() as f64;
        }
        self.maybe_log();
    }

    /// K and the lane count for the lone stream's drafted block (plan section
    /// 6): the stopping rule for sampled drafts, the global search for
    /// point-mass ones (2.4), both pricing a block of `rows` rows at the
    /// cheapest lane count `rule` allows it (`LaneTables::best`); the lanes are
    /// then that cheapest count for the chosen rows. `rule` = this step's
    /// snapshot; the step runs what this returns.
    pub fn k_for(&mut self, conf: &[f32; DRAFT_BLOCK], cap: usize, sampled: bool, rule: LaneRule) -> (usize, usize) {
        if fixed_k().is_none() {
            // Drawn before `conf` is read (block comment at `explore_p`): one
            // (K, lanes) of the rule's, from the time-aged weight of the cost
            // cell each would feed.
            let t = &self.lanes;
            let cands = t.block_choices(cap, rule);
            if let Some(i) = explore(&mut self.rng, cands.len(), explore_p(), |i| t.weight(cands[i].0 + 1, cands[i].1)) {
                let (k, l) = cands[i];
                self.stats.explored_hist[k] += 1;
                // A K = 0 draw runs as a plain step (`record_k0`): counted apart.
                if k == 0 {
                    self.stats.explored_k0 += 1;
                    self.k0_explored = true;
                } else {
                    self.stats.explored += 1;
                }
                if l >= 2 {
                    self.stats.explored_two += 1;
                }
                return (k, l);
            }
        }
        let cost = self.lanes.priced(rule);
        let k = if sampled { choose_k_stopping(conf, cap, &cost) } else { choose_k(conf, cap, &cost) };
        if k == 0 {
            self.stats.policy_k0 += 1;
            self.k0_explored = false;
        }
        let lanes = self.lanes.best(1 + k, rule).1;
        if k >= 1 && self.switches.note(1 + k, lanes) {
            self.lanes.log_switch("dspark", 1 + k, lanes);
        }
        (k, lanes)
    }

    /// One lone step's sample: BOTH tables age by one step, then `rows`/`ms`
    /// joins the one it ran in (`LaneTables::observe`). Fits forget by time
    /// (lone steps), not by their
    /// own samples: a fit the policy stops using must not keep stale data --
    /// e.g. a cold-start level -- for good (10-01: the one-lane line froze at
    /// 135 + 15/row while every block ran two lanes). "Time" = LONE steps:
    /// multi-stream steps age nothing, so a lone stream after a long
    /// multi-stream stretch starts from the fits it left (aging and exploration
    /// resume at once). The regime the policy does not use thus stays YOUNG for
    /// good (~8 exploration samples per memory: a strong level prior and the
    /// tight clamp) -- intended: that is what undoes a gross trap either way.
    fn observe(&mut self, lanes: usize, rows: usize, ms: f64) {
        self.lanes.observe(rows, lanes, ms);
    }

    fn slot(&mut self, slot: u32) -> eyre::Result<&mut SlotDraft> {
        self.slots.get_mut(slot as usize).ok_or_else(|| eyre!("ms dspark: slot {slot} out of range"))
    }

    /// Run `f` with `slot`'s ring swapped into `mtp.state` (pointer swap, no
    /// copy), and swap it back whatever `f` returns.
    fn with_ring<T>(
        &mut self,
        drafter: &mut DrafterCtx,
        slot: u32,
        rewind: bool,
        f: impl FnOnce(&mut DrafterCtx) -> eyre::Result<T>,
    ) -> eyre::Result<T> {
        let sd = self.slot(slot)?;
        std::mem::swap(&mut drafter.state.rings, &mut sd.rings);
        drafter.state.set_ring_writes(sd.writes - usize::from(rewind && sd.writes > 0));
        let r = f(drafter);
        sd.writes = drafter.state.ring_writes();
        std::mem::swap(&mut drafter.state.rings, &mut sd.rings);
        r
    }

    /// Wait for every slot's outstanding async ring write and attribute an
    /// error to the slot that issued it (its ring restarts). Called before
    /// anything that blocks on the iGPU on the step path -- the top of
    /// `decode_rows` (before the pager's blocking copies), `keep_rows` (its
    /// blocking `rows_in` upload), `seed`, `draft` -- because such a call
    /// would return the earlier write's fault as its own. Normally free: the
    /// write finished under the previous step's host tail. A sticky device
    /// fault poisons the context and fails the step regardless, as before.
    pub fn settle_writes(&mut self) -> f64 {
        let t = Instant::now();
        for i in 0..self.slots.len() {
            if !self.slots[i].write_pending {
                continue;
            }
            self.slots[i].write_pending = false;
            if let Err(e) = self.slots[i].write_done.synchronize() {
                tracing::warn!(slot = i, error = %e, "ms dspark: async ring write failed; the stream's ring restarts");
                self.stats.ring_errors += 1;
                let _ = self.reset(i as u32);
            }
        }
        t.elapsed().as_secs_f64() * 1e3
    }

    /// Forget `slot`'s stream (a new request took the slot).
    pub fn reset(&mut self, slot: u32) -> eyre::Result<()> {
        let sd = self.slot(slot)?;
        if sd.write_pending {
            // The old stream's write targets this slot's ring: let it land
            // before the ring is reused (its outcome no longer matters).
            sd.write_pending = false;
            let _ = sd.write_done.synchronize();
        }
        sd.writes = 0;
        sd.last_ring_pos = None;
        sd.hidden = None;
        sd.gates = [Gate::default(); 2];
        Ok(())
    }

    /// Stage 1: should `slot`'s stream draft this step? False while it backs
    /// off after its drafted blocks stopped paying (each call counts a step).
    /// `multi` = other live streams may draft beside it (its MULTI gate).
    pub fn should_draft(&mut self, slot: u32, multi: bool) -> bool {
        let Ok(sd) = self.slot(slot) else { return false };
        let gate = &mut sd.gates[if multi { MULTI } else { LONE }];
        if gate.skip_left > 0 {
            gate.skip_left -= 1;
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
        drafter: &mut DrafterCtx,
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
            self.with_ring(drafter, slot, false, |m| engine.dspark_ring_write_only(&mut m.state, &m.w, pos, &hidden))?;
            self.slot(slot)?.last_ring_pos = Some(pos);
        }
        self.slot(slot)?.hidden = Some((pos, hidden));
        Ok(())
    }

    /// `keep_row` for the consecutive rows `pos0..pos0 + rows.len()` a step
    /// kept: rows not yet in the ring go in with ONE batched write per
    /// `RING_ROWS_MAX` (`DrafterState::ring_write_rows`: one upload, one read of
    /// each projection), and the last row's residual feeds the next draft. The
    /// write is enqueued WITHOUT a sync (`V41_MS_DSPARK_RING_ASYNC`, default
    /// on): it runs under the step's host tail, and the next draft's blocking
    /// upload finds it done; its event goes to `settle_writes`. Ring contents,
    /// order and the draft's rewind are unchanged.
    pub fn keep_rows(
        &mut self,
        engine: &HeterogeneousEngine,
        drafter: &mut DrafterCtx,
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
        let t = Instant::now();
        let last = pos0 + rows.len() as u32 - 1;
        if write_ring {
            // The upload below blocks on the iGPU: settle every slot's write
            // first, so an async fault is charged to the slot that issued it.
            self.settle_writes();
            let skip = match self.slot(slot)?.last_ring_pos {
                Some(p) if p >= pos0 => ((p - pos0) as usize + 1).min(rows.len()),
                _ => 0,
            };
            let asynchronous = ring_async();
            for (c, chunk) in rows[skip..].chunks(RING_ROWS_MAX).enumerate() {
                let p = pos0 + (skip + c * RING_ROWS_MAX) as u32;
                let flat: Vec<f32> = chunk.concat();
                if asynchronous {
                    self.with_ring(drafter, slot, false, |m| engine.dspark_ring_write_rows_async(&mut m.state, &m.w, p, &flat))?;
                    // After the write in `igpu.compute` order: its completion.
                    let sd = self.slot(slot)?;
                    sd.write_done.record(&engine.igpu.compute)?;
                    sd.write_pending = true;
                } else {
                    self.with_ring(drafter, slot, false, |m| engine.dspark_ring_write_rows(&mut m.state, &m.w, p, &flat))?;
                }
            }
            if skip < rows.len() {
                self.slot(slot)?.last_ring_pos = Some(last);
            }
        }
        self.stats.keeps += 1;
        self.stats.keep_ms += t.elapsed().as_secs_f64() * 1e3;
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
        drafter: &mut DrafterCtx,
        slot: u32,
        mut rows: BTreeMap<u32, Vec<f32>>,
        last: u32,
    ) -> eyre::Result<usize> {
        self.settle_writes();
        self.reset(slot)?;
        let mut first = last;
        while first > 0 && rows.contains_key(&(first - 1)) && (last - first + 1) < DRAFT_WINDOW as u32 {
            first -= 1;
        }
        if !rows.contains_key(&last) {
            return Ok(0);
        }
        let run: Vec<Vec<f32>> = (first..=last).map(|p| rows.remove(&p).expect("contiguous run")).collect();
        let n = run.len();
        self.keep_rows(engine, drafter, slot, first, run, true)?;
        Ok(n)
    }

    /// Draft a block for `slot`'s stream: its last row in KV is `pos`, its
    /// next input `next` (at `pos + 1`, embedded in `token_row`). Returns the
    /// drafts for `pos + 2 ..= pos + 1 + DRAFT_BLOCK`, their confidence logits
    /// and (sampled drafts) their q, or `None` when the stream has no residual
    /// for `pos` (then it steps without drafts).
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &mut self,
        engine: &HeterogeneousEngine,
        weights: &HetModelWeights,
        drafter: &mut DrafterCtx,
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
        let ids = self.with_ring(drafter, slot, rewind, |m| {
            let DrafterCtx { state, exit, w, xw, markov_embd, markov_dtype, noise_row, .. } = m;
            engine
                .dspark_draft(state, exit, &hidden, w, xw, weights, markov_embd, *markov_dtype, pos, token_row, noise_row, next, false, sampling)
                .map(|(ids, _plain)| ids)
        })?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let sd = self.slot(slot)?;
        sd.last_ring_pos = Some(pos);
        sd.last_draft_ms = ms;
        self.stats.draft_ms += ms;
        self.lanes.one.observe_draft(ms);
        let q = sampling.map(|_| std::mem::take(&mut drafter.exit.q));
        Ok(Some(Drafted { ids, conf: drafter.exit.conf, q }))
    }

    /// A plain one-row decode step took `ms` (no drafts): the stage-1 baseline.
    pub fn note_plain_step(&mut self, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            self.plain_ms = 0.9 * self.plain_ms + 0.1 * ms;
        }
        self.observe(1, 1, ms);
    }

    /// Account one verified block of `slot` (stage-1 gain, step cost,
    /// calibration, stats) and log a rollup every 50 blocks. `conf` = the
    /// block's confidence logits; `lanes` = how many lanes its verify ran on
    /// (the cost fit it feeds).
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self, slot: u32, conf: &[f32; DRAFT_BLOCK], k: usize, accepted: usize, emitted: usize, step_ms: f64, lanes: usize) {
        let plain_ms = if self.lanes.one.live { self.lanes.one.cost(1) } else { self.plain_ms };
        self.observe(lanes, 1 + k, step_ms);
        if lanes >= 2 {
            self.stats.two_lane += 1;
        }
        self.calib.observe(conf, k, accepted);
        if self.calib.blocks % CALIB_EVERY == 0 {
            self.calib.log();
        }
        if let Ok(sd) = self.slot(slot) {
            let g = emitted as f64 * plain_ms / (step_ms + sd.last_draft_ms).max(1.0);
            sd.gates[LONE].update(g);
        }
        let s = &mut self.stats;
        s.blocks += 1;
        s.drafts_verified += k as u64;
        s.accepted += accepted as u64;
        s.emitted += emitted as u64;
        s.k_hist[k.min(DRAFT_BLOCK)] += 1;
        s.lanes_by_rows[k.min(DRAFT_BLOCK)][(lanes >= 2) as usize] += 1;
        s.step_ms += step_ms;
        self.maybe_log();
    }

    /// A drafted step whose K came out 0 (`k_for`: the policy's, or an
    /// exploration draw) ran as a plain one-row step but paid for its draft:
    /// `note_plain_step`'s sample, a block's accounting (`Stats::note_k0`)
    /// and -- a POLICY K = 0 only -- a stage-1 gain sample with that draft
    /// charged, so a stream whose blocks keep coming out empty backs off
    /// instead of paying the draft for every token (issue #3 finding 2;
    /// before, these steps reached neither the gain nor any counter). An
    /// exploration draw measures a choice the policy did not make: no gain.
    pub fn record_k0(&mut self, slot: u32, step_ms: f64) {
        let plain_ms = if self.lanes.one.live { self.lanes.one.cost(1) } else { self.plain_ms };
        self.note_plain_step(step_ms);
        if !self.k0_explored {
            if let Ok(sd) = self.slot(slot) {
                let g = plain_ms / (step_ms + sd.last_draft_ms).max(1.0);
                sd.gates[LONE].update(g);
            }
        }
        self.stats.note_k0(step_ms);
        self.maybe_log();
    }

    /// The `ms dspark: blocks` rollup, every 50 drafted steps.
    fn maybe_log(&mut self) {
        let s = &self.stats;
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
                cost_ms = format!("{:.1}+{:.1}/row", self.lanes.one.a, self.lanes.one.b),
                cost2_ms = format!("{:.1}+{:.1}/row", self.lanes.two.a, self.lanes.two.b),
                cost_shape = ?self.lanes.one.shape,
                cells = cells_str(self.lanes.one.cell_costs(), 1),
                cells_w = cells_str(&self.lanes.one.cell_weights(), 1),
                cells2 = cells_str(self.lanes.two.cell_costs(), 2),
                cells2_w = cells_str(&self.lanes.two.cell_weights(), 2),
                two_lane_blocks = s.two_lane,
                explored = s.explored,
                explored_k0 = s.explored_k0,
                policy_k0 = s.policy_k0,
                explored_hist = ?s.explored_hist,
                explored_two = s.explored_two,
                lane_switches = self.switches.take(),
                lanes_by_rows = s.lanes_by_rows.iter().enumerate().skip(1).map(|(i, h)| format!("{}:{}/{}", i + 1, h[0], h[1])).collect::<Vec<_>>().join(" "),
                keep_ms = format!("{:.2}", s.keep_ms / (s.keeps as f64).max(1.0)),
                ring_errors = s.ring_errors,
                draft_est_ms = format!("{:.1}", self.lanes.one.draft_ms()),
                cost_samples = self.lanes.one.samples,
                cost2_samples = self.lanes.two.samples,
                multi_steps = s.multi_steps,
                multi_spec_steps = s.multi_spec_steps,
                multi_blocks = s.multi_blocks,
                multi_explored = s.multi_explored,
                mcells = cells_str(self.multi.one.cell_costs(), 2),
                mcells2 = cells_str(self.multi.two.cell_costs(), 2),
                mcells_w = cells_str(&self.multi.one.cell_weights(), 2),
                mcells2_w = cells_str(&self.multi.two.cell_weights(), 2),
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
/// (`drafter_src` is slot-major `[3][DRAFT_CAP_ROWS][N_EMBD]`). The step must have
/// synchronized its stream.
pub fn lane_captures(bd: &BatchDgpuScratch, n: usize) -> eyre::Result<Vec<Vec<f32>>> {
    if n > DRAFT_CAP_ROWS || bd.drafter_captured != n {
        return Err(eyre!("ms dspark: lane captured {} rows, the step ran {n}", bd.drafter_captured));
    }
    let ne = N_EMBD as usize;
    let mut out = vec![vec![0f32; HIDDEN]; n];
    let mut tmp = vec![0f32; n * ne];
    for s in 0..DRAFT_SRC_LAYERS.len() {
        bd.drafter_src.slice_view(s * DRAFT_CAP_ROWS * ne, n * ne).copy_to_host(&mut tmp)?;
        for (k, row) in out.iter_mut().enumerate() {
            row[s * ne..(s + 1) * ne].copy_from_slice(&tmp[k * ne..(k + 1) * ne]);
        }
    }
    Ok(out)
}

/// A prefill's captures by absolute position (both lanes; `drafter_captured_pos0`
/// is each lane's first captured position).
pub fn prefill_captures(lanes: &[&BatchDgpuScratch]) -> eyre::Result<BTreeMap<u32, Vec<f32>>> {
    let mut m = BTreeMap::new();
    for bd in lanes {
        let n = bd.drafter_captured;
        if n == 0 {
            continue;
        }
        for (k, row) in lane_captures(bd, n)?.into_iter().enumerate() {
            m.insert(bd.drafter_captured_pos0 + k as u32, row);
        }
    }
    Ok(m)
}

/// `a,b,c,...` (one value per row count, rows `first..`) for the log lines.
fn cells_str(c: &[f64], first: usize) -> String {
    c[(first.max(1) - 1).min(c.len())..].iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(",")
}



/// The 2026-09-30..10-01 production p50 ladder (ms of 1, 2, ... rows): the
/// default PRIOR of `StepCost`.
const DEFAULT_LADDER: [f64; 8] = [60.0, 81.0, 101.0, 107.0, 125.0, 139.0, 156.0, 172.0];

/// 2026-09-30 production step means by rows (27 h, 543K steps; rows 1-3 one
/// lane, 4-8 two): the default two-lane prior (`StepCost::from_env_two_lane`
/// reads rows `>= first_row` only).
const DEFAULT_LADDER_TWO_LANE: [f64; 8] = [62.4, 83.9, 103.4, 109.1, 127.4, 141.4, 158.6, 176.7];

/// What the K policy prices a block with.
pub trait CostModel {
    /// Step time (ms) of `rows` rows of one stream.
    fn cost(&self, rows: usize) -> f64;
    /// Draft time (ms).
    fn draft_ms(&self) -> f64;
}

impl CostModel for StepCost {
    fn cost(&self, rows: usize) -> f64 {
        StepCost::cost(self, rows)
    }
    fn draft_ms(&self) -> f64 {
        StepCost::draft_ms(self)
    }
}

/// How a step's lane count is chosen (one snapshot per step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneRule {
    /// One lane at any row count.
    Off,
    /// Two lanes from this many rows on (`V41_MS_PIPELINE_MIN_ROWS`).
    Threshold(usize),
    /// Per row count, whichever lane count its table prices cheaper
    /// (`V41_MS_LANES_LEARNED`; `LaneTables`).
    Learned,
}

impl LaneRule {
    /// The lane counts this rule may run `rows` rows on: the threshold leaves
    /// no choice, and one row cannot be split.
    pub fn choices(self, rows: usize) -> &'static [usize] {
        match self {
            LaneRule::Learned if rows >= 2 => &[1, 2],
            LaneRule::Threshold(m) if rows >= m.max(2) => &[2],
            _ => &[1],
        }
    }
}

/// A cell's start on the lane count the startup threshold would NOT pick is
/// this much dearer than the other's start: until the first samples arrive the
/// learned rule makes the threshold's choice (`LaneTables::new`).
const START_MARGIN: f64 = 1.03;

/// Step cost by row count on ONE lane and on TWO (owner, 10-01: "if we have
/// cells1 and cells2, why have a fixed min_rows at all?"). Under
/// `LaneRule::Learned` each row count runs on whichever lane count its own
/// cell prices cheaper -- no threshold, and no assumption that the cheaper
/// count changes only once (the two-lane step hides box-2 waits and costs a
/// longer dGPU chain, both of which move with load). Both tables only learn
/// from steps that ran them, so exploration (`explore`, at `explore_p`) draws
/// over (rows, lanes) pairs: the lane count the rule avoids at some row count
/// keeps being sampled, and the time-aged cells follow drift (a cold pool, a
/// warm one, context length). Time = the steps this pair of tables prices:
/// each step ages BOTH (`observe`). Two pairs exist: a lone stream's DSpark
/// blocks (`MsDspark`; rows = 1 + K of ONE stream, an ordered verify cut) and
/// the plain multi-stream steps (`PlainLanes`; rows = streams).
pub struct LaneTables {
    pub one: StepCost,
    pub two: StepCost,
    /// The startup threshold (`new`); stands in for `Learned` when the costs
    /// are static (`V41_MS_DSPARK_COST_LIVE=0`: nothing to learn from).
    start_from: usize,
}

impl LaneTables {
    /// Cells start from the tables' ladders, except that at each row count
    /// the lane count `Threshold(start_from)` would not pick starts at the
    /// other's start x `START_MARGIN`: a cold `Learned` rule reproduces the
    /// threshold (the ladders are mixed measurements -- the production means
    /// had rows 1-3 on one lane and 4-8 on two -- and comparing them directly
    /// would flip 4-row steps to one lane at start). One sample of each cell
    /// outweighs its start (`CELL_START_W`).
    pub fn new(mut one: StepCost, mut two: StepCost, start_from: usize) -> Self {
        let start_from = start_from.max(2);
        for rows in 2..=one.cells.len().min(two.cells.len()) {
            if rows >= start_from {
                let c = two.cells[rows - 1].1 * START_MARGIN;
                one.set_start(rows, c);
            } else {
                let c = one.cells[rows - 1].1 * START_MARGIN;
                two.set_start(rows, c);
            }
        }
        Self { one, two, start_from }
    }

    fn rule(&self, rule: LaneRule) -> LaneRule {
        if rule == LaneRule::Learned && !self.one.live { LaneRule::Threshold(self.start_from) } else { rule }
    }

    /// The lane counts `rule` may run `rows` rows on (`LaneRule::choices`).
    pub fn choices(&self, rows: usize, rule: LaneRule) -> &'static [usize] {
        self.rule(rule).choices(rows)
    }

    /// A drafted block's `(K, lanes)` candidates under `rule`, K in
    /// `0..=cap` (exploration draws one).
    pub fn block_choices(&self, cap: usize, rule: LaneRule) -> Vec<(usize, usize)> {
        (0..=cap.min(DRAFT_BLOCK)).flat_map(|k| self.choices(k + 1, rule).iter().map(move |&l| (k, l))).collect()
    }

    fn table(&self, lanes: usize) -> &StepCost {
        if lanes >= 2 { &self.two } else { &self.one }
    }

    /// Step time (ms) of `rows` rows on `lanes` lanes.
    pub fn cost(&self, rows: usize, lanes: usize) -> f64 {
        self.table(lanes).cost(rows)
    }

    /// One `ms lanes: switch` line per policy switch (`Switches`): both cells,
    /// so a switch can be lined up with the `ms.phase` burst boundaries.
    fn log_switch(&self, what: &'static str, rows: usize, lanes: usize) {
        tracing::info!(
            what,
            rows,
            from = if lanes >= 2 { 1 } else { 2 },
            to = lanes,
            one_ms = format!("{:.1}", self.cost(rows, 1)),
            two_ms = format!("{:.1}", self.cost(rows, 2)),
            one_w = format!("{:.1}", self.weight(rows, 1)),
            two_w = format!("{:.1}", self.weight(rows, 2)),
            "ms lanes: switch"
        );
    }

    /// The time-aged sample weight of the cell `(rows, lanes)` feeds.
    pub fn weight(&self, rows: usize, lanes: usize) -> f64 {
        self.table(lanes).cell_weight(rows)
    }

    /// The cheapest of `rule`'s lane counts for `rows` rows: `(cost, lanes)`
    /// (a tie keeps one lane).
    pub fn best(&self, rows: usize, rule: LaneRule) -> (f64, usize) {
        let mut best = (f64::INFINITY, 1);
        for &l in self.choices(rows, rule) {
            let c = self.cost(rows, l);
            if c < best.0 {
                best = (c, l);
            }
        }
        best
    }

    /// The lane count for a step whose rows are given (a plain step): with
    /// probability `explore_p` an exploration draw among `rule`'s choices
    /// (`true`), else the cheapest (`false`).
    pub fn pick(&self, rng: &mut impl Rng, rows: usize, rule: LaneRule) -> (usize, bool) {
        let opts = self.choices(rows, rule);
        match explore(rng, opts.len(), explore_p(), |i| self.weight(rows, opts[i])) {
            Some(i) => (opts[i], true),
            None => (self.best(rows, rule).1, false),
        }
    }

    /// One step (time): both tables age, then `(rows, ms)` joins the table of
    /// the lane count it ran on.
    pub fn observe(&mut self, rows: usize, lanes: usize, ms: f64) {
        self.one.age();
        self.two.age();
        if lanes >= 2 { self.two.add(rows, ms) } else { self.one.add(rows, ms) }
    }

    /// What the K policy prices a block with under `rule`: `best`.
    pub fn priced(&self, rule: LaneRule) -> Priced<'_> {
        Priced { t: self, rule }
    }
}

/// `LaneTables` under one rule, as the K policy's `CostModel`. Under
/// `Learned` the min of two noisy cells reads slightly LOW where they are
/// close (the winner's curse): a little over-drafting, bounded by the cells'
/// noise (review round 12).
pub struct Priced<'a> {
    t: &'a LaneTables,
    rule: LaneRule,
}

impl CostModel for Priced<'_> {
    fn cost(&self, rows: usize) -> f64 {
        self.t.best(rows, self.rule).0
    }
    fn draft_ms(&self) -> f64 {
        self.t.one.draft_ms()
    }
}

/// The cheapest of `rule`'s lane counts for a speculating step of `rows`
/// rows that keeps every lane at most `SPEC_ROWS_PER_LANE`: `(cost, lanes)`,
/// `None` when no allowed lane count can hold the rows.
fn multi_best(t: &LaneTables, rows: usize, rule: LaneRule) -> Option<(f64, usize)> {
    let mut best: Option<(f64, usize)> = None;
    for &l in t.choices(rows, rule) {
        if rows > l * SPEC_ROWS_PER_LANE {
            continue;
        }
        let c = t.cost(rows, l);
        if best.is_none_or(|b| c < b.0) {
            best = Some((c, l));
        }
    }
    best
}

/// Drop drafts from the deepest block (the first on a tie) until the step's
/// rows fit a lane count `rule` allows at `SPEC_ROWS_PER_LANE` per lane. The
/// policies never pick an unfit step (it costs infinity); a fixed K
/// (`V41_MS_DSPARK_K`) ignores costs, and this is its cap.
fn fit_rows(ks: &mut [usize], base_rows: usize, t: &LaneTables, rule: LaneRule) {
    while ks.iter().sum::<usize>() > 0 && multi_best(t, base_rows + ks.iter().sum::<usize>(), rule).is_none() {
        let deepest = (0..ks.len()).max_by_key(|&s| (ks[s], std::cmp::Reverse(s))).expect("a block");
        ks[deepest] -= 1;
    }
}

/// Every `(K per stream, rows, lanes)` a multi-stream speculating step may
/// run under `rule` (exploration draws one): K_s in `0..=cap_s` with at least
/// one draft (a step with none is a PLAIN step: it feeds `PlainLanes`, never
/// a multi cell, so a multi cell for it would stay stale and draw the
/// staleness half of the exploration for good), lanes among the rule's
/// choices for the rows, at most `SPEC_ROWS_PER_LANE` per lane. Reads the
/// caps only, never a confidence.
fn multi_choices(t: &LaneTables, blocks: &[BlockConf], base_rows: usize, rule: LaneRule) -> Vec<(Vec<usize>, usize, usize)> {
    let caps: Vec<usize> = blocks.iter().map(|b| b.cap.min(DRAFT_BLOCK)).collect();
    let mut out = Vec::new();
    let mut ks = vec![0usize; blocks.len()];
    loop {
        let rows = base_rows + ks.iter().sum::<usize>();
        for &l in t.choices(rows, rule) {
            if rows > base_rows && rows <= l * SPEC_ROWS_PER_LANE {
                out.push((ks.clone(), rows, l));
            }
        }
        let mut s = 0;
        while s < ks.len() {
            if ks[s] < caps[s] {
                ks[s] += 1;
                break;
            }
            ks[s] = 0;
            s += 1;
        }
        if s == ks.len() {
            break;
        }
    }
    out
}

/// The multi-stream tables under one rule as the joint policy's
/// `CostModel`: the step's TOTAL rows at the cheapest allowed lane count
/// (`multi_best`; infinite when none can hold them, so no rule picks them),
/// `draft` = the step's total draft time.
struct MultiPriced<'a> {
    t: &'a LaneTables,
    rule: LaneRule,
    draft: f64,
}

impl CostModel for MultiPriced<'_> {
    fn cost(&self, rows: usize) -> f64 {
        multi_best(self.t, rows, self.rule).map(|b| b.0).unwrap_or(f64::INFINITY)
    }
    fn draft_ms(&self) -> f64 {
        self.draft
    }
}

/// How often the POLICY's lane count for a row count changed (exploration
/// draws excluded), since the last log line: `LaneRule::Learned` flip-flop
/// (review round 12: a slow spell lands on the active lane's mature cell
/// while the other keeps its pre-spell mean, so the rule may flip and sit on
/// the worse lane for about a memory; measure how often before correcting
/// for it). `"rows:switches ..."`, nonzero only.
#[derive(Default)]
struct Switches {
    last: Vec<usize>,
    n: Vec<u64>,
}

impl Switches {
    /// Whether this choice switched the row count's lane count.
    fn note(&mut self, rows: usize, lanes: usize) -> bool {
        if rows == 0 {
            return false;
        }
        if self.last.len() < rows {
            self.last.resize(rows, 0);
            self.n.resize(rows, 0);
        }
        let last = &mut self.last[rows - 1];
        let switched = *last != 0 && *last != lanes;
        if switched {
            self.n[rows - 1] += 1;
        }
        *last = lanes;
        switched
    }

    fn take(&mut self) -> String {
        let out: Vec<String> = self.n.iter().enumerate().filter(|&(_, &c)| c > 0).map(|(i, c)| format!("{}:{c}", i + 1)).collect();
        self.n.iter_mut().for_each(|c| *c = 0);
        out.join(" ")
    }
}

/// Plain multi-stream steps between `ms lanes` log lines.
const PLAIN_LOG_EVERY: u64 = 2000;

/// The plain multi-stream steps' lane choice: a `LaneTables` of their own,
/// rows = the step's streams (2 ..= the arena's slots), fed by every plain
/// step of two or more streams (`observe`; three-lane steps feed nothing) and
/// aged by those steps alone (`V41_MS_LANES_MEMORY` steps, default 1000 =
/// ~2 min). Not the DSpark tables: a block's rows are ONE stream's (one KV, an
/// ordered verify cut), a plain step's one per stream (the cross-stream split).
/// Always live cells (`V41_MS_DSPARK_COST_LIVE` / `_COST_SHAPE` are the DSpark
/// policy's knobs). Logs both tables every `PLAIN_LOG_EVERY` steps. A step's
/// time runs to the end of its accept / ring tail, like a block's: the same
/// for both lane counts, so comparisons are fair, but the logged absolutes
/// include it.
pub struct PlainLanes {
    t: LaneTables,
    rng: StdRng,
    rule: LaneRule,
    switches: Switches,
    /// Since the last log line, by rows: steps on one lane, on two.
    hist: Vec<[u64; 2]>,
    explored: u64,
    steps: u64,
    since: Instant,
}

impl PlainLanes {
    /// Cells for 1 ..= `max_rows` streams; `start_from` = the startup threshold
    /// (`LaneTables::new`).
    pub fn from_env(max_rows: usize, start_from: usize) -> Self {
        let memory = crate::knobs::MS_LANES_MEMORY.f64();
        let rows = max_rows.max(2);
        let one = StepCost::new(DEFAULT_LADDER.to_vec(), 0.0, true, memory).with_shape(CostShape::Cells).with_rows(rows);
        let two = StepCost::with_first_row(DEFAULT_LADDER_TWO_LANE.to_vec(), start_from, 0.0, true, memory).with_shape(CostShape::Cells).with_rows(rows);
        Self {
            t: LaneTables::new(one, two, start_from),
            // Not the DSpark stream under a fixed seed.
            rng: explore_rng_salted(0x9e37_79b9_7f4a_7c15),
            rule: LaneRule::Off,
            switches: Switches::default(),
            hist: vec![[0; 2]; rows],
            explored: 0,
            steps: 0,
            since: Instant::now(),
        }
    }

    /// The lane count for a plain step of `rows` streams under `rule`.
    pub fn pick(&mut self, rows: usize, rule: LaneRule) -> usize {
        self.rule = rule;
        let (lanes, explored) = self.t.pick(&mut self.rng, rows, rule);
        if explored {
            self.explored += 1;
        } else if self.switches.note(rows, lanes) {
            self.t.log_switch("plain", rows, lanes);
        }
        lanes
    }

    /// What a plain step of `rows` streams costs under `rule` (the cheaper
    /// allowed lane count): the stage-1 baseline of a multi-stream DSpark step.
    pub fn cost(&self, rows: usize, rule: LaneRule) -> f64 {
        self.t.best(rows, rule).0
    }

    /// A plain step of `rows` streams on `lanes` lanes took `ms`.
    pub fn observe(&mut self, rows: usize, lanes: usize, ms: f64) {
        self.t.observe(rows, lanes, ms);
        if let Some(h) = self.hist.get_mut(rows.wrapping_sub(1)) {
            h[(lanes >= 2) as usize] += 1;
        }
        self.steps += 1;
        if self.steps >= PLAIN_LOG_EVERY {
            let hist: Vec<String> = self.hist.iter().enumerate().skip(1).map(|(i, h)| format!("{}:{}/{}", i + 1, h[0], h[1])).collect();
            tracing::info!(
                steps = self.steps,
                rule = ?self.rule,
                one = cells_str(self.t.one.cell_costs(), 2),
                two = cells_str(self.t.two.cell_costs(), 2),
                one_w = cells_str(&self.t.one.cell_weights(), 2),
                two_w = cells_str(&self.t.two.cell_weights(), 2),
                lanes_by_rows = hist.join(" "),
                explored = self.explored,
                switches = self.switches.take(),
                window_s = self.since.elapsed().as_secs(),
                "ms lanes: plain steps"
            );
            self.hist.iter_mut().for_each(|h| *h = [0; 2]);
            self.explored = 0;
            self.steps = 0;
            self.since = Instant::now();
        }
    }
}

/// `ladder[rows - 1]`, extrapolated linearly past its end.
fn ladder_cost(ladder: &[f64], rows: usize) -> f64 {
    if rows == 0 {
        return 0.0;
    }
    if rows <= ladder.len() {
        return ladder[rows - 1];
    }
    let n = ladder.len();
    ladder[n - 1] + (ladder[n - 1] - ladder[n - 2]) * (rows - n) as f64
}

/// The live fit's prior is the configured ladder's least-squares line over
/// rows 1..=DRAFT_BLOCK+1, held with SEPARATE weights: its slope as strongly as
/// 24 pseudo-samples spread over those rows (24/6 * sum (x - 3.5)^2 = 70), so
/// the slope stays put while the data sit at one row count; its level (at the
/// ladder's centroid) as `level_prior(n)` samples: ~4 while the fit has seen
/// only a handful (n = the decayed sample count), so a few samples cannot
/// swing it, falling to `PRIOR_LEVEL` = half a sample once the memory fills,
/// so the level then follows the data at whatever row count they are without
/// bending the slope (a level held at 4 for good steepened the slope 15 -> 21
/// ms/row when every sample sat at 6 rows, 60 ms slow). At half a sample from
/// the start (until 10-01) ONE 222 ms warm-up step -- the first two-lane
/// verify after a restart -- lifted the two-lane 4-row estimate from 110 to
/// 185 ms, and the policy never verified 4+ rows again in that process (no
/// new two-lane samples could correct it). One ladder held as points instead
/// turned a uniform slowdown seen only at 6 rows into a doubled slope (60 ms
/// pinned at 1 row): K would shrink exactly when a fixed cost grew.
const PRIOR_SLOPE: f64 = 70.0;
const PRIOR_LEVEL: f64 = 0.5;
/// The extra level weight while the fit is young, and the sample count at
/// which it has halved.
const PRIOR_LEVEL_EARLY: f64 = 4.0;
const PRIOR_LEVEL_HALF: f64 = 16.0;

/// The level prior's weight after `n` (decayed) samples.
fn level_prior(n: f64) -> f64 {
    PRIOR_LEVEL + PRIOR_LEVEL_EARLY * PRIOR_LEVEL_HALF / (PRIOR_LEVEL_HALF + n)
}

/// A step sample counts at most as this multiple of the CURRENT estimate: one
/// stall (box-2 paging, a warm-up) moves the fit a bounded step, while a real,
/// lasting slowdown is still followed (the bound rises with the estimate).
/// Tight (1.5x) only while the fit is YOUNG (`young`: what went wrong on 10-01
/// was a young fit's first samples); a mature fit keeps 3x, so the routine
/// paging tail -- which grows with rows -- still counts toward its mean and
/// slope instead of being trimmed into an under-estimate (over-drafting).
fn stall_clamp(n: f64) -> f64 {
    if young(n) { 1.5 } else { 3.0 }
}

/// A fit with fewer than `PRIOR_LEVEL_HALF` (decayed) samples.
fn young(n: f64) -> bool {
    n < PRIOR_LEVEL_HALF
}

/// EXPLORATION (owner, 10-01: "if we have a few really bad samples we
/// shouldn't give up on larger batches forever"). The fits only learn from the
/// rows the policy picks, so a few bad samples could make it avoid a row count
/// -- or a whole regime -- and never sample it again. It happened both ways on
/// 10-01: a slow first two-lane verify kept K <= 2 for good; after a cold
/// restart a one-lane line frozen at its cold level kept K = 5 on two lanes for
/// good. So with probability `explore_p()` a drafting block verifies a random
/// number of drafts in `0..=cap` instead of the policy's K -- weighted toward
/// the cost cells sampled least lately (`explore`; uniform until 10-01
/// evening) -- and, under `LaneRule::Learned`, a random lane count for them;
/// a plain multi-stream step a random lane count (`LaneTables::pick`).
///
/// Chosen over Thompson sampling: TS explores only where the posterior is
/// uncertain, and ours is a hand-set Gaussian (decayed pseudo-counts, a clamped
/// heavy-tailed stall distribution, warm-up drift) -- a confidently WRONG fit,
/// the failure here, is what TS rarely revisits. The staleness weighting reads
/// only how much recent data each cell has, never what the model believes it
/// costs: every row count of both regimes keeps a known minimum rate, and the
/// draws go where data is thinnest. Costs well under 1% at 1/32. Exact under
/// the stopping rule: K is drawn before `conf` is read, independent of the
/// drafts.
fn explore_p() -> f64 {
    crate::knobs::MS_DSPARK_EXPLORE.f64()
}

/// `V41_MS_DSPARK_EXPLORE_SEED` (a u64) seeds the exploration draws; else entropy.
fn explore_rng() -> StdRng {
    explore_rng_salted(0)
}

/// `explore_rng` with the seed XOR `salt` (a second stream under one seed).
fn explore_rng_salted(salt: u64) -> StdRng {
    match crate::knobs::MS_DSPARK_EXPLORE_SEED.str().and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(seed) => StdRng::seed_from_u64(seed ^ salt),
        None => StdRng::from_entropy(),
    }
}

/// With probability `p`, one of `n` candidates (0..n; none when `n < 2`):
/// half the draws UNIFORM, half by STALENESS (owner, 10-01: "focus exploration
/// on cells we haven't explored in a while"). A candidate -- a K, or a (K,
/// lanes) pair, or a lane count -- feeds one cost cell, and a staleness draw
/// picks it with probability proportional to `1 / (1 + w)`, `w = weight(i)`
/// that cell's time-aged sample weight: cells nobody has sampled lately -- an
/// idle
/// regime's rows, a row count the policy avoids, a cell left behind by a
/// restart (the 1-row cell held at its cold-start mean while every block
/// drafted) -- get most of those. The uniform half is the floor (review round
/// 10): a cell the policy has just ABANDONED while its weight is still high
/// (say ~250, inflated by a slow spell) gets almost no staleness draws until
/// its weight has decayed for a few hundred lone steps, and cells do not
/// revert their mean -- without the floor its stale estimate could stand for
/// minutes. Still epsilon-exploration (`explore_p` sets how often); `weight`
/// reads only past samples, so the draw is exact under the stopping rule.
pub fn explore(rng: &mut impl Rng, n: usize, p: f64, weight: impl Fn(usize) -> f64) -> Option<usize> {
    if n < 2 || p <= 0.0 || rng.gen::<f64>() >= p {
        return None;
    }
    if rng.gen::<bool>() {
        return Some(rng.gen_range(0..n));
    }
    let scores: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + weight(i).max(0.0))).collect();
    let mut x = rng.gen::<f64>() * scores.iter().sum::<f64>();
    for (i, &sc) in scores.iter().enumerate() {
        if x < sc {
            return Some(i);
        }
        x -= sc;
    }
    Some(n - 1)
}

/// Rows the per-row cost CELLS cover by default (1 ..= DRAFT_BLOCK + 1: every
/// block's rows; `StepCost::with_rows` for more).
const CELLS: usize = DRAFT_BLOCK + 1;
/// A cell starts at its ladder value, weighted as this many samples (it
/// washes out with the first real ones; bounds a first warm-up stall).
const CELL_START_W: f64 = 1.0;

/// `V41_MS_DSPARK_COST_SHAPE`: `cells` (default since 2026-10-01) or `line`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostShape {
    /// One independent estimate per row count (`StepCost` doc).
    Cells,
    /// `a + b * rows` with the ladder line as prior (the 10-01 daytime model).
    Line,
}

fn cost_shape() -> CostShape {
    if crate::knobs::MS_DSPARK_COST_SHAPE.pick() == 1 { CostShape::Line } else { CostShape::Cells }
}

/// What a lone stream's step and draft cost (ms), for the K policy and the
/// stage-1 baseline.
///
/// CELLS (default since 2026-10-01 evening, `V41_MS_DSPARK_COST_SHAPE`): one
/// independent estimate per row count 1..=6 -- a time-aged mean of that row
/// count's own steps, starting at its ladder value (`CELL_START_W`) -- with no
/// prior BETWEEN row counts (owner: "we only have 6 possible options here",
/// and epsilon-exploration now samples every one). A line missed the measured
/// concavity (each extra row costs less: one lane +15.7 ms at 1->2 rows, +10.6
/// at 5->6; two lanes +11.7, +7.5 -- more rows share more experts) and so
/// overpriced 6-row steps by ~5 ms, under-drafting the 5th draft. No coupling
/// at all, monotonicity included (owner: the cost may legitimately fall with
/// rows, e.g. where the second lane switches on): a noisy rare cell is fixed
/// by its own next samples, which exploration keeps supplying.
///
/// LINE (`=line`): `cost(rows) = a + b * rows`, least squares over the lone
/// stream's recent steps (exponential forgetting, `V41_MS_DSPARK_COST_MEMORY`
/// samples, default 500 = ~75 s of steps) with the configured ladder
/// (`V41_MS_DSPARK_COST`) as the prior (`PRIOR_SLOPE`, `PRIOR_LEVEL`); the
/// draft is an EWMA of measured drafts starting at `V41_MS_DSPARK_DRAFT_MS`.
/// MEASURED 2026-10-01 over 3,128 warm lone steps: 53.2 + 15.1 ms/row, every
/// row count within 1.6 ms of the line, draft 12.0 ms. The static p50 ladder
/// had its 4-row entry 8 ms low, which made a 3rd draft look nearly free
/// (verified at predicted acceptance >= 0.15 where the line says 0.42), and
/// assumed a 20 ms draft. A LINE rather than per-row means: row counts the
/// policy rarely picks (1 and 3 rows, ~2% of steps each) borrow strength from
/// the others, and a noise bump cannot carve a kink into K. It also follows
/// a pool warming after a restart (steps 1.3-1.9x slower for ~5 min).
///
/// `V41_MS_DSPARK_COST_LIVE=0`: the configured ladder and draft as given,
/// never updated (the policy before 10-01).
///
/// Only past steps feed it, so a K decision still reads nothing of its own
/// block but `conf` (exactness of the stopping rule, plan 2.4).
pub struct StepCost {
    live: bool,
    ladder: Vec<f64>,
    decay: f64,
    /// Exponentially weighted `[n, sum x, sum x^2, sum y, sum x*y]` of the
    /// observed `(rows, ms)` samples.
    data: [f64; 5],
    /// The prior line: slope, and its level at `prior_xc`.
    prior_b: f64,
    prior_xc: f64,
    prior_yc: f64,
    a: f64,
    b: f64,
    draft: f64,
    samples: u64,
    shape: CostShape,
    /// Per row count (index rows - 1): `(time-aged weight, mean)`. The MEAN is
    /// stored, not a weighted sum: aging shrinks the weight only, so an idle
    /// cell keeps its value however long it idles (10-01: as `(weight, sum)`
    /// with `sum / weight.max(1e-9)`, a cell idle for ~30K lone steps -- one-
    /// lane rows 4-6 while two lanes ran them -- underflowed below the clamp
    /// and read ~0 ms). Rows 1..=`CELLS` unless `with_rows`.
    cells: Vec<(f64, f64)>,
    /// `cells`' means (what `cost` returns).
    cell_cost: Vec<f64>,
}

impl StepCost {
    pub fn new(ladder: Vec<f64>, draft_ms: f64, live: bool, memory: f64) -> Self {
        Self::with_first_row(ladder, 1, draft_ms, live, memory)
    }

    /// `new` with the prior line fitted over rows `first_row..=DRAFT_BLOCK+1`
    /// of the ladder only: the two-lane regime never prices fewer rows.
    pub fn with_first_row(ladder: Vec<f64>, first_row: usize, draft_ms: f64, live: bool, memory: f64) -> Self {
        let ladder = if ladder.len() >= 2 { ladder } else { DEFAULT_LADDER.to_vec() };
        let first_row = first_row.clamp(1, DRAFT_BLOCK);
        let pts: Vec<(f64, f64)> = (first_row..=DRAFT_BLOCK + 1).map(|r| (r as f64, ladder_cost(&ladder, r))).collect();
        let xc = pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64;
        let yc = pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64;
        let prior_b = pts.iter().map(|&(x, y)| (x - xc) * (y - yc)).sum::<f64>() / pts.iter().map(|&(x, _)| (x - xc) * (x - xc)).sum::<f64>();
        let cells = (1..=CELLS).map(|r| (CELL_START_W, ladder_cost(&ladder, r))).collect();
        let mut c = Self {
            live,
            ladder,
            decay: 1.0 - 1.0 / memory.max(10.0),
            data: [0.0; 5],
            prior_b,
            prior_xc: xc,
            prior_yc: yc,
            a: 0.0,
            b: 0.0,
            draft: draft_ms,
            samples: 0,
            shape: cost_shape(),
            cells,
            cell_cost: vec![0.0; CELLS],
        };
        c.refit();
        c
    }

    /// This fit with `shape` (tests; production reads `V41_MS_DSPARK_COST_SHAPE`).
    pub fn with_shape(mut self, shape: CostShape) -> Self {
        self.shape = shape;
        self
    }

    /// This fit with cells for rows `1..=rows` (at least 2), each starting at
    /// its ladder value: before any sample (a constructor step).
    pub fn with_rows(mut self, rows: usize) -> Self {
        self.cells = (1..=rows.max(2)).map(|r| (CELL_START_W, ladder_cost(&self.ladder, r))).collect();
        self.cell_cost = vec![0.0; self.cells.len()];
        self.refit();
        self
    }

    /// Cell `rows`' starting mean: before any sample (`LaneTables::new`).
    fn set_start(&mut self, rows: usize, ms: f64) {
        if let Some(c) = self.cells.get_mut(rows.wrapping_sub(1)) {
            c.1 = ms;
        }
        self.refit();
    }

    /// The per-row cells' costs (logging): rows 1, 2, ...
    pub fn cell_costs(&self) -> &[f64] {
        &self.cell_cost
    }

    /// The per-row cells' (time-aged) sample weights (logging): a cell under
    /// ~16 is young -- its mean is a handful of recent samples.
    pub fn cell_weights(&self) -> Vec<f64> {
        self.cells.iter().map(|c| c.0).collect()
    }

    /// Cell `rows`' weight (0 past the cells: never sampled).
    pub fn cell_weight(&self, rows: usize) -> f64 {
        self.cells.get(rows.wrapping_sub(1)).map_or(0.0, |c| c.0)
    }

    /// The line's `(a, b)` (logging).
    pub fn line(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    /// From `V41_MS_DSPARK_COST`, `_DRAFT_MS`, `_COST_LIVE`, `_COST_MEMORY`.
    pub fn from_env() -> Self {
        let ladder = crate::knobs::MS_DSPARK_COST
            .str()
            .and_then(|v| v.split(',').map(|x| x.trim().parse().ok()).collect::<Option<Vec<f64>>>())
            .filter(|v| v.len() >= 2)
            .unwrap_or_else(|| DEFAULT_LADDER.to_vec());
        Self::new(
            ladder,
            crate::knobs::MS_DSPARK_DRAFT_MS.f64(),
            crate::knobs::MS_DSPARK_COST_LIVE.on(),
            crate::knobs::MS_DSPARK_COST_MEMORY.f64(),
        )
    }

    /// The TWO-lane regime (an ordered verify cut from `first_row` rows):
    /// `V41_MS_DSPARK_COST2`, a ladder for rows 1, 2, ... of which only rows
    /// `>= first_row` shape the prior (default: the 2026-09-30 production
    /// step means, rows 4..8 on two lanes). Its own fit, so the step change in
    /// cost where the second lane switches on cannot bend the one-lane line.
    pub fn from_env_two_lane(first_row: usize) -> Self {
        let ladder = crate::knobs::MS_DSPARK_COST2
            .str()
            .and_then(|v| v.split(',').map(|x| x.trim().parse().ok()).collect::<Option<Vec<f64>>>())
            .filter(|v| v.len() >= 2)
            .unwrap_or_else(|| DEFAULT_LADDER_TWO_LANE.to_vec());
        Self::with_first_row(
            ladder,
            first_row,
            crate::knobs::MS_DSPARK_DRAFT_MS.f64(),
            crate::knobs::MS_DSPARK_COST_LIVE.on(),
            crate::knobs::MS_DSPARK_COST_MEMORY.f64(),
        )
    }

    /// Step time (ms) of `rows` rows of one stream.
    pub fn cost(&self, rows: usize) -> f64 {
        if rows == 0 {
            return 0.0;
        }
        if !self.live {
            return ladder_cost(&self.ladder, rows);
        }
        let n = self.cell_cost.len();
        match self.shape {
            CostShape::Line => (self.a + self.b * rows as f64).max(1.0),
            CostShape::Cells if rows <= n => self.cell_cost[rows - 1].max(1.0),
            // Past the cells (never a block's rows): extend the last step.
            CostShape::Cells => {
                let (l, p) = (self.cell_cost[n - 1], self.cell_cost[n - 2]);
                (l + (l - p).max(0.0) * (rows - n) as f64).max(1.0)
            }
        }
    }

    /// Draft time (ms).
    pub fn draft_ms(&self) -> f64 {
        self.draft
    }

    /// A lone stream's step of `rows` rows took `ms` (draft excluded): `age`
    /// then `add`. Tests only: production goes through `MsDspark::observe`,
    /// which ages BOTH fits once per step (calling this there would age twice).
    #[cfg(test)]
    pub fn observe_step(&mut self, rows: usize, ms: f64) {
        self.age();
        self.add(rows, ms);
    }

    /// One step of forgetting (exponential, `V41_MS_DSPARK_COST_MEMORY` steps).
    pub fn age(&mut self) {
        if !self.live {
            return;
        }
        for v in self.data.iter_mut() {
            *v *= self.decay;
        }
        for c in self.cells.iter_mut() {
            // Flush a long-idle weight to 0 rather than age a subnormal forever
            // (the mean is unaffected; the next sample carries the cell).
            c.0 = if c.0 < 1e-30 { 0.0 } else { c.0 * self.decay };
        }
        self.refit();
    }

    /// Add one sample (no forgetting: `age` does that).
    pub fn add(&mut self, rows: usize, ms: f64) {
        if !self.live || rows == 0 || !(ms.is_finite() && ms > 0.0) {
            return;
        }
        // A stall (box-2 hiccup, seconds of paging, a warm-up) counts, but at
        // most as `stall_clamp` x the estimate: one sample must not drag the fit.
        let line = (self.a + self.b * rows as f64).max(1.0);
        let (x, y) = (rows as f64, ms.min(stall_clamp(self.data[0]) * line));
        for (s, v) in self.data.iter_mut().zip([1.0, x, x * x, y, x * y]) {
            *s += v;
        }
        // Each cell clamps against ITS OWN estimate and youth.
        if rows <= self.cells.len() {
            let c = &mut self.cells[rows - 1];
            let y = ms.min(stall_clamp(c.0) * c.1);
            c.0 += 1.0;
            c.1 += (y - c.1) / c.0;
        }
        self.samples += 1;
        self.refit();
    }

    /// Fewer than `PRIOR_LEVEL_HALF` (decayed) step samples so far.
    pub fn young(&self) -> bool {
        young(self.data[0])
    }

    /// A draft took `ms`. (A plain 3x clamp: drafts are device-bound and
    /// low-variance, and the EWMA sees every draft, so it cannot freeze.)
    pub fn observe_draft(&mut self, ms: f64) {
        if self.live && ms.is_finite() && ms > 0.0 {
            self.draft = 0.95 * self.draft + 0.05 * ms.min(3.0 * self.draft);
        }
    }

    /// The line: minimize `sum_data w (y - a - b x)^2 + level_prior(n) (a + b xc
    /// - yc)^2 + PRIOR_SLOPE (b - prior_b)^2` (2x2 normal equations). The cells:
    /// each its own mean.
    fn refit(&mut self) {
        for (cost, c) in self.cell_cost.iter_mut().zip(&self.cells) {
            *cost = c.1;
        }
        let [n, sx, sxx, sy, sxy] = self.data;
        let (l, s, xc, yc) = (level_prior(n), PRIOR_SLOPE, self.prior_xc, self.prior_yc);
        let (m00, m01, m11) = (n + l, sx + l * xc, sxx + l * xc * xc + s);
        let (r0, r1) = (sy + l * yc, sxy + l * xc * yc + s * self.prior_b);
        let det = m00 * m11 - m01 * m01; // > 0: l, s > 0
        // Never cheaper per extra row than free: b >= 0, `a` refitted for it.
        let b = ((m00 * r1 - m01 * r0) / det).max(0.0);
        self.b = b;
        self.a = (r0 - b * m01) / m00;
    }
}

fn sigmoid(c: f32) -> f64 {
    1.0 / (1.0 + (-(c as f64)).exp())
}

/// `V41_MS_DSPARK_K`: verify exactly this many drafts (capped like the policy).
fn fixed_k() -> Option<usize> {
    crate::knobs::MS_DSPARK_K.str().and_then(|v| v.trim().parse().ok())
}

/// Most drafts a block may verify: `V41_MS_DSPARK_K` if set (0 = never draft,
/// the plain-decode control on the same binary), else `V41_MS_DSPARK_KMAX`
/// (default the block size).
pub fn k_max() -> usize {
    fixed_k().unwrap_or(crate::knobs::MS_DSPARK_KMAX.usize()).min(DRAFT_BLOCK)
}

/// K for SAMPLED drafts (plan 2.4 / section 6): a STOPPING rule. Whether draft
/// k is verified is decided from `conf[..=k]` only -- conf_k reads the markov
/// row of draft k-1, never draft k -- with the depths beyond k forecast at
/// sigmoid(conf_k). So no decision reads the value it would test. The global
/// search of `choose_k` lets conf_{k+1}, which depends on d_k, decide whether
/// d_k is verified: exact for point-mass tests only (review N1).
///
/// With per-row cost cells (`CostShape::Cells`) the cost need not grow with
/// rows; nothing here assumes it does (`go` is the best of EVERY deeper k). Being
/// sequential, the rule can head for a cheap far row count and then stop one
/// short once conf_{k+1} arrives, on a dearer count than stopping earlier --
/// inherent to a stopping rule, and bounded.
pub fn choose_k_stopping(conf: &[f32; DRAFT_BLOCK], cap: usize, cost: &dyn CostModel) -> usize {
    let cap = cap.min(DRAFT_BLOCK);
    if let Some(k) = fixed_k() {
        return k.min(cap);
    }
    let draft = cost.draft_ms();
    let (mut k, mut run, mut e) = (0usize, 1.0f64, 1.0f64);
    while k < cap {
        let p = sigmoid(conf[k]);
        let stop = e / (cost.cost(1 + k) + draft);
        let (mut r, mut ee, mut go) = (run, e, f64::MIN);
        for kk in k + 1..=cap {
            r *= p;
            ee += r;
            go = go.max(ee / (cost.cost(1 + kk) + draft));
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
pub fn choose_k(conf: &[f32; DRAFT_BLOCK], cap: usize, cost: &dyn CostModel) -> usize {
    let cap = cap.min(DRAFT_BLOCK);
    if let Some(k) = fixed_k() {
        return k.min(cap);
    }
    let draft = cost.draft_ms();
    let (mut best_k, mut best) = (0usize, 1.0 / (cost.cost(1) + draft));
    let (mut run, mut e) = (1.0f64, 1.0f64);
    for k in 1..=cap {
        run *= sigmoid(conf[k - 1]);
        e += run;
        let r = e / (cost.cost(1 + k) + draft);
        if r > best {
            best = r;
            best_k = k;
        }
    }
    best_k
}

/// One drafting stream's block as the JOINT policy sees it
/// (docs/v41/MS_DSPARK_STREAMS_DESIGN.md 2.2): its confidence logits and the
/// most drafts it may verify.
#[derive(Clone, Copy, Debug)]
pub struct BlockConf {
    pub conf: [f32; DRAFT_BLOCK],
    pub cap: usize,
}

/// K per stream for several streams' blocks verified in ONE step, SAMPLED
/// drafts allowed: the S-stream form of `choose_k_stopping`. `base_rows` = the
/// step's rows before any draft (one per live stream, drafting or not); `cost`
/// prices the step's TOTAL rows and `cost.draft_ms()` is the TOTAL draft time
/// (paid either way). From K = 0 everywhere it prices every extension of the
/// frontier -- each stream's depths beyond its current K forecast at
/// `sigmoid(conf[K])`, as the one-stream rule does -- and, while the best
/// extension beats stopping, adds ONE draft: the next draft of the stream with
/// the highest immediate value `P(its drafts so far accepted) x sigmoid(conf[K])`
/// among the streams that extension extends.
///
/// Exact: including draft k of stream s reads `conf_s[..=k]` (conf_k reads
/// draft k-1, never draft k) and other streams' confidences, which do not
/// depend on s's drafts (separate drafter runs, separate draft RNGs). No
/// draft's inclusion reads its own value. One stream with `base_rows` = 1 gives
/// exactly `choose_k_stopping` (the same sums in the same order).
pub fn choose_ks_stopping(blocks: &[BlockConf], base_rows: usize, cost: &dyn CostModel) -> Vec<usize> {
    let n = blocks.len();
    let caps: Vec<usize> = blocks.iter().map(|b| b.cap.min(DRAFT_BLOCK)).collect();
    if let Some(k) = fixed_k() {
        return caps.iter().map(|&c| k.min(c)).collect();
    }
    let draft = cost.draft_ms();
    let mut k = vec![0usize; n];
    let mut run = vec![1.0f64; n];
    let mut e = base_rows as f64;
    loop {
        let rows = base_rows + k.iter().sum::<usize>();
        let stop = e / (cost.cost(rows) + draft);
        let p: Vec<f64> = (0..n).map(|s| if k[s] < caps[s] { sigmoid(blocks[s].conf[k[s]]) } else { 0.0 }).collect();
        // Every extension m (m_s more drafts of stream s), m != 0.
        let mut best: Option<(f64, Vec<usize>)> = None;
        let mut m = vec![0usize; n];
        loop {
            // Next m in mixed radix (m_s in 0..=caps[s] - k[s]).
            let mut s = 0;
            while s < n {
                if m[s] < caps[s] - k[s] {
                    m[s] += 1;
                    break;
                }
                m[s] = 0;
                s += 1;
            }
            if s == n {
                break;
            }
            let mut ee = e;
            for t in 0..n {
                let mut r = run[t];
                for _ in 0..m[t] {
                    r *= p[t];
                    ee += r;
                }
            }
            let rate = ee / (cost.cost(rows + m.iter().sum::<usize>()) + draft);
            if best.as_ref().is_none_or(|b| rate > b.0) {
                best = Some((rate, m.clone()));
            }
        }
        let Some((go, bm)) = best else { break };
        if go <= stop {
            break;
        }
        let s = (0..n)
            .filter(|&s| bm[s] > 0)
            .max_by(|&a, &b| (run[a] * p[a]).partial_cmp(&(run[b] * p[b])).unwrap_or(std::cmp::Ordering::Equal).then(b.cmp(&a)))
            .expect("the best extension extends some stream");
        run[s] *= p[s];
        e += run[s];
        k[s] += 1;
    }
    k
}

/// K per stream for several streams' blocks in ONE step, POINT-MASS drafts
/// only (any K rule is exact for point-mass tests, plan 2.4): the global
/// search of `choose_k` over every `(K_1, .., K_S)`. One stream with
/// `base_rows` = 1 gives exactly `choose_k`.
pub fn choose_ks(blocks: &[BlockConf], base_rows: usize, cost: &dyn CostModel) -> Vec<usize> {
    let n = blocks.len();
    let caps: Vec<usize> = blocks.iter().map(|b| b.cap.min(DRAFT_BLOCK)).collect();
    if let Some(k) = fixed_k() {
        return caps.iter().map(|&c| k.min(c)).collect();
    }
    let draft = cost.draft_ms();
    // runs[s][j]: P(stream s's first j drafts all accepted), j >= 1.
    let runs: Vec<Vec<f64>> = (0..n)
        .map(|s| {
            let mut v = vec![1.0f64; caps[s] + 1];
            for j in 1..=caps[s] {
                v[j] = v[j - 1] * sigmoid(blocks[s].conf[j - 1]);
            }
            v
        })
        .collect();
    let mut best_k = vec![0usize; n];
    let mut best = base_rows as f64 / (cost.cost(base_rows) + draft);
    let mut ks = vec![0usize; n];
    loop {
        let mut s = 0;
        while s < n {
            if ks[s] < caps[s] {
                ks[s] += 1;
                break;
            }
            ks[s] = 0;
            s += 1;
        }
        if s == n {
            break;
        }
        // Added term by term onto the base: one stream is `choose_k`'s
        // running `e`, bit for bit.
        let mut e = base_rows as f64;
        for t in 0..n {
            for &r in &runs[t][1..=ks[t]] {
                e += r;
            }
        }
        let r = e / (cost.cost(base_rows + ks.iter().sum::<usize>()) + draft);
        if r > best {
            best = r;
            best_k = ks.clone();
        }
    }
    best_k
}

/// Calibration of the confidence head: sigmoid(conf_k) is read as P(draft k
/// accepted | drafts before it accepted). Draft k of a block is OBSERVED when
/// the block verified it and accepted every draft before it. Per decile of
/// the prediction and per depth: observed drafts, the predictions' sum, and
/// the accepted ones. Cumulative since start; logged every `CALIB_EVERY`
/// blocks.
#[derive(Default)]
struct Calib {
    blocks: u64,
    n: [u64; 10],
    pred: [f64; 10],
    acc: [u64; 10],
    depth_n: [u64; DRAFT_BLOCK],
    depth_pred: [f64; DRAFT_BLOCK],
    depth_acc: [u64; DRAFT_BLOCK],
}

const CALIB_EVERY: u64 = 500;

impl Calib {
    fn observe(&mut self, conf: &[f32; DRAFT_BLOCK], k: usize, accepted: usize) {
        self.blocks += 1;
        for j in 0..k.min(accepted + 1).min(DRAFT_BLOCK) {
            let p = sigmoid(conf[j]);
            let b = ((p * 10.0) as usize).min(9);
            let hit = u64::from(j < accepted);
            self.n[b] += 1;
            self.pred[b] += p;
            self.acc[b] += hit;
            self.depth_n[j] += 1;
            self.depth_pred[j] += p;
            self.depth_acc[j] += hit;
        }
    }

    /// `"label:predicted/accepted/n"` per non-empty cell.
    fn table(n: &[u64], pred: &[f64], acc: &[u64], label: impl Fn(usize) -> String) -> String {
        (0..n.len())
            .filter(|&i| n[i] > 0)
            .map(|i| format!("{}:{:.2}/{:.2}/{}", label(i), pred[i] / n[i] as f64, acc[i] as f64 / n[i] as f64, n[i]))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn log(&self) {
        let (n, p, a) = (self.n.iter().sum::<u64>(), self.pred.iter().sum::<f64>(), self.acc.iter().sum::<u64>());
        tracing::info!(
            blocks = self.blocks,
            drafts = n,
            predicted = format!("{:.3}", p / (n as f64).max(1.0)),
            accepted = format!("{:.3}", a as f64 / (n as f64).max(1.0)),
            by_decile = %Self::table(&self.n, &self.pred, &self.acc, |i| format!("{:.1}", i as f64 / 10.0)),
            by_depth = %Self::table(&self.depth_n, &self.depth_pred, &self.depth_acc, |i| format!("d{i}")),
            "ms dspark: confidence calibration (predicted/accepted/n)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-10-01 policy: the static p50 ladder, a 20 ms draft.
    fn static_cost() -> StepCost {
        StepCost::new(DEFAULT_LADDER.to_vec(), 20.0, false, 500.0)
    }

    /// Deterministic noise in [-1, 1).
    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }

    #[test]
    fn choose_k_follows_confidence() {
        let c = static_cost();
        // Confident blocks go deep, unconfident ones verify nothing.
        assert_eq!(choose_k(&[6.0; DRAFT_BLOCK], DRAFT_BLOCK, &c), DRAFT_BLOCK);
        assert_eq!(choose_k(&[-6.0; DRAFT_BLOCK], DRAFT_BLOCK, &c), 0);
        // The cap binds.
        assert_eq!(choose_k(&[6.0; DRAFT_BLOCK], 2, &c), 2);
        assert_eq!(choose_k(&[6.0; DRAFT_BLOCK], 0, &c), 0);
        // One confident draft then noise: stop after it.
        assert_eq!(choose_k(&[6.0, -6.0, -6.0, -6.0, -6.0], DRAFT_BLOCK, &c), 1);
    }

    #[test]
    fn stopping_rule_never_reads_past_its_stop() {
        // Changing any confidence AFTER the chosen K must not change K: the
        // decision on draft k reads conf[..=k] only (exactness for sampled
        // drafts, plan 2.4). Under the static ladder and a live-fitted one.
        let mut live = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0);
        let mut st = 7u64;
        for _ in 0..2000 {
            let rows = 1 + (lcg(&mut st).abs() * 6.0) as usize;
            live.observe_step(rows, 53.2 + 15.1 * rows as f64 + 10.0 * lcg(&mut st));
        }
        for c in [static_cost(), live] {
            let mut rng = 0x1234_5678u64;
            let mut next = || {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng >> 33) as f32 / (1u64 << 31) as f32) * 12.0 - 6.0
            };
            for _ in 0..2000 {
                let conf: [f32; DRAFT_BLOCK] = std::array::from_fn(|_| next());
                let k = choose_k_stopping(&conf, DRAFT_BLOCK, &c);
                for j in (k + 1)..DRAFT_BLOCK {
                    let mut c2 = conf;
                    c2[j] = next();
                    assert_eq!(choose_k_stopping(&c2, DRAFT_BLOCK, &c), k, "conf {conf:?}: K moved when conf[{j}] changed");
                }
            }
            assert_eq!(choose_k_stopping(&[6.0; DRAFT_BLOCK], DRAFT_BLOCK, &c), DRAFT_BLOCK);
            assert_eq!(choose_k_stopping(&[-6.0; DRAFT_BLOCK], DRAFT_BLOCK, &c), 0);
        }
    }

    /// A live-fitted cost model (cells), as production prices blocks.
    fn live_cost(seed: u64) -> StepCost {
        let mut live = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0).with_rows(16);
        let mut st = seed;
        for _ in 0..3000 {
            let rows = 1 + (lcg(&mut st).abs() * 12.0) as usize;
            live.observe_step(rows, 53.2 + 9.4 * rows as f64 + 10.0 * lcg(&mut st));
        }
        live
    }

    fn random_conf(st: &mut u64) -> [f32; DRAFT_BLOCK] {
        std::array::from_fn(|_| (lcg(st) * 6.0) as f32)
    }

    /// One stream, one base row: the joint rules ARE the one-stream rules, bit
    /// for bit (same sums in the same order), on any cost model and cap.
    #[test]
    fn joint_rules_with_one_stream_are_the_lone_rules() {
        let mut st = 99u64;
        for c in [static_cost(), live_cost(3)] {
            for _ in 0..4000 {
                let conf = random_conf(&mut st);
                let cap = (lcg(&mut st).abs() * 6.0) as usize;
                let b = [BlockConf { conf, cap }];
                assert_eq!(choose_ks_stopping(&b, 1, &c), vec![choose_k_stopping(&conf, cap, &c)], "conf {conf:?} cap {cap}");
                assert_eq!(choose_ks(&b, 1, &c), vec![choose_k(&conf, cap, &c)], "conf {conf:?} cap {cap}");
            }
        }
    }

    /// The joint stopping rule never reads a confidence past a stream's
    /// frontier: changing `conf_s[j]` for any `j > K_s` changes NO stream's K
    /// (conf_j reads draft j-1, which is then not verified). Reading
    /// `conf_s[K_s]` is allowed: it reads draft K_s - 1, which IS verified.
    #[test]
    fn joint_stopping_rule_never_reads_past_any_frontier() {
        let mut st = 12345u64;
        for c in [static_cost(), live_cost(5)] {
            for _ in 0..1500 {
                let blocks = [BlockConf { conf: random_conf(&mut st), cap: DRAFT_BLOCK }, BlockConf { conf: random_conf(&mut st), cap: DRAFT_BLOCK }];
                let base = 2;
                let ks = choose_ks_stopping(&blocks, base, &c);
                for s in 0..2 {
                    for j in (ks[s] + 1)..DRAFT_BLOCK {
                        let mut b2 = blocks;
                        b2[s].conf[j] = (lcg(&mut st) * 6.0) as f32;
                        assert_eq!(choose_ks_stopping(&b2, base, &c), ks, "blocks {blocks:?}: K moved when stream {s} conf[{j}] changed");
                    }
                }
            }
            // Confident blocks go to their caps, unconfident ones verify nothing.
            let sure = BlockConf { conf: [6.0; DRAFT_BLOCK], cap: DRAFT_BLOCK };
            let unsure = BlockConf { conf: [-6.0; DRAFT_BLOCK], cap: DRAFT_BLOCK };
            assert_eq!(choose_ks_stopping(&[sure, sure], 2, &c), vec![DRAFT_BLOCK, DRAFT_BLOCK]);
            assert_eq!(choose_ks_stopping(&[unsure, unsure], 2, &c), vec![0, 0]);
            assert_eq!(choose_ks_stopping(&[sure, unsure], 2, &c), vec![DRAFT_BLOCK, 0]);
            assert_eq!(choose_ks_stopping(&[BlockConf { cap: 2, ..sure }, sure], 2, &c), vec![2, DRAFT_BLOCK]);
        }
    }

    /// Changing `conf_s[m]` with `m <= K_s` may change K, but never the first
    /// `m` inclusion decisions of stream s: `min(K_s, m)` stays (they were
    /// made before `conf_s[m]` was read).
    #[test]
    fn joint_stopping_rule_decides_each_draft_before_reading_past_it() {
        let mut st = 4242u64;
        let c = live_cost(9);
        for _ in 0..2000 {
            let blocks = [BlockConf { conf: random_conf(&mut st), cap: DRAFT_BLOCK }, BlockConf { conf: random_conf(&mut st), cap: DRAFT_BLOCK }];
            let ks = choose_ks_stopping(&blocks, 2, &c);
            for s in 0..2 {
                for m in 0..DRAFT_BLOCK {
                    let mut b2 = blocks;
                    b2[s].conf[m] = (lcg(&mut st) * 6.0) as f32;
                    let k2 = choose_ks_stopping(&b2, 2, &c);
                    assert_eq!(k2[s].min(m), ks[s].min(m), "blocks {blocks:?}: stream {s}'s first {m} decisions moved with conf[{m}]");
                }
            }
        }
    }

    /// A multi-stream step's accounting feeds the MULTI tables only: the lone
    /// blocks' cells, their weights (aging) and `plain_ms` stay as they were.
    #[test]
    fn a_multi_step_feeds_only_the_multi_tables() {
        let mut d = MsDspark::bare();
        let (one, two, w1, w2, plain) = (
            d.lanes.one.cell_costs().to_vec(),
            d.lanes.two.cell_costs().to_vec(),
            d.lanes.one.cell_weights(),
            d.lanes.two.cell_weights(),
            d.plain_ms,
        );
        let before = d.multi.two.cell_weight(8);
        d.record_multi(&[(0, [1.0; DRAFT_BLOCK], 3, 2, 3), (1, [1.0; DRAFT_BLOCK], 3, 3, 4)], 8, 2, LaneRule::Learned, 120.0, 62.5);
        d.record_k0_multi(&[0, 1], 70.0);
        assert_eq!(d.lanes.one.cell_costs(), &one[..]);
        assert_eq!(d.lanes.two.cell_costs(), &two[..]);
        assert_eq!(d.lanes.one.cell_weights(), w1);
        assert_eq!(d.lanes.two.cell_weights(), w2);
        assert_eq!(d.plain_ms, plain);
        assert!(d.multi.two.cell_weight(8) > before, "the multi step's cell took the sample");
        assert_eq!((d.stats.multi_blocks, d.stats.multi_spec_steps, d.stats.blocks), (2, 1, 4));
    }

    /// A fixed K is capped at 8 rows per lane; the policies' candidates never
    /// exceed it either.
    #[test]
    fn speculating_steps_keep_every_lane_within_the_cap() {
        let t = multi_tables(4);
        let mut ks = vec![5, 5];
        fit_rows(&mut ks, 2, &t, LaneRule::Off);
        assert_eq!(ks, vec![3, 3], "one lane holds 8 rows");
        let mut ks = vec![5, 5];
        fit_rows(&mut ks, 2, &t, LaneRule::Learned);
        assert_eq!(ks, vec![5, 5], "two lanes hold 12 rows");
        let blocks = [BlockConf { conf: [0.0; DRAFT_BLOCK], cap: DRAFT_BLOCK }; 2];
        for rule in [LaneRule::Off, LaneRule::Threshold(4), LaneRule::Learned] {
            let cands = multi_choices(&t, &blocks, 2, rule);
            assert!(!cands.is_empty());
            for (ks, rows, lanes) in cands {
                assert_eq!(rows, 2 + ks.iter().sum::<usize>());
                assert!(rows > 2, "a candidate without drafts is a plain step, not a multi cell");
                assert!(rows <= lanes * SPEC_ROWS_PER_LANE, "{rule:?}: {rows} rows on {lanes} lanes");
                assert!(t.choices(rows, rule).contains(&lanes));
            }
        }
    }

    /// A cost model with a row cap: rows past `max` cost infinity.
    struct Capped<'a>(&'a StepCost, usize);
    impl CostModel for Capped<'_> {
        fn cost(&self, rows: usize) -> f64 {
            if rows > self.1 { f64::INFINITY } else { self.0.cost(rows) }
        }
        fn draft_ms(&self) -> f64 {
            self.0.draft_ms()
        }
    }

    /// The global search is the best of every (K_1, K_2) (point-mass drafts),
    /// and both rules respect a row cap priced as infinite cost.
    #[test]
    fn joint_search_is_optimal_and_rules_respect_a_row_cap() {
        let mut st = 777u64;
        let c = live_cost(11);
        for _ in 0..1500 {
            let blocks = [BlockConf { conf: random_conf(&mut st), cap: DRAFT_BLOCK }, BlockConf { conf: random_conf(&mut st), cap: DRAFT_BLOCK }];
            let rate = |ks: &[usize]| {
                let e: f64 = 2.0 + (0..2).map(|s| (1..=ks[s]).map(|k| (0..k).map(|j| sigmoid(blocks[s].conf[j])).product::<f64>()).sum::<f64>()).sum::<f64>();
                e / (c.cost(2 + ks[0] + ks[1]) + c.draft_ms())
            };
            let got = choose_ks(&blocks, 2, &c);
            let best = (0..=DRAFT_BLOCK).flat_map(|a| (0..=DRAFT_BLOCK).map(move |b| [a, b])).map(|k| rate(&k)).fold(f64::MIN, f64::max);
            assert!(rate(&got) >= best * (1.0 - 1e-12), "{got:?} rate {} < best {best}", rate(&got));
            let capped = Capped(&c, 8);
            assert!(choose_ks(&blocks, 2, &capped).iter().sum::<usize>() + 2 <= 8);
            assert!(choose_ks_stopping(&blocks, 2, &capped).iter().sum::<usize>() + 2 <= 8);
        }
    }

    #[test]
    fn ladder_extrapolates() {
        assert_eq!(ladder_cost(&DEFAULT_LADDER, 1), 60.0);
        assert_eq!(ladder_cost(&DEFAULT_LADDER, 8), 172.0);
        assert_eq!(ladder_cost(&DEFAULT_LADDER, 9), 188.0);
    }

    #[test]
    fn static_cost_is_the_ladder_and_ignores_samples() {
        let mut c = static_cost();
        for _ in 0..1000 {
            c.observe_step(4, 500.0);
            c.observe_draft(5.0);
        }
        for rows in 1..=9 {
            assert_eq!(c.cost(rows), ladder_cost(&DEFAULT_LADDER, rows));
        }
        assert_eq!(c.draft_ms(), 20.0);
    }

    #[test]
    fn live_fit_recovers_the_measured_line() {
        // The 10-01 production mix (rows 4 and 6 dominate; 1, 2, 3, 5 rare),
        // true cost 53.2 + 15.1 * rows with +-20 ms noise, prior = the
        // static ladder (4-row entry 8 ms low).
        let mut c = StepCost::new(DEFAULT_LADDER.to_vec(), 20.0, true, 500.0).with_shape(CostShape::Line);
        let mix = [(1usize, 2), (2, 12), (3, 3), (4, 52), (5, 6), (6, 25)];
        let mut st = 42u64;
        for _ in 0..40 {
            for &(rows, n) in &mix {
                for _ in 0..n {
                    c.observe_step(rows, 53.2 + 15.1 * rows as f64 + 20.0 * lcg(&mut st));
                }
            }
            for _ in 0..100 {
                c.observe_draft(12.0 + lcg(&mut st));
            }
        }
        for rows in 1..=6 {
            let want = 53.2 + 15.1 * rows as f64;
            assert!((c.cost(rows) - want).abs() < 3.0, "rows {rows}: {} vs {want}", c.cost(rows));
        }
        assert!((c.draft_ms() - 12.0).abs() < 0.5, "draft {}", c.draft_ms());
        // Monotone: no kink can make a deeper block cheaper.
        for rows in 1..6 {
            assert!(c.cost(rows + 1) > c.cost(rows));
        }
    }

    #[test]
    fn one_row_count_shifts_the_level_and_keeps_the_prior_slope() {
        // Every block verifies 5 drafts: the data pin cost(6), the prior the slope.
        let mut c = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0).with_shape(CostShape::Line);
        let prior_slope = c.b;
        for _ in 0..3000 {
            c.observe_step(6, 200.0);
        }
        assert!((c.cost(6) - 200.0).abs() < 3.0, "cost(6) {}", c.cost(6));
        assert!((c.b - prior_slope).abs() < 2.0, "slope {} vs prior {prior_slope}", c.b);
    }

    #[test]
    fn two_lane_prior_starts_at_its_first_row_and_the_rule_routes_by_rows() {
        // The two-lane ladder is indexed from row 1 like the one-lane one; only
        // rows >= first_row shape its prior line (09-30: 109.1 / 127.4 / 141.4).
        let c2 = StepCost::with_first_row(DEFAULT_LADDER_TWO_LANE.to_vec(), 4, 12.0, true, 500.0);
        for (rows, want) in [(4, 109.1), (5, 127.4), (6, 141.4)] {
            assert!((c2.cost(rows) - want).abs() < 3.0, "cost2({rows}) = {} vs {want}", c2.cost(rows));
        }
        let c1 = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0);
        let t = LaneTables::new(c1, c2, 4);
        let on = t.priced(LaneRule::Threshold(4));
        assert_eq!(on.cost(3), t.one.cost(3));
        assert_eq!(on.cost(4), t.two.cost(4));
        assert_eq!(on.draft_ms(), t.one.draft_ms());
        // Lanes off: the one-lane table only.
        let off = t.priced(LaneRule::Off);
        assert_eq!(off.cost(5), t.one.cost(5));
        // Cheaper deep blocks on two lanes push K deeper for the same confidences.
        let conf = [2.0f32, 1.0, 0.5, 0.2, 0.0];
        assert!(choose_k_stopping(&conf, DRAFT_BLOCK, &on) >= choose_k_stopping(&conf, DRAFT_BLOCK, &off));
    }

    #[test]
    fn a_stall_is_clamped() {
        let mut c = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0);
        for _ in 0..3000 {
            c.observe_step(4, 114.0);
        }
        let before = c.cost(4);
        c.observe_step(4, 10_000.0);
        assert!(c.cost(4) - before < 2.0, "{before} -> {}", c.cost(4));
    }

    #[test]
    fn a_few_bad_samples_cannot_close_a_regime() {
        // 10-01: the first two-lane verifies after restarts took 222, 1709,
        // 565, 738 ms (warm-up, a cold box 2) where ~110 is normal. They may
        // raise the estimate a bounded step, and a run of normal steps (which
        // exploration keeps supplying) must bring it back.
        let mut c2 = StepCost::with_first_row(DEFAULT_LADDER_TWO_LANE.to_vec(), 4, 12.0, true, 500.0).with_shape(CostShape::Line);
        let prior4 = c2.cost(4);
        for (rows, ms) in [(4, 222.0), (4, 1709.0), (5, 565.0), (6, 738.0)] {
            c2.observe_step(rows, ms);
        }
        // Each clamp is relative to the risen estimate, so four in a row compound
        // (~1.38x here) -- bounded; the old fit sat at ~3.4x (377.5 + 21.4/row).
        assert!(c2.cost(4) < 1.45 * prior4, "4 stalls: cost2(4) {} vs prior {prior4}", c2.cost(4));
        for _ in 0..40 {
            c2.observe_step(4, 110.0);
            c2.observe_step(5, 126.0);
        }
        assert!((c2.cost(4) - 110.0).abs() < 8.0, "after 80 normal steps: cost2(4) {}", c2.cost(4));
        // At half a sample of level prior and a 3x clamp the first stall alone
        // put the 4-row estimate at ~185 (what production logged).
        let mut one = StepCost::with_first_row(DEFAULT_LADDER_TWO_LANE.to_vec(), 4, 12.0, true, 500.0).with_shape(CostShape::Line);
        one.observe_step(4, 222.0);
        assert!(one.cost(4) < 130.0, "one warm-up step: cost2(4) {}", one.cost(4));
    }

    #[test]
    fn exploration_favours_stale_cells_at_its_rate() {
        let mut rng = StdRng::seed_from_u64(7);
        let (n, p) = (64_000, 1.0 / 32.0);
        // Cells K=0 (w 0, stale), K=1..4 (w 100, well measured), K=5 (w 1).
        let w = [0.0, 100.0, 100.0, 100.0, 100.0, 1.0];
        let mut hist = [0u32; DRAFT_BLOCK + 1];
        let mut hits = 0;
        for _ in 0..n {
            if let Some(k) = explore(&mut rng, DRAFT_BLOCK + 1, p, |k| w[k]) {
                hist[k] += 1;
                hits += 1;
            }
        }
        let rate = hits as f64 / n as f64;
        assert!((rate - p).abs() < 0.004, "rate {rate}");
        // Shares: half uniform (1/6 each) + half by 1 / (1 + w) (K=0 ~ 1, K=5 ~
        // 0.5, the measured ~ 0.0099 each) -- every K keeps >= 1/12 (the floor).
        let total: f64 = w.iter().map(|x| 1.0 / (1.0 + x)).sum();
        for (k, &c) in hist.iter().enumerate() {
            let share = 0.5 / 6.0 + 0.5 * (1.0 / (1.0 + w[k])) / total;
            let want = hits as f64 * share;
            assert!((c as f64 - want).abs() < 0.15 * want + 6.0, "K={k}: {c} vs {want:.0} ({hist:?})");
            assert!(c as f64 >= 0.8 * hits as f64 / 12.0, "K={k} below the uniform floor: {c} ({hist:?})");
        }
        assert!(hist.iter().all(|&c| c > 0), "every K stays possible: {hist:?}");
        // Equal weights: uniform. Never past the cap; nothing without room; off at p = 0.
        let mut rng = StdRng::seed_from_u64(1);
        let mut h2 = [0u32; 3];
        for k in (0..30_000).filter_map(|_| explore(&mut rng, 3, 1.0, |_| 5.0)) {
            h2[k] += 1;
        }
        assert!(h2.iter().all(|&c| (c as f64 - 10_000.0).abs() < 600.0), "{h2:?}");
        assert_eq!(explore(&mut rng, 1, 1.0, |_| 0.0), None);
        assert_eq!(explore(&mut rng, DRAFT_BLOCK + 1, 0.0, |_| 0.0), None);
    }

    #[test]
    fn an_unused_fit_forgets_by_time() {
        // The one-lane line learned a cold level, then every block ran two
        // lanes: aging alone (others' steps) must bring it back to its prior.
        let mut c = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0).with_shape(CostShape::Line);
        let prior3 = c.cost(3);
        for _ in 0..3000 {
            c.observe_step(3, 1.4 * prior3);
        }
        assert!(c.cost(3) > 1.3 * prior3);
        // ~8 memories of other regimes' steps (e^-8 of the data left).
        for _ in 0..4000 {
            c.age();
        }
        assert!(c.young(), "data weight {}", c.data[0]);
        assert!((c.cost(3) - prior3).abs() < 0.05 * prior3, "cost(3) {} vs prior {prior3}", c.cost(3));
    }

    #[test]
    fn a_mature_fit_keeps_the_paging_tail() {
        // Steps of 100 ms with every 10th a 250 ms paging stall (mean 115):
        // the mature fit must land near the mean, not near a tail trimmed at
        // 1.5x (~107.5); a young one may trim.
        let mut c = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0);
        assert!(c.young());
        for i in 0..3000 {
            c.observe_step(4, if i % 10 == 9 { 250.0 } else { 100.0 });
        }
        assert!(!c.young());
        assert!((c.cost(4) - 115.0).abs() < 4.0, "cost(4) {}", c.cost(4));
    }

    /// The 10-01 evening means (one lane, rows 1..6; MEASURED over 9,736 warm
    /// lone steps), ~+-8 ms noise, at an epsilon-exploring policy's row mix
    /// (rows 1 and 2 seen only through exploration).
    const MEASURED_ONE_LANE: [f64; 6] = [63.1, 78.8, 93.6, 107.2, 119.6, 130.2];

    #[test]
    fn cells_recover_a_concave_curve_a_line_cannot() {
        let mut st = 7u64;
        let mix = [(1usize, 1), (2, 1), (3, 12), (4, 10), (5, 6), (6, 10)];
        let mut cells = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0).with_shape(CostShape::Cells);
        let mut line = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0).with_shape(CostShape::Line);
        for _ in 0..200 {
            for &(rows, n) in &mix {
                for _ in 0..n {
                    let ms = MEASURED_ONE_LANE[rows - 1] + 8.0 * lcg(&mut st);
                    cells.observe_step(rows, ms);
                    line.observe_step(rows, ms);
                }
            }
        }
        for rows in 1..=6 {
            let want = MEASURED_ONE_LANE[rows - 1];
            assert!((cells.cost(rows) - want).abs() < 3.0, "cells rows {rows}: {} vs {want}", cells.cost(rows));
        }
        // The line cannot bend to the concavity: it misses some row count
        // (which one depends on where the row mix puts its weight).
        let worst = (1..=6).map(|r| (line.cost(r) - MEASURED_ONE_LANE[r - 1]).abs()).fold(0.0, f64::max);
        assert!(worst > 3.0, "line's worst row error {worst}");
    }

    #[test]
    fn cells_are_independent_and_a_young_cell_bounds_a_stall() {
        let mut c = StepCost::with_first_row(DEFAULT_LADDER_TWO_LANE.to_vec(), 4, 12.0, true, 500.0).with_shape(CostShape::Cells);
        // Neighbours do not pull a cell: a 5-row cell measured below the 4-row
        // one stays below it (the cost may legitimately fall with rows).
        for _ in 0..300 {
            c.observe_step(4, 105.0);
            c.observe_step(5, 100.0);
            c.observe_step(6, 125.0);
        }
        assert!((c.cost(4) - 105.0).abs() < 0.5 && (c.cost(5) - 100.0).abs() < 0.5 && (c.cost(6) - 125.0).abs() < 0.5, "{:?}", c.cell_costs());
        // One 1709 ms step in a young cell moves it a bounded step (1.5x clamp
        // against the cell's own start value), and normal steps bring it back.
        let mut y = StepCost::with_first_row(DEFAULT_LADDER_TWO_LANE.to_vec(), 4, 12.0, true, 500.0).with_shape(CostShape::Cells);
        let start = y.cost(4);
        y.observe_step(4, 1709.0);
        assert!(y.cost(4) <= 1.26 * start, "one stall: {} vs start {start}", y.cost(4));
        for _ in 0..30 {
            y.observe_step(4, 110.0);
        }
        assert!((y.cost(4) - 110.0).abs() < 5.0, "after 30 normal steps: {}", y.cost(4));
    }

    #[test]
    fn an_idle_cell_keeps_its_mean_but_a_new_sample_dominates() {
        // No prior between cells and no reversion: aging shrinks a cell's weight,
        // not its value; the next sample then carries it (bounded by the young clamp).
        let mut c = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0).with_shape(CostShape::Cells);
        for _ in 0..1000 {
            c.observe_step(2, 80.0);
        }
        for _ in 0..4000 {
            c.age();
        }
        // (The ladder start's pseudo-sample is all but gone: ~0.1 of ~500.)
        assert!((c.cost(2) - 80.0).abs() < 0.01, "{}", c.cost(2));
        // Far longer than a weight can stay above any clamp (e^-60): the value
        // stays put; and an untouched cell keeps its ladder start.
        let mut long = StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0).with_shape(CostShape::Cells);
        for _ in 0..1000 {
            long.observe_step(4, 107.0);
        }
        for _ in 0..30_000 {
            long.age();
        }
        assert!((long.cost(4) - 107.0).abs() < 0.01, "idle 30K steps: {}", long.cost(4));
        assert!((long.cost(6) - ladder_cost(&DEFAULT_LADDER, 6)).abs() < 1e-9, "never-sampled cell: {}", long.cost(6));
        long.observe_step(4, 120.0);
        assert!((long.cost(4) - 120.0).abs() < 0.01, "first sample after the idle: {}", long.cost(4));
        // ~0.17 samples of weight left (e^-8 of ~500): the new sample carries it.
        c.observe_step(2, 100.0);
        assert!(c.cost(2) > 96.0, "{}", c.cost(2));
    }

    fn cold_tables(start_from: usize) -> LaneTables {
        LaneTables::new(
            StepCost::new(DEFAULT_LADDER.to_vec(), 12.0, true, 500.0).with_shape(CostShape::Cells),
            StepCost::with_first_row(DEFAULT_LADDER_TWO_LANE.to_vec(), start_from, 12.0, true, 500.0).with_shape(CostShape::Cells),
            start_from,
        )
    }

    #[test]
    fn a_cold_learned_rule_makes_the_threshold_choice_then_follows_the_cells() {
        // Cold: the startup threshold's choice at every row count (the ladders
        // alone would put 4 rows on one lane: 107 vs 109.1).
        for m in [2, 3, 4, 6] {
            let t = cold_tables(m);
            for rows in 1..=CELLS {
                let want = if rows >= m { 2 } else { 1 };
                assert_eq!(t.best(rows, LaneRule::Learned).1, want, "start {m}, rows {rows}");
                assert_eq!(t.best(rows, LaneRule::Threshold(m)).1, want, "start {m}, rows {rows}");
            }
        }
        // Then row by row, whichever cell measured cheaper -- non-monotone in
        // rows included (owner: "the cost may legit be nonmonotonic").
        let mut t = cold_tables(4);
        let truth = |rows: usize, lanes: usize| -> f64 {
            match (rows, lanes) {
                (1, _) => 60.0,
                (2, 1) => 80.0,
                (2, _) => 75.0,
                (3, 1) => 95.0,
                (3, _) => 99.0,
                (4, 1) => 104.0,
                (4, _) => 100.0,
                (5, 1) => 110.0,
                (5, _) => 118.0,
                (6, 1) => 140.0,
                _ => 130.0,
            }
        };
        for _ in 0..50 {
            for rows in 1..=CELLS {
                for &l in LaneRule::Learned.choices(rows) {
                    t.observe(rows, l, truth(rows, l));
                }
            }
        }
        let picks: Vec<usize> = (2..=CELLS).map(|r| t.best(r, LaneRule::Learned).1).collect();
        assert_eq!(picks, [2, 1, 2, 1, 2]);
        for r in 2..=CELLS {
            let want = truth(r, 1).min(truth(r, 2));
            assert!((t.best(r, LaneRule::Learned).0 - want).abs() < 1.0, "rows {r}: {:?} vs {want}", t.best(r, LaneRule::Learned));
            assert!((t.priced(LaneRule::Learned).cost(r) - want).abs() < 1.0);
        }
        // The other rules ignore what the cells say.
        assert_eq!((2..=CELLS).map(|r| t.best(r, LaneRule::Threshold(4)).1).collect::<Vec<_>>(), [1, 1, 2, 2, 2]);
        assert!((1..=CELLS).all(|r| t.best(r, LaneRule::Off).1 == 1));
        // Static costs (`V41_MS_DSPARK_COST_LIVE=0`): nothing to learn from, so
        // `Learned` is the startup threshold.
        let t = LaneTables::new(static_cost(), StepCost::with_first_row(DEFAULT_LADDER_TWO_LANE.to_vec(), 3, 20.0, false, 500.0), 3);
        assert_eq!((1..=CELLS).map(|r| t.best(r, LaneRule::Learned).1).collect::<Vec<_>>(), [1, 1, 2, 2, 2, 2]);
        assert_eq!(t.choices(5, LaneRule::Learned), &[2]);
    }

    #[test]
    fn learned_exploration_reaches_every_row_and_lane_cell() {
        let t = cold_tables(4);
        // A block's candidates: (K, lanes), one lane only at K = 0 (one row).
        let cands = t.block_choices(DRAFT_BLOCK, LaneRule::Learned);
        assert_eq!(cands.len(), 1 + 2 * DRAFT_BLOCK);
        assert_eq!(cands[0], (0, 1));
        assert_eq!(t.block_choices(DRAFT_BLOCK, LaneRule::Threshold(4)), [(0, 1), (1, 1), (2, 1), (3, 2), (4, 2), (5, 2)]);
        assert_eq!(t.block_choices(2, LaneRule::Off), [(0, 1), (1, 1), (2, 1)]);
        // Every pair keeps the uniform floor (half the draws, 1/11 each).
        let mut rng = StdRng::seed_from_u64(3);
        let mut hist = vec![0u32; cands.len()];
        let n = 44_000;
        for _ in 0..n {
            if let Some(i) = explore(&mut rng, cands.len(), 1.0, |i| t.weight(cands[i].0 + 1, cands[i].1)) {
                hist[i] += 1;
            }
        }
        let floor = n as f64 / (2.0 * cands.len() as f64);
        assert!(hist.iter().all(|&c| c as f64 >= 0.9 * floor), "{hist:?}");
        // A plain step: no choice (no draw) under the threshold; under Learned
        // the cheaper count, and the other at the exploration rate.
        let mut pl = PlainLanes::from_env(8, 4);
        assert!((0..2000).all(|_| pl.pick(3, LaneRule::Threshold(4)) == 1 && pl.pick(5, LaneRule::Threshold(4)) == 2));
        assert_eq!(pl.explored, 0);
        let ones = (0..20_000).filter(|_| pl.pick(5, LaneRule::Learned) == 1).count();
        // p/4 from the uniform half + p/4 from the staleness half (equal
        // weights: neither cell has a sample yet).
        let want = 20_000.0 * explore_p() * 0.5;
        assert!((ones as f64 - want).abs() < 0.3 * want + 20.0, "{ones} vs {want:.0}");
        // Switches count policy changes per row count, exploration excluded.
        let mut sw = Switches::default();
        let flips: Vec<bool> = [(3, 1), (3, 1), (3, 2), (5, 2), (3, 1), (5, 2)].iter().map(|&(r, l)| sw.note(r, l)).collect();
        assert_eq!(flips, [false, false, true, false, true, false]);
        assert_eq!(sw.take(), "3:2");
        assert_eq!(sw.take(), "");
        // Rows past the table extend it; rows within are counted.
        pl.observe(8, 2, 170.0);
        pl.observe(9, 2, 180.0);
        assert_eq!(pl.hist[7], [0, 1]);
        assert!(pl.t.cost(9, 2) > pl.t.cost(8, 2));
    }

    #[test]
    fn calibration_observes_up_to_the_first_rejection() {
        let mut cal = Calib::default();
        // p ~ 0.95, 0.73, 0.5, 0.27, 0.05; 4 verified, 1 accepted: draft 0
        // accepted, draft 1 rejected, drafts 2-3 never tested.
        cal.observe(&[3.0, 1.0, 0.0, -1.0, -3.0], 4, 1);
        assert_eq!(cal.n.iter().sum::<u64>(), 2);
        assert_eq!((cal.n[9], cal.acc[9]), (1, 1));
        assert_eq!((cal.n[7], cal.acc[7]), (1, 0));
        assert_eq!(cal.depth_n, [1, 1, 0, 0, 0]);
        assert_eq!(cal.depth_acc, [1, 0, 0, 0, 0]);
        // All K accepted: every verified draft observed and accepted.
        cal.observe(&[3.0; DRAFT_BLOCK], 3, 3);
        assert_eq!(cal.depth_n, [2, 2, 1, 0, 0]);
        assert_eq!(cal.depth_acc, [2, 1, 1, 0, 0]);
        assert_eq!(cal.blocks, 2);
    }

    /// KNOWN_BUGS #50: with a request's min-p, every token a sampled draft can
    /// propose has target p > 0 when the drafter's logits are the target's.
    /// Before the fix q ignored min-p: here it kept ids 16-19, which the
    /// target's min-p (0.05) then top-p (over the survivors) removes, so such
    /// a draft was always rejected and ended its block.
    #[test]
    fn sampled_draft_q_stays_inside_the_targets_min_p_support() {
        let mode = SampleMode::Multinomial { temperature: 1.0, min_p_rel: 0.05, top_p: 0.95 };
        let ds = draft_sampling(&mode, [0.5; DRAFT_BLOCK]).expect("sampled drafts by default");
        assert_eq!(ds.min_p_rel, 0.05);
        let logits: Vec<f32> = (0..64).map(|i| -0.15 * i as f32).collect();
        let ids: Vec<i32> = (0..64).collect();
        let target = crate::spec_sample::TargetDist::from_logits(&logits, &mode);
        let (q, _) = v4flash_kernels::het::drafter::draft_dist(&ids, &logits, ds.temperature, ds.top_p, ds.min_p_rel, 0.5).unwrap();
        for &(t, _) in &q {
            assert!(target.prob(t) > 0.0, "draft {t} has q > 0 but p = 0");
        }
        assert_eq!(q.len(), 16, "q's support is the target's: min-p keeps 0..=19, top-p 0..=15");
    }

    /// Issue #3 finding 2: drafted steps that keep coming out K = 0 pay the
    /// draft for one token each; charged to the gain, the stream backs off.
    #[test]
    fn k0_steps_with_the_draft_charged_back_off() {
        let (mut gain, mut backoff, mut skip) = (GAIN0, 0u32, 0u32);
        // plain step 80 ms, draft 12 ms: one token for 92 ms
        let g = 80.0 / (80.0 + 12.0);
        let mut n = 0;
        while backoff == 0 {
            gate_update(&mut gain, &mut backoff, &mut skip, g, 1.0);
            n += 1;
            assert!(n < 20, "never backed off");
        }
        assert_eq!((n, backoff, skip), (6, 4, 4));
        // The probe after the back-off moves half way: a good block clears it.
        gate_update(&mut gain, &mut backoff, &mut skip, 2.5, 1.0);
        assert_eq!(backoff, 0);
        // The step counts like a block, with nothing verified.
        let mut s = Stats::default();
        s.note_k0(80.0);
        assert_eq!((s.blocks, s.emitted, s.k_hist[0], s.drafts_verified, s.accepted), (1, 1, 1, 0, 0));
        assert_eq!(s.lanes_by_rows[0], [1, 0]);
        assert_eq!(s.step_ms, 80.0);
    }
}
