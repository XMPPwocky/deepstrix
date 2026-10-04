//! The server's knobs (`v4flash_kernels::knobs`: one implementation, values
//! in atomics, live ones re-read from `V41_KNOBS_FILE` and their legacy
//! files). Each keeps the env name, default and parse it had before 10-01's
//! move here. Only the eight knobs that were live before are live: the others
//! size or choose something once, or were never tested changing under a
//! running scheduler.

use v4flash_kernels::het::mtp::MTP_BLOCK;

const MAX: u64 = u32::MAX as u64;

v4flash_kernels::knobs! {
    // ---- OpenAI API (static)
    /// `V41_API_STRICT` (`1`): a request with a sampling parameter the server
    /// does not implement (`top_k`, the penalties, `logit_bias`) is a 400.
    /// Default off: it is served without them and a warning names them (client
    /// presets send `top_k: 40` / `repetition_penalty: 1.1` on every request).
    pub static API_STRICT = Knob::flag("V41_API_STRICT", false);
    // ---- multistream: scheduler and arena (static)
    /// `V41_MULTISTREAM` (`1|on`): the multistream scheduler.
    pub static MULTISTREAM = Knob::flag("V41_MULTISTREAM", false);
    /// `V41_MS_SLOTS` (default 8): arena slots.
    pub static MS_SLOTS = Knob::int("V41_MS_SLOTS", 8, 1, 1024);
    /// `V41_MS_CTX_ROWS`: arena rows per stream; default (`Source::Default`) =
    /// twice the model's KV maximum.
    pub static MS_CTX_ROWS = Knob::int("V41_MS_CTX_ROWS", 0, 0, MAX);
    /// `V41_MS_CHUNK_ROWS` (default 1024): prefill chunk rows while streams
    /// decode; also sizes the prefill scratch.
    pub static MS_CHUNK_ROWS = Knob::int("V41_MS_CHUNK_ROWS", 1024, 1, MAX);
    /// `V41_MS_CHUNK_ROWS_IDLE` (default 1024): prefill chunk rows with no
    /// stream decoding.
    pub static MS_CHUNK_ROWS_IDLE = Knob::int("V41_MS_CHUNK_ROWS_IDLE", 1024, 1, MAX);
    /// `V41_MS_PREFILL_JOBS` (default 2): prefill jobs in flight.
    pub static MS_PREFILL_JOBS = Knob::int("V41_MS_PREFILL_JOBS", 2, 1, 64);
    /// `V41_MS_KV_HEADROOM` (default 16384): `multistream::kv_headroom`.
    pub static MS_KV_HEADROOM = Knob::int("V41_MS_KV_HEADROOM", 16384, 0, MAX);
    /// `V41_MS_KV_GROW` (default 16384): `multistream::kv_grow_step`.
    pub static MS_KV_GROW = Knob::int("V41_MS_KV_GROW", 16384, 1, MAX);
    /// `V41_MS_KV_GROW_AT` (default 256): `multistream::kv_grow_at`.
    pub static MS_KV_GROW_AT = Knob::int("V41_MS_KV_GROW_AT", 256, 0, MAX);
    /// `V41_MS_KV_SPARE`: `multistream::kv_spare`; default (`Source::Default`)
    /// = `V41_MS_KV_GROW`.
    pub static MS_KV_SPARE = Knob::int("V41_MS_KV_SPARE", 0, 0, MAX);
    /// `V41_MS_PREFILL_BURST_MS` (default 120000): `Sched::burst_budget`.
    pub static MS_PREFILL_BURST_MS = Knob::int("V41_MS_PREFILL_BURST_MS", 120_000, 0, MAX);
    /// `V41_MS_PREFILL_BURST_MIN_MS` (default 10000).
    pub static MS_PREFILL_BURST_MIN_MS = Knob::int("V41_MS_PREFILL_BURST_MIN_MS", 10_000, 0, MAX);
    /// `V41_MS_DECODE_BURST_MS` (default 30000).
    pub static MS_DECODE_BURST_MS = Knob::int("V41_MS_DECODE_BURST_MS", 30_000, 0, MAX);
    /// `V41_MS_DECODE_BURST_MIN_MS` (default 3000).
    pub static MS_DECODE_BURST_MIN_MS = Knob::int("V41_MS_DECODE_BURST_MIN_MS", 3_000, 0, MAX);
    /// `V41_MS_BURST_SCALE` (`1`): scale the bursts by load.
    pub static MS_BURST_SCALE = Knob::flag("V41_MS_BURST_SCALE", false);
    /// `V41_MS_AGING_S` (default 60): queue aging.
    pub static MS_AGING_S = Knob::int("V41_MS_AGING_S", 60, 0, MAX);
    /// `V41_MS_STARVE_S` (default 600): queue starvation bound.
    pub static MS_STARVE_S = Knob::int("V41_MS_STARVE_S", 600, 0, MAX);
    /// `V41_MS_CHECKPOINT_EVERY` (default 32768; `0` = no periodic checkpoints):
    /// prefill checkpoint stride.
    pub static MS_CHECKPOINT_EVERY = Knob::int("V41_MS_CHECKPOINT_EVERY", 32768, 0, MAX);
    /// `V41_MS_CHECKPOINT_MIN_ROWS` (default 4096).
    pub static MS_CHECKPOINT_MIN_ROWS = Knob::int("V41_MS_CHECKPOINT_MIN_ROWS", 4096, 0, MAX);
    /// `V41_MS_FINISH_GROUP` (default on): `multistream::finish_group`.
    pub static MS_FINISH_GROUP = Knob::flag("V41_MS_FINISH_GROUP", true);
    /// `V41_MS_ENGRAM_AHEAD` (default on): the Engram look-ahead gather.
    pub static MS_ENGRAM_AHEAD = Knob::flag("V41_MS_ENGRAM_AHEAD", true);
    /// `V41_B1_HOT_REFRESH` (default 500): ticks between box-1 hot-set refreshes.
    pub static B1_HOT_REFRESH = Knob::int("V41_B1_HOT_REFRESH", 500, 1, MAX);
    /// `V41_MS_PROFILE` (`1`): per-stage GPU timing (`ms.stage`).
    pub static MS_PROFILE = Knob::flag("V41_MS_PROFILE", false);
    /// `V41_MS_PROFILE_EVERY` (default 20): steps per `ms.stage` rollup.
    pub static MS_PROFILE_EVERY = Knob::int("V41_MS_PROFILE_EVERY", 20, 1, MAX);

    // ---- multistream: decode step drivers (static)
    /// `V41_MS_PIPELINE` (default on; `0` = one lane always).
    pub static MS_PIPELINE = Knob::flag("V41_MS_PIPELINE", true);
    /// `V41_MS_STAGGER`: `0` lockstep (default), `1` staggered, `2` ready-first.
    pub static MS_STAGGER = Knob::choice("V41_MS_STAGGER", 0, &[&["0", "off"], &["1", "on"], &["2", "ready-first"]]);
    /// `V41_MS_LANES` (default 2): `3` = three lanes from `V41_MS_LANES3_MIN_ROWS`.
    pub static MS_LANES = Knob::int("V41_MS_LANES", 2, 1, 3);
    /// `V41_MS_LANES3_MIN_ROWS` (default 6).
    pub static MS_LANES3_MIN_ROWS = Knob::int("V41_MS_LANES3_MIN_ROWS", 6, 3, MAX);

    // ---- multistream: LIVE (the 10-01 `_FILE` knobs; their files still work)
    /// `V41_MS_HEAD_CANDS`: `1` (default) candidates, `0` full rows, `check` both.
    pub static MS_HEAD_CANDS = Knob::choice("V41_MS_HEAD_CANDS", 0, &[&["1", "on"], &["0", "off"], &["check"]]).legacy("V41_MS_HEAD_CANDS_FILE");
    /// `V41_MS_SPEC_LANES` (default on): `multistream::spec_lanes_on`.
    pub static MS_SPEC_LANES = Knob::flag("V41_MS_SPEC_LANES", true).legacy("V41_MS_SPEC_LANES_FILE");
    /// `V41_MS_PIPELINE_MIN_ROWS` (default 6, >= 2): `multistream::pipeline_min_rows`.
    pub static MS_PIPELINE_MIN_ROWS = Knob::int("V41_MS_PIPELINE_MIN_ROWS", 6, 2, 64).legacy("V41_MS_PIPELINE_MIN_ROWS_FILE");
    /// `V41_MS_LANES_LEARNED` (default off): `multistream::lanes_learned`.
    pub static MS_LANES_LEARNED = Knob::flag("V41_MS_LANES_LEARNED", false).legacy("V41_MS_LANES_LEARNED_FILE");
    /// `V41_MS_ENGRAM_THREADS` (default 32, 1..=512): `multistream::engram_threads`.
    pub static MS_ENGRAM_THREADS = Knob::int("V41_MS_ENGRAM_THREADS", 32, 1, 512).legacy("V41_MS_ENGRAM_THREADS_FILE");

    // ---- live perfetto traces (multistream `LiveTrace`)
    /// `V41_PERFETTO_STEPS` (live, default 0): set it to N > 0 -- a new value,
    /// or 0 then N again -- and the scheduler writes a perfetto trace of the
    /// next N decode steps (and the prefill units between them) to
    /// `V41_PERFETTO_DIR`, then detaches. Device stages of both GPUs, box-1
    /// paging, the remote-expert round trips and box 2's paging as reported
    /// over the wire, and every knob. Ignored while `V41_PERFETTO_OUT` holds a
    /// trace open. A traced step reads a little long: each re-anchors the
    /// device clocks (four stream syncs, one may wait on in-flight box-1
    /// prefetch copies).
    pub static PERFETTO_STEPS = Knob::int("V41_PERFETTO_STEPS", 0, 0, 1_000_000).live();
    /// `V41_PERFETTO_KERNELS` (live, default off): also each kernel (`k.*`
    /// sub-stages). Several x the events: a full pool (16384 per device)
    /// fails the step, so keep N small with it.
    pub static PERFETTO_KERNELS = Knob::flag("V41_PERFETTO_KERNELS", false).live();
    /// `V41_PERFETTO_DIR` (default `~/traces`): where live traces go.
    pub static PERFETTO_DIR = Knob::text("V41_PERFETTO_DIR");

    // ---- DSpark on the arena (static)
    /// `V41_MS_DSPARK` (`accept|1|on`): `ms_dspark::enabled`.
    pub static MS_DSPARK = Knob::choice("V41_MS_DSPARK", 0, &[&["0", "off"], &["accept", "1", "on"]]);
    /// `V41_MS_DSPARK_RING`: `all` (default) or `solo`: `ms_dspark::ring_all`.
    pub static MS_DSPARK_RING = Knob::choice("V41_MS_DSPARK_RING", 0, &[&["all"], &["solo"]]);
    /// `V41_MS_DSPARK_DRAFTS`: `sampled` (default) or `argmax`.
    pub static MS_DSPARK_DRAFTS = Knob::choice("V41_MS_DSPARK_DRAFTS", 0, &[&["sampled"], &["argmax"]]);
    /// `V41_MS_DSPARK_RING_ASYNC` (default on).
    pub static MS_DSPARK_RING_ASYNC = Knob::flag("V41_MS_DSPARK_RING_ASYNC", true);
    /// `V41_LONG_PREFILL_TOKENS` (default 16384): a prefill job longer than this
    /// flags its box-2 requests `REQ_FLAG_LONG_JOB` (box 2's phase keeps the
    /// job's experts: `knobs::prefill_budget_long` there); 0 = never.
    pub static LONG_PREFILL_TOKENS = Knob::int("V41_LONG_PREFILL_TOKENS", 16384, 0, u32::MAX as u64);
    /// `V41_MS_DSPARK_MIN_GAIN` (default 1.0): the stage-1 back-off bar.
    pub static MS_DSPARK_MIN_GAIN = Knob::real("V41_MS_DSPARK_MIN_GAIN", 1.0, 0.0, 1e9);
    /// `V41_MS_DSPARK_EXPLORE` (default 1/32): `ms_dspark::explore_p`.
    pub static MS_DSPARK_EXPLORE = Knob::real("V41_MS_DSPARK_EXPLORE", 1.0 / 32.0, 0.0, 1.0);
    /// `V41_MS_DSPARK_EXPLORE_SEED` (a u64; unset = entropy).
    pub static MS_DSPARK_EXPLORE_SEED = Knob::text("V41_MS_DSPARK_EXPLORE_SEED");
    /// `V41_MS_DSPARK_COST_SHAPE`: `cells` (default) or `line`.
    pub static MS_DSPARK_COST_SHAPE = Knob::choice("V41_MS_DSPARK_COST_SHAPE", 0, &[&["cells"], &["line"]]);
    /// `V41_MS_DSPARK_COST`: the one-lane ladder (ms of 1, 2, ... rows).
    pub static MS_DSPARK_COST = Knob::text("V41_MS_DSPARK_COST");
    /// `V41_MS_DSPARK_COST2`: the two-lane ladder.
    pub static MS_DSPARK_COST2 = Knob::text("V41_MS_DSPARK_COST2");
    /// `V41_MS_DSPARK_DRAFT_MS` (default 20): the draft's starting estimate.
    pub static MS_DSPARK_DRAFT_MS = Knob::real("V41_MS_DSPARK_DRAFT_MS", 20.0, 0.0, 1e9);
    /// `V41_MS_DSPARK_COST_LIVE` (default on; `0` = the static ladder).
    pub static MS_DSPARK_COST_LIVE = Knob::flag("V41_MS_DSPARK_COST_LIVE", true);
    /// `V41_MS_DSPARK_COST_MEMORY` (default 500 lone steps).
    pub static MS_DSPARK_COST_MEMORY = Knob::real("V41_MS_DSPARK_COST_MEMORY", 500.0, 10.0, 1e12);
    /// `V41_MS_LANES_MEMORY` (default 1000 plain multi-stream steps).
    pub static MS_LANES_MEMORY = Knob::real("V41_MS_LANES_MEMORY", 1000.0, 10.0, 1e12);
    /// `V41_MS_DSPARK_K`: verify exactly this many drafts (unset = the policy).
    pub static MS_DSPARK_K = Knob::text("V41_MS_DSPARK_K");
    /// `V41_MS_DSPARK_KMAX` (default the block size).
    pub static MS_DSPARK_KMAX = Knob::int("V41_MS_DSPARK_KMAX", MTP_BLOCK as u64, 0, MTP_BLOCK as u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use v4flash_kernels::knobs::{Kind, Knob};

    #[test]
    fn the_table_is_sane_and_only_the_legacy_file_knobs_are_live() {
        let all: Vec<&Knob> = ALL.iter().chain(v4flash_kernels::knobs::ALL.iter()).copied().collect();
        let mut names = std::collections::HashSet::new();
        for k in &all {
            assert!(k.name.starts_with("V41_"), "{}", k.name);
            assert!(names.insert(k.name), "{} declared twice", k.name);
            assert!(k.live || k.legacy.is_none(), "{}: a legacy _FILE knob is live", k.name);
            if let Some(f) = k.legacy {
                assert!(f.ends_with("_FILE"), "{f}");
            }
            assert!(!(k.live && matches!(k.kind, Kind::Text)), "{}: text knobs are static", k.name);
        }
        // Live: the eight 10-01 `_FILE` knobs, the live-trace trigger, Tier
        // B's device timing (its kill switch), the box-2 partial upload and the
        // layer-major group prefetch (A/B'd per turn).
        let mut live: Vec<&str> = all.iter().filter(|k| k.live).map(|k| k.name).collect();
        live.sort();
        assert_eq!(live, [
            "V41_B2_PIN_PREFILL_BAND", "V41_EVTRACE_DEV", "V41_LM_PREFETCH", "V41_LM_PREFETCH_PER_REQ", "V41_LM_PREFILL",
            "V41_MS_ENGRAM_THREADS", "V41_MS_HEAD_CANDS", "V41_MS_LANES_LEARNED",
            "V41_MS_PIPELINE_MIN_ROWS", "V41_MS_SPEC_LANES", "V41_PERFETTO_KERNELS", "V41_PERFETTO_STEPS",
            "V41_REMOTE_PARTIAL_ASYNC", "V41_SUB_LAMBDA",
        ]);
        // The parses these knobs had before.
        let p = |k: &Knob, s: &str| k.kind.parse(s);
        assert_eq!(p(&MS_DSPARK, "accept"), Some(1));
        assert_eq!(p(&MS_DSPARK, "on"), Some(1));
        assert_eq!(p(&MS_HEAD_CANDS, "check"), Some(2));
        assert_eq!(p(&MS_HEAD_CANDS, "0"), Some(1));
        assert_eq!(p(&MS_STAGGER, "2"), Some(2));
        assert_eq!(p(&MS_PIPELINE_MIN_ROWS, "1"), Some(2), "never below 2");
        assert_eq!(p(&MS_ENGRAM_THREADS, "9999"), Some(512));
        assert_eq!(p(&MS_DSPARK_RING, "solo"), Some(1));
        assert_eq!(p(&MS_CHECKPOINT_EVERY, "0"), Some(0), "0 = periodic checkpoints off");
    }
}
