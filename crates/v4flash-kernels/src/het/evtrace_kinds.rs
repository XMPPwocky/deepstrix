//! Event kinds for `het::evtrace` (ids are stable: the reader keys on the
//! header, but keep ids unique and never reuse one for different fields).
//! `t_*` fields are CLOCK_MONOTONIC_RAW ns of the emitting box, except the
//! hub's `t2_b2`/`t3_b2`, which are box 2's stamps echoed in the reply.

use super::evtrace::Kind;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Every hub/box-2 kind (added to `evtrace`'s header list).
pub static ALL: &[&Kind] = &[&HUB_REQ, &HUB_STEP, &HUB_PHASE, &B2_REQ, &B2_READ, &B2_ENSURE, &B2_WRITE];

// ---- hub: step context for per-request records ----

/// Decode step number (multistream), and that step's rows / lanes, stamped
/// into every `hub_req` so requests group by step.
pub static STEP: AtomicU64 = AtomicU64::new(u64::MAX);
pub static STEP_ROWS: AtomicU64 = AtomicU64::new(u64::MAX);

pub fn set_step(step: u64, rows: u64) {
    STEP.store(step, Relaxed);
    STEP_ROWS.store(rows, Relaxed);
}

/// Outside a decode step (prefill ticks run on the same thread): `hub_req`
/// records then carry NaN step / rows instead of the last decode step's.
pub fn clear_step() {
    STEP.store(u64::MAX, Relaxed);
    STEP_ROWS.store(u64::MAX, Relaxed);
}

pub fn step_f64() -> f64 {
    match STEP.load(Relaxed) {
        u64::MAX => f64::NAN,
        s => s as f64,
    }
}

pub fn step_rows_f64() -> f64 {
    match STEP_ROWS.load(Relaxed) {
        u64::MAX => f64::NAN,
        s => s as f64,
    }
}

/// One box-2 request as the hub saw it, emitted when its reply is consumed
/// (arena / multistream path only; the legacy decode path in forward_layer
/// is not instrumented).
/// `t_submit..t_submit_end` = the client call (encode + queue to the writer),
/// `t1` writer stamp just before `write()`, `t4` reader stamp on receipt,
/// `t_wait_enter/exit` the hub thread's blocking wait. `blocked` = the reply
/// landed after the wait began (the leg was exposed). `n_pred_*` = the
/// mirror's view of this request's distinct experts at submit.
pub static HUB_REQ: Kind = Kind {
    id: 10,
    name: "hub_req",
    fields: &[
        "t_submit", "t_submit_end", "t1", "t4", "t_wait_enter", "t_wait_exit", "t2_b2", "t3_b2",
        "step", "lane", "layer", "b", "seq", "flags", "partner", "unmasked", "n_hints", "n_pf_words",
        "n_picks", "n_distinct", "n_pred_miss", "n_pred_incoming", "n_pred_pending",
        "rtt_us", "srv_us", "page_us", "compute_us", "n_miss", "miss_bits", "bytes_out", "bytes_in",
        "blocked", "clock_offset_ns", "clock_delay_ns", "step_rows",
    ],
};

/// One decode step (multistream `step`), all per-step values -- the numbers
/// `ms.stage` folds into 20-step means. `NaN` where profiling is off. The
/// emitter fills fields BY NAME: `lh.x` -> `lh_x`, `dgpu.a.b` -> `d_a_b`,
/// `igpu.a` -> `i_a`, so a stage added or renamed upstream reads NaN here
/// rather than shifting columns.
pub static HUB_STEP: Kind = Kind {
    id: 11,
    name: "hub_step",
    fields: &[
        // identity + wall
        "t_start", "t_end", "step", "rows", "live", "lanes", "fwd_ms", "fwd_all_ms", "engram_ms", "sample_ms", "step_ms", "profiled",
        // box-2 leg, summed over the step's requests (ms / counts)
        "remote_wait_ms", "remote_rtt_ms", "remote_srv_ms", "b2_page_ms", "b2_service_ms", "b2_misses", "b2_paged_replies",
        // hop split (remote_experts::take_hop_stats)
        "hop_submit_to_write_us", "hop_wake_us", "hop_slack_us", "hop_blocked", "hop_waits",
        // cache-prior / substitution (b2_mirror::take_sub_stats)
        "sub_predicted_miss", "sub_reads_avoided", "sub_picks_swapped", "sub_blocked", "sub_plan_failed", "sub_admits_queued", "sub_incoming_covered",
        // box-1 pager (deltas; a prefill between steps lands in the next step)
        "b1_misses", "b1_read_ms", "b1_pf_queued", "b1_pf_admitted", "b1_pf_dropped_full", "b1_pf_admit_ms",
        // layer-host timers (lh.*, ms)
        "lh_pre_moe", "lh_post_moe", "lh_engram", "lh_pager_block", "lh_sel_sync", "lh_remote_submit", "lh_ensure", "lh_owns",
        "lh_excl", "lh_audit", "lh_remap_h2d", "lh_work_items_sync", "lh_remote_wait", "lh_pager_sync_igpu", "lh_engram_join",
        "lh_work_items_count", "lh_sel_d2h", "lh_sub", "lh_wic_busy_x1e3", "lh_wic_idle_x1e3", "lh_seld2h_busy_x1e3",
        "lh_seld2h_idle_x1e3", "lh_remote_sync",
        // device busy (event time, parent stages)
        "dgpu_busy_ms", "igpu_busy_ms",
        // named dGPU stages (ms)
        "d_output_proj", "d_attn_compute", "d_q_chain", "d_shared_expert", "d_mhc_pre_attn", "d_mhc_pre_ffn", "d_mhc_mix_ffn_late",
        "d_router", "d_prefill_indexer", "d_prefill_indexer_reuse", "d_peer_push_ffn_input_norm", "d_kv_chain", "d_rb_pack",
        "d_head_batch", "d_engram", "d_ffn_combine_local", "d_ffn_combine_remote", "d_kv_append_compressor_serial", "d_mhc_post_attn",
        // named iGPU stages (ms)
        "i_pair_kwide", "i_q2k_down", "i_moe_group_builder", "i_moe_work_items", "i_q8k_quantize_pre_iq2", "i_q8k_quantize_post_iq2",
        "i_peer_push_ffn_moe",
        // parent stages not named above (busy minus the named ones; the names
        // are logged once as `evtrace: hub_step stages not named`)
        "d_other", "i_other",
        // context
        "pos_min", "pos_max",
    ],
};

/// Scheduler phase change (multistream `ms.phase`). Phases: 0 Decode,
/// 1 Prefill.
pub static HUB_PHASE: Kind = Kind {
    id: 12,
    name: "hub_phase",
    fields: &["t", "from", "to", "live", "prefills", "queued", "burst_ms", "next_budget_ms", "starved"],
};

// ---- box 2 (expertd) ----

/// One request on box 2 (`serve_connection`), emitted when its reply is
/// handed to the writer. Stamps in order: header read, frame complete (`t2`),
/// dequeued by the compute loop, merge done, hints/prefetch words queued,
/// `run_path` start/end (paging + kernels), D2H done, ready. `d_*` = this
/// layer's page-stat deltas across the request, as separate components (the
/// reported `page_us` sums read+h2d+repack_gpu, which double-counts the repack
/// -- read_ns already contains h2d_ns -- and includes blocked prefetch waits).
/// `pf_*` = the background readers at dequeue (`run_*`/`q_*` running/queued
/// certain/speculative) and their counter deltas across the request.
/// `merged`: 0 alone, 1 carried a partner (`partner_*`), 2 the partner itself
/// (its stamps are the carrier's pass -- `t_d2h_end`/`t_ready` included, its
/// own D2H runs after them -- and its per-pass fields are NaN: page/miss/d_*/
/// exec/pf deltas are reported on the carrier). `served_under` = the parked
/// request this one was served inside (`knobs::park`), else NaN; such records
/// have only the fields that path measures. A PARKED request's page deltas
/// and `n_miss` include the paging of the requests served inside it on the
/// same layer (which have their own rows): subtract theirs to get its own.
pub static B2_REQ: Kind = Kind {
    id: 20,
    name: "b2_req",
    fields: &[
        "seq", "layer", "b", "flags", "merged", "partner_seq", "partner_b", "partner_promised", "served_under",
        "t_hdr", "t_frame", "t_dequeue", "t_merge_end", "t_hints_end", "t_run_start", "t_run_end", "t_d2h_end", "t_ready",
        "depth_on_take", "pending_after", "idle_before_us",
        "n_sel", "n_distinct", "n_hint_admit", "n_prefetch_words",
        "n_miss", "page_us", "compute_us", "server_us",
        "d_misses", "d_read_ns", "d_h2d_ns", "d_repack_gpu_ns", "d_pread_ns", "d_repack_cpu_ns", "d_prefetch_wait_ns",
        "park_wait_ns", "park_serve_ns",
        "path_decode", "two_pass", "n_work_items", "n_missing", "exec_h2d_us", "exec_gpu_us",
        "pf_run_certain", "pf_run_spec", "pf_q_certain", "pf_q_spec", "pf_free_sets", "pf_pending",
        "pf_d_hinted", "pf_d_admitted", "pf_d_dropped", "pf_d_waited", "pf_d_promoted",
        "pool_resident",
    ],
};

/// One `ensure_layer_inner` call (a request's paging for one layer).
/// `admit_*` = landing finished background reads first (`wait_ns` blocked on
/// this request's own in-flight picks); then hit/miss + victim search, the
/// demand reads, and the remap uploads.
pub static B2_ENSURE: Kind = Kind {
    id: 22,
    name: "b2_ensure",
    fields: &[
        "seq", "layer", "n_ids", "n_want", "n_hits", "n_miss", "prefill_shaped",
        "t_start", "t_admit_end", "t_dirty_end", "t_victims_end", "t_reads_end", "t_end",
        "admit_wait_ns", "admit_landed", "admit_landed_wanted", "admit_blocking_recvs",
        "victim_scan_ns", "evicted_foreign", "took_free", "k_par", "n_chunks",
        "dirty_upload", "remap_upload_ns",
        "pf_run_certain", "pf_run_spec", "pf_q_certain", "pf_q_spec",
    ],
};

/// One expert read on box 2. `src`: 0 demand miss (`ensure`), 1 background
/// read popped as certain, 2 popped speculative but made certain before its
/// read, 3 speculative. Demand: emitted after its repack (`t_hint` = the
/// ensure start, `t_pop` = its chunk start). Background: emitted when the
/// compute thread LANDS it (`t_recv` = received, `t_land_*` = victim + repack
/// + commit). `rK_start/end` = the three role reader threads; `pause_ns` =
/// io_throttle pauses (speculative chunks) SUMMED over the role threads (can
/// exceed the read's wall), `yield_ns` = the wait before a speculative read
/// starts. The concurrency fields are sampled at READ START (after the yield);
/// `demand_reads_at_start` is NaN for demand reads (only their own thread
/// changes it). Demand: `set` = its staging set (= index in the chunk),
/// `chunk_idx` = the chunk's number. `wanted`/`blocked_on` = the request being served
/// needed it / the compute thread was blocked waiting for it.
pub static B2_READ: Kind = Kind {
    id: 21,
    name: "b2_read",
    fields: &[
        "src", "seq", "layer", "expert", "slot", "victim_layer", "victim_expert", "set",
        "t_hint", "t_pop", "t_read_start", "t_read_end", "t_recv", "t_land_start", "t_land_end",
        "yield_ns", "pause_ns", "demand_reads_at_start", "run_certain_at_start", "run_spec_at_start",
        "r0_start", "r0_end", "r1_start", "r1_end", "r2_start", "r2_end",
        "wanted", "blocked_on", "coalesced", "victim_scan_ns", "repack_ns", "chunk_n", "chunk_idx", "already_resident",
    ],
};

/// One reply written on box 2 (writer thread): `t3` stamp and `write()` end.
pub static B2_WRITE: Kind = Kind { id: 23, name: "b2_write", fields: &["seq", "t3", "t_written", "bytes"] };
