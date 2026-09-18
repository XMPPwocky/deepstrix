import sys
def patch(path, edits):
    src = open(path).read()
    for old, new in edits:
        n = src.count(old)
        assert n == 1, f"{path}: anchor count {n} != 1 for:\n{old[:120]}"
        src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print(f"patched {path}: {len(edits)} edits")

# ---------------- trace.rs ----------------
T = "crates/v4flash-kernels/src/het/trace.rs"
patch(T, [
(
"""    pub static REMOTE_RTT_NS: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {""",
"""    pub static REMOTE_RTT_NS: AtomicU64 = AtomicU64::new(0);

    /// Host phases OUTSIDE `forward_token_impl`'s `token_start..sync` bracket
    /// but ON the decode loop's critical path (engine_worker.rs
    /// `finish_decode` / `forward_one!`): the caller adds to these between one
    /// forward's return and the next forward's entry. Deliberately NOT cleared
    /// by `reset()` — that runs at forward entry, i.e. AFTER the caller has
    /// already added to them — the token summary drains them with `take()`,
    /// so each `het.token.summary` line reports the glue that ran since the
    /// previous line (attributed to the token whose forward follows it).
    ///
    /// Why: the lever-1 A/B logs (2026-09-14, l1_1.log) put the decode loop's
    /// wall at 60.8 ms/token against `total_us` 57.6 ms — 3.3 ms/token that no
    /// counter covered. (The rest of that A/B's "88 ms/token" was the 36-token
    /// prompt's 6.9 s CED-replay prefill amortised over 256 tokens by a
    /// curl-wall harness — see `decode.loop.summary` / `request.summary`.)
    pub static CALLER_ENGRAM_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLER_EMBED_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLER_SAMPLE_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLER_STREAM_NS: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {"""
),
(
"""    pub fn get(c: &AtomicU64) -> u64 {
        c.load(Relaxed)
    }
}
""",
"""    pub fn get(c: &AtomicU64) -> u64 {
        c.load(Relaxed)
    }
    /// Read-and-clear, for the `CALLER_*` counters (see above).
    pub fn take(c: &AtomicU64) -> u64 {
        c.swap(0, Relaxed)
    }
}

/// Process-wide monotonic clock (ns since first use) for gaps that span two
/// calls, e.g. `TokenTiming::gap_us` between consecutive `forward_token_impl`s.
pub fn epoch_ns() -> u64 {
    static EPOCH: std::sync::LazyLock<std::time::Instant> =
        std::sync::LazyLock::new(std::time::Instant::now);
    EPOCH.elapsed().as_nanos() as u64
}
"""
),
(
"""    /// against `total_us` to see whether the round trip is hidden.
    pub remote_rtt_us: u64,
}
""",
"""    /// against `total_us` to see whether the round trip is hidden.
    pub remote_rtt_us: u64,
    /// `stage_engram_rows` H2D inside the bracket (V4.1 Engram layers 1, 14).
    pub engram_stage_us: u64,
    /// Bracket edges inside `forward_token_impl`, NOT part of `total_us`:
    /// `pre_us` = fn entry -> `token_start` (residual H2D + per-token scalar
    /// writes); `post_us` = final sync -> summary (perfetto export + event
    /// harvest).
    pub pre_us: u64,
    pub post_us: u64,
    /// Wall from the previous `forward_token_impl`'s return to this one's
    /// entry: EVERYTHING the caller did between two tokens. On a request's
    /// first token this spans the whole prefill — exclude it from averages.
    pub gap_us: u64,
    /// Caller phases drained from `phase::CALLER_*` (a subset of `gap_us`):
    /// Engram SSD gather (`EngramCtx::rows_for`), embedding lookup, sampler
    /// (kernels + sync + 4 B D2H), detokenise + chunk send.
    /// `gap_us - (engram_us + embed_us + sample_us + stream_us)` is the
    /// unattributed caller glue.
    pub engram_us: u64,
    pub embed_us: u64,
    pub sample_us: u64,
    pub stream_us: u64,
}
"""
),
(
"""            pager_misses = self.pager_misses,
            "het.token.summary"
        );""",
"""            pager_misses = self.pager_misses,
            engram_stage_us = self.engram_stage_us,
            pre_us = self.pre_us,
            post_us = self.post_us,
            gap_us = self.gap_us,
            engram_us = self.engram_us,
            embed_us = self.embed_us,
            sample_us = self.sample_us,
            stream_us = self.stream_us,
            "het.token.summary"
        );"""
),
])

# ---------------- engine.rs ----------------
E = "crates/v4flash-kernels/src/het/engine.rs"
patch(E, [
(
"""    /// Diagnostic: time the host spent inside the final `synchronize()`.
    pub last_sync_us: std::sync::atomic::AtomicU64,
""",
"""    /// Diagnostic: time the host spent inside the final `synchronize()`.
    pub last_sync_us: std::sync::atomic::AtomicU64,
    /// Diagnostic: `trace::epoch_ns()` at which the last `forward_token_impl`
    /// returned; the next call reports the elapsed gap as `gap_us`. 0 = none yet.
    pub last_token_end_ns: std::sync::atomic::AtomicU64,
"""
),
(
"""            last_sync_us: std::sync::atomic::AtomicU64::new(0),
""",
"""            last_sync_us: std::sync::atomic::AtomicU64::new(0),
            last_token_end_ns: std::sync::atomic::AtomicU64::new(0),
"""
),
(
"""        let _token_span = debug_span!("het.token", pos, token_id).entered();

        // Reset event pools for this token.""",
"""        // Caller-side gap since the previous token's return (see
        // `TokenTiming::gap_us`), taken BEFORE anything else in this call.
        let fn_entry = std::time::Instant::now();
        let gap_us = {
            let prev = self.last_token_end_ns.load(std::sync::atomic::Ordering::Relaxed);
            if prev == 0 { 0 } else { super::trace::epoch_ns().saturating_sub(prev) / 1000 }
        };
        let _token_span = debug_span!("het.token", pos, token_id).entered();

        // Reset event pools for this token."""
),
(
"""        let token_start = std::time::Instant::now();
        let dump_subtensor_layers: Vec<usize> = subtensor_dump_spec()""",
"""        let token_start = std::time::Instant::now();
        let pre_us = token_start.duration_since(fn_entry).as_micros() as u64;
        let dump_subtensor_layers: Vec<usize> = subtensor_dump_spec()"""
),
(
"""                match rows {
                    Some(r) => self.stage_engram_rows(dgpu_scratch, r)?,
                    None => {""",
"""                match rows {
                    Some(r) => {
                        let t = std::time::Instant::now();
                        self.stage_engram_rows(dgpu_scratch, r)?;
                        super::trace::phase::add(
                            &super::trace::phase::ENGRAM_STAGE_NS,
                            t.elapsed().as_nanos() as u64,
                        );
                    }
                    None => {"""
),
(
"""        let summary = super::trace::TokenTiming {
            token_pos: pos,
            total_us: token_elapsed_us,""",
"""        // Everything between the bracket's closing sync and this line
        // (perfetto export, event harvest) — outside `total_us` by construction.
        let post_us = (token_start.elapsed().as_micros() as u64).saturating_sub(token_elapsed_us);
        let summary = super::trace::TokenTiming {
            token_pos: pos,
            total_us: token_elapsed_us,"""
),
(
"""            remote_rtt_us: super::trace::phase::get(&super::trace::phase::REMOTE_RTT_NS) / 1000,
        };
        summary.emit();""",
"""            remote_rtt_us: super::trace::phase::get(&super::trace::phase::REMOTE_RTT_NS) / 1000,
            engram_stage_us: super::trace::phase::get(&super::trace::phase::ENGRAM_STAGE_NS) / 1000,
            pre_us,
            post_us,
            gap_us,
            // Drained (read-and-clear): these were added by the caller between
            // the previous forward's return and this one's entry.
            engram_us: super::trace::phase::take(&super::trace::phase::CALLER_ENGRAM_NS) / 1000,
            embed_us: super::trace::phase::take(&super::trace::phase::CALLER_EMBED_NS) / 1000,
            sample_us: super::trace::phase::take(&super::trace::phase::CALLER_SAMPLE_NS) / 1000,
            stream_us: super::trace::phase::take(&super::trace::phase::CALLER_STREAM_NS) / 1000,
        };
        summary.emit();"""
),
(
"""                    "het.stage"
                );
            }
        }
        Ok(())
    }

    /// Build a het engine over (dgpu, igpu).""",
"""                    "het.stage"
                );
            }
        }
        // Stamp the return so the next call can report the caller-side gap.
        self.last_token_end_ns
            .store(super::trace::epoch_ns(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Build a het engine over (dgpu, igpu)."""
),
])

# ---------------- engine_worker.rs ----------------
W = "crates/deepstrix-server/src/engine_worker.rs"
patch(W, [
(
"""            let engram_rows = match $state.engram.as_mut() {
                Some(ec) => Some(ec.rows_for(pg.raw(), $tok, $pos)?),
                None => None,
            };""",
"""            // Timed into `phase::CALLER_ENGRAM_NS` -> `engram_us` on the next
            // `het.token.summary`: 2 layers x 24 spawned threads x 2 preads
            // against a 98 GB NVMe table, all on this thread's critical path.
            let engram_rows = match $state.engram.as_mut() {
                Some(ec) => {
                    let t = std::time::Instant::now();
                    let rows = ec.rows_for(pg.raw(), $tok, $pos)?;
                    v4flash_kernels::het::trace::phase::add(
                        &v4flash_kernels::het::trace::phase::CALLER_ENGRAM_NS,
                        t.elapsed().as_nanos() as u64,
                    );
                    Some(rows)
                }
                None => None,
            };"""
),
(
"""    let mut pos = start_pos;
""",
"""    let mut pos = start_pos;
    // Decode-loop wall, reported as `decode.loop.summary` at the end. Its
    // ms/token is what a harness must compare against `het.token.summary`;
    // curl-wall / completion_tokens ALSO carries the prefill and snapshot save.
    let loop_t0 = std::time::Instant::now();
"""
),
(
"""    let mut rng = SamplerRng::new(req.seed);
    let mut next = state
        .engine
        .sample_next(&mut state.dgpu_scratch, sample_mode, rng.next_f32())?;
    let mut completion_tokens: u32 = 1;""",
"""    let mut rng = SamplerRng::new(req.seed);
    let t_sample = std::time::Instant::now();
    let mut next = state
        .engine
        .sample_next(&mut state.dgpu_scratch, sample_mode, rng.next_f32())?;
    v4flash_kernels::het::trace::phase::add(
        &v4flash_kernels::het::trace::phase::CALLER_SAMPLE_NS,
        t_sample.elapsed().as_nanos() as u64,
    );
    let mut completion_tokens: u32 = 1;"""
),
(
"""            let raw = gpt2_decode_token(bytes, &state.byte_decoder);
            // Always emit, even for empty raw""",
"""            let t_stream = std::time::Instant::now();
            let raw = gpt2_decode_token(bytes, &state.byte_decoder);
            // Always emit, even for empty raw"""
),
(
"""                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        send_failed = true;
                        break;
                    }
                }
            }
            if send_failed {""",
"""                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        send_failed = true;
                        break;
                    }
                }
            }
            // Detokenise + chunk hand-off -> `stream_us` on the next summary.
            v4flash_kernels::het::trace::phase::add(
                &v4flash_kernels::het::trace::phase::CALLER_STREAM_NS,
                t_stream.elapsed().as_nanos() as u64,
            );
            if send_failed {"""
),
(
"""        embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, next, &mut residual);
        forward_one!(state, residual, pos, next)?;""",
"""        let t_embed = std::time::Instant::now();
        embed_lookup(&state.token_embd_bytes, state.token_embd_dtype, next, &mut residual);
        v4flash_kernels::het::trace::phase::add(
            &v4flash_kernels::het::trace::phase::CALLER_EMBED_NS,
            t_embed.elapsed().as_nanos() as u64,
        );
        forward_one!(state, residual, pos, next)?;"""
),
(
"""        next = state
            .engine
            .sample_next(&mut state.dgpu_scratch, sample_mode, rng.next_f32())?;
        completion_tokens += 1;
    };""",
"""        let t_sample = std::time::Instant::now();
        next = state
            .engine
            .sample_next(&mut state.dgpu_scratch, sample_mode, rng.next_f32())?;
        v4flash_kernels::het::trace::phase::add(
            &v4flash_kernels::het::trace::phase::CALLER_SAMPLE_NS,
            t_sample.elapsed().as_nanos() as u64,
        );
        completion_tokens += 1;
    };
    // The loop's own wall. Measured 2026-09-14 (l1_1.log): 60.8 ms/token here
    // vs 88 ms/token by curl-wall/256 — the difference was the 6.9 s prefill of
    // a 36-token prompt (CED replay paging ~1500 decoder experts on box 1).
    {
        let wall = loop_t0.elapsed();
        tracing::info!(
            completion_tokens,
            loop_ms = wall.as_millis() as u64,
            ms_per_tok = format!("{:.2}", wall.as_secs_f64() * 1e3 / completion_tokens.max(1) as f64),
            finish = ?finish,
            "decode.loop.summary"
        );
    }"""
),
(
"""                if let Err(e) = handle_generate_stream(&mut state, req, session_id, cancel, &tx) {
                    let _ = tx.blocking_send(WorkerEvent::Error(format!("{e:#}")));
                }""",
"""                let t_req = std::time::Instant::now();
                if let Err(e) = handle_generate_stream(&mut state, req, session_id, cancel, &tx) {
                    let _ = tx.blocking_send(WorkerEvent::Error(format!("{e:#}")));
                }
                // Engine-side end-to-end wall for the request (prefill + decode
                // loop + snapshot saves). `e2e_ms - decode.loop.summary.loop_ms`
                // is the per-request fixed cost a curl-wall harness amortises
                // over completion_tokens.
                tracing::info!(
                    e2e_ms = t_req.elapsed().as_millis() as u64,
                    "request.summary"
                );"""
),
])
