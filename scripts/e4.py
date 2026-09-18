p='/home/claude-code/deepstrix/crates/deepstrix-server/src/engine_worker.rs'
s=open(p).read()
def rep(old,new,cnt=1):
    global s
    assert s.count(old)==cnt,(s.count(old),old[:100])
    s=s.replace(old,new,cnt)

# 1. logits shape vs batch rows
rep("""            // Accept the longest prefix whose argmax matches the draft.
            let nv = v4flash_kernels::config::N_VOCAB as usize;""",
"""            // Accept the longest prefix whose argmax matches the draft.
            let nv = v4flash_kernels::config::N_VOCAB as usize;
            // `row_sample(j)` slices `logits[j*nv .. (j+1)*nv]` for every row up
            // to and including row `k`, so the verify must have returned ONE
            // FULL ROW PER INPUT TOKEN. A short buffer (e.g. a last_only driver)
            // would silently read one row's distribution as another's.
            assert_eq!(
                logits.len(),
                toks.len() * nv,
                "dspark accept: verify returned {} logits for {} rows ({nv} per row expected); \
                 per-row indexing would read the wrong row",
                logits.len(),
                toks.len()
            );""")

# 2. lane mapping for mtp_src
rep("""            let cut = state.bd_a.mtp_lane_cut;
            let mut whole = vec![0.0f32; nsrc * cap * ne];
            let mut whole_b = vec![0.0f32; nsrc * cap * ne];
            state.bd_a.mtp_src.copy_to_host(&mut whole)?;
            if cut < toks.len() {
                state.bd_b.mtp_src.copy_to_host(&mut whole_b)?;
            }
            let lane_row = |r: usize| -> (&Vec<f32>, usize) {
                if r < cut { (&whole, r) } else { (&whole_b, r - cut) }
            };""",
"""            let cut = state.bd_a.mtp_lane_cut;
            // The cut is the ONLY thing tying a global batch row to the lane
            // that captured its residual, and each lane's `mtp_src` is indexed
            // from 0. If the cut does not cover the rows each lane actually
            // captured, `main_hidden` is read out of the wrong lane's buffer and
            // is silently STALE -- the failure that looked like "the drafter is
            // degenerate" until it was root-caused to this mapping.
            assert!(
                state.bd_a.mtp_captured >= cut.min(toks.len()),
                "dspark accept: lane A captured {} mtp_src rows, but the recorded lane cut claims \
                 rows [0,{}) of this {}-row verify came from lane A",
                state.bd_a.mtp_captured,
                cut.min(toks.len()),
                toks.len()
            );
            assert!(
                cut >= toks.len() || state.bd_b.mtp_captured >= toks.len() - cut,
                "dspark accept: lane B captured {} mtp_src rows, but the recorded lane cut claims \
                 rows [{cut},{}) of this verify came from lane B",
                state.bd_b.mtp_captured,
                toks.len()
            );
            let mut whole = vec![0.0f32; nsrc * cap * ne];
            let mut whole_b = vec![0.0f32; nsrc * cap * ne];
            state.bd_a.mtp_src.copy_to_host(&mut whole)?;
            if cut < toks.len() {
                state.bd_b.mtp_src.copy_to_host(&mut whole_b)?;
            }
            let lane_row = |r: usize| -> (&Vec<f32>, usize) {
                let (buf, lr) = if r < cut { (&whole, r) } else { (&whole_b, r - cut) };
                // LANE-LOCAL row, never a global one: `mtp_src` holds at most
                // MTP_CAP_ROWS rows per lane.
                assert!(
                    lr < cap,
                    "dspark accept: lane-local mtp_src row {lr} (global row {r}, lane cut {cut}) \
                     >= MTP_CAP_ROWS {cap}"
                );
                (buf, lr)
            };""")

# 3. keep bound
rep("""            let keep = (n + 1) as u32; // `next` plus the n accepted drafts""",
"""            let keep = (n + 1) as u32; // `next` plus the n accepted drafts
            // The accepted prefix can never be longer than the rows the verify
            // actually appended to KV: the batch is `next` + k drafts, so
            // `keep <= toks.len()`. Claiming more rows than were appended
            // misaligns the cache against the emitted stream permanently.
            assert!(
                n <= k && (keep as usize) <= toks.len(),
                "dspark accept: keeping {keep} rows (n={n} of k={k}) from a {}-row verify",
                toks.len()
            );""")

# 4. main_hidden shape
rep("""                m.main_hidden.extend_from_slice(&buf[o..o + ne]);
            }
            m.confirmed.clear();""",
"""                m.main_hidden.extend_from_slice(&buf[o..o + ne]);
            }
            // The drafter's entry projection consumes exactly one N_EMBD row per
            // MTP source layer; a short/long vector means the capture layout and
            // the reader disagree.
            assert_eq!(
                m.main_hidden.len(),
                nsrc * ne,
                "dspark accept: main_hidden has {} floats, expected {} ({nsrc} MTP source layers \
                 x {ne} N_EMBD)",
                m.main_hidden.len(),
                nsrc * ne
            );
            m.confirmed.clear();""")

# 5. confirmed/ingested established
rep("""            m.ingested = n + 1;
            let head = if n < k { corrected } else { row_sample(k, &mut rng) };""",
"""            m.ingested = n + 1;
            // The step yields the n confirmed drafts and then the head.
            // `ingested` counts the rows this verify LEFT IN KV that the decode
            // loop must not forward again: `next` (row 0) plus those n drafts.
            assert_eq!(
                m.confirmed.len() + 1,
                m.ingested,
                "dspark accept: {} confirmed drafts queued but ingested={} (must be \
                 confirmed + 1, the head row `next`)",
                m.confirmed.len(),
                m.ingested
            );
            let head = if n < k { corrected } else { row_sample(k, &mut rng) };""")

# 6. the per-step lockstep of confirmed vs ingested at the consume site
rep("""        let spec_next: Option<i32> = state.mtp.as_mut().and_then(|m| {
            m.confirmed
                .pop_front()""",
"""        let spec_next: Option<i32> = state.mtp.as_mut().and_then(|m| {
            // By this point in the iteration `ingested` has already been
            // decremented for the token just emitted, so the queue of confirmed
            // drafts and the count of verify rows still sitting in KV must be
            // EQUAL. Drift either emits a confirmed token with no KV row behind
            // it, or skips a forward for a row that was never appended -- both
            // desynchronise the cache from the emitted stream silently.
            // Holds trivially (0 == 0) when DSpark accept is off.
            assert_eq!(
                m.confirmed.len(),
                m.ingested,
                "dspark accept: {} confirmed drafts pending but {} verify rows still marked \
                 ingested in KV",
                m.confirmed.len(),
                m.ingested
            );
            m.confirmed
                .pop_front()""")

# 7. seed_mtp_ring: captured rows must fit the capture buffer
rep("""    let mut whole = vec![0.0f32; nsrc * cap * ne];
    if from_b {""",
"""    // `whole` is read at `sl * cap * ne + r * ne` for r in [0, n), so the
    // capture count the driver recorded must fit the buffer it wrote into.
    assert!(
        n <= cap,
        "dspark seed: prefill recorded {n} captured mtp_src rows but the buffer holds {cap}"
    );
    let mut whole = vec![0.0f32; nsrc * cap * ne];
    if from_b {""")
open(p,'w').write(s)
print("ok")
