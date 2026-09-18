p='/home/claude-code/deepstrix/crates/v4flash-kernels/src/het/forward_prefill.rs'
s=open(p).read()
def rep(old,new,cnt=1):
    global s
    assert s.count(old)==cnt,(s.count(old),old[:90])
    s=s.replace(old,new,cnt)

# lane cut record
rep("""        let b_b = b - b_a;
        // Record where the cut fell. `mtp_src` is captured per lane and indexed
        // lane-locally, so anything selecting a GLOBAL batch row needs this.
        bd_a.mtp_lane_cut = b_a;""",
"""        // Lane A takes rows [0, b_a) and lane B rows [b_a, b); a cut outside the
        // batch would both underflow `b - b_a` and mis-map every global batch
        // row onto a lane below.
        assert!(
            b_a <= b,
            "prefill lane split: lane A cut {b_a} exceeds the batch of {b} rows"
        );
        let b_b = b - b_a;
        // Record where the cut fell. `mtp_src` is captured per lane and indexed
        // lane-locally, so anything selecting a GLOBAL batch row needs this.
        bd_a.mtp_lane_cut = b_a;""")

# mtp capture site
rep("""                let n = tokens.len().min(rows);
                if n > 0 {
                    let de = &self.dgpu;""",
"""                let n = tokens.len().min(rows);
                if n > 0 {
                    // `src` below slices `bd.residual` at `skip * nhc * ne` for
                    // `n * nhc * ne` floats, i.e. the WHOLE lane batch has to fit
                    // the lane scratch; `dst` / `wsrc` are sized for
                    // MTP_CAP_ROWS. Either bound broken reads (or writes) past
                    // the buffer into whatever scratch follows it.
                    assert!(
                        tokens.len() <= bd.rows,
                        "mtp capture L{layer}: lane batch of {} rows exceeds lane scratch \\
                         capacity {} rows; the `residual` slice would run past the buffer",
                        tokens.len(),
                        bd.rows
                    );
                    assert!(
                        n <= super::batch_scratch::MTP_CAP_ROWS,
                        "mtp capture L{layer}: capturing {n} rows > MTP_CAP_ROWS {}; mtp_src and \\
                         mtp_hc_mean are only sized for MTP_CAP_ROWS",
                        super::batch_scratch::MTP_CAP_ROWS
                    );
                    let de = &self.dgpu;""")
open(p,'w').write(s)
print("ok")
