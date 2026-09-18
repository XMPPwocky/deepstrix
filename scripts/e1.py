import io,sys
p='/home/claude-code/deepstrix/crates/v4flash-kernels/src/het/state.rs'
s=open(p).read()
def rep(old,new,cnt=1):
    global s
    assert s.count(old)==cnt, (s.count(old), old[:80])
    s=s.replace(old,new,cnt)

# 1. rollback_kv: per_layer_comp must be layer-indexed too
rep("""        if !mark.slid {
            for (i, (_, raw_off)) in mark.per_layer.iter().copied().enumerate() {""",
"""        // `per_layer_comp` is indexed BY LAYER below (`self.layers[i]`), so a
        // mark whose compressor snapshot has a different length than the layer
        // list would restore the wrong layer's compressor. The empty vec is the
        // documented "mark taken before this field existed" case.
        if !mark.per_layer_comp.is_empty() && mark.per_layer_comp.len() != self.layers.len() {
            return Err(eyre!(
                "rollback_kv: mark's compressor snapshot covers {} layers, state has {}; \
                 per_layer_comp is indexed by layer id",
                mark.per_layer_comp.len(),
                self.layers.len()
            ));
        }
        if !mark.slid {
            for (i, (_, raw_off)) in mark.per_layer.iter().copied().enumerate() {""")

# 2. advanced_by: per_layer / per_layer_comp are both layer-indexed
rep("""        let per_layer = self
            .per_layer
            .iter()
            .map(|&(nr, off)| {
                let end = off + nr + keep; // append pointer after the kept rows
                let new_nr = end.min(SWA_WINDOW);
                (new_nr, end - new_nr)
            })
            .collect();""",
"""        // Both vectors are indexed BY LAYER below (`COMPRESS_RATIOS[layer]`
        // reads the second one by the first one's index space), so they must
        // describe the same layer list.
        assert_eq!(
            self.per_layer.len(),
            self.per_layer_comp.len(),
            "KvMark::advanced_by: per_layer has {} entries but per_layer_comp has {}; both are \\
             indexed by layer id",
            self.per_layer.len(),
            self.per_layer_comp.len()
        );
        let per_layer = self
            .per_layer
            .iter()
            .map(|&(nr, off)| {
                let end = off + nr + keep; // append pointer after the kept rows
                // The kept rows were physically appended at
                // `[off+nr, off+nr+keep)` of the oversized cache, so the new
                // append pointer must still lie inside it. Past the end means
                // the verify already wrote out of bounds.
                assert!(
                    (end as usize) <= KV_CACHE_ROWS,
                    "KvMark::advanced_by: append pointer {end} (raw_off {off} + n_raw {nr} + \\
                     keep {keep}) exceeds raw KV capacity {KV_CACHE_ROWS}"
                );
                let new_nr = end.min(SWA_WINDOW);
                (new_nr, end - new_nr)
            })
            .collect();""")

# 3. bump(): capture the pre-value and assert the lockstep postcondition
rep("""                        let before = m.n_comp;
                        m.n_comp = n.max(before).min(before + keep);""",
"""                        let before = m.n_comp;
                        let before_index = m.n_index_comp;
                        m.n_comp = n.max(before).min(before + keep);""")

rep("""                            (m.n_index_comp + delta).min(m.n_comp)
                        };
                    }
                    m
                };""",
"""                            (m.n_index_comp + delta).min(m.n_comp)
                        };
                        // `index_k` rows are indexed exactly like `comp_kv`
                        // rows, so the two counters move in LOCKSTEP: the
                        // indexer can never claim more rows than the main store
                        // holds, and a partial rollback that ADVANCES the main
                        // store must never move the indexer BACKWARDS (the
                        // `.min(m.n_comp)` above can clamp it if the indexer was
                        // ever ahead of the main store).
                        assert!(
                            m.n_index_comp <= m.n_comp,
                            "KvMark::advanced_by: n_index_comp {} > n_comp {} after keep={keep} \\
                             (before: n_comp {before}, n_index_comp {before_index})",
                            m.n_index_comp,
                            m.n_comp
                        );
                        assert!(
                            m.n_index_comp >= before_index,
                            "KvMark::advanced_by: n_index_comp went BACKWARDS {before_index} -> \\
                             {} while n_comp went {before} -> {} (keep={keep})",
                            m.n_index_comp,
                            m.n_comp
                        );
                    }
                    m
                };""")
open(p,'w').write(s)
print("ok")
