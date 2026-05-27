//! M50 Phase 3 expert stats — user-requested instrumentation.
//!
//! Collects two complementary views of routed-MoE expert selection during
//! batched prefill:
//!
//! 1. **Per-batch reuse** — within one chunk of B tokens (= one
//!    `forward_layer_batch_v2` call), how concentrated is the top-6
//!    selection? Compares actual unique-experts-touched vs. the
//!    uniform-distribution expectation, and reports per-expert reuse
//!    (max / mean) for each batch we see.
//!
//! 2. **Per-layer distribution** — across the whole prefill (every chunk,
//!    every token at this layer), how many times was each expert picked?
//!    Reveals skew that uniform-distribution math hides.
//!
//! Wiring: pass `Some(&mut PrefillStats)` to `forward_prompt_batch_v2` or
//! `forward_prefill`. After Stage 9 (router topk) in each layer, the
//! prefill driver copies `bd.d_selected[0..B*N_USED]` to host and calls
//! `stats.record_batch(layer, &d_sel_host, B)`. The copy is sync and
//! fences the device — fine for one-off measurement runs, NOT for
//! production prefill (off by default — `None`).
//!
//! Why this matters for perf: our current iq2/q2_k batched-MoE kernels
//! group by **token** (each WG handles one (row_block, slot, token) triple
//! and re-reads the picked expert's weight rows independently). If actual
//! expert reuse is high — meaning many tokens picked the same expert — a
//! token-by-expert grouping (SGLang-style) would amortize weight reads and
//! likely unlock WMMA on the GEMM portion. The uniform-distribution model
//! predicts ~1.9× reuse at B=64 and ~6× at B=256; this instrumentation
//! tells us if reality is more or less skewed.

use std::cmp::Reverse;

/// Cumulative stats across an entire prefill run.
#[derive(Clone)]
pub struct PrefillStats {
    pub n_used: u32,
    pub n_expert: u32,
    /// Per-layer counters (length = N_LAYER).
    pub layers: Vec<LayerStats>,
}

#[derive(Clone)]
pub struct LayerStats {
    /// `[n_expert]` — cumulative pick count across every (batch, token, slot)
    /// at this layer in this prefill run.
    pub pick_counts: Vec<u32>,
    /// Total picks recorded at this layer = `n_used × sum_chunks B`.
    pub total_picks: u32,
    /// Number of tokens we've recorded at this layer (= sum of B across chunks).
    pub total_tokens: u32,
    /// Per-chunk reuse snapshots — one entry per call to `record_batch`.
    pub per_chunk: Vec<PerChunkReuse>,
}

#[derive(Clone, Copy)]
pub struct PerChunkReuse {
    pub batch_size: u32,
    pub total_picks: u32,
    /// Number of distinct experts that appeared in this chunk's selection.
    pub unique_experts: u32,
    /// Max times any single expert was picked in this chunk.
    pub max_reuse: u32,
    /// Mean reuse = total_picks / unique_experts.
    pub mean_reuse: f32,
}

impl PrefillStats {
    pub fn new(n_layer: u32, n_used: u32, n_expert: u32) -> Self {
        let layers = (0..n_layer)
            .map(|_| LayerStats {
                pick_counts: vec![0u32; n_expert as usize],
                total_picks: 0,
                total_tokens: 0,
                per_chunk: Vec::new(),
            })
            .collect();
        Self {
            n_used,
            n_expert,
            layers,
        }
    }

    /// Record one batch of selections at the given layer.
    /// `d_selected_host` is `[batch_size × n_used]` i32 — what router_topk
    /// (or hash router) picked for this chunk's tokens at this layer.
    pub fn record_batch(&mut self, layer: usize, d_selected_host: &[i32], batch_size: u32) {
        let n_used = self.n_used as usize;
        let n_expert = self.n_expert as usize;
        assert_eq!(d_selected_host.len(), (batch_size as usize) * n_used);
        let layer_stats = &mut self.layers[layer];

        // Per-chunk pick count to derive max/unique without an alloc per call.
        // 256 experts fits comfortably; use a stack-ish buffer.
        let mut chunk_pick: Vec<u32> = vec![0u32; n_expert];
        for &eid_signed in d_selected_host {
            let e = eid_signed as usize;
            if e >= n_expert {
                continue; // defensive — shouldn't happen
            }
            chunk_pick[e] += 1;
            layer_stats.pick_counts[e] += 1;
        }
        let total = (batch_size as u32) * self.n_used;
        let unique = chunk_pick.iter().filter(|&&c| c > 0).count() as u32;
        let max_reuse = chunk_pick.iter().copied().max().unwrap_or(0);
        let mean_reuse = if unique > 0 {
            total as f32 / unique as f32
        } else {
            0.0
        };
        layer_stats.per_chunk.push(PerChunkReuse {
            batch_size,
            total_picks: total,
            unique_experts: unique,
            max_reuse,
            mean_reuse,
        });
        layer_stats.total_picks += total;
        layer_stats.total_tokens += batch_size;
    }

    /// Theoretical uniform-distribution expected unique experts at batch B:
    /// `n_expert × (1 - (1 - n_used/n_expert)^B)`.
    pub fn expected_unique_uniform(&self, batch_size: u32) -> f32 {
        let p_picked = self.n_used as f32 / self.n_expert as f32;
        let p_not_picked_b = (1.0_f32 - p_picked).powi(batch_size as i32);
        (self.n_expert as f32) * (1.0 - p_not_picked_b)
    }

    /// Write per-expert pick_counts (one line per expert) for a chosen layer to a file.
    /// Format: JSON {"layer": L, "n_used": ..., "n_expert": ..., "total_tokens": T,
    ///                "picks": [count_for_expert_0, count_for_expert_1, ...]}.
    pub fn dump_layer_picks(&self, layer: usize, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let ls = &self.layers[layer];
        let mut f = std::fs::File::create(path)?;
        write!(f, "{{\"layer\": {}, \"n_used\": {}, \"n_expert\": {}, \"total_tokens\": {}, \"picks\": [",
               layer, self.n_used, self.n_expert, ls.total_tokens)?;
        for (i, &c) in ls.pick_counts.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", c)?;
        }
        write!(f, "]}}")?;
        Ok(())
    }

    /// Print a human-readable summary.
    pub fn print_summary(&self) {
        eprintln!("\n=== PrefillStats (n_layer={}, n_used={}, n_expert={}) ===",
            self.layers.len(), self.n_used, self.n_expert);

        // ---- Per-chunk reuse: aggregate across layers ----
        let mut by_chunk_idx: Vec<Vec<&PerChunkReuse>> = Vec::new();
        for layer_stats in &self.layers {
            for (i, pc) in layer_stats.per_chunk.iter().enumerate() {
                if i >= by_chunk_idx.len() {
                    by_chunk_idx.push(Vec::new());
                }
                by_chunk_idx[i].push(pc);
            }
        }
        eprintln!("\nPER-CHUNK REUSE (averaged across layers):");
        eprintln!("  chunk_idx  B  picks  unique(actual)  unique(uniform-model)  mean_reuse  max_reuse");
        for (i, chunks_at_i) in by_chunk_idx.iter().enumerate() {
            if chunks_at_i.is_empty() { continue; }
            let n = chunks_at_i.len() as f32;
            let batch_size = chunks_at_i[0].batch_size;
            let avg_unique: f32 = chunks_at_i.iter().map(|c| c.unique_experts as f32).sum::<f32>() / n;
            let avg_max: f32 = chunks_at_i.iter().map(|c| c.max_reuse as f32).sum::<f32>() / n;
            let avg_mean_reuse: f32 = chunks_at_i.iter().map(|c| c.mean_reuse).sum::<f32>() / n;
            let expected = self.expected_unique_uniform(batch_size);
            let total_picks = chunks_at_i[0].total_picks;
            eprintln!(
                "  {:>9}  {:>3} {:>5}   {:>7.1} ({:>5.1}%)         {:>7.1} ({:>5.1}%)         {:>5.2}      {:>5.1}",
                i, batch_size, total_picks,
                avg_unique, 100.0 * avg_unique / self.n_expert as f32,
                expected,   100.0 * expected   / self.n_expert as f32,
                avg_mean_reuse, avg_max
            );
        }

        // ---- Per-layer distribution skew ----
        eprintln!("\nPER-LAYER DISTRIBUTION SKEW (across all chunks of the prefill):");
        eprintln!("  layer  total_picks  top-10 share  Gini  most-picked-expert  count");
        for (layer, ls) in self.layers.iter().enumerate() {
            if ls.total_picks == 0 { continue; }
            let total = ls.total_picks as f32;
            let mut sorted_desc: Vec<u32> = ls.pick_counts.clone();
            sorted_desc.sort_by_key(|&c| Reverse(c));
            let top10_sum: u32 = sorted_desc.iter().take(10).sum();
            let top10_share = top10_sum as f32 / total;
            let top_expert_idx = ls.pick_counts.iter().enumerate()
                .max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
            let top_count = ls.pick_counts[top_expert_idx];
            // Gini coefficient: 0 = perfect equality, 1 = max inequality.
            // Standard form expects ascending sort.
            let mut sorted_asc: Vec<u32> = ls.pick_counts.clone();
            sorted_asc.sort();
            let n = self.n_expert as f32;
            let mean = total / n;
            let mut wd: f32 = 0.0;
            for (i, &c) in sorted_asc.iter().enumerate() {
                let rank = (i + 1) as f32;
                wd += (2.0 * rank - n - 1.0) * c as f32;
            }
            let gini = if mean > 0.0 { wd / (n * n * mean) } else { 0.0 };
            eprintln!(
                "  {:>5}  {:>11}  {:>11.1}%  {:>4.2}  {:>17}  {:>5}",
                layer, ls.total_picks, 100.0 * top10_share, gini,
                top_expert_idx, top_count
            );
        }
    }
}
