#!/usr/bin/env python3
"""M7 CED prefill: apply the forward_prefill.rs / config.rs edits (exact-string, asserted)."""
import sys

def rep(src, old, new, count=1, name=""):
    n = src.count(old)
    if n != count:
        sys.exit(f"anchor {name!r}: expected {count} occurrence(s), found {n}")
    return src.replace(old, new)

# ---------------------------------------------------------------- config.rs
P = "crates/v4flash-kernels/src/config.rs"
s = open(P).read()
s = rep(s, '''#[cfg(feature = "v41")]
pub const ENGRAM_LAYERS: &[i32] = &[1, 14];
''', '''#[cfg(feature = "v41")]
pub const ENGRAM_LAYERS: &[i32] = &[1, 14];
/// V4.1 Causal Encoder-Decoder split (tech report §2.2): layers below are the
/// causal encoder; this layer is the decoder's Full-mode layer whose global KV
/// is projected from the final encoder hidden state. CED prefill runs the
/// encoder over every prompt token and the decoder over the last `SWA_WINDOW`
/// only (`het::forward_prefill::CedMode`, docs/v41/ENGINE_PORT.md "M7 CED").
#[cfg(feature = "v41")]
pub const CED_DECODER_START: usize = 20;
#[cfg(not(feature = "v41"))]
pub const CED_DECODER_START: usize = N_LAYER as usize;
''', name="config engram")
open(P, "w").write(s)

# ---------------------------------------------------------- forward_prefill.rs
P = "crates/v4flash-kernels/src/het/forward_prefill.rs"
s = open(P).read()

# 1. enum + helpers before chunk_visibility
s = rep(s, '''fn chunk_visibility(pos0: u32, b: usize, spans: &[ImageSpan]) -> eyre::Result<Option<Vec<(u32, u32)>>> {''',
'''/// M7 CED: mode of one batched layer call under V4.1 Causal Encoder-Decoder
/// prefill (tech report §2.2 / §3.2.2, docs/v41/ENGINE_PORT.md "M7 CED").
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CedMode {
    /// The exact all-stage layer (what the reference forward runs everywhere).
    Exact,
    /// The decoder's Full-mode layer (`CED_DECODER_START`) over encoder-only
    /// prompt rows: mHC pre-mix + attn norm + the global-KV projection into
    /// the shared store, nothing else (no window KV, attention, MoE, and no
    /// carry update — the residual/carry entering the layer are left as-is).
    KvSourceOnly,
    /// The same layer over the bounded-replay segment: the store already holds
    /// these positions (written by `KvSourceOnly`), so the projection is
    /// skipped and the causal comp counts are positional (like a reuse
    /// layer); every other stage runs.
    Replay,
}

/// `V41_CED` (default on under the `v41` feature): encoder-only prefill +
/// Decoder SWA Bounded Replay in `forward_prefill_pipelined` (last-token
/// path). `V41_CED=0` restores the exact all-40-layer prefill.
pub fn ced_enabled() -> bool {
    cfg!(feature = "v41") && std::env::var("V41_CED").map(|v| v != "0").unwrap_or(true)
}

/// One prompt row entering the decoder (`CED_DECODER_START`), kept on the host
/// for the bounded replay: token id, residual `[HC_DIM]`, mHC carry
/// `[HC_MIX_DIM]`.
struct ReplayRow {
    tok: i32,
    hc: Vec<f32>,
    carry: Vec<f32>,
}

/// `V41_PREFILL_LOGITS_DUMP=<path>`: append the last-token prefill logits
/// (raw f32 LE, `N_VOCAB` per call) — the CED-vs-exact bit-equality gate.
fn dump_prefill_logits(logits: &[f32]) -> eyre::Result<()> {
    let Ok(path) = std::env::var("V41_PREFILL_LOGITS_DUMP") else { return Ok(()) };
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes)?;
    Ok(())
}

fn chunk_visibility(pos0: u32, b: usize, spans: &[ImageSpan]) -> eyre::Result<Option<Vec<(u32, u32)>>> {''',
name="enum insert")

# 2. pipelined: wrapper + range fn header
s = rep(s, '''    #[allow(clippy::too_many_arguments)]
    pub fn forward_prompt_batch_v2_pipelined(
        &self,
        bd_a: &mut BatchDgpuScratch,
''', '''    #[allow(clippy::too_many_arguments)]
    pub fn forward_prompt_batch_v2_pipelined(
        &self,
        bd_a: &mut BatchDgpuScratch,
        bi_a: &mut BatchIgpuScratch,
        bd_b: &mut BatchDgpuScratch,
        bi_b: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        stats: Option<&mut PrefillStats>,
        image_spans: Option<&[ImageSpan]>,
        pager: Option<&mut super::expert_pager::ExpertPager>,
        engram_rows: Option<&[Vec<f32>]>,
    ) -> eyre::Result<()> {
        self.forward_prompt_batch_v2_pipelined_range(
            bd_a, bi_a, bd_b, bi_b, sd, si, state, weights, input_hcs, tokens, pos0, stats,
            image_spans, pager, engram_rows, 0..N_LAYER as usize, CedMode::Exact, None,
        )
        .map(|_| ())
    }

    /// `forward_prompt_batch_v2_pipelined` over the layer range `layers` (M7
    /// CED). `ced` is the mode of `CED_DECODER_START` when it lies in the range
    /// (every other layer runs `Exact`): `KvSourceOnly` requires the range to
    /// END at that layer (no post-MoE / residual swap after it, so on return
    /// `residual` and `hc_pre_carry` are the rows ENTERING it); `Replay`
    /// requires the range to START there. `seed_carry` seeds each row's mHC
    /// carry (`[B, HC_MIX_DIM]`) for a range that does not start at layer 0.
    /// Returns the lane cut `b_a` (rows `[0, b_a)` are in lane A).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prompt_batch_v2_pipelined_range(
        &self,
        bd_a: &mut BatchDgpuScratch,
''', name="pipelined header")

s = rep(s, '''        engram_rows: Option<&[Vec<f32>]>,
    ) -> eyre::Result<()> {
        let b = tokens.len();
        if b == 0 {
            return Ok(());
        }
        if input_hcs.len() != b {
            return Err(eyre!(
                "forward_prompt_batch_v2_pipelined: input_hcs len {} != tokens len {b}",
                input_hcs.len()
            ));
        }
        // For chunks too small to bother pipelining, fall back to single-lane.
        if b < 2 {
            return self.forward_prompt_batch_v2(
                bd_a, bi_a, sd, si, state, weights, input_hcs, tokens, pos0, stats, image_spans,
            pager.as_deref_mut(),
);
        }
''', '''        engram_rows: Option<&[Vec<f32>]>,
        layers: std::ops::Range<usize>,
        ced: CedMode,
        seed_carry: Option<&[Vec<f32>]>,
    ) -> eyre::Result<usize> {
        let b = tokens.len();
        if b == 0 {
            return Ok(0);
        }
        if input_hcs.len() != b {
            return Err(eyre!(
                "forward_prompt_batch_v2_pipelined: input_hcs len {} != tokens len {b}",
                input_hcs.len()
            ));
        }
        let (lo, hi) = (layers.start, layers.end);
        let split = crate::config::CED_DECODER_START;
        if lo >= hi || hi > N_LAYER as usize {
            return Err(eyre!("forward_prompt_batch_v2_pipelined: bad layer range {lo}..{hi}"));
        }
        match ced {
            CedMode::Exact => {}
            CedMode::KvSourceOnly if hi == split + 1 => {}
            CedMode::Replay if lo == split => {}
            _ => {
                return Err(eyre!(
                    "forward_prompt_batch_v2_pipelined: {ced:?} over layers {lo}..{hi} (split {split})"
                ))
            }
        }
        let mode_of = |l: usize| if l == split { ced } else { CedMode::Exact };
        if let Some(c) = seed_carry {
            if lo == 0 {
                return Err(eyre!(
                    "forward_prompt_batch_v2_pipelined: seed_carry with layer 0 (reset to one-hot there)"
                ));
            }
            if c.len() != b || c.iter().any(|r| r.len() != HC_MIX_DIM as usize) {
                return Err(eyre!(
                    "forward_prompt_batch_v2_pipelined: seed_carry must be [{b}, {HC_MIX_DIM}]"
                ));
            }
        }
        // For chunks too small to bother pipelining, fall back to single-lane
        // (exact full-depth only: the single-lane driver has no layer range).
        if b < 2 && ced == CedMode::Exact && lo == 0 && hi == N_LAYER as usize {
            self.forward_prompt_batch_v2(
                bd_a, bi_a, sd, si, state, weights, input_hcs, tokens, pos0, stats, image_spans,
                pager.as_deref_mut(),
            )?;
            return Ok(b);
        }
''', name="pipelined body head")

s = rep(s, '''        for i in 0..b_b {
            let mut slot = bd_b
                .residual
                .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(&input_b[i])?;
        }
''', '''        for i in 0..b_b {
            let mut slot = bd_b
                .residual
                .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(&input_b[i])?;
        }
        if let Some(c) = seed_carry {
            let m = HC_MIX_DIM as usize;
            for i in 0..b_a {
                bd_a.hc_pre_carry.slice_view_mut(i * m, m).copy_from_host(&c[i])?;
            }
            for i in 0..b_b {
                bd_b.hc_pre_carry.slice_view_mut(i * m, m).copy_from_host(&c[b_a + i])?;
            }
        }
''', name="seed carry")

# warmup
s = rep(s, '''        let layer0 = 0usize;
        self.stage_engram_lane(bd_a, layer0, engram_rows, 0, b_a)?;
        self.forward_layer_pre_moe_v2(
            bd_a,
            bi_a,
            sd,
            si,
            &mut state.layers[layer0],
            &weights.dgpu_layers[layer0],
            &weights.igpu_layers[layer0],
            pos0_a,
            tokens_a,
            vis_a.as_deref(),
            stats_a.as_deref_mut(),
            &self.sync_events.layers[layer0],
        pager.as_deref_mut(),
)?;
        self.stage_engram_lane(bd_b, layer0, engram_rows, b_a, b_b)?;
        self.forward_layer_pre_moe_v2(
            bd_b,
            bi_b,
            sd,
            si,
            &mut state.layers[layer0],
            &weights.dgpu_layers[layer0],
            &weights.igpu_layers[layer0],
            pos0_b,
            tokens_b,
            vis_b.as_deref(),
            None,
            &self.sync_events_t1.layers[layer0],
        pager.as_deref_mut(),
)?;
''', '''        let layer0 = lo;
        self.stage_engram_lane(bd_a, layer0, engram_rows, 0, b_a)?;
        self.forward_layer_pre_moe_v2(
            bd_a,
            bi_a,
            sd,
            si,
            &mut state.layers[layer0],
            &weights.dgpu_layers[layer0],
            &weights.igpu_layers[layer0],
            pos0_a,
            tokens_a,
            vis_a.as_deref(),
            stats_a.as_deref_mut(),
            &self.sync_events.layers[layer0],
            pager.as_deref_mut(),
            mode_of(layer0),
        )?;
        if b_b > 0 {
            self.stage_engram_lane(bd_b, layer0, engram_rows, b_a, b_b)?;
            self.forward_layer_pre_moe_v2(
                bd_b,
                bi_b,
                sd,
                si,
                &mut state.layers[layer0],
                &weights.dgpu_layers[layer0],
                &weights.igpu_layers[layer0],
                pos0_b,
                tokens_b,
                vis_b.as_deref(),
                None,
                &self.sync_events_t1.layers[layer0],
                pager.as_deref_mut(),
                mode_of(layer0),
            )?;
        }
''', name="warmup")

# steady state
s = rep(s, '''        for layer in 0..(N_LAYER as usize - 1) {
            let sev_a_cur = &self.sync_events.layers[layer];''',
'''        for layer in lo..(hi - 1) {
            let sev_a_cur = &self.sync_events.layers[layer];''', name="steady loop")

s = rep(s, '''                vis_a.as_deref(),
                stats_a.as_deref_mut(),
                &self.sync_events.layers[layer + 1],
            pager.as_deref_mut(),
)?;

            // Lane B: same.
            self.forward_layer_post_moe_v2(bd_b, b_b as u32, sev_b_cur, hot_cur)?;
            std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);
            self.stage_engram_lane(bd_b, layer + 1, engram_rows, b_a, b_b)?;
            self.forward_layer_pre_moe_v2(
                bd_b,
                bi_b,
                sd,
                si,
                &mut state.layers[layer + 1],
                &weights.dgpu_layers[layer + 1],
                &weights.igpu_layers[layer + 1],
                pos0_b,
                tokens_b,
                vis_b.as_deref(),
                None,
                &self.sync_events_t1.layers[layer + 1],
            pager.as_deref_mut(),
)?;
''', '''                vis_a.as_deref(),
                stats_a.as_deref_mut(),
                &self.sync_events.layers[layer + 1],
                pager.as_deref_mut(),
                mode_of(layer + 1),
            )?;

            // Lane B: same.
            if b_b > 0 {
                self.forward_layer_post_moe_v2(bd_b, b_b as u32, sev_b_cur, hot_cur)?;
                std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);
                self.stage_engram_lane(bd_b, layer + 1, engram_rows, b_a, b_b)?;
                self.forward_layer_pre_moe_v2(
                    bd_b,
                    bi_b,
                    sd,
                    si,
                    &mut state.layers[layer + 1],
                    &weights.dgpu_layers[layer + 1],
                    &weights.igpu_layers[layer + 1],
                    pos0_b,
                    tokens_b,
                    vis_b.as_deref(),
                    None,
                    &self.sync_events_t1.layers[layer + 1],
                    pager.as_deref_mut(),
                    mode_of(layer + 1),
                )?;
            }
''', name="steady lanes")

# cooldown
s = rep(s, '''        // Cooldown: post-MoE for the final layer on both lanes.
        let last = N_LAYER as usize - 1;
        let hot_last = prefill_hot_active(
            &weights.dgpu_layers[last],
            &weights.igpu_layers[last],
            bd_a,
            sd,
        );
        self.forward_layer_post_moe_v2(bd_a, b_a as u32, &self.sync_events.layers[last], hot_last)?;
        std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
        self.forward_layer_post_moe_v2(bd_b, b_b as u32, &self.sync_events_t1.layers[last], hot_last)?;
        std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);

        self.dgpu.compute.synchronize()?;
        Ok(())
    }
''', '''        // Cooldown: post-MoE for the final layer on both lanes. A source-only
        // last layer has no MoE and leaves `residual` = its input rows.
        let last = hi - 1;
        if mode_of(last) != CedMode::KvSourceOnly {
            let hot_last = prefill_hot_active(
                &weights.dgpu_layers[last],
                &weights.igpu_layers[last],
                bd_a,
                sd,
            );
            self.forward_layer_post_moe_v2(bd_a, b_a as u32, &self.sync_events.layers[last], hot_last)?;
            std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
            if b_b > 0 {
                self.forward_layer_post_moe_v2(bd_b, b_b as u32, &self.sync_events_t1.layers[last], hot_last)?;
                std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);
            }
        }

        self.dgpu.compute.synchronize()?;
        Ok(b_a)
    }
''', name="cooldown")

# 3. forward_layer_batch_v2 passes Exact
s = rep(s, '''        self.forward_layer_pre_moe_v2(bd, bi, sd, si, ls, dlw, ilw, pos0, tokens, vis, stats, sev, pager.as_deref_mut())?;''',
'''        self.forward_layer_pre_moe_v2(bd, bi, sd, si, ls, dlw, ilw, pos0, tokens, vis, stats, sev, pager.as_deref_mut(), CedMode::Exact)?;''',
name="batch_v2 call")

# 4. pre_moe_v2 signature
s = rep(s, '''        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<()> {
        let layer = dlw.layer_idx;
        if ilw.layer_idx != layer {
            return Err(eyre!(
                "forward_layer_pre_moe_v2: dgpu L{} != igpu L{}",''',
'''        mut pager: Option<&mut super::expert_pager::ExpertPager>,
        // M7 CED mode of this call (only `CED_DECODER_START` is ever called
        // with anything but `Exact`).
        ced: CedMode,
    ) -> eyre::Result<()> {
        let layer = dlw.layer_idx;
        if ilw.layer_idx != layer {
            return Err(eyre!(
                "forward_layer_pre_moe_v2: dgpu L{} != igpu L{}",''', name="pre_moe sig")

# 5. stage-1 carry update skipped in KvSourceOnly
s = rep(s, '''            de.hc_weighted.launch_batched(&de.compute, &mut sd.attn_cur, &bd.residual, w, N_EMBD, N_HC, HC_MIX_DIM, b)?;
            if cfg!(feature = "v41") {
                let rows = b as usize * HC_MIX_DIM as usize;
                let cur = bd.split.slice_view(0, rows);
                bd.hc_pre_carry.slice_view_mut(0, rows).copy_from_buffer_async(&cur, &de.compute)?;
            }
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.rms_w", &de.compute)?;''',
'''            de.hc_weighted.launch_batched(&de.compute, &mut sd.attn_cur, &bd.residual, w, N_EMBD, N_HC, HC_MIX_DIM, b)?;
            // M7 CED: a source-only call leaves the carry as it entered the
            // layer (the replay re-runs this sub-block and carries it then).
            if cfg!(feature = "v41") && ced != CedMode::KvSourceOnly {
                let rows = b as usize * HC_MIX_DIM as usize;
                let cur = bd.split.slice_view(0, rows);
                bd.hc_pre_carry.slice_view_mut(0, rows).copy_from_buffer_async(&cur, &de.compute)?;
            }
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.rms_w", &de.compute)?;''', name="carry guard")

# 6. stages 2-3 wrapped
s = rep(s, '''        let _t_q = de.events.stage("dgpu.q_chain", &de.compute)?;
        {
            let _t = de.events.stage("k.q_chain.cast_input_f16", &de.compute)?;''',
'''        // M7 CED: a source-only call needs neither Q nor the window KV.
        if ced != CedMode::KvSourceOnly {
        let _t_q = de.events.stage("dgpu.q_chain", &de.compute)?;
        {
            let _t = de.events.stage("k.q_chain.cast_input_f16", &de.compute)?;''', name="q_chain open")
s = rep(s, '''        drop(_t_kv);
        let _t_kv_append_comp = de.events.stage("dgpu.kv_append_compressor_serial", &de.compute)?;''',
'''        drop(_t_kv);
        } // ced != KvSourceOnly (stages 2-3)
        let _t_kv_append_comp = de.events.stage("dgpu.kv_append_compressor_serial", &de.compute)?;''', name="kv_chain close")

# 7. raw window + kv_append wrapped
s = rep(s, '''        let n_raw_before = ls.n_raw;
        let mut n_raw_offset_after: Vec<u32> = Vec::with_capacity(b as usize);
        match vis {''',
'''        let n_raw_before = ls.n_raw;
        let mut n_raw_offset_after: Vec<u32> = Vec::with_capacity(b as usize);
        if ced != CedMode::KvSourceOnly {
        match vis {''', name="raw window open")
s = rep(s, '''        let n_raw_during_chunk = n_raw_before + b;
''', '''        } // ced != KvSourceOnly (window KV append)
        let n_raw_during_chunk = n_raw_before + b;
''', name="raw window close")

# 8. compressor: skip the projection + store write in Replay
s = rep(s, '''        let own_compressor = ratio > 0 && dlw.compressor.is_some();
        if own_compressor {''',
'''        // M7 CED Replay: the store already holds these positions (written by
        // the source-only pass), so the projection + store write are skipped
        // and the causal counts come from the positional reuse formula below.
        let own_compressor = ratio > 0 && dlw.compressor.is_some() && ced != CedMode::Replay;
        if own_compressor {''', name="own_compressor def")

# 9. reuse-layer causal counts: positional (fixes the two-lane leak)
s = rep(s, '''            let in_chunk = (pos0 + b) / ratio - pos0 / ratio;
            let before = cs.n_comp.checked_sub(in_chunk).ok_or_else(|| eyre!("L{layer}: source store n_comp {} < boundaries in chunk {in_chunk}", cs.n_comp))?;
            for k in 0..b {
                n_comp_after.push(before + ((pos0 + k + 1) / ratio - pos0 / ratio));
            }''',
'''            // The store is 1:1 with compressor boundaries from position 0, so
            // row k's causal count is positional. It must NOT be derived from
            // `cs.n_comp - boundaries_in_this_call`: in the two-lane driver the
            // source layer has already appended the OTHER lane's rows by the
            // time this lane's reuse layer runs, which let lane A attend to
            // lane B's (future) compressed rows.
            let need = (pos0 + b) / ratio;
            if cs.n_comp < need {
                return Err(eyre!(
                    "L{layer}: source store n_comp {} < {need} boundaries up to pos {}",
                    cs.n_comp,
                    pos0 + b
                ));
            }
            for k in 0..b {
                n_comp_after.push((pos0 + k + 1) / ratio);
            }''', name="reuse positional")

# 10. KvSourceOnly returns before attention
s = rep(s, '''        drop(_t_kv_append_comp);

        // ========================================================
        // Stage 5: Attention (BATCHED — grid (n_head, B, 1))''',
'''        drop(_t_kv_append_comp);
        if ced == CedMode::KvSourceOnly {
            // M7 CED: the decoder's global KV for these rows is in the store;
            // nothing below (window KV, attention, MoE) runs for them.
            return Ok(());
        }

        // ========================================================
        // Stage 5: Attention (BATCHED — grid (n_head, B, 1))''', name="source-only return")

# 11. head helper + use in both drivers
s = rep(s, '''    /// Two-lane pipelined chunked prefill. Same contract as
    /// `forward_prefill` but takes two BatchDgpu/BatchIgpu scratch sets''',
'''    /// Head over one batched row `idx` of `bd`: residual (+ under V4.1 the mHC
    /// carry the head's collapse reads — decode twin: `forward_head` after
    /// layer N-1 reads `dgpu_scratch.hc_pre_carry`) → logits `[N_VOCAB]`.
    fn head_from_row(
        &self,
        head_scratch: &mut DgpuScratch,
        bd: &BatchDgpuScratch,
        idx: usize,
        weights: &HetModelWeights,
    ) -> eyre::Result<Vec<f32>> {
        let cs_hc = HC_DIM as usize;
        head_scratch
            .residual
            .copy_from_buffer(&bd.residual.slice_view(idx * cs_hc, cs_hc))?;
        if cfg!(feature = "v41") {
            let m = HC_MIX_DIM as usize;
            head_scratch
                .hc_pre_carry
                .copy_from_buffer(&bd.hc_pre_carry.slice_view(idx * m, m))?;
        }
        self.forward_head(head_scratch, &weights.global)?;
        let mut logits = vec![0f32; N_VOCAB as usize];
        head_scratch.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    /// Two-lane pipelined chunked prefill. Same contract as
    /// `forward_prefill` but takes two BatchDgpu/BatchIgpu scratch sets''', name="head helper")

# single-lane driver heads
s = rep(s, '''            if last_only {
                if is_last_chunk {
                    let last_b = chunk_b - 1;
                    head_scratch.residual.copy_from_buffer(
                        &bd.residual.slice_view(last_b * cs_hc, cs_hc),
                    )?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits = logits;
                }
            } else {
                for i in 0..chunk_b {
                    head_scratch
                        .residual
                        .copy_from_buffer(&bd.residual.slice_view(i * cs_hc, cs_hc))?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits.extend_from_slice(&logits);
                }
            }
''', '''            if last_only {
                if is_last_chunk {
                    out_logits = self.head_from_row(head_scratch, bd, chunk_b - 1, weights)?;
                }
            } else {
                for i in 0..chunk_b {
                    let logits = self.head_from_row(head_scratch, bd, i, weights)?;
                    out_logits.extend_from_slice(&logits);
                }
            }
''', name="single-lane heads")

# pipelined driver: CED setup
s = rep(s, '''        let prefill_start = std::time::Instant::now();
        let total_chunks = t.div_ceil(chunk_size);
        let mut chunk_idx = 0usize;
        let mut chunk_start = 0usize;
''', '''        let prefill_start = std::time::Instant::now();
        let total_chunks = t.div_ceil(chunk_size);
        let mut chunk_idx = 0usize;
        let mut chunk_start = 0usize;

        // M7 CED (tech report §2.2 / §3.2.2): the causal encoder runs over every
        // prompt token and the decoder's global KV is projected from the final
        // encoder hidden state at `CED_DECODER_START`; the decoder itself runs
        // only over the last SWA_WINDOW tokens (Decoder SWA Bounded Replay),
        // after the last chunk. Per-token logits need the decoder over every
        // row, so only the last-token path takes it.
        let ced = ced_enabled() && last_only;
        let split = crate::config::CED_DECODER_START;
        let mut replay: std::collections::VecDeque<ReplayRow> = std::collections::VecDeque::new();
''', name="ced setup")

s = rep(s, '''            self.forward_prompt_batch_v2_pipelined(
                bd_a,
                bi_a,
                bd_b,
                bi_b,
                sd,
                si,
                state,
                weights,
                chunk_input,
                chunk_tokens,
                chunk_pos0,
                stats.as_deref_mut(),
                image_spans,
                pager.as_deref_mut(),
                // Slice the prompt-wide Engram rows down to this chunk.
                chunk_engram.as_deref(),
            )?;
''', '''            if ced {
                // Encoder + the decoder's KV projection; `residual` /
                // `hc_pre_carry` come back holding the rows ENTERING `split`.
                let cut = self.forward_prompt_batch_v2_pipelined_range(
                    bd_a,
                    bi_a,
                    bd_b,
                    bi_b,
                    sd,
                    si,
                    state,
                    weights,
                    chunk_input,
                    chunk_tokens,
                    chunk_pos0,
                    stats.as_deref_mut(),
                    image_spans,
                    pager.as_deref_mut(),
                    chunk_engram.as_deref(),
                    0..split + 1,
                    CedMode::KvSourceOnly,
                    None,
                )?;
                if cut != b_a {
                    return Err(eyre!("CED prefill: lane cut {cut} != planned {b_a}"));
                }
                // Keep the last SWA_WINDOW rows entering the decoder (host ring).
                let take = chunk_b.min(SWA_WINDOW as usize);
                let (m, hc) = (HC_MIX_DIM as usize, HC_DIM as usize);
                for i in chunk_b - take..chunk_b {
                    let (src, idx) = if i < b_a { (&*bd_a, i) } else { (&*bd_b, i - b_a) };
                    let mut row = ReplayRow { tok: chunk_tokens[i], hc: vec![0f32; hc], carry: vec![0f32; m] };
                    src.residual.slice_view(idx * hc, hc).copy_to_host(&mut row.hc)?;
                    src.hc_pre_carry.slice_view(idx * m, m).copy_to_host(&mut row.carry)?;
                    replay.push_back(row);
                    if replay.len() > SWA_WINDOW as usize {
                        replay.pop_front();
                    }
                }
            } else {
                self.forward_prompt_batch_v2_pipelined(
                    bd_a,
                    bi_a,
                    bd_b,
                    bi_b,
                    sd,
                    si,
                    state,
                    weights,
                    chunk_input,
                    chunk_tokens,
                    chunk_pos0,
                    stats.as_deref_mut(),
                    image_spans,
                    pager.as_deref_mut(),
                    // Slice the prompt-wide Engram rows down to this chunk.
                    chunk_engram.as_deref(),
                )?;
            }
''', name="chunk call")

s = rep(s, '''            if last_only {
                if is_last_chunk {
                    // Last token: lives in lane B if b_b > 0, else lane A.
                    let (src_bd, last_idx) = if b_b > 0 {
                        (&*bd_b, b_b - 1)
                    } else {
                        (&*bd_a, b_a - 1)
                    };
                    head_scratch
                        .residual
                        .copy_from_buffer(&src_bd.residual.slice_view(last_idx * cs_hc, cs_hc))?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits = logits;
                }
            } else {
                for i in 0..b_a {
                    head_scratch
                        .residual
                        .copy_from_buffer(&bd_a.residual.slice_view(i * cs_hc, cs_hc))?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits.extend_from_slice(&logits);
                }
                for i in 0..b_b {
                    head_scratch
                        .residual
                        .copy_from_buffer(&bd_b.residual.slice_view(i * cs_hc, cs_hc))?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits.extend_from_slice(&logits);
                }
            }
''', '''            if last_only {
                if is_last_chunk && !ced {
                    // Last token: lives in lane B if b_b > 0, else lane A.
                    let (src_bd, last_idx) = if b_b > 0 {
                        (&*bd_b, b_b - 1)
                    } else {
                        (&*bd_a, b_a - 1)
                    };
                    let logits = self.head_from_row(head_scratch, src_bd, last_idx, weights)?;
                    dump_prefill_logits(&logits)?;
                    out_logits = logits;
                }
            } else {
                for i in 0..b_a {
                    let logits = self.head_from_row(head_scratch, bd_a, i, weights)?;
                    out_logits.extend_from_slice(&logits);
                }
                for i in 0..b_b {
                    let logits = self.head_from_row(head_scratch, bd_b, i, weights)?;
                    out_logits.extend_from_slice(&logits);
                }
            }
''', name="pipelined heads")

s = rep(s, '''            chunk_start = chunk_end;
            chunk_idx += 1;
            if let Some(f) = on_chunk_done {
                f();
            }
        }
        Ok(out_logits)
    }
''', '''            chunk_start = chunk_end;
            chunk_idx += 1;
            if let Some(f) = on_chunk_done {
                f();
            }
        }

        if ced {
            // Decoder SWA Bounded Replay (§3.2.2): feed the last SWA_WINDOW
            // rows' encoder outputs through the decoder with the decoder rings
            // emptied, so a segment query at index i sees window keys in
            // [max(s, i-W+1), i] and the complete global KV. Approximate by
            // design for N > SWA_WINDOW (identical to the exact path otherwise).
            if let Some(c) = cancel {
                if c.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok(Vec::new());
                }
            }
            let b_seg = replay.len();
            if b_seg == 0 || b_seg > t {
                return Err(eyre!("CED prefill: replay segment {b_seg} of {t} rows"));
            }
            let seg_pos0 = pos0 + (t - b_seg) as u32;
            for l in split..N_LAYER as usize {
                state.layers[l].n_raw = 0;
                state.layers[l].raw_off = 0;
            }
            let mut seg_hcs: Vec<Vec<f32>> = Vec::with_capacity(b_seg);
            let mut seg_carry: Vec<Vec<f32>> = Vec::with_capacity(b_seg);
            let mut seg_tokens: Vec<i32> = Vec::with_capacity(b_seg);
            for r in replay.drain(..) {
                seg_hcs.push(r.hc);
                seg_carry.push(r.carry);
                seg_tokens.push(r.tok);
            }
            let t0 = std::time::Instant::now();
            self.dgpu.events.reset();
            self.igpu.events.reset();
            let b_a = self.forward_prompt_batch_v2_pipelined_range(
                bd_a,
                bi_a,
                bd_b,
                bi_b,
                sd,
                si,
                state,
                weights,
                &seg_hcs,
                &seg_tokens,
                seg_pos0,
                None,
                // Text-causal replay: an image span's widened window is an
                // encoder-side (chunk) property; V4.1 vision is not wired yet.
                None,
                pager.as_deref_mut(),
                None,
                split..N_LAYER as usize,
                CedMode::Replay,
                Some(&seg_carry),
            )?;
            let (src_bd, last_idx) = if b_seg > b_a { (&*bd_b, b_seg - b_a - 1) } else { (&*bd_a, b_a - 1) };
            let logits = self.head_from_row(head_scratch, src_bd, last_idx, weights)?;
            tracing::info!(
                replay_tokens = b_seg,
                seg_pos0,
                elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f32()),
                total_s = format!("{:.1}", prefill_start.elapsed().as_secs_f32()),
                "ced_replay"
            );
            dump_prefill_logits(&logits)?;
            out_logits = logits;
            if let Some(f) = on_chunk_done {
                f();
            }
        }
        Ok(out_logits)
    }
''', name="replay block")

open(P, "w").write(s)
print("ok")
