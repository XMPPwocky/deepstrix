//! Per-layer state for the het orchestrator (KV cache + compressor state).
//!
//! All state lives on the dGPU (the device that runs attention and reads
//! the KV cache). The compressor's iGPU-resident era was rolled back —
//! see [`HetCompressorState`].

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer};

use crate::comp_kv_fp8::{FP8_KV_HEAD_ROWS, FP8_KV_ROW_BYTES};
use crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES;
use crate::config::{COMPRESS_RATIOS, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LAYER, NEG_INF, SWA_WINDOW};
use crate::het::batch_scratch::B_MAX;

/// During batched prefill we need to hold the prior SWA-window AND the
/// current chunk's freshly-computed KVs together in cache so each token in
/// the batch can attend to its causally-valid window (which spans both)
/// — see the n_raw_offset_per attention parameter in forward_prefill.rs.
/// Outside of prefill chunks only the first SWA_WINDOW rows are used.
pub const KV_CACHE_ROWS: usize = SWA_WINDOW as usize + B_MAX;

/// Storage of a compressor's cumulative pooled cache.
///
/// `F16`: `[n_comp_max, head_dim]` f16 (held as `u16`) — the ratio-128
/// main compressors (tiny caches, read directly by the dense path) and
/// the indexer compressors (head_dim 128; lever 3 of the VRAM plan).
///
/// `Fp8`: the ratio-4 main compressors. `rows` holds packed
/// [`FP8_KV_ROW_BYTES`]-byte rows (E4M3 codes + block exponents + f16 RoPE
/// tail, `comp_kv_fp8.rs`), bit-identical to the f16 rows after expansion;
/// `head` is an f16 shadow of rows `[0, FP8_KV_HEAD_ROWS)` that the dense
/// attention path reads in place of the old full f16 cache (the dense path
/// is only taken while `n_comp <= INDEXER_TOP_K == FP8_KV_HEAD_ROWS`).
/// The sparse path expands selected rows into `active_comp_kv` via
/// `indexer_gather_fp8`, so the attention kernels never see the packed
/// format. -42% on the compressed cache (0.42 GiB at 192K).
/// `E2m1`: the ratio-4 INDEXER compressors (lever 3). `rows` holds packed
/// [`E2M1_KEY_ROW_BYTES`]-byte rows (E2M1 nibbles + block exponents,
/// `index_kv_e2m1.rs`), bit-identical to the f16 key rows after expansion;
/// the score kernels expand at their loads (`*_e2m1` twins), so no f16 copy
/// exists at all. -69% on the index key cache.
pub enum CompKvStore {
    F16(DeviceBuffer<u16>),
    Fp8 {
        rows: DeviceBuffer<u8>,
        head: DeviceBuffer<u16>,
    },
    E2m1(DeviceBuffer<u8>),
}

impl CompKvStore {
    /// Whether the packed format is selected for the ratio-4 main
    /// compressor. Read at every allocation (so a one-load A/B test can
    /// flip it between states). `COMP_KV_FP8=0` keeps the f16 cache: a
    /// rollback / in-process A-B knob, not a tuning parameter (every
    /// consumer dispatches on the variant, so both are always correct;
    /// snapshots convert either way on restore).
    pub fn fp8_enabled() -> bool {
        std::env::var("COMP_KV_FP8")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("off")))
            .unwrap_or(true)
    }

    /// Whether the packed E2M1 format is selected for the ratio-4 indexer
    /// compressor. Read at every allocation. `INDEXER_KEYS_E2M1=0` keeps the
    /// f16 key cache (rollback / one-load A/B knob).
    pub fn e2m1_enabled() -> bool {
        std::env::var("INDEXER_KEYS_E2M1")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("off")))
            .unwrap_or(true)
    }

    pub fn is_fp8(&self) -> bool {
        matches!(self, CompKvStore::Fp8 { .. })
    }

    pub fn is_e2m1(&self) -> bool {
        matches!(self, CompKvStore::E2m1(_))
    }

    /// The packed rows of an `E2m1` store.
    pub fn e2m1(&self) -> Option<&DeviceBuffer<u8>> {
        match self {
            CompKvStore::E2m1(b) => Some(b),
            _ => None,
        }
    }

    pub fn e2m1_mut(&mut self) -> Option<&mut DeviceBuffer<u8>> {
        match self {
            CompKvStore::E2m1(b) => Some(b),
            _ => None,
        }
    }

    /// The f16 buffer of an `F16` store.
    pub fn f16(&self) -> Option<&DeviceBuffer<u16>> {
        match self {
            CompKvStore::F16(b) => Some(b),
            _ => None,
        }
    }

    pub fn f16_mut(&mut self) -> Option<&mut DeviceBuffer<u16>> {
        match self {
            CompKvStore::F16(b) => Some(b),
            _ => None,
        }
    }

    /// The f16 buffer the DENSE attention path reads rows `[0, n_comp)`
    /// from: the whole cache for `F16`, the head shadow for `Fp8`. Errors
    /// (rather than reading past the shadow) when the caller's `n_comp`
    /// exceeds what the shadow holds — the `DECODE_INDEXER=off`-at-depth
    /// and FAKE_POS-overrun cases.
    pub fn dense_f16(&self, n_comp: u32, what: &str) -> eyre::Result<&DeviceBuffer<u16>> {
        match self {
            CompKvStore::F16(b) => Ok(b),
            CompKvStore::Fp8 { head, .. } => {
                if n_comp as usize > FP8_KV_HEAD_ROWS {
                    return Err(eyre::eyre!(
                        "{what}: dense attention over {n_comp} compressed rows but the FP8 \
                         cache keeps only {FP8_KV_HEAD_ROWS} f16 rows for the dense path \
                         (sparse/indexer path required above INDEXER_TOP_K; \
                         DECODE_INDEXER=off is unsupported at this depth)"
                    ));
                }
                Ok(head)
            }
            CompKvStore::E2m1(_) => Err(eyre::eyre!(
                "{what}: dense attention over a packed-E2M1 indexer key store (never a main compressor)"
            )),
        }
    }

    /// Row capacity of the store.
    pub fn capacity_rows(&self, head_dim: u32) -> usize {
        match self {
            CompKvStore::F16(b) => b.len() / head_dim as usize,
            CompKvStore::Fp8 { rows, .. } => rows.len() / FP8_KV_ROW_BYTES,
            CompKvStore::E2m1(rows) => rows.len() / E2M1_KEY_ROW_BYTES,
        }
    }
}

/// Per-layer compressor state. All buffers live on the dGPU: the
/// compressor kernels run alongside attn_input_norm on dGPU, so
/// `state_kv` / `state_score` (sliding compressor state) and `comp_kv`
/// (the cumulative pooled cache `attn_mixed` reads) are all local —
/// no peer push needed when the compressor fires at a boundary.
pub struct HetCompressorState {
    /// iGPU-resident: compressor sliding state.
    pub state_kv: DeviceBuffer<f32>,
    pub state_score: DeviceBuffer<f32>,
    /// dGPU-resident: cumulative pooled comp-KV cache consumed by
    /// `attn_mixed`. See [`CompKvStore`] for the two storage formats;
    /// values come out of the compressor as f32 and are cast (or packed)
    /// at the append.
    pub comp_kv: CompKvStore,
    pub n_comp: u32,
    pub width: u32,
    pub head_dim: u32,
    /// V4.1 CSA2 index-K cache: `k_norm(wk(latent))` per compressed row, stored
    /// PACKED E2M1 + one E8M0 scale per 32 (`E2M1_KEY_ROW_BYTES` = 80 B/row),
    /// which is exactly the reference's `fp4_act_quant(k, 32, True)`. One 128-dim
    /// key per ROW (shared across the 32 index heads, MLA-style), not per head.
    /// 80 B/row vs 256 B for f16, so the whole cache is ~26 MB at --ctx 130688.
    ///
    /// It lives HERE, in the main compressor state, on purpose. `compressor` is
    /// allocated exactly on the 4 KV-SOURCE layers, and `with_kv_source` already
    /// moves that state into the reuse layers for the forward — so putting the
    /// index-K cache inside it makes reuse work for free. V4.1 has NO indexer
    /// compressor, so allocating a second `HetCompressorState` (the V4-Flash
    /// ratio-4 shape) would have been the wrong structure entirely.
    ///
    /// `None` on V4-Flash and on the ratio-4 indexer compressor itself.
    pub index_k: Option<DeviceBuffer<u8>>,
    /// Rows valid in `index_k`. Tracks `n_comp` once the indexer writes it;
    /// separate counter so a half-built cache can never be read as complete.
    pub n_index_comp: u32,
}

impl HetCompressorState {
    pub fn alloc(
        igpu_device: Device,
        dgpu_device: Device,
        ratio: u32,
        head_dim: u32,
        n_kv_max: u32,
    ) -> eyre::Result<Self> {
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let state_rows = ratio * coff;
        let n_state = (state_rows as usize) * (width as usize);
        let zeros = vec![0f32; n_state];
        let neg_inf = vec![NEG_INF; n_state];

        // State on iGPU.
        igpu_device.set_current()?;
        let mut state_kv: DeviceBuffer<f32> = DeviceBuffer::new(igpu_device.id, n_state)?;
        let mut state_score: DeviceBuffer<f32> = DeviceBuffer::new(igpu_device.id, n_state)?;
        state_kv.copy_from_host(&zeros)?;
        state_score.copy_from_host(&neg_inf)?;

        // comp_kv on dGPU.
        dgpu_device.set_current()?;
        let max_n_comp = (n_kv_max + ratio - 1) / ratio;
        let comp_kv = if ratio == 4 && head_dim == N_HEAD_DIM && CompKvStore::fp8_enabled() {
            CompKvStore::Fp8 {
                rows: DeviceBuffer::new(dgpu_device.id, (max_n_comp as usize) * FP8_KV_ROW_BYTES)?,
                head: DeviceBuffer::new(dgpu_device.id, FP8_KV_HEAD_ROWS * (head_dim as usize))?,
            }
        } else if ratio == 4 && head_dim == N_INDEXER_HEAD_DIM && CompKvStore::e2m1_enabled() {
            CompKvStore::E2m1(DeviceBuffer::new(dgpu_device.id, (max_n_comp as usize) * E2M1_KEY_ROW_BYTES)?)
        } else {
            let comp_kv_capacity = (max_n_comp as usize) * (head_dim as usize);
            CompKvStore::F16(DeviceBuffer::new(dgpu_device.id, comp_kv_capacity)?)
        };
        // Index-K only for V4.1's MAIN compressor (head_dim 512). The ratio-4
        // indexer compressor passes N_INDEXER_HEAD_DIM and must not get one.
        let index_k = if head_dim == N_HEAD_DIM {
            dgpu_device.set_current()?;
            Some(DeviceBuffer::new(
                dgpu_device.id,
                (max_n_comp as usize) * crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES,
            )?)
        } else {
            None
        };
        Ok(Self {
            state_kv,
            state_score,
            comp_kv,
            n_comp: 0,
            width,
            head_dim,
            index_k,
            n_index_comp: 0,
        })
    }
}

impl HetModelState {
    /// Snapshot every layer's KV position so a speculative batch can be undone.
    pub fn mark_kv(&self) -> KvMark {
        self.try_mark_kv().expect("mark_kv: compressor snapshot")
    }

    /// `mark_kv`, surfacing the compressor-snapshot copy error instead of
    /// panicking.
    pub fn try_mark_kv(&self) -> eyre::Result<KvMark> {
        let per_layer = self.layers.iter().map(|l| (l.n_raw, l.raw_off)).collect();
        let mut per_layer_comp = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            if l.compressor.is_none() && l.indexer_compressor.is_none() {
                per_layer_comp.push(None);
                continue;
            }
            per_layer_comp.push(Some(CompMark {
                main: l.compressor.as_ref().map(CompStateMark::capture).transpose()?,
                indexer: l.indexer_compressor.as_ref().map(CompStateMark::capture).transpose()?,
            }));
        }
        Ok(KvMark { per_layer, per_layer_comp, slid: false })
    }

    /// Undo the KV appends made since `mark`.
    ///
    /// This is the reject half of speculative decoding: a verify step appends B
    /// tokens, the first rejected one and everything after it must go away.
    ///
    /// # What this does NOT restore
    /// **The compressor and indexer-compressor streaming state.** Those advance
    /// with accepted AND rejected tokens and would need
    /// `compressor_state_snapshot` to roll back properly. Deliberately out of
    /// scope for now, so a rejected token leaves the compressor slightly ahead of
    /// the raw KV. Fine while the compressed store is an approximation used for
    /// scoring, NOT fine if bit-exact continuation is required — wire the
    /// snapshot before claiming that.
    ///
    /// # Errors
    /// If any layer's `raw_off` went BACKWARDS, the oversized cache wrapped since
    /// the mark: the eviction-down copy physically relocated the live window, so
    /// the marked counters no longer address the same rows and the rollback would
    /// silently serve wrong KV. Refuses instead. A wrap happens roughly once per
    /// `B_MAX` tokens per layer, so a verify batch of <=8 almost never straddles
    /// one — but "almost never" is exactly the bug that survives testing.
    pub fn rollback_kv(&mut self, mark: &KvMark) -> color_eyre::eyre::Result<()> {
        use color_eyre::eyre::eyre;
        if mark.per_layer.len() != self.layers.len() {
            return Err(eyre!(
                "rollback_kv: mark covers {} layers, state has {}",
                mark.per_layer.len(),
                self.layers.len()
            ));
        }
        // `per_layer_comp` is indexed BY LAYER below (`self.layers[i]`), so a
        // mark whose compressor snapshot has a different length than the layer
        // list would restore the wrong layer's compressor. The empty vec is the
        // documented "mark taken before this field existed" case.
        if !mark.per_layer_comp.is_empty() && mark.per_layer_comp.len() != self.layers.len() {
            return Err(eyre!(
                "rollback_kv: mark's compressor snapshot covers {} layers, state has {};                  per_layer_comp is indexed by layer id",
                mark.per_layer_comp.len(),
                self.layers.len()
            ));
        }
        if !mark.slid {
            for (i, (_, raw_off)) in mark.per_layer.iter().copied().enumerate() {
                if self.layers[i].raw_off < raw_off {
                    return Err(eyre!(
                        "rollback_kv: layer {i} wrapped since the mark (raw_off {} < marked {raw_off}); \
                         the eviction-down copy moved the window, so the mark no longer addresses it",
                        self.layers[i].raw_off
                    ));
                }
            }
        }
        // A slid mark must still fit the oversized cache. If the append pointer
        // is within MTP_BLOCK of the end, the next verify append would OOB — the
        // window needs compacting down, which the caller must do before the next
        // step (decode's own wrap path). Refuse loudly rather than corrupt.
        for (i, (n_raw, raw_off)) in mark.per_layer.iter().copied().enumerate() {
            if mark.slid
                && (raw_off + n_raw) as usize + crate::het::mtp::MTP_BLOCK > KV_CACHE_ROWS
            {
                return Err(eyre!(
                    "rollback_kv: layer {i} slid append pointer {} within MTP_BLOCK of cache \
                     capacity {KV_CACHE_ROWS} — needs compaction (not yet wired for accept)",
                    raw_off + n_raw
                ));
            }
        }
        for (i, (n_raw, raw_off)) in mark.per_layer.iter().copied().enumerate() {
            self.layers[i].n_raw = n_raw;
            self.layers[i].raw_off = raw_off;
        }
        // Compressed KV too, or the raw window rewinds while the compressed
        // store keeps the speculative rows. A mark taken before this field
        // existed (empty vec) rolls back the raw window only, as it used to.
        // `V41_COMP_ROLLBACK=0` restores the OLD (buggy) behaviour: raw window
        // only. Kept as a flag because the damage is CUMULATIVE across steps,
        // so the two arms cannot be interleaved inside one run — they have to
        // be separate runs of the same binary.
        if !comp_rollback_enabled() {
            return Ok(());
        }
        for (i, cm) in mark.per_layer_comp.iter().enumerate() {
            let Some(cm) = cm.as_ref() else { continue };
            if let (Some(m), Some(cs)) = (cm.main.as_ref(), self.layers[i].compressor.as_mut()) {
                m.restore(cs)?;
            }
            if let (Some(m), Some(cs)) =
                (cm.indexer.as_ref(), self.layers[i].indexer_compressor.as_mut())
            {
                m.restore(cs)?;
            }
        }
        Ok(())
    }

    /// Run `f` on layer `layer` with its KV source's compressor state moved in
    /// (V4.1 reuse layers), moving it back afterwards. A no-op wrapper when the
    /// layer owns its store. The forward skips the compressor stage for a layer
    /// without compressor *weights* and only reads the store.
    /// Restore the "every compressor lives at its KV-source layer" invariant.
    ///
    /// The steady-state loops in `forward_prefill` and `engine` lend a source
    /// layer's compressor store to its V4.1 reuse layer at the top of an
    /// iteration and hand it back at the bottom. That manual pair is NOT
    /// exception-safe: any `?` in between leaves the store parked on the reuse
    /// layer with `layers[src].compressor == None`, and then EVERY later request
    /// fails with "L{src}: missing compressor state". One real error becomes a
    /// permanent one, and the reported layer is not the one that broke — which
    /// is exactly how a dropped-pick error at L18 surfaced as "L14: missing
    /// compressor state". (`with_kv_source` below gets this right by construction.)
    ///
    /// Called at forward entry, where the invariant must hold. 40 Option moves,
    /// no device work.
    pub fn restore_compressor_lending(&mut self) {
        for layer in 0..self.layers.len() {
            let Some(src) = crate::config::kv_source_of(layer) else {
                continue;
            };
            if self.layers[src].compressor.is_none()
                && self.layers[layer].compressor.is_some()
            {
                let st = self.layers[layer].compressor.take();
                self.layers[src].compressor = st;
            }
        }
    }

    pub fn with_kv_source<R>(
        &mut self,
        layer: usize,
        f: impl FnOnce(&mut HetLayerState) -> eyre::Result<R>,
    ) -> eyre::Result<R> {
        match crate::config::kv_source_of(layer) {
            Some(src) => {
                debug_assert!(self.layers[layer].compressor.is_none());
                let st = self.layers[src].compressor.take();
                self.layers[layer].compressor = st;
                let r = f(&mut self.layers[layer]);
                let st = self.layers[layer].compressor.take();
                self.layers[src].compressor = st;
                r
            }
            None => f(&mut self.layers[layer]),
        }
    }
}

pub struct HetLayerState {
    /// SWA raw KV cache. f16-stored (see `HetCompressorState::comp_kv` rationale).
    pub kv_cache: DeviceBuffer<u16>,
    pub n_raw: u32,
    /// M55: first valid row of the SWA window inside `kv_cache`. The decode
    /// path appends MONOTONICALLY at slot `raw_off + n_raw` and advances
    /// `raw_off` instead of sliding 127 rows per token (the old
    /// kv_cache_append evict path: 254 barriers/layer/token). When the
    /// window reaches the end of the oversized cache, an eviction-down
    /// copy (same two-hop pattern as prefill's) resets it to 0. Readers
    /// take a `slice_view` starting at `raw_off * head_dim`. Also the
    /// prerequisite for MTP rollback: rejected appends just decrement,
    /// the "evicted" row is still in place.
    pub raw_off: u32,
    pub compressor: Option<HetCompressorState>,
    /// CSA indexer's parallel compressor (head_dim=128 vs main's 512).
    /// Only present at ratio==4 layers. Layout / lifecycle mirror the
    /// main compressor exactly; `n_comp` here is what ds4 calls
    /// `cache->n_index_comp`.
    pub indexer_compressor: Option<HetCompressorState>,
}

/// A per-layer KV position mark, for speculative rollback.
///
/// The decode append is monotonic — `n_raw` grows until `SWA_WINDOW`, then
/// `raw_off` slides — and the row it "evicts" is still physically there, which is
/// what `HetLayerState::raw_off` means by "the prerequisite for MTP rollback".
/// So undoing k speculative tokens is restoring two counters per layer; no data
/// moves and nothing is rewritten.
fn comp_rollback_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("V41_COMP_ROLLBACK").as_deref() != Ok("0"))
}

#[derive(Clone, Debug, Default)]
pub struct KvMark {
    /// `(n_raw, raw_off)` per layer at mark time.
    pub per_layer: Vec<(u32, u32)>,
    /// True when produced by [`Self::advanced_by`]: the target `raw_off` is a
    /// SLID window pointer that legitimately moved FORWARD, so the wrap check in
    /// `rollback_kv` (which exists to catch a compaction that moved bytes
    /// BACKWARD) must not fire on it. A plain `mark_kv` leaves this false.
    pub slid: bool,
    /// Compressor store per layer at mark time, for the layers that OWN one
    /// (`with_kv_source` lends it to the reuse layers, so only the 4 KV-source
    /// layers are `Some`).
    ///
    /// Without this a rollback restored the raw window but left the COMPRESSED
    /// KV advanced: a speculative batch fires ~b/ratio compressor boundaries,
    /// and those rows stayed in the store for decode to attend to, permanently
    /// and cumulatively. MEASURED as a verify-vs-decode fidelity cliff at the
    /// batch width where a second boundary can fire (ratio 2, so B>=3):
    /// argmax agreement 0.875 at B<=2 against 0.12-0.39 at B>=3.
    ///
    /// Only the running segment accumulators and the counters need saving:
    /// `comp_kv` / `index_k` rows past `n_comp` / `n_index_comp` are never
    /// read, so truncating the counters is enough to discard them. The
    /// accumulators are `ratio * coff * width` floats — 4 KB a layer.
    pub per_layer_comp: Vec<Option<CompMark>>,
}

/// Snapshot of one layer's compressor stores (main + CSA indexer). See
/// [`KvMark::per_layer_comp`].
#[derive(Clone, Debug)]
pub struct CompMark {
    pub main: Option<CompStateMark>,
    pub indexer: Option<CompStateMark>,
}

#[derive(Clone, Debug)]
pub struct CompStateMark {
    pub n_comp: u32,
    pub n_index_comp: u32,
    pub state_kv: Vec<f32>,
    pub state_score: Vec<f32>,
}

impl KvMark {
    /// A mark advanced by `keep` rows, for the speculative accept path's
    /// PARTIAL rollback: the first `keep` rows of the verify batch became real
    /// context and everything after them is discarded.
    ///
    /// `abs_pos` is the absolute position of row 0 of the batch.
    ///
    /// The raw window is just the mark advanced by `keep`. The compressor needs
    /// more care, because it is a running accumulator over a `ratio`-position
    /// segment and cannot be rewound to a row in the middle of a batch:
    ///
    ///   - `n_comp` IS positional (`(pos + 1) / ratio`), and every compressed
    ///     row a boundary wrote inside the accepted prefix was computed from
    ///     real tokens, so those rows are valid and we keep exactly them.
    ///   - `state_kv` / `state_score` are restored to their pre-verify values.
    ///     That is exact when the accepted prefix ends ON a segment boundary
    ///     (the open segment is empty either way) and approximate otherwise, by
    ///     at most the `ratio - 1` positions of one open segment — which affects
    ///     only the NEXT compressed row. The alternative, keeping the batch's
    ///     accumulator, is wrong by the REJECTED rows, which is strictly worse.
    ///     Exactness here needs a per-row accumulator snapshot, which the
    ///     batched compressor (one launch for the whole chunk) cannot provide.
    pub fn advanced_by(&self, keep: u32, abs_pos: u32) -> Self {
        // SLIDING WINDOW, not a growing count. Decode keeps a monotonic append
        // region of size SWA_WINDOW + B_MAX: the append pointer is `off + nr`,
        // and the live window is the LAST min(count, SWA_WINDOW) rows. Growing
        // `nr` past SWA_WINDOW (the old behaviour) let attention read beyond the
        // window and eventually ran off the buffer, corrupting long
        // speculative generations. Advance the pointer by `keep`, then re-derive
        // (nr, off) as the trailing window — no byte movement, because the kept
        // rows already sit at `[off+nr .. off+nr+keep)` from the verify append.
        // Both vectors are indexed BY LAYER below (`COMPRESS_RATIOS[layer]`
        // reads the second one by the first one's index space), so they must
        // describe the same layer list.
        assert_eq!(
            self.per_layer.len(),
            self.per_layer_comp.len(),
            "KvMark::advanced_by: per_layer has {} entries but per_layer_comp has {}; both are \
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
                    "KvMark::advanced_by: append pointer {end} (raw_off {off} + n_raw {nr} + \
                     keep {keep}) exceeds raw KV capacity {KV_CACHE_ROWS}"
                );
                let new_nr = end.min(SWA_WINDOW);
                (new_nr, end - new_nr)
            })
            .collect();
        let per_layer_comp = self
            .per_layer_comp
            .iter()
            .enumerate()
            .map(|(layer, cm)| {
                let cm = cm.as_ref()?;
                let ratio = crate::config::COMPRESS_RATIOS[layer];
                let n_comp = if ratio == 0 {
                    None
                } else {
                    // Rows for absolute positions [0, abs_pos + keep).
                    Some((abs_pos + keep) / ratio)
                };
                let bump = |m: &CompStateMark| {
                    let mut m = m.clone();
                    if let Some(n) = n_comp {
                        // Never go backwards past what the mark already held,
                        // and never past what the batch actually wrote.
                        let before = m.n_comp;
                        let before_index = m.n_index_comp;
                        m.n_comp = n.max(before).min(before + keep);
                        // `index_k` rows are indexed the same way `comp_kv` rows
                        // are, so `n_index_comp` advances IN LOCKSTEP with
                        // `n_comp` -- decode does `cs.n_index_comp += 1` beside
                        // its comp_kv append, and batched prefill sets
                        // `n_comp_start + n_boundaries`. Clamping it to the new
                        // `n_comp` (what this did) advances the main store while
                        // leaving the indexer at its PRE-VERIFY count, so every
                        // boundary that fired inside the kept rows was written
                        // but not counted. The indexer then selects over fewer
                        // rows than decode would, which changes attention and
                        // compounds across steps.
                        let delta = m.n_comp - before;
                        m.n_index_comp = if m.n_index_comp == 0 {
                            // The indexer is not storing at all (`V41_INDEX_K`
                            // off, the default) -- it must stay at zero.
                            0
                        } else if m.n_index_comp == before {
                            m.n_comp
                        } else {
                            (m.n_index_comp + delta).min(m.n_comp)
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
                            "KvMark::advanced_by: n_index_comp {} > n_comp {} after keep={keep} \
                             (before: n_comp {before}, n_index_comp {before_index})",
                            m.n_index_comp,
                            m.n_comp
                        );
                        assert!(
                            m.n_index_comp >= before_index,
                            "KvMark::advanced_by: n_index_comp went BACKWARDS {before_index} -> \
                             {} while n_comp went {before} -> {} (keep={keep})",
                            m.n_index_comp,
                            m.n_comp
                        );
                    }
                    m
                };
                Some(CompMark {
                    main: cm.main.as_ref().map(&bump),
                    indexer: cm.indexer.as_ref().map(&bump),
                })
            })
            .collect();
        Self { per_layer, per_layer_comp, slid: true }
    }
}

impl CompStateMark {
    fn capture(cs: &HetCompressorState) -> eyre::Result<Self> {
        let mut state_kv = vec![0f32; cs.state_kv.len()];
        let mut state_score = vec![0f32; cs.state_score.len()];
        cs.state_kv.copy_to_host(&mut state_kv)?;
        cs.state_score.copy_to_host(&mut state_score)?;
        Ok(Self { n_comp: cs.n_comp, n_index_comp: cs.n_index_comp, state_kv, state_score })
    }

    fn restore(&self, cs: &mut HetCompressorState) -> eyre::Result<()> {
        cs.n_comp = self.n_comp;
        cs.n_index_comp = self.n_index_comp;
        cs.state_kv.copy_from_host(&self.state_kv)?;
        cs.state_score.copy_from_host(&self.state_score)?;
        Ok(())
    }
}

pub struct HetModelState {
    pub layers: Vec<HetLayerState>,
    pub n_kv_max: u32,
}

impl HetModelState {
    /// Reset the state in place so it's equivalent to a freshly-`alloc`ed
    /// state of the same dimensions — without re-creating the underlying
    /// device buffers. Used by the server's KV-cache management to wipe
    /// a live conversation before reloading from disk or starting fresh.
    ///
    /// Re-initialises compressor scratch to match `HetCompressorState::alloc`
    /// (state_kv→0, state_score→NEG_INF). The raw `kv_cache` and the
    /// cumulative `comp_kv` buffers don't need clearing — the per-layer
    /// kernels never read past `n_raw` / `n_comp` slots, so zeroing the
    /// counters is sufficient.
    pub fn reset_in_place(&mut self, dgpu_device: Device, igpu_device: Device) -> eyre::Result<()> {
        for layer in &mut self.layers {
            layer.n_raw = 0;
            layer.raw_off = 0;
            if let Some(comp) = &mut layer.compressor {
                comp.n_comp = 0;
                comp.n_index_comp = 0;
                let n_state = comp.state_kv.len();
                let zeros = vec![0f32; n_state];
                let neg_inf = vec![NEG_INF; n_state];
                igpu_device.set_current()?;
                comp.state_kv.copy_from_host(&zeros)?;
                comp.state_score.copy_from_host(&neg_inf)?;
            }
            // Indexer compressor (ratio==4 only) uses the same reset shape.
            if let Some(comp) = &mut layer.indexer_compressor {
                comp.n_comp = 0;
                comp.n_index_comp = 0;
                let n_state = comp.state_kv.len();
                let zeros = vec![0f32; n_state];
                let neg_inf = vec![NEG_INF; n_state];
                igpu_device.set_current()?;
                comp.state_kv.copy_from_host(&zeros)?;
                comp.state_score.copy_from_host(&neg_inf)?;
            }
        }
        // `V41_RESET_ZERO=1`: also wipe the cumulative KV buffers.
        //
        // The claim above ("don't need wiping — the extent is gated by n_raw /
        // n_comp") is what this flag TESTS. Measured 2026-09-14: a 37-token
        // request poisons a later 104K request, deterministically, and the
        // corruption CHANGES with the poisoner's prefill size (37-tok ->
        // sha 5799afaa4959, 32K -> 6d1dbafa35e6) while being INDEPENDENT of its
        // decode length. The two runs also agree for their first ~11 generated
        // tokens, so the error is SMALL and compounding, not gross. That is the
        // signature of a kernel reading a little past n_comp into tile padding:
        // harmless when those rows are freshly-allocated zeros, poisonous when
        // they hold the previous request's values.
        //
        // If this flag makes the corruption vanish, the gating claim is false
        // somewhere and the real fix is to find the over-read (this memset is
        // ~3.9 GB at --ctx 130688, far too expensive to keep on).
        if std::env::var("V41_RESET_ZERO").as_deref() == Ok("1") {
            dgpu_device.set_current()?;
            for layer in &mut self.layers {
                layer.kv_cache.fill_zero()?;
                for comp in [layer.compressor.as_mut(), layer.indexer_compressor.as_mut()]
                    .into_iter()
                    .flatten()
                {
                    match &mut comp.comp_kv {
                        CompKvStore::F16(b) => b.fill_zero()?,
                        CompKvStore::Fp8 { rows, head } => {
                            rows.fill_zero()?;
                            head.fill_zero()?;
                        }
                        CompKvStore::E2m1(b) => b.fill_zero()?,
                    }
                }
            }
        }
        // Set the current device back to dGPU to leave the engine in the
        // expected state for the next forward pass.
        dgpu_device.set_current()?;
        Ok(())
    }

    pub fn alloc(dgpu_device: Device, _igpu_device: Device, n_kv_max: u32) -> eyre::Result<Self> {
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for layer in 0..N_LAYER {
            let ratio = COMPRESS_RATIOS[layer as usize];
            // V4.1 reuse layers borrow their source's state at forward time
            // (`HetModelState::with_kv_source`), so they allocate none.
            let compressor = if ratio > 0 && crate::config::kv_source_of(layer as usize).is_none() {
                // Attn compressor state lives on dGPU alongside attn_input_norm
                // (no peer push needed for the boundary `comp_row` write).
                Some(HetCompressorState::alloc(
                    dgpu_device,
                    dgpu_device,
                    ratio,
                    N_HEAD_DIM,
                    n_kv_max,
                )?)
            } else {
                None
            };
            // CSA indexer compressor — second compressor with head_dim=128,
            // only on ratio==4 layers. State layout / lifecycle identical to
            // the main compressor; allocator reused via head_dim parameter.
            let indexer_compressor = if ratio == 4 {
                Some(HetCompressorState::alloc(
                    dgpu_device,
                    dgpu_device,
                    ratio,
                    N_INDEXER_HEAD_DIM,
                    n_kv_max,
                )?)
            } else {
                None
            };
            // Raw KV cache is sized SWA_WINDOW + B_MAX rows. The first
            // SWA_WINDOW slots hold the steady-state SWA-window contents
            // (which is all that decode/single-token attention sees).
            // During batched prefill we additionally use slots
            // [SWA_WINDOW .. SWA_WINDOW + chunk_b) for the chunk's
            // freshly-computed KVs so that each token's per-token
            // n_raw_offset_per can see its causally-valid window across
            // the prior+current boundary. After each chunk the last
            // SWA_WINDOW rows are evicted back down to slot [0..W).
            dgpu_device.set_current()?;
            let raw_rows = KV_CACHE_ROWS as u32;
            layers.push(HetLayerState {
                kv_cache: DeviceBuffer::<u16>::new(
                    dgpu_device.id,
                    (raw_rows as usize) * (N_HEAD_DIM as usize),
                )?,
                n_raw: 0,
                raw_off: 0,
                compressor,
                indexer_compressor,
            });
        }
        Ok(Self {
            layers,
            n_kv_max,
        })
    }
}
