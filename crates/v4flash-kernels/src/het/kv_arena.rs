//! Multi-stream KV arena (docs/v41/MULTISTREAM_DECODE_PLAN.md 3.2 / 3.7).
//!
//! `HetModelState` holds ONE sequence's KV. For S live streams the batched
//! kernels want every row's KV somewhere in ONE allocation per store, addressed
//! by per-row base arrays (the `*_rows` wrappers; `tests/multistream_row_bases.rs`
//! shows them bit-identical to the single-sequence forms). This module owns:
//!
//!   * per layer, one RAW SWA window buffer of `n_slots * KV_CACHE_ROWS` rows —
//!     stream `s` owns rows `[s*KV_CACHE_ROWS, (s+1)*KV_CACHE_ROWS)`, and its
//!     live window is `[raw_off, raw_off + n_raw)` inside that region exactly as
//!     `HetLayerState` keeps it (monotonic append, compaction at the region end);
//!   * per KV-SOURCE layer (`config::KV_SOURCE_LAYERS`), one compressed store:
//!     `comp_kv` (f16 rows), `index_k` (packed E2M1 keys, one per comp row) and
//!     the compressor accumulators (`ratio * width` floats per stream), with a
//!     first-fit region allocator over comp rows;
//!   * per stream, the counters the kernels' tables are derived from.
//!
//! The buffers live inside a `HetModelState` (`KvArena::state`): layer `l`'s
//! `kv_cache` IS the arena's raw buffer for that layer, and the four KV-source
//! layers carry a `HetCompressorState` whose `state_kv`/`state_score` hold every
//! stream's accumulator block and whose `comp_kv`/`index_k` are the shared
//! stores. That is what lets the batched layer driver
//! (`forward_layer_pre_moe_v2`) run S streams through the SAME `&mut
//! HetLayerState` argument it runs one sequence through, with a
//! `RowLayout::Arena` telling it to take bases and counts from the per-row
//! tables instead of the state's scalars. The state's own counters (`n_raw`,
//! `raw_off`, `n_comp`, `n_index_comp`) are meaningless in the arena and stay 0;
//! `with_kv_source` lends the store to the reuse layers exactly as for one
//! sequence.
//!
//! The raw counters are kept ONCE per stream, not per layer: decode advances
//! every layer's window in lockstep (`forward_token_impl` takes the slot from
//! layer 0 for that reason), and a multi-stream step keeps that invariant.
//!
//! Nothing here launches a kernel except the region compaction copy; the
//! tables are plain host vectors (`RowTables`) the step uploads once into a
//! `RowTablesDev`.
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer, Stream};

use crate::config::{
    kv_source_of, COMPRESS_RATIOS, KV_SOURCE_LAYERS, NEG_INF, N_HEAD_DIM, N_LAYER, SWA_WINDOW,
};
use crate::het::state::{CompKvStore, HetCompressorState, HetLayerState, HetModelState, KV_CACHE_ROWS};
use crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES;

/// Index into `KvArena::stores` / `RowTables::stores` of the store `layer`
/// reads (its KV source's), `None` for the dense layers.
pub fn store_index_of(layer: usize) -> Option<usize> {
    let src = kv_source_of(layer)?;
    KV_SOURCE_LAYERS.iter().position(|&l| l as usize == src)
}

/// One stream's region in one compressed store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompRegion {
    /// First row of the region (rows of `comp_kv` / `index_k`).
    pub base: u32,
    /// Rows reserved.
    pub cap: u32,
    /// Rows written so far (`HetCompressorState::n_comp`).
    pub n_comp: u32,
    /// Index keys written so far (`n_index_comp`); tracks `n_comp`.
    pub n_index_comp: u32,
}

/// One live stream's KV counters.
#[derive(Clone, Debug)]
pub struct StreamKv {
    /// Next token's position (the KV position of the row this stream will run).
    pub pos: u32,
    /// Live raw window inside the stream's raw region, per `HetLayerState`.
    pub raw_off: u32,
    pub n_raw: u32,
    /// One region per KV-source store, in `KV_SOURCE_LAYERS` order.
    pub comp: Vec<CompRegion>,
}

/// First-fit row allocator with a sorted, coalesced free list.
#[derive(Clone, Debug)]
pub struct RowFreeList {
    free: Vec<(u32, u32)>,
}

impl RowFreeList {
    pub fn new(rows: u32) -> Self {
        Self { free: if rows > 0 { vec![(0, rows)] } else { Vec::new() } }
    }
    pub fn carve(&mut self, rows: u32) -> Option<u32> {
        let i = self.free.iter().position(|&(_, len)| len >= rows)?;
        let (base, len) = self.free[i];
        if len == rows {
            self.free.remove(i);
        } else {
            self.free[i] = (base + rows, len - rows);
        }
        Some(base)
    }
    pub fn give_back(&mut self, base: u32, rows: u32) {
        self.free.push((base, rows));
        self.free.sort_unstable();
        let mut out: Vec<(u32, u32)> = Vec::with_capacity(self.free.len());
        for &(b, l) in &self.free {
            if let Some(last) = out.last_mut() {
                if last.0 + last.1 == b {
                    last.1 += l;
                    continue;
                }
            }
            out.push((b, l));
        }
        self.free = out;
    }
    pub fn fits(&self, rows: u32) -> bool {
        self.free.iter().any(|&(_, l)| l >= rows)
    }
    pub fn free_rows(&self) -> u32 {
        self.free.iter().map(|&(_, l)| l).sum()
    }
    pub fn largest_run(&self) -> u32 {
        self.free.iter().map(|&(_, l)| l).max().unwrap_or(0)
    }
}

/// One compressed store (one KV-source layer) for all streams: allocator +
/// geometry. Its buffers are `KvArena::state.layers[layer].compressor`.
pub struct CompStore {
    pub layer: usize,
    pub ratio: u32,
    pub width: u32,
    pub rows_cap: u32,
    pub free: RowFreeList,
}

/// Per-row tables for one step, in the order the rows were given. Host
/// vectors; the step uploads them once (`RowTablesDev::upload`). Names match
/// the kernel parameters. Every value is PRE-step: the row's stream has
/// `n_raw` raw rows and `n_comp` comp rows before this row runs; the driver
/// derives the causal counts (`min(n_raw + 1, SWA_WINDOW)`, `n_comp + fires`).
#[derive(Clone, Debug, Default)]
pub struct RowTables {
    pub pos_per: Vec<i32>,
    /// Raw window per row BEFORE the append: rows valid and the window start in
    /// the LAYER buffer (`slot * KV_CACHE_ROWS + raw_off`), the same for every
    /// layer.
    pub n_raw_per: Vec<i32>,
    pub n_raw_offset_per: Vec<i32>,
    /// Raw append destination per row (`window start + n_raw`), every layer.
    pub slot_per: Vec<i32>,
    /// One entry per KV-source store, `KV_SOURCE_LAYERS` order.
    pub stores: Vec<StoreTables>,
}

#[derive(Clone, Debug, Default)]
pub struct StoreTables {
    /// Comp rows written before this step (the row's own boundary, if it
    /// fires this step, is NOT counted).
    pub n_comp_per: Vec<i32>,
    pub comp_base_per: Vec<i32>,
    /// Same rows as `comp_base_per` (index keys parallel the comp rows), as
    /// the u32 the score kernel takes.
    pub keys_base_per: Vec<u32>,
    /// Float-element base of the row's accumulator block; and the block index
    /// for the pool kernel.
    pub state_base_per: Vec<i32>,
    pub state_idx_per: Vec<i32>,
    /// Rows whose compressor boundary FIRES this step (`(pos + 1) % ratio == 0`),
    /// as row indices into the batch, with — per firing row, same order — the
    /// stream's accumulator block (`state_idx_per[row]`), the comp destination
    /// row (`base + n_comp`) and the boundary's position (`pos + 1 - ratio`,
    /// what decode ropes the pooled row at: forward_layer.rs `comp_pos`).
    pub fire_rows: Vec<i32>,
    pub fire_state_idx: Vec<i32>,
    pub fire_dst_row: Vec<i32>,
    pub fire_comp_pos: Vec<i32>,
}

/// Device copies of the per-row base arrays the `*_rows` launches take.
/// Allocated once for `rows_cap` rows; `upload` refills the used prefix.
pub struct StoreTablesDev {
    pub comp_base_per: DeviceBuffer<i32>,
    pub keys_base_per: DeviceBuffer<u32>,
    pub state_base_per: DeviceBuffer<i32>,
    pub fire_state_idx: DeviceBuffer<i32>,
    pub fire_dst_row: DeviceBuffer<i32>,
}

pub struct RowTablesDev {
    pub rows_cap: u32,
    pub slot_per: DeviceBuffer<i32>,
    pub stores: Vec<StoreTablesDev>,
}

impl RowTablesDev {
    pub fn alloc(dgpu: Device, rows_cap: u32, n_stores: usize) -> eyre::Result<Self> {
        dgpu.set_current()?;
        let n = rows_cap.max(1) as usize;
        let mut stores = Vec::with_capacity(n_stores);
        for _ in 0..n_stores {
            stores.push(StoreTablesDev {
                comp_base_per: DeviceBuffer::<i32>::new(dgpu.id, n)?,
                keys_base_per: DeviceBuffer::<u32>::new(dgpu.id, n)?,
                state_base_per: DeviceBuffer::<i32>::new(dgpu.id, n)?,
                fire_state_idx: DeviceBuffer::<i32>::new(dgpu.id, n)?,
                fire_dst_row: DeviceBuffer::<i32>::new(dgpu.id, n)?,
            });
        }
        Ok(Self { rows_cap, slot_per: DeviceBuffer::<i32>::new(dgpu.id, n)?, stores })
    }

    /// Async copies on `stream`, so they FIFO ahead of the step's launches.
    pub fn upload(&mut self, t: &RowTables, stream: &Stream) -> eyre::Result<()> {
        let b = t.pos_per.len();
        if b > self.rows_cap as usize {
            return Err(eyre!("row tables: {b} rows > capacity {}", self.rows_cap));
        }
        if t.stores.len() != self.stores.len() {
            return Err(eyre!("row tables: {} stores, device has {}", t.stores.len(), self.stores.len()));
        }
        fn up<T: Copy>(dst: &mut DeviceBuffer<T>, src: &[T], stream: &Stream) -> eyre::Result<()> {
            if !src.is_empty() {
                let mut v = dst.slice_view_mut(0, src.len());
                v.copy_from_host_async(src, stream)?;
            }
            Ok(())
        }
        up(&mut self.slot_per, &t.slot_per, stream)?;
        for (d, h) in self.stores.iter_mut().zip(&t.stores) {
            up(&mut d.comp_base_per, &h.comp_base_per, stream)?;
            up(&mut d.keys_base_per, &h.keys_base_per, stream)?;
            up(&mut d.state_base_per, &h.state_base_per, stream)?;
            up(&mut d.fire_state_idx, &h.fire_state_idx, stream)?;
            up(&mut d.fire_dst_row, &h.fire_dst_row, stream)?;
        }
        Ok(())
    }
}

pub struct KvArena {
    pub dgpu: Device,
    pub n_slots: u32,
    /// The buffers, as the layer driver takes them (module doc). Layer `l`:
    /// `kv_cache` = `n_slots * KV_CACHE_ROWS * N_HEAD_DIM` f16; KV-source
    /// layers: `compressor = Some(..)` with `state_kv`/`state_score` =
    /// `[n_slots, ratio * width]` f32, `comp_kv = F16([rows_cap, width])`,
    /// `index_k = Some([rows_cap, E2M1_KEY_ROW_BYTES])`. Counters stay 0.
    pub state: HetModelState,
    pub stores: Vec<CompStore>,
    streams: Vec<Option<StreamKv>>,
}

impl KvArena {
    /// `comp_rows_cap`: rows per compressed store shared by all streams (a
    /// stream at context `c` needs `ceil(c / ratio)` rows in each store).
    pub fn alloc(dgpu: Device, n_slots: u32, comp_rows_cap: u32) -> eyre::Result<Self> {
        if n_slots == 0 {
            return Err(eyre!("kv arena: n_slots must be >= 1"));
        }
        dgpu.set_current()?;
        let raw_rows = (n_slots as usize) * KV_CACHE_ROWS;
        let mut stores = Vec::with_capacity(KV_SOURCE_LAYERS.len());
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for l in 0..N_LAYER as usize {
            let kv_cache = DeviceBuffer::<u16>::new(dgpu.id, raw_rows * N_HEAD_DIM as usize)?;
            let compressor = if KV_SOURCE_LAYERS.iter().any(|&s| s as usize == l) {
                let ratio = COMPRESS_RATIOS[l];
                if ratio == 0 || ratio == 4 {
                    // ratio 4 is V4-Flash's FP8/shuffle shape; the arena's f16
                    // store + `ratio * width` accumulators are V4.1's (1, 2).
                    return Err(eyre!("kv arena: KV-source layer {l} has ratio {ratio} (need 1 or 2)"));
                }
                let width = N_HEAD_DIM;
                let n_state = (n_slots as usize) * (ratio * width) as usize;
                let mut state_kv = DeviceBuffer::<f32>::new(dgpu.id, n_state)?;
                let mut state_score = DeviceBuffer::<f32>::new(dgpu.id, n_state)?;
                state_kv.copy_from_host(&vec![0f32; n_state])?;
                state_score.copy_from_host(&vec![NEG_INF; n_state])?;
                stores.push(CompStore {
                    layer: l,
                    ratio,
                    width,
                    rows_cap: comp_rows_cap,
                    free: RowFreeList::new(comp_rows_cap),
                });
                Some(HetCompressorState {
                    state_kv,
                    state_score,
                    comp_kv: CompKvStore::F16(DeviceBuffer::<u16>::new(
                        dgpu.id,
                        (comp_rows_cap as usize) * width as usize,
                    )?),
                    n_comp: 0,
                    width,
                    head_dim: N_HEAD_DIM,
                    index_k: Some(DeviceBuffer::<u8>::new(
                        dgpu.id,
                        (comp_rows_cap as usize) * E2M1_KEY_ROW_BYTES,
                    )?),
                    n_index_comp: 0,
                })
            } else {
                None
            };
            layers.push(HetLayerState { kv_cache, n_raw: 0, raw_off: 0, compressor, indexer_compressor: None });
        }
        let state = HetModelState { layers, n_kv_max: comp_rows_cap };
        Ok(Self { dgpu, n_slots, state, stores, streams: vec![None; n_slots as usize] })
    }

    pub fn stream(&self, slot: u32) -> Option<&StreamKv> {
        self.streams.get(slot as usize).and_then(|s| s.as_ref())
    }
    pub fn stream_mut(&mut self, slot: u32) -> Option<&mut StreamKv> {
        self.streams.get_mut(slot as usize).and_then(|s| s.as_mut())
    }
    pub fn live(&self) -> usize {
        self.streams.iter().filter(|s| s.is_some()).count()
    }

    /// Admit a stream that will run up to `ctx_cap` positions: a free slot plus
    /// `ceil(ctx_cap / ratio)` rows in every store. Fails without touching
    /// anything if any store cannot fit it (first-fit, no compaction).
    pub fn admit(&mut self, ctx_cap: u32, pos0: u32) -> eyre::Result<u32> {
        let slot = self
            .streams
            .iter()
            .position(|s| s.is_none())
            .ok_or_else(|| eyre!("kv arena: all {} slots live", self.n_slots))? as u32;
        let need: Vec<u32> = self.stores.iter().map(|st| ctx_cap.div_ceil(st.ratio).max(1)).collect();
        if let Some((i, _)) = self.stores.iter().enumerate().find(|(i, st)| !st.free.fits(need[*i])) {
            return Err(eyre!(
                "kv arena: store L{} cannot fit {} rows (free {}, largest run {})",
                self.stores[i].layer,
                need[i],
                self.stores[i].free.free_rows(),
                self.stores[i].free.largest_run()
            ));
        }
        let mut comp = Vec::with_capacity(self.stores.len());
        for (i, st) in self.stores.iter_mut().enumerate() {
            let base = st.free.carve(need[i]).expect("checked above");
            comp.push(CompRegion { base, cap: need[i], n_comp: 0, n_index_comp: 0 });
        }
        self.streams[slot as usize] = Some(StreamKv { pos: pos0, raw_off: 0, n_raw: 0, comp });
        Ok(slot)
    }

    /// Admit a stream whose prompt was prefilled into `src` (a single-sequence
    /// state, compressors at rest on their source layers) and copy its KV in:
    /// every layer's live raw window to the start of the slot's region, each
    /// store's comp rows / index keys to the carved region and its accumulator
    /// block to the slot's. `pos` is the NEXT position (the prompt length).
    /// D2D copies on `stream`; the caller synchronizes before reading.
    pub fn admit_from_state(
        &mut self,
        src: &HetModelState,
        ctx_cap: u32,
        pos: u32,
        stream: &Stream,
    ) -> eyre::Result<u32> {
        if src.layers.len() != self.state.layers.len() {
            return Err(eyre!("kv arena: source state has {} layers, arena {}", src.layers.len(), self.state.layers.len()));
        }
        let (n_raw, raw_off) = (src.layers[0].n_raw, src.layers[0].raw_off);
        if src.layers.iter().any(|l| l.n_raw != n_raw) {
            return Err(eyre!("kv arena: source state's raw windows are not in lockstep"));
        }
        if n_raw > SWA_WINDOW || pos < n_raw {
            return Err(eyre!("kv arena: source window {n_raw} rows at pos {pos}"));
        }
        let slot = self.admit(ctx_cap.max(pos + 1), pos)?;
        let hd = N_HEAD_DIM as usize;
        self.dgpu.set_current()?;
        let region = Self::raw_region_base(slot) as usize;
        for (dst, s) in self.state.layers.iter_mut().zip(&src.layers) {
            if s.n_raw == 0 {
                continue;
            }
            let win = s.n_raw as usize * hd;
            let sv = s.kv_cache.slice_view(s.raw_off as usize * hd, win);
            let mut dv = dst.kv_cache.slice_view_mut(region * hd, win);
            dv.copy_from_buffer_async(&sv, stream)?;
        }
        let _ = raw_off;
        let mut comp = Vec::with_capacity(self.stores.len());
        for (si, st) in self.stores.iter().enumerate() {
            let l = st.layer;
            let scs = src.layers[l].compressor.as_ref().ok_or_else(|| {
                eyre!("kv arena: source state has no compressor at rest on L{l} (lent to a reuse layer?)")
            })?;
            let dcs = self.state.layers[l].compressor.as_mut().expect("arena store");
            let region = self.streams[slot as usize].as_ref().expect("just admitted").comp[si];
            if scs.n_comp > region.cap || scs.n_index_comp > scs.n_comp {
                return Err(eyre!(
                    "kv arena: L{l} source has {} comp rows ({} keys), region holds {}",
                    scs.n_comp, scs.n_index_comp, region.cap
                ));
            }
            let width = st.width as usize;
            if scs.width != st.width {
                return Err(eyre!("kv arena: L{l} source width {} != arena {}", scs.width, st.width));
            }
            if scs.n_comp > 0 {
                let sbuf = scs.comp_kv.f16().ok_or_else(|| eyre!("kv arena: L{l} source store is not f16"))?;
                let dbuf = dcs.comp_kv.f16_mut().expect("arena store is f16");
                let n = scs.n_comp as usize * width;
                let mut dv = dbuf.slice_view_mut(region.base as usize * width, n);
                dv.copy_from_buffer_async(&sbuf.slice_view(0, n), stream)?;
            }
            if scs.n_index_comp > 0 {
                let sk = scs.index_k.as_ref().ok_or_else(|| eyre!("kv arena: L{l} source has no index keys"))?;
                let dk = dcs.index_k.as_mut().expect("arena store has keys");
                let n = scs.n_index_comp as usize * E2M1_KEY_ROW_BYTES;
                let mut dv = dk.slice_view_mut(region.base as usize * E2M1_KEY_ROW_BYTES, n);
                dv.copy_from_buffer_async(&sk.slice_view(0, n), stream)?;
            }
            let block = (st.ratio * st.width) as usize;
            if scs.state_kv.len() != block {
                return Err(eyre!("kv arena: L{l} source accumulator has {} floats, arena block {block}", scs.state_kv.len()));
            }
            {
                let mut dv = dcs.state_kv.slice_view_mut(slot as usize * block, block);
                dv.copy_from_buffer_async(&scs.state_kv.slice_view(0, block), stream)?;
                let mut dv = dcs.state_score.slice_view_mut(slot as usize * block, block);
                dv.copy_from_buffer_async(&scs.state_score.slice_view(0, block), stream)?;
            }
            comp.push(CompRegion { n_comp: scs.n_comp, n_index_comp: scs.n_index_comp, ..region });
        }
        let s = self.streams[slot as usize].as_mut().expect("just admitted");
        s.n_raw = n_raw;
        s.raw_off = 0;
        s.comp = comp;
        Ok(slot)
    }

    pub fn release(&mut self, slot: u32) -> eyre::Result<()> {
        let s = self.streams.get_mut(slot as usize).and_then(|s| s.take())
            .ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
        for (st, r) in self.stores.iter_mut().zip(&s.comp) {
            st.free.give_back(r.base, r.cap);
        }
        Ok(())
    }

    /// Row start of `slot`'s raw region in every layer buffer, in rows.
    pub fn raw_region_base(slot: u32) -> u32 {
        slot * KV_CACHE_ROWS as u32
    }

    /// The step's tables for `slots` (one row per slot, in order). Every row is
    /// at its stream's `pos`; draft rows (K>1) are the caller's business.
    pub fn tables(&self, slots: &[u32]) -> eyre::Result<RowTables> {
        let mut t = RowTables { stores: vec![StoreTables::default(); self.stores.len()], ..Default::default() };
        for (b, &slot) in slots.iter().enumerate() {
            // K=1 in v1: one row per stream (two rows of one stream would
            // write the same accumulator block / append slot in one launch).
            if slots[..b].contains(&slot) {
                return Err(eyre!("kv arena: slot {slot} appears twice in the step"));
            }
            let s = self.stream(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
            let region = Self::raw_region_base(slot);
            t.pos_per.push(s.pos as i32);
            t.n_raw_per.push(s.n_raw as i32);
            t.n_raw_offset_per.push((region + s.raw_off) as i32);
            t.slot_per.push((region + s.raw_off + s.n_raw) as i32);
            for (si, st) in self.stores.iter().enumerate() {
                let r = s.comp[si];
                let ts = &mut t.stores[si];
                ts.n_comp_per.push(r.n_comp as i32);
                ts.comp_base_per.push(r.base as i32);
                ts.keys_base_per.push(r.base);
                ts.state_base_per.push((slot * st.ratio * st.width) as i32);
                ts.state_idx_per.push(slot as i32);
                if (s.pos + 1) % st.ratio == 0 {
                    if r.n_comp >= r.cap {
                        return Err(eyre!(
                            "kv arena: slot {slot} store L{} region full ({} rows) at pos {}",
                            st.layer, r.cap, s.pos
                        ));
                    }
                    ts.fire_rows.push(b as i32);
                    ts.fire_state_idx.push(slot as i32);
                    ts.fire_dst_row.push((r.base + r.n_comp) as i32);
                    ts.fire_comp_pos.push((s.pos + 1 - st.ratio) as i32);
                }
            }
        }
        Ok(t)
    }

    /// True when the next append of `slot` would run off its raw region: the
    /// caller must `compact_raw` first (a D2D copy per layer, on `stream`).
    pub fn needs_compaction(&self, slot: u32) -> bool {
        self.stream(slot).is_some_and(|s| (s.raw_off + s.n_raw) as usize >= KV_CACHE_ROWS)
    }

    /// Move `slot`'s live window to the start of its region in every layer, the
    /// same two-hop copy `forward_layer` and `normalize_raw_windows` do, using
    /// `scratch` (>= `SWA_WINDOW * N_HEAD_DIM` f16) as the bounce buffer.
    pub fn compact_raw(&mut self, slot: u32, stream: &Stream, scratch: &mut DeviceBuffer<u16>) -> eyre::Result<()> {
        let (raw_off, n_raw) = {
            let s = self.stream(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
            (s.raw_off, s.n_raw)
        };
        if raw_off == 0 || n_raw == 0 {
            if let Some(s) = self.stream_mut(slot) { s.raw_off = 0; }
            return Ok(());
        }
        let hd = N_HEAD_DIM as usize;
        let win = n_raw as usize * hd;
        if scratch.len() < win {
            return Err(eyre!("kv arena: compaction scratch {} < window {}", scratch.len(), win));
        }
        let region = Self::raw_region_base(slot) as usize * hd;
        self.dgpu.set_current()?;
        for ls in self.state.layers.iter_mut() {
            let buf = &mut ls.kv_cache;
            {
                let src = buf.slice_view(region + raw_off as usize * hd, win);
                let mut sc = scratch.slice_view_mut(0, win);
                sc.copy_from_buffer_async(&src, stream)?;
            }
            {
                let sc = scratch.slice_view(0, win);
                let mut dst = buf.slice_view_mut(region, win);
                dst.copy_from_buffer_async(&sc, stream)?;
            }
        }
        if let Some(s) = self.stream_mut(slot) { s.raw_off = 0; }
        Ok(())
    }

    /// Advance `slot` by one appended token: the raw window (monotonic append
    /// with the SWA_WINDOW cap, as `forward_layer` keeps it), each store's
    /// counters where its boundary fired at this position, and `pos`.
    pub fn advance(&mut self, slot: u32) -> eyre::Result<()> {
        let ratios: Vec<u32> = self.stores.iter().map(|st| st.ratio).collect();
        let s = self.stream_mut(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
        if s.n_raw < SWA_WINDOW {
            s.n_raw += 1;
        } else {
            s.raw_off += 1;
        }
        for (r, ratio) in s.comp.iter_mut().zip(ratios) {
            if (s.pos + 1) % ratio == 0 {
                r.n_comp += 1;
                r.n_index_comp += 1;
            }
        }
        s.pos += 1;
        Ok(())
    }

    /// Restore `slot` to a saved copy of its counters (speculative rollback;
    /// the accumulators are the caller's, per `KvMark::per_layer_comp`).
    pub fn restore(&mut self, slot: u32, saved: &StreamKv) -> eyre::Result<()> {
        let s = self.stream_mut(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
        if s.comp.iter().zip(&saved.comp).any(|(a, b)| a.base != b.base || a.cap != b.cap) {
            return Err(eyre!("kv arena: restore of slot {slot} with different regions"));
        }
        *s = saved.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_list_first_fit_and_coalesce() {
        let mut fl = RowFreeList::new(100);
        assert_eq!(fl.carve(30), Some(0));
        assert_eq!(fl.carve(30), Some(30));
        assert_eq!(fl.carve(50), None);
        fl.give_back(0, 30);
        assert_eq!(fl.free, vec![(0, 30), (60, 40)]);
        assert_eq!(fl.carve(40), Some(60));
        fl.give_back(30, 30);
        assert_eq!(fl.free, vec![(0, 60)]);
        assert_eq!(fl.free_rows(), 60);
        assert!(fl.fits(60) && !fl.fits(61));
    }
}
