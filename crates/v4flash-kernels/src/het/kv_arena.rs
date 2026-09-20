//! Multi-stream KV arena (docs/v41/MULTISTREAM_DECODE_PLAN.md 3.2).
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
//! The raw counters are kept ONCE per stream, not per layer: decode advances
//! every layer's window in lockstep (`forward_token_impl` takes the slot from
//! layer 0 for that reason), and a multi-stream step keeps that invariant.
//!
//! Nothing here launches a kernel except the region compaction copy; the
//! tables are plain host vectors the step uploads once.
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer, Stream};

use crate::config::{COMPRESS_RATIOS, KV_SOURCE_LAYERS, N_HEAD_DIM, N_LAYER, SWA_WINDOW};
use crate::het::state::KV_CACHE_ROWS;
use crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES;

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

/// One compressed store (one KV-source layer) for all streams.
pub struct CompStore {
    pub layer: usize,
    pub ratio: u32,
    pub width: u32,
    pub comp_kv: DeviceBuffer<u16>,
    pub index_k: DeviceBuffer<u8>,
    /// `[n_slots, ratio * width]` running segment accumulators, one block per slot.
    pub state_kv: DeviceBuffer<f32>,
    pub state_score: DeviceBuffer<f32>,
    pub rows_cap: u32,
    pub free: RowFreeList,
}

/// Per-row tables for one step, in the order the rows were given. Host
/// vectors; the step uploads them once. Names match the kernel parameters.
#[derive(Clone, Debug, Default)]
pub struct RowTables {
    pub pos_per: Vec<i32>,
    /// Raw window per row: rows valid and the row's window start in the LAYER
    /// buffer (`slot * KV_CACHE_ROWS + raw_off`), the same for every layer.
    pub n_raw_per: Vec<i32>,
    pub n_raw_offset_per: Vec<i32>,
    /// Raw append destination per row (`window start + n_raw`), every layer.
    pub slot_per: Vec<i32>,
    /// One entry per KV-source store, `KV_SOURCE_LAYERS` order.
    pub stores: Vec<StoreTables>,
}

#[derive(Clone, Debug, Default)]
pub struct StoreTables {
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
    /// as row indices into the batch, and their comp destination rows
    /// (`base + n_comp`) — what the batched pool/append/index-k launches take.
    pub fire_rows: Vec<i32>,
    pub dst_row_per: Vec<i32>,
}

pub struct KvArena {
    pub dgpu: Device,
    pub n_slots: u32,
    /// Per layer, `n_slots * KV_CACHE_ROWS * N_HEAD_DIM` f16.
    pub raw: Vec<DeviceBuffer<u16>>,
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
        let mut raw = Vec::with_capacity(N_LAYER as usize);
        for _ in 0..N_LAYER {
            raw.push(DeviceBuffer::<u16>::new(dgpu.id, raw_rows * N_HEAD_DIM as usize)?);
        }
        let mut stores = Vec::with_capacity(KV_SOURCE_LAYERS.len());
        for &l in KV_SOURCE_LAYERS {
            let ratio = COMPRESS_RATIOS[l as usize];
            if ratio == 0 {
                return Err(eyre!("kv arena: KV-source layer {l} has ratio 0"));
            }
            let width = N_HEAD_DIM;
            let state_per = (ratio * width) as usize;
            stores.push(CompStore {
                layer: l as usize,
                ratio,
                width,
                comp_kv: DeviceBuffer::<u16>::new(dgpu.id, (comp_rows_cap as usize) * width as usize)?,
                index_k: DeviceBuffer::<u8>::new(dgpu.id, (comp_rows_cap as usize) * E2M1_KEY_ROW_BYTES)?,
                state_kv: DeviceBuffer::<f32>::new(dgpu.id, (n_slots as usize) * state_per)?,
                state_score: DeviceBuffer::<f32>::new(dgpu.id, (n_slots as usize) * state_per)?,
                rows_cap: comp_rows_cap,
                free: RowFreeList::new(comp_rows_cap),
            });
        }
        Ok(Self { dgpu, n_slots, raw, stores, streams: vec![None; n_slots as usize] })
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
                    ts.dst_row_per.push((r.base + r.n_comp) as i32);
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
        for buf in self.raw.iter_mut() {
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
