//! The prefill dGPU VRAM loan (docs/v41/PREFILL_VRAM_LOAN_DESIGN.md, rev 2 approved 2026-10-10).
//!
//! Decode runs on small resident lanes (`V41_PREFILL_VRAM_LOAN=1`: sized for the arena driver's rows); PREFILL runs on its
//! own `V41_PREFILL_LANE_ROWS` (1024) lane set -- two `BatchDgpuScratch` + one `BatchDgpuShared` -- whose buffers are
//! VIEWS into (a) a permanent pool (the memory the resident shrink freed) and (b) the END rows of the KV arena's stores,
//! LENT for the duration of prefill work ([`PrefillLoan::lend`] / [`PrefillLoan::end`]; `KvArena::lend` compacts and
//! swaps out any live decode KV that overlaps).
//!
//! The set is built ONCE with the unchanged allocation code: run under `alloc_scope::record` (phantoms: sizes in call
//! order, and which buffers the routine WRITES -- those must persist, so they go to the pool), plan first-fit-decreasing
//! (pool first, then the stores' comp_kv end ranges largest first, then their index_k end ranges), then run again under
//! `alloc_scope::place`. A size that drifts between the passes fails the build. The prefill set is never
//! `register_static`'d: no captured graph references it.
use color_eyre::eyre::{self, eyre};
use v4flash_hip::alloc_scope::{self, Recorded};
use v4flash_hip::{sys, Device, DeviceBuffer, Stream};

use super::batch_scratch::{BatchDgpuScratch, BatchDgpuShared};
use super::kv_arena::{EndStats, KvArena, LendStats};
use crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES;

/// `V41_PREFILL_VRAM_LOAN=1` (default off; STATIC: changes allocation at startup).
pub fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| matches!(std::env::var("V41_PREFILL_VRAM_LOAN").as_deref(), Ok("1") | Ok("on")));
    *ON
}

/// Free dGPU memory kept OUT of the pool (lazily instantiated stage graphs, review r1 finding 2). `V41_PREFILL_LOAN_HEADROOM_MB`.
pub fn headroom_bytes() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PREFILL_LOAN_HEADROOM_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(512) << 20
    });
    *N
}

/// Optional cap on the pool (`V41_PREFILL_LOAN_POOL_MB`; default: all the free memory minus the headroom). The GPU test
/// sets it to a production-like 2048 so the store tails are exercised with the hub down.
pub fn pool_cap_bytes() -> usize {
    std::env::var("V41_PREFILL_LOAN_POOL_MB").ok().and_then(|v| v.parse::<usize>().ok()).map_or(usize::MAX, |mb| mb << 20)
}

const ALIGN: usize = 256;

fn align_up(x: usize) -> usize {
    x.div_ceil(ALIGN) * ALIGN
}

/// Where one allocation lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bin {
    Pool,
    /// Store `si`'s `comp_kv`.
    Comp(usize),
    /// Store `si`'s `index_k`.
    Keys(usize),
}

/// A plan: per allocation its bin and BYTE offset from the bin's start (store bins are filled from their END down).
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub at: Vec<(Bin, usize)>,
    pub pool_used: usize,
    /// Per store: rows lent at the end (`max` over its comp and key bins, in rows).
    pub lent_rows: Vec<u32>,
}

/// First-fit-decreasing. `pool_cap` bytes; per store `(comp_bytes, comp_row_bytes, key_bytes, key_row_bytes)`. Written
/// allocations only go to the pool. Errors if anything does not fit.
pub fn plan(items: &[Recorded], pool_cap: usize, stores: &[(usize, usize, usize, usize)]) -> eyre::Result<Plan> {
    // Bin ENDS aligned down to 256 B (review r3 finding 5: `index_k` is `rows_cap x 80` B, not a multiple of 256), so every
    // buffer carved downward from an end starts 256-aligned.
    // The slack between the true end and the aligned end lies INSIDE the lent range (review r4 NEW-2): `lent_rows`
    // counts it, so a live row there is compacted or swapped like any other overlap.
    let slack: Vec<(usize, usize)> = stores.iter().map(|&(c, _, k, _)| (c % ALIGN, k % ALIGN)).collect();
    let stores: Vec<(usize, usize, usize, usize)> = stores.iter().map(|&(c, cr, k, kr)| (c / ALIGN * ALIGN, cr, k / ALIGN * ALIGN, kr)).collect();
    let stores = &stores[..];
    // Must-persist (written) items FIRST (review r4 NEW-1: packed after the big unwritten ones they found the pool full),
    // then first-fit-decreasing within each class.
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&i| (!items[i].written, std::cmp::Reverse(items[i].bytes), i));
    // store bins, largest comp first, then keys largest first
    let mut comp_bins: Vec<usize> = (0..stores.len()).collect();
    comp_bins.sort_by_key(|&s| (std::cmp::Reverse(stores[s].0), s));
    let mut key_bins = comp_bins.clone();
    key_bins.sort_by_key(|&s| (std::cmp::Reverse(stores[s].2), s));
    let mut pool_used = 0usize;
    let mut comp_used = vec![0usize; stores.len()];
    let mut key_used = vec![0usize; stores.len()];
    let mut at = vec![(Bin::Pool, 0usize); items.len()];
    for i in order {
        let sz = align_up(items[i].bytes.max(1));
        if pool_used + sz <= pool_cap {
            at[i] = (Bin::Pool, pool_used);
            pool_used += sz;
            continue;
        }
        if items[i].written {
            return Err(eyre!("prefill loan: allocation #{i} ({} B) must persist but the pool is full ({pool_used} of {pool_cap})", items[i].bytes));
        }
        let mut placed = false;
        for &s in &comp_bins {
            if comp_used[s] + sz <= stores[s].0 {
                comp_used[s] += sz;
                at[i] = (Bin::Comp(s), stores[s].0 - comp_used[s]);
                placed = true;
                break;
            }
        }
        if !placed {
            for &s in &key_bins {
                if key_used[s] + sz <= stores[s].2 {
                    key_used[s] += sz;
                    at[i] = (Bin::Keys(s), stores[s].2 - key_used[s]);
                    placed = true;
                    break;
                }
            }
        }
        if !placed {
            return Err(eyre!("prefill loan: allocation #{i} ({} B) fits neither the pool nor any store end range", items[i].bytes));
        }
    }
    let lent_rows = (0..stores.len())
        .map(|s| {
            let c = if comp_used[s] > 0 { comp_used[s] + slack[s].0 } else { 0 };
            let k = if key_used[s] > 0 { key_used[s] + slack[s].1 } else { 0 };
            c.div_ceil(stores[s].1).max(k.div_ceil(stores[s].3)) as u32
        })
        .collect();
    Ok(Plan { at, pool_used, lent_rows })
}

/// The loaned prefill lane set and its bookkeeping.
pub struct PrefillLoan {
    pub bd_a: BatchDgpuScratch,
    pub bd_b: BatchDgpuScratch,
    pub sd: BatchDgpuShared,
    pub rows: usize,
    pub lent_rows: Vec<u32>,
    /// Owns the pool; the set's pool-resident buffers and the layer-major store region are views into it.
    _pool: DeviceBuffer<u8>,
    pub pool_bytes: usize,
    pub set_bytes: usize,
    dgpu: Device,
}

/// The layer-major window store region (`lm_store` / `lm_idx`) inside the pool: `forward_prefill` places its window
/// buffers here when set (`lm_pool`).
static LM_POOL: std::sync::Mutex<Option<(usize, usize, i32)>> = std::sync::Mutex::new(None);

/// `(address, bytes, device)` of the layer-major store region, if the loan reserved one. ONE region shared by every
/// prefill job: reuse across jobs is fine -- the store holds nothing live between windows (review r3 nit 7).
pub fn lm_pool() -> Option<(sys::hipDeviceptr_t, usize, i32)> {
    LM_POOL.lock().unwrap_or_else(|p| p.into_inner()).map(|(a, b, d)| (a as sys::hipDeviceptr_t, b, d))
}

fn alloc_set(dgpu: Device, rows: usize, n_kv_max: u32) -> eyre::Result<(BatchDgpuShared, BatchDgpuScratch, BatchDgpuScratch)> {
    let sd = BatchDgpuShared::alloc_rows_ctx_opts(dgpu, rows, n_kv_max, false)?;
    let a = BatchDgpuScratch::alloc_rows(dgpu, rows)?;
    let b = BatchDgpuScratch::alloc_rows(dgpu, rows)?;
    Ok((sd, a, b))
}

impl PrefillLoan {
    /// Build the set over a new pool + the arena's store end ranges. `lm_bytes` = the layer-major store region reserved
    /// at the pool's start. Call LAST at startup (the pool takes the free dGPU memory minus `headroom_bytes()`).
    pub fn build(dgpu: Device, rows: usize, n_kv_max: u32, arena: &KvArena, lm_bytes: usize) -> eyre::Result<Self> {
        let id = dgpu.id;
        // 1. record
        let rec = {
            let _g = alloc_scope::record(id)?;
            let set = alloc_set(dgpu, rows, n_kv_max)?;
            let rec = alloc_scope::recorded()?;
            drop(set); // phantom views: nothing is freed
            rec
        };
        let set_bytes: usize = rec.iter().map(|r| align_up(r.bytes)).sum();
        // 2. pool: the free memory now, minus the headroom
        dgpu.set_current()?;
        let (mut free, mut total) = (0usize, 0usize);
        v4flash_hip::check_eyre(unsafe { sys::hipMemGetInfo(&mut free, &mut total) }, "hipMemGetInfo")?;
        let pool_bytes = free.saturating_sub(headroom_bytes()).min(pool_cap_bytes()) / (1 << 20) * (1 << 20);
        if pool_bytes <= lm_bytes {
            return Err(eyre!("prefill loan: {} MiB free leaves no pool beyond the {} MiB layer-major store",
                free >> 20, lm_bytes >> 20));
        }
        let pool = DeviceBuffer::<u8>::new(id, pool_bytes)?;
        let lm_region = align_up(lm_bytes);
        // 3. plan over pool (after the lm region) + store end ranges
        let mut stores = Vec::with_capacity(arena.stores.len());
        for si in 0..arena.stores.len() {
            let (c, k) = arena.store_buffers(si)?;
            stores.push((c.byte_len(), arena.stores[si].width as usize * 2, k.byte_len(), E2M1_KEY_ROW_BYTES));
        }
        let p = plan(&rec, pool_bytes - lm_region, &stores)?;
        let mut ptrs = Vec::with_capacity(rec.len());
        for (i, &(bin, off)) in p.at.iter().enumerate() {
            let base = match bin {
                Bin::Pool => pool.raw() as usize + lm_region + off,
                Bin::Comp(s) => arena.store_buffers(s)?.0.raw() as usize + off,
                Bin::Keys(s) => arena.store_buffers(s)?.1.raw() as usize + off,
            };
            ptrs.push((base as sys::hipDeviceptr_t, rec[i].bytes));
        }
        // 4. place
        let (sd, bd_a, bd_b) = {
            let _g = alloc_scope::place(id, ptrs)?;
            let set = alloc_set(dgpu, rows, n_kv_max)?;
            let (n, planned) = alloc_scope::placed()?;
            if n != planned {
                return Err(eyre!("prefill loan: placed {n} of {planned} planned allocations"));
            }
            set
        };
        *LM_POOL.lock().unwrap_or_else(|p| p.into_inner()) = Some((pool.raw() as usize, lm_region, id));
        let arena_lent: u64 = p.lent_rows.iter().zip(&stores).map(|(&r, s)| r as u64 * (s.1 + s.3) as u64).sum();
        tracing::info!(
            rows, set_mib = set_bytes >> 20, pool_mib = pool_bytes >> 20, pool_used_mib = (lm_region + p.pool_used) >> 20,
            lm_mib = lm_region >> 20, arena_lent_mib = arena_lent >> 20, lent_rows = ?p.lent_rows, allocations = rec.len(),
            persistent = rec.iter().filter(|r| r.written).count(), headroom_mib = headroom_bytes() >> 20,
            "prefill VRAM loan: prefill lane set built over the pool + the arena's store end ranges"
        );
        Ok(Self { bd_a, bd_b, sd, rows, lent_rows: p.lent_rows, _pool: pool, pool_bytes, set_bytes, dgpu })
    }

    /// Lend (idempotent): synchronize the whole dGPU, then `KvArena::lend`.
    pub fn lend(&mut self, arena: &mut KvArena, stream: &Stream, bounce_f16: &mut DeviceBuffer<u16>, bounce_u8: &mut DeviceBuffer<u8>) -> eyre::Result<Option<LendStats>> {
        if arena.is_lent() {
            return Ok(None);
        }
        let _cur = self.dgpu.scoped_current()?;
        v4flash_hip::device_synchronize()?;
        arena.lend(&self.lent_rows, stream, bounce_f16, bounce_u8).map(Some)
    }

    /// End the loan (idempotent): synchronize the whole dGPU (the prefill pass's writes into the lent rows are done),
    /// then restore the swapped rows.
    pub fn end(&mut self, arena: &mut KvArena) -> eyre::Result<Option<EndStats>> {
        if !arena.is_lent() {
            return Ok(None);
        }
        let _cur = self.dgpu.scoped_current()?;
        v4flash_hip::device_synchronize()?;
        arena.end_lend().map(Some)
    }
}

/// Bytes of the layer-major window store for `rows` (`lm_store` + `lm_idx`): `rows x (HC_DIM + HC_MIX_DIM) f32` and
/// `rows x (INDEXER_TOP_K + 1) i32`, each buffer 256-aligned.
pub fn lm_store_bytes(rows: usize) -> usize {
    let h = crate::config::HC_DIM as usize;
    let m = crate::config::HC_MIX_DIM as usize;
    let k = crate::indexer::INDEXER_TOP_K as usize;
    align_up(rows * h * 4) + align_up(rows * m * 4) + align_up(rows * k * 4) + align_up(rows * 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(bytes: usize, written: bool) -> Recorded {
        Recorded { bytes, written }
    }

    #[test]
    fn ffd_pool_first_then_largest_store_from_its_end() {
        // pool 1024 B; stores: (comp 2048 B, 128 B rows, keys 512 B, 16 B rows), (comp 1024, 128, keys 256, 16)
        let st = [(2048, 128, 512, 16), (1024, 128, 256, 16)];
        let items = vec![r(900, false), r(1500, false), r(600, false), r(200, false)];
        let p = plan(&items, 1024, &st).unwrap();
        // size order: 1500 (#1, 1536 aligned) > 900 (#0, 1024) > 600 (#2, 768) > 200 (#3, 256)
        assert_eq!(p.at[1], (Bin::Comp(0), 2048 - 1536), "too big for the pool: the largest store, at its END");
        assert_eq!(p.at[0], (Bin::Pool, 0), "fills the pool exactly");
        assert_eq!(p.at[2], (Bin::Comp(1), 1024 - 768), "store 0's comp has 512 left: the next store");
        assert_eq!(p.at[3], (Bin::Comp(0), 2048 - 1536 - 256), "stacks below the 1536 in store 0");
        assert_eq!(p.pool_used, 1024);
        assert_eq!(p.lent_rows, vec![(1536 + 256) / 128, 768 / 128]);
    }

    #[test]
    fn persistent_items_only_go_to_the_pool() {
        let st = [(4096, 128, 1024, 16)];
        // written first: 100 (#0, 256) -> pool 0; then size order: 600 (#1, 768) -> pool 256; 300 (#2, 512) -> comp
        let p = plan(&[r(100, true), r(600, false), r(300, false)], 1024, &st).unwrap();
        assert_eq!(p.at[0], (Bin::Pool, 0), "a written allocation stays in the pool");
        assert_eq!(p.at[1], (Bin::Pool, 256));
        assert_eq!(p.at[2], (Bin::Comp(0), 4096 - 512));
        // a written allocation the pool cannot hold is refused, never lent
        let e = plan(&[r(2000, true)], 1024, &st).unwrap_err().to_string();
        assert!(e.contains("must persist"), "{e}");
    }

    #[test]
    fn bin_ends_are_aligned_down_so_every_buffer_starts_256_aligned() {
        // keys bin of 1000 B (e.g. rows_cap x 80): usable end 768 -> three 256 B slots
        let p = plan(&[r(100, false); 3], 0, &[(0, 128, 1000, 80)]).unwrap();
        for &(bin, off) in &p.at {
            assert_eq!(bin, Bin::Keys(0));
            assert_eq!(off % 256, 0, "offset {off}");
            assert!(off + 256 <= 768, "offset {off} past the aligned end");
        }
        assert!(plan(&[r(100, false); 4], 0, &[(0, 128, 1000, 80)]).is_err(), "a 4th slot would cross the aligned end");
    }

    /// Review r4 NEW-1: a small written item sorted after big unwritten ones still gets the pool.
    #[test]
    fn written_items_are_placed_first() {
        // pool 1024: an unwritten 1000 (-> 1024) would fill it; the written 10 B must still land in the pool
        let p = plan(&[r(1000, false), r(10, true)], 1024, &[(4096, 128, 0, 16)]).unwrap();
        assert_eq!(p.at[1], (Bin::Pool, 0), "the written item goes first");
        assert_eq!(p.at[0].0, Bin::Comp(0), "the big unwritten one moves to a store");
    }

    /// Review r4 NEW-2: with an unaligned key bin the lent rows cover the placed buffer, slack included.
    #[test]
    fn lent_rows_cover_the_alignment_slack() {
        // keys bin 1000 B, 80 B rows: aligned end 768, slack 232; one 256 B buffer at [512, 768)
        let p = plan(&[r(200, false)], 0, &[(0, 128, 1000, 80)]).unwrap();
        assert_eq!(p.at[0], (Bin::Keys(0), 512));
        let lent_from_byte = 1000 - p.lent_rows[0] as usize * 80;
        assert!(lent_from_byte <= 512, "lent range starts at byte {lent_from_byte}, the buffer at 512");
    }

    #[test]
    fn keys_bins_take_what_the_comp_bins_cannot() {
        let items = vec![r(200, false), r(200, false)];
        let p = plan(&items, 0, &[(256, 128, 256, 16)]).unwrap();
        assert_eq!(p.at[0], (Bin::Comp(0), 0));
        assert_eq!(p.at[1], (Bin::Keys(0), 0));
        assert_eq!(p.lent_rows, vec![(256usize / 128).max(256 / 16) as u32]);
        assert!(plan(&[r(300, false)], 0, &[(256, 128, 256, 16)]).is_err());
    }
}
