//! Row gather from the V4.1 Engram hash tables (`layers.{1,14}.engram.embed`:
//! `.weight` fp8 e4m3 `[rows, 256]` + `.scale` e8m0 `[rows, 8]`, block 32),
//! 98 GB per layer on SSD, read with a parallel pread pool and dequantised to
//! f32 (bf16-rounded, as the reference `ParallelEngramEmbedding` returns
//! bf16). Rows are NOT fadvise-dropped after reading: n-grams recur within a
//! conversation and the page cache is the cheapest prefetch there is.

use color_eyre::eyre::{self, eyre};

use crate::engram_hash::{ENGRAM_COLS, ENGRAM_ROW_DIM};
use crate::hf_v41::{e4m3_to_f32, e8m0_to_f32};
use crate::safetensors::{SafetensorsDir, StTensor};

pub const ENGRAM_ROW_BYTES: u64 = ENGRAM_ROW_DIM as u64; // 1 B per fp8 element
pub const ENGRAM_SCALE_BYTES: u64 = (ENGRAM_ROW_DIM / 32) as u64; // 8 e8m0 per row
/// One RAW cached row: the 256 fp8 weights then the 8 e8m0 scales.
const RAW_ROW: usize = ENGRAM_ROW_DIM + ENGRAM_SCALE_BYTES as usize;
/// Cache bytes per row: the payload + its hashbrown `(i64, u32)` entry (16 B + a
/// control byte at <= 7/8 load, rounded up), with both generations pre-sized.
const RAW_ROW_COST: usize = RAW_ROW + 24;

/// Dequantise one RAW row (`RAW_ROW` bytes) into 256 f32 -- the exact formula the
/// disk path always used, so cached and uncached rows are bit-identical.
#[inline]
fn dequant_raw(raw: &[u8], out: &mut [f32]) {
    let (w, sc) = raw.split_at(ENGRAM_ROW_DIM);
    for (j, o) in out.iter_mut().enumerate() {
        *o = bf16_round(e4m3_to_f32(w[j]) * e8m0_to_f32(sc[j / 32]));
    }
}

/// One generation of the raw-row cache: id -> slot, rows packed in `data`.
/// PRE-SIZED (`with_capacity`): a growing Vec doubles past its need (a 718 MB
/// generation would sit in a 1 GiB allocation) and every doubling / rehash runs
/// inside `insert` under the cache mutex (review r6 MAJOR-1).
#[derive(Default)]
struct RowGen {
    idx: std::collections::HashMap<i64, u32>,
    data: Vec<u8>,
}

impl RowGen {
    fn with_capacity(rows: usize) -> Self {
        Self { idx: std::collections::HashMap::with_capacity(rows), data: Vec::with_capacity(rows * RAW_ROW) }
    }
    fn get(&self, id: i64) -> Option<&[u8]> {
        self.idx.get(&id).map(|&i| &self.data[i as usize * RAW_ROW..(i as usize + 1) * RAW_ROW])
    }
    fn push(&mut self, id: i64, raw: &[u8]) {
        let i = (self.data.len() / RAW_ROW) as u32;
        self.data.extend_from_slice(raw);
        self.idx.insert(id, i);
    }
    fn len(&self) -> usize {
        self.idx.len()
    }
}

/// RAW Engram row cache (owner 10-11): rows as their 264 stored bytes, ~15x denser
/// than the page cache (a 4 KiB page per 264 B row) and ~4x denser than the old
/// decoded-f32 cache. TWO GENERATIONS instead of clear-on-full: inserts go to
/// `young`; when it holds `gen_cap` rows it becomes `old` (the previous `old` is
/// dropped); a hit in `old` is copied into `young`. An LRU approximation with no
/// per-access bookkeeping, bounded at ~2 x `gen_cap` rows.
///
/// WHY (measured 10-11, box 1): a 4096-token layer-major window gathers ~196K rows
/// (4096 x 24 cols x 2 layers); with ~8 GB of page cache left beside 112 GB of GTT
/// the 2 x 98 GB tables thrashed it -- window time tracked ~0.4-3 GB of ~10 KB reads
/// per window (2.6 s + ~0.9 s/GB) and long jobs "ramped" as the conversation's rows
/// warmed. The old 262K-row (256 MB) cache cleared itself every ~1.3 windows.
struct RawRowCache {
    young: RowGen,
    old: RowGen,
    gen_cap: usize,
}

impl RawRowCache {
    fn new(cap_rows: usize) -> Self {
        let gen_cap = (cap_rows / 2).max(1);
        Self { young: RowGen::with_capacity(gen_cap), old: RowGen::default(), gen_cap }
    }
    /// Copy the row into `dst` if cached (promoting an `old` hit).
    fn lookup(&mut self, id: i64, dst: &mut [u8]) -> bool {
        if let Some(r) = self.young.get(id) {
            dst.copy_from_slice(r);
            return true;
        }
        let Some(r) = self.old.get(id) else { return false };
        dst.copy_from_slice(r);
        self.insert(id, dst);
        true
    }
    fn insert(&mut self, id: i64, raw: &[u8]) {
        if self.young.idx.contains_key(&id) {
            return;
        }
        if self.young.len() >= self.gen_cap {
            self.old = std::mem::replace(&mut self.young, RowGen::with_capacity(self.gen_cap));
        }
        self.young.push(id, raw);
    }
    fn rows(&self) -> usize {
        self.young.len() + self.old.len()
    }
}

/// Round an f32 to the nearest bf16 (ties to even) and back.
#[inline]
pub fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 {
        return x; // inf / nan
    }
    let lsb = (b >> 16) & 1;
    let rounded = b.wrapping_add(0x7fff + lsb) & 0xffff_0000;
    f32::from_bits(rounded)
}

pub struct EngramTable {
    pub layer: usize,
    weight: StTensor,
    scale: StTensor,
    pub rows: u64,
    /// Decoded-row cache: `row id -> ENGRAM_ROW_DIM bf16-rounded f32`.
    ///
    /// WHY. A row is 264 B (256 B weight + 8 B scale) but the page cache works in
    /// 4 KiB pages, so every miss pulls ~15x what it needs out of a 98 GB table that
    /// cannot stay resident in 96 GB of RAM. Decode issues 2 layers x 24 cols x 2 reads
    /// = 96 tiny random reads PER TOKEN and blocks on them: measured **7.9 ms/token
    /// p50** (`engram_us`), which is latency, not bandwidth — ~12 KB moved in 7.9 ms.
    /// Threads already hide some of it (24-way fan-out; narrowing it once cost decode
    /// 14.4 -> 11.9 tok/s) but cannot fix the granularity.
    ///
    /// n-gram hashes repeat heavily within a conversation. RAW rows (`RawRowCache`),
    /// sized by `ENGRAM_CACHE_MB` (default 1024 per table, ~3.7M rows, both
    /// generations full ~1 GB; box 1 has ~7 GB available beside 112 GB of GTT) or
    /// `ENGRAM_CACHE_ROWS` (RAW rows -- was decoded rows; wins if set); `0` disables it.
    cache: std::sync::Mutex<RawRowCache>,
    cache_cap: usize,
    pub cache_hits: std::sync::atomic::AtomicU64,
    pub cache_misses: std::sync::atomic::AtomicU64,
    /// Stats line at most once a minute (`gather`).
    last_log: std::sync::Mutex<std::time::Instant>,
    /// HEDGED gathers (`gather_hedged`): owned handles on both copies of the weight / scale shards, opened at `open`
    /// when `V41_EXPERT_MIRROR_DIR` is set, else at the first hedged gather (`None` inside = no usable mirror).
    hedge_files: std::sync::OnceLock<Option<std::sync::Arc<HedgeFiles>>>,
}

/// Both copies of a table's two tensors (`SafetensorsDir::hedge_handles`) and their absolute offsets.
struct HedgeFiles {
    w_primary: std::sync::Arc<std::fs::File>,
    w_mirror: std::sync::Arc<std::fs::File>,
    s_primary: std::sync::Arc<std::fs::File>,
    s_mirror: std::sync::Arc<std::fs::File>,
    w_off: u64,
    s_off: u64,
}

impl EngramTable {
    pub fn open(st: &SafetensorsDir, layer: usize) -> eyre::Result<Self> {
        let weight = st.get(&format!("layers.{layer}.engram.embed.weight"))?.clone();
        let scale = st.get(&format!("layers.{layer}.engram.embed.scale"))?.clone();
        if weight.shape.len() != 2 || weight.shape[1] != ENGRAM_ROW_DIM as u64
            || scale.shape.len() != 2 || scale.shape[1] != ENGRAM_SCALE_BYTES || scale.shape[0] != weight.shape[0]
        {
            return Err(eyre!("engram table L{layer}: unexpected shapes {:?} / {:?}", weight.shape, scale.shape));
        }
        let cache_cap = std::env::var("ENGRAM_CACHE_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                let mb = std::env::var("ENGRAM_CACHE_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024);
                mb * (1 << 20) / RAW_ROW_COST
            });
        // No readahead on the tables (`ENGRAM_FADV_RANDOM`, default on): a row miss
        // reads its page, not ~10 KB around it (measured 10-11: ~10 KB per IO during
        // prefill windows). Advisory; a failure only loses the hint.
        if std::env::var("ENGRAM_FADV_RANDOM").as_deref() != Ok("0") {
            for t in [&weight, &scale] {
                if let Err(e) = st.advise_random(t) {
                    eprintln!("engram L{layer}: {e:#}");
                }
            }
        }
        eprintln!(
            "engram L{layer}: raw row cache {} rows (~{} MB at full); FADV_RANDOM on shards {} / {}",
            cache_cap, cache_cap * RAW_ROW_COST >> 20, st.shard_name(weight.shard), st.shard_name(scale.shard)
        );
        let t = Self {
            layer,
            rows: weight.shape[0],
            weight,
            scale,
            cache: std::sync::Mutex::new(RawRowCache::new(cache_cap)),
            cache_cap,
            cache_hits: std::sync::atomic::AtomicU64::new(0),
            cache_misses: std::sync::atomic::AtomicU64::new(0),
            last_log: std::sync::Mutex::new(std::time::Instant::now()),
            hedge_files: std::sync::OnceLock::new(),
        };
        // Hedge handles (and the mirror's header + sampled-data check) at OPEN, not inside the first decode gather.
        if std::env::var_os("V41_EXPERT_MIRROR_DIR").is_some() {
            let _ = t.hedge_files(st);
        }
        Ok(t)
    }

    /// The table's hedge handles (first call opens them; `None` = no usable mirror copy).
    fn hedge_files(&self, st: &SafetensorsDir) -> Option<std::sync::Arc<HedgeFiles>> {
        self.hedge_files
            .get_or_init(|| {
                let open = || -> eyre::Result<Option<HedgeFiles>> {
                    let (Some((w_primary, w_mirror)), Some((s_primary, s_mirror))) = (st.hedge_handles(self.weight.shard)?, st.hedge_handles(self.scale.shard)?) else {
                        return Ok(None);
                    };
                    Ok(Some(HedgeFiles { w_primary, w_mirror, s_primary, s_mirror, w_off: self.weight.offset, s_off: self.scale.offset }))
                };
                match open() {
                    Ok(Some(h)) => {
                        eprintln!("engram L{}: hedged reads ready (mirror copies of {} / {})", self.layer, st.shard_name(self.weight.shard), st.shard_name(self.scale.shard));
                        Some(std::sync::Arc::new(h))
                    }
                    Ok(None) => {
                        eprintln!("engram L{}: no usable mirror copy: hedged gathers read the primary only", self.layer);
                        None
                    }
                    Err(e) => {
                        eprintln!("engram L{}: hedge handles: {e:#}; hedged gathers read the primary only", self.layer);
                        None
                    }
                }
            })
            .clone()
    }

    /// [`Self::gather`] for a DECODE step, with HEDGED disk reads (`V41_ENGRAM_HEDGE_MS`, `crate::hedge`): row misses
    /// not read from the primary within `hedge` are re-read from the mirror copy and the first answer is used (same
    /// bytes either way: bit-identical rows). `None`, no row cache (`ENGRAM_CACHE_ROWS=0`) or no usable mirror ->
    /// exactly `gather`.
    pub fn gather_hedged(&self, st: &SafetensorsDir, ids: &[i64], out: &mut [f32], threads: usize, hedge: Option<std::time::Duration>) -> eyre::Result<()> {
        self.gather_impl(st, ids, out, threads, hedge)
    }

    /// Gather + dequantise `ids.len()` rows into `out` (`ids.len() * 256` f32),
    /// `threads` concurrent preads (each row is two small reads: 256 B + 8 B).
    pub fn gather(&self, st: &SafetensorsDir, ids: &[i64], out: &mut [f32], threads: usize) -> eyre::Result<()> {
        self.gather_impl(st, ids, out, threads, None)
    }

    fn gather_impl(&self, st: &SafetensorsDir, ids: &[i64], out: &mut [f32], threads: usize, hedge: Option<std::time::Duration>) -> eyre::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len() * ENGRAM_ROW_DIM {
            return Err(eyre!("engram gather: out len {} != {} rows × {ENGRAM_ROW_DIM}", out.len(), ids.len()));
        }
        for &id in ids {
            if id < 0 || id as u64 >= self.rows {
                return Err(eyre!("engram gather L{}: row {id} outside [0, {})", self.layer, self.rows));
            }
        }
        // DO NOT cap this by "work per thread". It looks like over-parallelisation
        // — `gather_position` spawns 24 threads for 24 rows of 256 B — but those
        // rows are random reads into a 98 GB NVMe-resident table, so the fan-out is
        // LATENCY HIDING, not throughput. MEASURED 2026-09-13: capping to 1 thread
        // per position regressed decode **14.4 -> 11.9 tok/s** (-17%). A cold row
        // read is ~80 us and a spawn ~15 us, so 48 serial cold preads cost ~3.8 ms
        // per token versus ~160 us overlapped.
        //
        // MEASURED 2026-09-21 (`engram_gather_timing`, 24 cold rows, live box): a
        // persistent 32-thread reader pool gave p50 1,705 us against 1,663 us for
        // these scoped spawns. The cost is the cold 4 KB reads on the loaded
        // dm-crypt drive (min ~830 us per gather), not thread creation; the pool
        // was removed. The levers are the row cache and a faster/plaintext drive.
        //
        // A microbenchmark that says serial wins here is measuring a PAGE-CACHE-HOT
        // table (warm read ~2 us, so spawning dominates) and does not describe
        // production. The real fix for spawn cost is to BATCH the call — see
        // `EngramCtx::rows_for_chunk`, which now issues one gather per layer per run
        // instead of one per position (289k spawns -> ~64, 530 -> 59 us/token) —
        // not to narrow the fan-out of an individual gather.
        // Serve what the cache already has (RAW rows); only the misses go to disk;
        // every row is then dequantised (in parallel for big gathers).
        if self.cache_cap > 0 {
            let mut raw = vec![0u8; ids.len() * RAW_ROW];
            let mut miss_at: Vec<usize> = Vec::new();
            {
                let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                for (k, &id) in ids.iter().enumerate() {
                    if !c.lookup(id, &mut raw[k * RAW_ROW..(k + 1) * RAW_ROW]) {
                        miss_at.push(k);
                    }
                }
            }
            self.cache_hits.fetch_add((ids.len() - miss_at.len()) as u64, std::sync::atomic::Ordering::Relaxed);
            if !miss_at.is_empty() {
                let mut need: Vec<i64> = miss_at.iter().map(|&k| ids[k]).collect();
                need.sort_unstable();
                need.dedup();
                self.cache_misses.fetch_add(need.len() as u64, std::sync::atomic::Ordering::Relaxed);
                let mut fetched = vec![0u8; need.len() * RAW_ROW];
                match hedge.and_then(|d| self.hedge_files(st).map(|h| (d, h))) {
                    Some((d, h)) => Self::read_raw_hedged(h, &need, &mut fetched, threads, d)?,
                    None => self.read_raw(st, &need, &mut fetched, threads)?,
                }
                for &k in &miss_at {
                    let j = need.binary_search(&ids[k]).expect("every miss was fetched");
                    raw[k * RAW_ROW..(k + 1) * RAW_ROW].copy_from_slice(&fetched[j * RAW_ROW..(j + 1) * RAW_ROW]);
                }
                let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                for (j, &id) in need.iter().enumerate() {
                    c.insert(id, &fetched[j * RAW_ROW..(j + 1) * RAW_ROW]);
                }
            }
            Self::dequant_rows(&raw, out, threads);
            self.maybe_log();
            return Ok(());
        }
        self.gather_uncached(st, ids, out, threads)
    }

    /// Dequantise `raw` (rows of `RAW_ROW`) into `out`; split over up to `threads`
    /// scoped threads when there is enough work (a window gathers ~98K rows/layer).
    fn dequant_rows(raw: &[u8], out: &mut [f32], threads: usize) {
        let n = raw.len() / RAW_ROW;
        let parts = if n >= 4096 { threads.clamp(1, 16).min(n / 2048).max(1) } else { 1 };
        let per = n.div_ceil(parts);
        if parts == 1 {
            for (r, o) in raw.chunks_exact(RAW_ROW).zip(out.chunks_exact_mut(ENGRAM_ROW_DIM)) {
                dequant_raw(r, o);
            }
            return;
        }
        std::thread::scope(|sc| {
            for (rc, oc) in raw.chunks(per * RAW_ROW).zip(out.chunks_mut(per * ENGRAM_ROW_DIM)) {
                sc.spawn(move || {
                    for (r, o) in rc.chunks_exact(RAW_ROW).zip(oc.chunks_exact_mut(ENGRAM_ROW_DIM)) {
                        dequant_raw(r, o);
                    }
                });
            }
        });
    }

    /// RAW rows for `ids` from disk into `dst` (`ids.len() * RAW_ROW`), `threads`
    /// concurrent preads (each row: 256 B weight + 8 B scale).
    fn read_raw(&self, st: &SafetensorsDir, ids: &[i64], dst: &mut [u8], threads: usize) -> eyre::Result<()> {
        let threads = threads.clamp(1, ids.len().max(1));
        let per = ids.len().div_ceil(threads);
        let results: Vec<eyre::Result<()>> = std::thread::scope(|sc| {
            let handles: Vec<_> = ids
                .chunks(per)
                .zip(dst.chunks_mut(per * RAW_ROW))
                .map(|(ids_c, dst_c)| {
                    sc.spawn(move || -> eyre::Result<()> {
                        for (&id, row) in ids_c.iter().zip(dst_c.chunks_exact_mut(RAW_ROW)) {
                            let (w, s) = row.split_at_mut(ENGRAM_ROW_DIM);
                            st.read_range_into_cached(&self.weight, id as u64 * ENGRAM_ROW_BYTES, w)?;
                            st.read_range_into_cached(&self.scale, id as u64 * ENGRAM_SCALE_BYTES, s)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("engram read thread panicked")).collect()
        });
        results.into_iter().collect::<eyre::Result<Vec<()>>>()?;
        Ok(())
    }

    /// `read_raw` HEDGED: the same `threads`-way chunking, each chunk read from the primary (buffered, as `read_raw`) on
    /// its own detached thread into an owned buffer; chunks not back within `hedge` are read again from the mirror
    /// (O_DIRECT, 4 KiB-aligned bounce per row: the mirror's pages stay out of the page cache) and the first answer is
    /// copied into `dst` (`crate::hedge::hedged`).
    fn read_raw_hedged(h: std::sync::Arc<HedgeFiles>, ids: &[i64], dst: &mut [u8], threads: usize, hedge: std::time::Duration) -> eyre::Result<()> {
        use std::os::unix::fs::FileExt;
        let threads = threads.clamp(1, ids.len().max(1));
        let per = ids.len().div_ceil(threads);
        let chunks: std::sync::Arc<Vec<Vec<i64>>> = std::sync::Arc::new(ids.chunks(per).map(|c| c.to_vec()).collect());
        let (hp, cp) = (h.clone(), chunks.clone());
        let primary = move |i: usize| -> eyre::Result<Vec<u8>> {
            crate::hedge::test_primary_delay();
            let c = &cp[i];
            let mut v = vec![0u8; c.len() * RAW_ROW];
            for (&id, row) in c.iter().zip(v.chunks_exact_mut(RAW_ROW)) {
                let (w, s) = row.split_at_mut(ENGRAM_ROW_DIM);
                hp.w_primary.read_exact_at(w, hp.w_off + id as u64 * ENGRAM_ROW_BYTES).map_err(|e| eyre!("engram primary row {id}: {e}"))?;
                hp.s_primary.read_exact_at(s, hp.s_off + id as u64 * ENGRAM_SCALE_BYTES).map_err(|e| eyre!("engram primary scale {id}: {e}"))?;
            }
            Ok(v)
        };
        let (hm, cm) = (h, chunks.clone());
        let mirror = move |i: usize| -> eyre::Result<Vec<u8>> {
            let c = &cm[i];
            let mut v = vec![0u8; c.len() * RAW_ROW];
            let mut sc = crate::hedge::AlignedBuf::new(2 * crate::hedge::AlignedBuf::ALIGN)?;
            for (&id, row) in c.iter().zip(v.chunks_exact_mut(RAW_ROW)) {
                let (w, s) = row.split_at_mut(ENGRAM_ROW_DIM);
                crate::hedge::read_direct_small(&hm.w_mirror, hm.w_off + id as u64 * ENGRAM_ROW_BYTES, w, &mut sc)?;
                crate::hedge::read_direct_small(&hm.s_mirror, hm.s_off + id as u64 * ENGRAM_SCALE_BYTES, s, &mut sc)?;
            }
            Ok(v)
        };
        let (parts, _) = crate::hedge::hedged(chunks.len(), hedge, std::sync::Arc::new(primary), std::sync::Arc::new(mirror), &crate::hedge::ENGRAM)?;
        for (part, d) in parts.iter().zip(dst.chunks_mut(per * RAW_ROW)) {
            d.copy_from_slice(part);
        }
        Ok(())
    }

    /// At most once a minute: hits / misses since start and the rows held.
    fn maybe_log(&self) {
        let Ok(mut t) = self.last_log.try_lock() else { return };
        if t.elapsed() < std::time::Duration::from_secs(60) {
            return;
        }
        *t = std::time::Instant::now();
        let h = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let m = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        let rows = self.cache.try_lock().map(|c| c.rows()).unwrap_or(0);
        eprintln!(
            "engram cache L{}: hits {h} misses {m} ({:.1}% hit) rows {rows}/{} ({} MB)",
            self.layer, 100.0 * h as f64 / (h + m).max(1) as f64, self.cache_cap, rows * RAW_ROW_COST >> 20
        );
    }

    /// The original disk path — unchanged; see `gather` for the cache in front of it.
    fn gather_uncached(&self, st: &SafetensorsDir, ids: &[i64], out: &mut [f32], threads: usize) -> eyre::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let threads = threads.clamp(1, ids.len().max(1));
        let per = ids.len().div_ceil(threads);
        let results: Vec<eyre::Result<()>> = std::thread::scope(|sc| {
            let handles: Vec<_> = ids
                .chunks(per)
                .zip(out.chunks_mut(per * ENGRAM_ROW_DIM))
                .map(|(ids_c, out_c)| {
                    sc.spawn(move || -> eyre::Result<()> {
                        let mut w = [0u8; ENGRAM_ROW_DIM];
                        let mut s = [0u8; ENGRAM_SCALE_BYTES as usize];
                        for (&id, row) in ids_c.iter().zip(out_c.chunks_exact_mut(ENGRAM_ROW_DIM)) {
                            st.read_range_into_cached(&self.weight, id as u64 * ENGRAM_ROW_BYTES, &mut w)?;
                            st.read_range_into_cached(&self.scale, id as u64 * ENGRAM_SCALE_BYTES, &mut s)?;
                            for (j, o) in row.iter_mut().enumerate() {
                                *o = bf16_round(e4m3_to_f32(w[j]) * e8m0_to_f32(s[j / 32]));
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("engram gather thread panicked")).collect()
        });
        results.into_iter().collect::<eyre::Result<Vec<()>>>()?;
        Ok(())
    }

    /// One position's 24 rows (`hash_ids` for this layer) → `out` (24 × 256 f32).
    pub fn gather_position(&self, st: &SafetensorsDir, ids: &[i64; ENGRAM_COLS], out: &mut [f32]) -> eyre::Result<()> {
        self.gather(st, ids, out, ENGRAM_COLS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram_hash::{EngramHash, ENGRAM_LAYERS};

    /// Cold random-row gather cost on the REAL tables (`ENGRAM_CACHE_ROWS=0` keeps
    /// the row cache out of it; needs `V41_MODEL`, the HF snapshot dir). 2026-09-21
    /// on the live box: p50 1.7 ms, min 0.83 ms per 24-row gather — disk latency,
    /// not spawn cost (a persistent pool measured the same and was removed).
    #[test]
    #[ignore]
    fn engram_gather_timing() -> eyre::Result<()> {
        let dir = std::env::var("V41_MODEL").expect("V41_MODEL=<hf snapshot dir>");
        let st = SafetensorsDir::open(&dir)?;
        let tbl = EngramTable::open(&st, 1)?;
        let rows = tbl.rows;
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % rows) as i64 };
        let mut out = vec![0f32; ENGRAM_COLS * ENGRAM_ROW_DIM];
        let mut us: Vec<u128> = Vec::new();
        for _ in 0..120 {
            let ids: Vec<i64> = (0..ENGRAM_COLS).map(|_| next()).collect();
            let t = std::time::Instant::now();
            tbl.gather(&st, &ids, &mut out, ENGRAM_COLS)?;
            us.push(t.elapsed().as_micros());
        }
        us.sort_unstable();
        let n = us.len();
        eprintln!(
            "engram gather 24 cold rows x{}: p50 {} us  p90 {} us  min {} us  (pool threads {:?}, cache rows {:?})",
            n, us[n / 2], us[n * 9 / 10], us[0],
            std::env::var("ENGRAM_GATHER_THREADS").ok(), std::env::var("ENGRAM_CACHE_ROWS").ok()  // pool env is historical
        );
        Ok(())
    }

    /// Two generations: bounded at ~2 x gen_cap rows, an `old` hit is promoted, a row
    /// two generations stale is gone; lookups return the inserted bytes.
    #[test]
    fn raw_row_cache_generations() {
        let row = |id: i64| -> Vec<u8> { (0..RAW_ROW).map(|j| (id as usize * 7 + j) as u8).collect() };
        let mut c = RawRowCache::new(8); // gen_cap 4
        for id in 0..4 {
            c.insert(id, &row(id));
        }
        let mut b = vec![0u8; RAW_ROW];
        assert!(c.lookup(2, &mut b) && b == row(2));
        for id in 4..8 {
            c.insert(id, &row(id)); // 4 rotates young(0..4) to old
        }
        assert!(c.rows() <= 8);
        assert!(c.lookup(1, &mut b) && b == row(1), "old hit");
        for id in 8..16 {
            c.insert(id, &row(id));
        }
        assert!(!c.lookup(0, &mut b), "two generations stale: gone");
        assert!(c.rows() <= 8);
        assert!(c.lookup(15, &mut b) && b == row(15));
    }

    /// The cached path's dequant == the disk path's formula, element for element, and
    /// the parallel split covers every row.
    #[test]
    fn dequant_raw_matches_formula() {
        let mut x: u64 = 0x1234_5678_9abc_def1;
        let n = 5000;
        let raw: Vec<u8> = (0..n * RAW_ROW).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect();
        let mut out = vec![0f32; n * ENGRAM_ROW_DIM];
        EngramTable::dequant_rows(&raw, &mut out, 8);
        for (r, o) in raw.chunks_exact(RAW_ROW).zip(out.chunks_exact(ENGRAM_ROW_DIM)) {
            for j in 0..ENGRAM_ROW_DIM {
                let want = bf16_round(e4m3_to_f32(r[j]) * e8m0_to_f32(r[ENGRAM_ROW_DIM + j / 32]));
                assert_eq!(o[j].to_bits(), want.to_bits());
            }
        }
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(1.00390625), 1.0); // half-ulp tie → even
        assert_eq!(bf16_round(1.0078125), 1.0078125);
        assert_eq!(bf16_round(-3.14159).to_bits() & 0xffff, 0);
    }

    /// Gathered rows for the dump's check prompt equal the reference's
    /// dequantised rows (`rows_L01.bin` / `rows_L14.bin`, written by
    /// export_engram_hash.py through the oracle's mmap table).
    #[test]
    fn gather_matches_reference_rows() -> eyre::Result<()> {
        let home = std::env::var("HOME").unwrap_or_default();
        let dump = std::path::PathBuf::from(&home).join(".cache/deepstrix/v41/engram");
        let model = std::env::var("V41_MODEL").unwrap_or_else(|_| format!("{home}/.cache/deepstrix/models/dsv4.1f"));
        if !dump.join("rows_L01.bin").exists() || !std::path::Path::new(&model).join("model.safetensors.index.json").exists() {
            // A fidelity test whose reference fixture is absent MUST say so unmistakably.
            // Returning Ok() quietly makes it green while testing nothing -- the survey found
            // six of these, and this class of silence is how #0b, the 77% submit mask and the
            // `ffn_norm`-left-at-1.0 bug all reached production. `V41_REQUIRE_FIXTURES=1`
            // turns any such skip into a hard failure (set it in CI); without it the skip is
            // loud but non-fatal, so a dev box without the 475 GiB checkpoint still builds.
            {
            if std::env::var("V41_REQUIRE_FIXTURES").as_deref() == Ok("1") {
                panic!("FIXTURE MISSING (V41_REQUIRE_FIXTURES=1): engram rows fixture or model missing");
            }
            eprintln!("*** SKIPPED, NOTHING TESTED: engram rows fixture or model missing ***");
        }
            return Ok(());
        }
        let hs = EngramHash::load(&dump)?;
        let st = SafetensorsDir::open(&model)?;
        let ids = hs.check.as_ref().unwrap().0.clone();
        let hashes = hs.hash_sequence(&ids);
        for l in 0..ENGRAM_LAYERS {
            let layer = hs.layer_ids[l];
            let want: Vec<f32> = std::fs::read(dump.join(format!("rows_L{layer:02}.bin")))?
                .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            assert_eq!(want.len(), ids.len() * ENGRAM_COLS * ENGRAM_ROW_DIM);
            let tbl = EngramTable::open(&st, layer)?;
            let flat: Vec<i64> = hashes.iter().flat_map(|h| h[l].iter().copied()).collect();
            let mut got = vec![0f32; flat.len() * ENGRAM_ROW_DIM];
            let t0 = std::time::Instant::now();
            tbl.gather(&st, &flat, &mut got, 32)?;
            let dt = t0.elapsed();
            let n_diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            let max_abs = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("L{layer}: {} rows gathered in {:.1} ms ({:.0} µs/row); {n_diff} of {} values differ, max |Δ| {max_abs:.3e}",
                flat.len(), dt.as_secs_f64() * 1e3, dt.as_secs_f64() * 1e6 / flat.len() as f64, got.len());
            assert_eq!(n_diff, 0, "L{layer}: gathered rows differ from the reference");
            // Again: now every row comes from the RAW row cache (when enabled).
            let mut again = vec![0f32; flat.len() * ENGRAM_ROW_DIM];
            tbl.gather(&st, &flat, &mut again, 32)?;
            let n_diff2 = again.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(n_diff2, 0, "L{layer}: cached rows differ from the reference");
            assert!(tbl.cache_cap == 0 || tbl.cache_hits.load(std::sync::atomic::Ordering::Relaxed) >= flat.len() as u64);
        }
        Ok(())
    }
}
