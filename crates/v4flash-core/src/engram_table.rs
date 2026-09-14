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
    /// n-gram hashes repeat heavily within a conversation, so a small cache should take
    /// most of that to zero. `ENGRAM_CACHE_ROWS=0` disables it; default 262144 rows
    /// ~= 256 MB of f32 (or set it lower — 65536 is ~64 MB).
    cache: std::sync::Mutex<std::collections::HashMap<i64, std::sync::Arc<Vec<f32>>>>,
    cache_cap: usize,
    pub cache_hits: std::sync::atomic::AtomicU64,
    pub cache_misses: std::sync::atomic::AtomicU64,
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
            .unwrap_or(262_144);
        Ok(Self {
            layer,
            rows: weight.shape[0],
            weight,
            scale,
            cache: std::sync::Mutex::new(std::collections::HashMap::with_capacity(
                cache_cap.min(1 << 16),
            )),
            cache_cap,
            cache_hits: std::sync::atomic::AtomicU64::new(0),
            cache_misses: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Gather + dequantise `ids.len()` rows into `out` (`ids.len() * 256` f32),
    /// `threads` concurrent preads (each row is two small reads: 256 B + 8 B).
    pub fn gather(&self, st: &SafetensorsDir, ids: &[i64], out: &mut [f32], threads: usize) -> eyre::Result<()> {
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
        // A microbenchmark that says serial wins here is measuring a PAGE-CACHE-HOT
        // table (warm read ~2 us, so spawning dominates) and does not describe
        // production. The real fix for spawn cost is to BATCH the call — see
        // `EngramCtx::rows_for_chunk`, which now issues one gather per layer per run
        // instead of one per position (289k spawns -> ~64, 530 -> 59 us/token) —
        // not to narrow the fan-out of an individual gather.
        // Serve what the cache already has; only the misses go to disk.
        if self.cache_cap > 0 {
            let mut need: Vec<i64> = Vec::new();
            {
                let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                for (k, &id) in ids.iter().enumerate() {
                    match c.get(&id) {
                        Some(row) => {
                            out[k * ENGRAM_ROW_DIM..(k + 1) * ENGRAM_ROW_DIM]
                                .copy_from_slice(row);
                            self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        None => need.push(id),
                    }
                }
            }
            if need.is_empty() {
                return Ok(());
            }
            need.sort_unstable();
            need.dedup();
            self.cache_misses
                .fetch_add(need.len() as u64, std::sync::atomic::Ordering::Relaxed);
            let mut fetched = vec![0f32; need.len() * ENGRAM_ROW_DIM];
            self.gather_uncached(st, &need, &mut fetched, threads)?;
            {
                let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                // Bounded, not LRU: n-gram locality is high and a clear is cheaper
                // than per-access bookkeeping on the critical path.
                if c.len() + need.len() > self.cache_cap {
                    c.clear();
                }
                for (j, &id) in need.iter().enumerate() {
                    c.insert(
                        id,
                        std::sync::Arc::new(
                            fetched[j * ENGRAM_ROW_DIM..(j + 1) * ENGRAM_ROW_DIM].to_vec(),
                        ),
                    );
                }
                for (k, &id) in ids.iter().enumerate() {
                    if let Some(row) = c.get(&id) {
                        out[k * ENGRAM_ROW_DIM..(k + 1) * ENGRAM_ROW_DIM].copy_from_slice(row);
                    }
                }
            }
            return Ok(());
        }
        self.gather_uncached(st, ids, out, threads)
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
            eprintln!("engram rows fixture or model missing; skipping");
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
        }
        Ok(())
    }
}
