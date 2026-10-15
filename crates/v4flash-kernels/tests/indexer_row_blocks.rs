//! Batched indexer at a PER-LAUNCH row pitch, in ROW BLOCKS (2026-10-15, branch ctx-1m) versus the old layout
//! (pitch = ATTN_MIXED_MAX_KEYS, all rows in one launch): the same selections bit for bit, through the production
//! chain score -> candidate build (source layer) / mask (later layer) -> top-k, for the prefill gemm and the
//! arena (decode) mw-rows score kernels. Also times the gemm at both pitches.
//!
//! Real packed keys (indexer_qat -> index_kv_append_e2m1), per-row n_idx spread below n_idx_max. Runs on the gfx12
//! device only (the dGPU). Needs the GPU to itself: a gate window, never next to a live hub.
//!
//! `CARGO_TARGET_DIR=target-v41 cargo test -p v4flash-kernels --features v41 --release --test indexer_row_blocks -- --nocapture`
//! (`IRB_N_IDX`, `IRB_BATCH`, `IRB_BLOCK` override the shape; the default forces 16-row blocks.)

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::candidate_blocks::CandidateBlocks;
use v4flash_kernels::config::{N_INDEXER_HEAD, N_INDEXER_HEAD_DIM};
use v4flash_kernels::het::batch_scratch::{candidate_bit_words, indexer_block_rows, indexer_scores_stride, indexer_topk_scratch_per_row};
use v4flash_kernels::index_kv_e2m1::E2M1_KEY_ROW_BYTES;
use v4flash_kernels::weight_contract::f32_to_f16_bits;
use v4flash_kernels::{IndexKvE2m1, IndexerQat, IndexerScoreWmma, IndexerTopkBitonic, ATTN_MIXED_MAX_KEYS, INDEXER_TOP_K};

const DIM: usize = N_INDEXER_HEAD_DIM as usize;
const NH: usize = N_INDEXER_HEAD as usize;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn env_u32(k: &str, d: u32) -> u32 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// One layer's inputs: packed keys + per-row Q (f32 and f16) and head weights.
struct Layer {
    packed: DeviceBuffer<u8>,
    q: DeviceBuffer<f32>,
    q16: DeviceBuffer<u16>,
    hw: DeviceBuffer<f32>,
}

fn layer(d: &Device, stream: &Stream, arch: &str, rng: &mut Rng, n_rows: usize, b: usize) -> eyre::Result<Layer> {
    let qat = IndexerQat::for_arch(arch)?;
    let packer = IndexKvE2m1::for_arch(arch)?;
    let rows: Vec<f32> = (0..n_rows * DIM)
        .map(|i| {
            let mag = if (i / DIM) % 97 == 0 { 30.0 } else { 1.0 };
            (rng.unit() * 2.0 - 1.0) * mag
        })
        .collect();
    let mut x = DeviceBuffer::<f32>::new(d.id, rows.len())?;
    x.copy_from_host(&rows)?;
    qat.launch(stream, &mut x, n_rows as u32)?;
    let mut packed = DeviceBuffer::<u8>::new(d.id, n_rows * E2M1_KEY_ROW_BYTES)?;
    let mut r0 = 0usize;
    while r0 < n_rows {
        let n = (n_rows - r0).min(16384);
        let xs = x.slice_view(r0 * DIM, n * DIM);
        packer.launch_append_batched(stream, &mut packed, &xs, r0 as u32, n as u32)?;
        r0 += n;
    }
    let q_host: Vec<f32> = (0..b * NH * DIM).map(|_| (rng.unit() * 2.0 - 1.0) * 0.5).collect();
    let q16_host: Vec<u16> = q_host.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let hw_host: Vec<f32> = (0..b * NH).map(|_| rng.unit() * 0.2).collect();
    let mut q = DeviceBuffer::<f32>::new(d.id, q_host.len())?;
    q.copy_from_host(&q_host)?;
    let mut q16 = DeviceBuffer::<u16>::new(d.id, q16_host.len())?;
    q16.copy_from_host(&q16_host)?;
    let mut hw = DeviceBuffer::<f32>::new(d.id, hw_host.len())?;
    hw.copy_from_host(&hw_host)?;
    stream.synchronize()?;
    Ok(Layer { packed, q, q16, hw })
}

#[derive(Clone, Copy, PartialEq)]
enum Score {
    Gemm,
    MwRows,
}

/// Buffers one path owns (the lane's selection / candidate state and the shared score, top-k and block-score scratch).
struct Path {
    scores: DeviceBuffer<f32>,
    topk_scratch: DeviceBuffer<u32>,
    done: DeviceBuffer<u32>,
    sel: [DeviceBuffer<i32>; 2],
    /// OLD: the lane's f32 block scores at the cap's block count. NEW: the shared transient (block-local).
    block_score: DeviceBuffer<f32>,
    threshold: DeviceBuffer<u32>,
    /// NEW only: the lane's keep bits at the cap's word pitch.
    bits: DeviceBuffer<u32>,
}

#[allow(clippy::too_many_arguments)]
/// One layer in production order: score -> (build | mask) -> top-k. `old`: the pre-2026-10-15 code (pitch = the cap,
/// f32 block scores carried at the cap's block count, `launch_build` / `launch_mask`). New: `stride` = the launch's
/// pitch, `block` rows at a time, block scores block-local at the launch's block count, keep bits carried.
fn run_layer(
    stream: &Stream,
    wmma: &IndexerScoreWmma,
    cand: &CandidateBlocks,
    topk: &IndexerTopkBitonic,
    p: &mut Path,
    l: &Layer,
    which: usize,
    kind: Score,
    n_idx: &DeviceBuffer<u32>,
    keys_base: &DeviceBuffer<u32>,
    n_idx_max: u32,
    stride: u32,
    block: usize,
    b: usize,
    old: bool,
) -> eyre::Result<()> {
    let q_row = NH * DIM;
    let nb_old = CandidateBlocks::n_blocks(ATTN_MIXED_MAX_KEYS);
    let nb_new = CandidateBlocks::n_blocks(stride);
    let nw = candidate_bit_words();
    let tk = INDEXER_TOP_K as usize;
    let mut r0 = 0usize;
    while r0 < b {
        let n = block.min(b - r0);
        let nb = n as u32;
        let n_per = n_idx.slice_view(r0, n);
        let hw_v = l.hw.slice_view(r0 * NH, n * NH);
        match kind {
            Score::Gemm => {
                let q16_v = l.q16.slice_view(r0 * q_row, n * q_row);
                wmma.launch_batched_gemm_e2m1(stream, &mut p.scores, &q16_v, &hw_v, &l.packed, &n_per, n_idx_max, stride, nb, 0)?;
            }
            Score::MwRows => {
                let q_v = l.q.slice_view(r0 * q_row, n * q_row);
                let kb_v = keys_base.slice_view(r0, n);
                wmma.launch_batched_mw_e2m1_rows(stream, &mut p.scores, &q_v, &hw_v, &l.packed, &n_per, n_idx_max, stride, nb, Some(&kb_v))?;
            }
        }
        match (old, which) {
            (true, 0) => {
                let nbs = nb_old as usize;
                let mut bs_v = p.block_score.slice_view_mut(r0 * nbs, n * nbs);
                let mut th_v = p.threshold.slice_view_mut(r0, n);
                cand.launch_build(stream, &p.scores, &mut bs_v, &mut th_v, n_per.raw(), stride, nb_old, n_idx_max, nb)?;
            }
            (true, _) => {
                let nbs = nb_old as usize;
                let bs_v = p.block_score.slice_view(r0 * nbs, n * nbs);
                let th_v = p.threshold.slice_view(r0, n);
                cand.launch_mask(stream, &mut p.scores, &bs_v, &th_v, n_per.raw(), stride, nb_old, n_idx_max, nb)?;
            }
            (false, 0) => {
                let mut th_v = p.threshold.slice_view_mut(r0, n);
                let mut bits_v = p.bits.slice_view_mut(r0 * nw, n * nw);
                cand.launch_build_bits(stream, &p.scores, &mut p.block_score, &mut th_v, &mut bits_v, n_per.raw(), stride, nb_new,
                    nw as u32, n_idx_max, nb)?;
            }
            (false, _) => {
                let bits_v = p.bits.slice_view(r0 * nw, n * nw);
                cand.launch_mask_bits(stream, &mut p.scores, &bits_v, n_per.raw(), stride, nw as u32, n_idx_max, nb)?;
            }
        }
        let mut sel_v = p.sel[which].slice_view_mut(r0 * tk, n * tk);
        let n_words = (ATTN_MIXED_MAX_KEYS + 31) / 32;
        topk.launch_batched(stream, &mut sel_v, None, &mut p.topk_scratch, &p.scores, &n_per, n_idx_max, stride, n_words,
            INDEXER_TOP_K, nb, Some(&mut p.done))?;
        r0 += n;
    }
    Ok(())
}

fn host<T: Copy + Default>(b: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); b.len()];
    b.copy_to_host(&mut v)?;
    Ok(v)
}

#[test]
fn indexer_row_blocks_match_full_pitch() -> eyre::Result<()> {
    install_panic_handler()?;
    let n_idx_max = env_u32("IRB_N_IDX", 40_003);
    let batch = env_u32("IRB_BATCH", 61) as usize;
    let block_req = env_u32("IRB_BLOCK", 16) as usize;
    let mut rng = Rng(0x1D3_B10C_5EED_0015);
    let mut ran = false;
    for d in Device::all()? {
        d.set_current()?;
        let arch = d.properties()?.gcn_arch_name.clone();
        if !arch.starts_with("gfx12") {
            continue;
        }
        ran = true;
        let stream = Stream::new(d.id)?;
        let wmma = IndexerScoreWmma::for_arch(&arch)?;
        let cand = CandidateBlocks::for_arch(&arch)?;
        let topk = IndexerTopkBitonic::for_arch(&arch)?;
        // Packed keys run KEY_BASE_SPAN rows past n_idx_max so the arena rows can start at distinct bases (a wrong or
        // unsliced per-row base then reads other keys and changes the selection).
        const KEY_BASE_SPAN: usize = 3 * 192;
        let layers = [layer(&d, &stream, &arch, &mut rng, n_idx_max as usize + KEY_BASE_SPAN, batch)?,
                      layer(&d, &stream, &arch, &mut rng, n_idx_max as usize + KEY_BASE_SPAN, batch)?];
        // Per-row n_idx, mixed depths behind one launch-wide n_idx_max (streams of different lengths in one step):
        // every 7th row at 3000 (one-sort top-k, vacuous candidates), the next at 12000 (vacuous candidates: t == 0),
        // the rest rising with the row like a prefill unit (row i sees i more keys), floor 600. The last row carries
        // n_idx_max (batch 61: 60 % 7 = 4).
        let n_idx_host: Vec<u32> = (0..batch)
            .map(|i| match i % 7 {
                0 => 3000.min(n_idx_max),
                1 => 12000.min(n_idx_max),
                _ => (n_idx_max - (batch - 1 - i) as u32 * 37).max(600),
            })
            .collect();
        if n_idx_host.iter().copied().max() != Some(n_idx_max) {
            return Err(eyre!("test shape: no row at n_idx_max (pick a batch whose last row is not a short one)"));
        }
        let mut n_idx = DeviceBuffer::<u32>::new(d.id, batch)?;
        n_idx.copy_from_host(&n_idx_host)?;
        let mut keys_base = DeviceBuffer::<u32>::new(d.id, batch)?;
        keys_base.copy_from_host(&(0..batch).map(|i| (i % 4) as u32 * 192).collect::<Vec<u32>>())?;

        let old_stride = ATTN_MIXED_MAX_KEYS;
        let new_stride = indexer_scores_stride(n_idx_max);
        let levels: usize = v4flash_kernels::indexer::topk_merge_levels(old_stride, INDEXER_TOP_K).iter().map(|&n| n as usize).sum();
        let nb_old = CandidateBlocks::n_blocks(ATTN_MIXED_MAX_KEYS) as usize;
        let nb_new = CandidateBlocks::n_blocks(new_stride) as usize;
        // NEW buffers hold `block_req` rows of every block-local member (scores, top-k, block scores) -- the forced block.
        let topk_new = block_req * indexer_topk_scratch_per_row(n_idx_max).max(1);
        let block = indexer_block_rows(batch, &[
            (block_req * new_stride as usize, new_stride as usize),
            (topk_new, indexer_topk_scratch_per_row(n_idx_max)),
            (block_req * nb_new, nb_new),
        ])
        .map_err(|e| eyre!(e))?;
        let mk = |old: bool| -> eyre::Result<Path> {
            let mut p = Path {
                scores: DeviceBuffer::new(d.id, if old { batch * old_stride as usize } else { block_req * new_stride as usize })?,
                topk_scratch: DeviceBuffer::new(d.id, if old { batch * levels.max(1) } else { topk_new })?,
                done: DeviceBuffer::new(d.id, batch)?,
                sel: [DeviceBuffer::new(d.id, batch * INDEXER_TOP_K as usize)?, DeviceBuffer::new(d.id, batch * INDEXER_TOP_K as usize)?],
                block_score: DeviceBuffer::new(d.id, if old { batch * nb_old } else { block_req * nb_new })?,
                threshold: DeviceBuffer::new(d.id, batch)?,
                bits: DeviceBuffer::new(d.id, batch * candidate_bit_words())?,
            };
            for s in p.sel.iter_mut() {
                s.fill_zero()?;
            }
            p.block_score.fill_zero()?;
            p.threshold.fill_zero()?;
            p.bits.fill_zero()?;
            Ok(p)
        };
        println!("device {} ({arch}): n_idx_max {n_idx_max}, batch {batch}, pitch {old_stride} -> {new_stride}, block {block} rows",
            d.id);
        for kind in [Score::Gemm, Score::MwRows] {
            let label = if kind == Score::Gemm { "gemm (prefill)" } else { "mw rows (arena/decode)" };
            let mut a = mk(true)?;
            let mut bp = mk(false)?;
            for (which, l) in layers.iter().enumerate() {
                run_layer(&stream, &wmma, &cand, &topk, &mut a, l, which, kind, &n_idx, &keys_base, n_idx_max, old_stride, batch, batch, true)?;
                run_layer(&stream, &wmma, &cand, &topk, &mut bp, l, which, kind, &n_idx, &keys_base, n_idx_max, new_stride, block, batch, false)?;
            }
            stream.synchronize()?;
            for which in 0..2 {
                let (sa, sb) = (host(&a.sel[which])?, host(&bp.sel[which])?);
                if let Some(i) = (0..sa.len()).find(|&i| sa[i] != sb[i]) {
                    return Err(eyre!("{label} layer {which}: selection differs at row {} slot {} ({} vs {})",
                        i / INDEXER_TOP_K as usize, i % INDEXER_TOP_K as usize, sa[i], sb[i]));
                }
            }
            if host(&a.threshold)? != host(&bp.threshold)? {
                return Err(eyre!("{label}: candidate thresholds differ"));
            }
            // The mask must have bitten (not the vacuous regime), or this compares nothing.
            let cut = host(&a.threshold)?.iter().filter(|&&t| t != 0).count();
            if cut == 0 {
                return Err(eyre!("{label}: no row past the candidate threshold: raise IRB_N_IDX above 16384"));
            }
            let picked = host(&a.sel[1])?.iter().filter(|&&v| v >= 0).count();
            println!("  {label}: both layers' selections bit-identical ({batch} rows x {INDEXER_TOP_K}, {picked} valid picks; {cut} rows cut by the candidate mask)");
        }

        // Timing: the gemm alone, all rows, at the old pitch vs the new one.
        let iters = env_u32("IRB_ITERS", 20);
        for (label, stride) in [("old pitch", old_stride), ("new pitch", new_stride)] {
            let mut s = DeviceBuffer::<f32>::new(d.id, batch * stride as usize)?;
            let l = &layers[0];
            wmma.launch_batched_gemm_e2m1(&stream, &mut s, &l.q16, &l.hw, &l.packed, &n_idx, n_idx_max, stride, batch as u32, 0)?;
            stream.synchronize()?;
            let t = Instant::now();
            for _ in 0..iters {
                wmma.launch_batched_gemm_e2m1(&stream, &mut s, &l.q16, &l.hw, &l.packed, &n_idx, n_idx_max, stride, batch as u32, 0)?;
            }
            stream.synchronize()?;
            println!("  gemm {label} {stride}: {:.1} us / launch", t.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
    }
    if !ran {
        println!("no gfx12 device: skipped");
    }
    Ok(())
}

#[test]
fn indexer_block_rows_arithmetic() {
    assert_eq!(indexer_scores_stride(0), 64);
    assert_eq!(indexer_scores_stride(1), 64);
    assert_eq!(indexer_scores_stride(64), 64);
    assert_eq!(indexer_scores_stride(65), 128);
    assert_eq!(indexer_scores_stride(368_640), 368_640);
    // Fits: all rows in one block (the ordinary case: nothing changes but the pitch).
    assert_eq!(indexer_block_rows(512, &[(512 * 4096, 4096), (usize::MAX, 4096)]), Ok(512));
    assert_eq!(indexer_block_rows(0, &[(100, 64), (0, 0)]), Ok(1));
    // Scores do not fit: whole 8-row gemm groups.
    assert_eq!(indexer_block_rows(512, &[(100 * 1024, 1024), (usize::MAX, 512)]), Ok(96));
    assert_eq!(indexer_block_rows(512, &[(5 * 1024, 1024), (usize::MAX, 512)]), Ok(5));
    // The top-k scratch binds instead; and a third bound (the candidate block scores) that binds.
    assert_eq!(indexer_block_rows(512, &[(512 * 1024, 1024), (usize::MAX, 0), (24 * 1000, 1000)]), Ok(24));
    assert_eq!(indexer_block_rows(512, &[(512 * 1024, 1024), (40 * 9216, 9216)]), Ok(40));
    assert!(indexer_block_rows(4, &[(1023, 1024), (usize::MAX, 0)]).is_err());
    assert!(indexer_block_rows(4, &[(4096, 1024), (100, 512)]).is_err());
    // The ladder: nothing at <= 4096 keys (one sort), one level to 8 chunks, then folds by 8.
    assert_eq!(indexer_topk_scratch_per_row(4096), 0);
    assert_eq!(indexer_topk_scratch_per_row(8192), 1024);
    assert!(indexer_topk_scratch_per_row(1_048_576) > indexer_topk_scratch_per_row(368_640));
}
