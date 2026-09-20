//! Multi-stream decode step harness (docs/v41/MULTISTREAM_DECODE_PLAN.md 3.6 / 3.7,
//! M1a step 3): S prompts of different lengths are prefilled through the ordinary
//! batched prefill into S single-sequence states, admitted into two `KvArena`s,
//! and decoded T teacher-forced tokens three ways:
//!
//!   dec    — today's decode path (`forward_token_paged`), one stream at a time,
//!            on the stream's own state; its greedy tokens are the forced
//!            continuation for all three;
//!   alone  — the arena step (`forward_step_arena`) with ONE row per step;
//!   batch  — the arena step with all S rows co-batched.
//!
//! Gates:
//!   G5a  alone == batch bit-exactly (a row's output does not depend on its
//!        co-rows), unless MS_ALLOW_INEXACT=1 (then reported only);
//!   G5b  KLD(dec || batch) per token under MS_KLD_MEAN / MS_KLD_MAX (defaults
//!        0.02 / 0.5 nats; the decode-vs-verify baseline is ~1e-3).
//!
//! Needs the model loaded, i.e. the server DOWN. Run:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 V41_PAGED_EXPERTS=1 V41_INDEX_K=1 V41_CANDIDATE_POOL=1 \
//!   MS_STREAMS=4 MS_STEPS=6 CARGO_TARGET_DIR=target-v41 \
//!   nix develop -c cargo test -p v4flash-kernels --features v41 --release \
//!     --test multistream_step -- --ignored --nocapture
//! ```
//! `V41_HF_DIR` (default: the production snapshot), `V41_ENGRAM_DIR`,
//! `V41_PAGER_POOL_GB` (default 40 here), `MS_LENS` (comma list of prompt lengths).
#![cfg(feature = "v41")]

use std::path::Path;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{EngramHash, EngramTable, V41HfWeights, WeightSrc};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{
    COMPRESS_RATIOS, ENGRAM_IN, ENGRAM_LAYERS, HC_DIM, KV_SOURCE_LAYERS, N_VOCAB, SWA_WINDOW,
};
use v4flash_kernels::embed::embed_lookup;
use v4flash_kernels::het::kv_arena::{KvArena, RowTablesDev};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, DgpuScratch, ExecMode,
    ExpertPager, HetModelState, HetModelWeights, HeterogeneousEngine, IgpuScratch,
};
use v4flash_kernels::RopeParams;

const ROPE_ORIG_CTX: u64 = 65536;
const HF_DIR_DEFAULT: &str =
    "/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277";

fn rope_for_layer(layer: i32) -> eyre::Result<RopeParams> {
    let compressed = COMPRESS_RATIOS[layer as usize] != 0;
    let freq_base = if compressed { 160000.0 } else { 10000.0 };
    let freq_scale = if compressed { 1.0 / 16.0 } else { 1.0 };
    let ext_factor = if compressed { 1.0 } else { 0.0 };
    let mut attn_factor = 1.0f32;
    if ext_factor != 0.0 && freq_scale > 0.0 {
        attn_factor /= 1.0 + 0.1 * (1.0f32 / freq_scale).ln();
    }
    let n_ctx_orig = if compressed { ROPE_ORIG_CTX } else { 0 };
    RopeParams::from_dump_blob(&[freq_base, freq_scale, ext_factor, attn_factor, 32.0, 1.0], n_ctx_orig)
}

fn pick(prefix: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(prefix) {
            return Ok(d);
        }
    }
    Err(eyre!("no {prefix} device"))
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Deterministic pseudo-prompt: ids in [1000, 30000), per-stream seed.
fn synth_prompt(seed: u64, len: usize) -> Vec<i32> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            1000 + ((x >> 33) % 29000) as i32
        })
        .collect()
}

struct Engram {
    hasher: EngramHash,
    tables: Vec<EngramTable>,
}

impl Engram {
    /// One `[len * ENGRAM_IN]` buffer per Engram layer for a whole prompt.
    fn rows_for_prompt(&self, st: &v4flash_core::SafetensorsDir, tokens: &[i32]) -> eyre::Result<Vec<Vec<f32>>> {
        let ein = ENGRAM_IN as usize;
        let hashes = self.hasher.hash_sequence(tokens);
        let mut out = vec![vec![0f32; tokens.len() * ein]; self.tables.len()];
        for (li, tbl) in self.tables.iter().enumerate() {
            for t in 0..tokens.len() {
                tbl.gather_position(st, &hashes[t][li], &mut out[li][t * ein..(t + 1) * ein])?;
            }
        }
        Ok(out)
    }
    /// One `[ENGRAM_IN]` row per Engram layer for the token at `pos` of `seq`
    /// (`seq[..=pos]` is the sequence so far).
    fn rows_at(&self, st: &v4flash_core::SafetensorsDir, seq: &[i32], pos: usize) -> eyre::Result<Vec<Vec<f32>>> {
        let ein = ENGRAM_IN as usize;
        let compressed: Vec<i32> = seq[..=pos].iter().map(|&t| self.hasher.compress(t)).collect();
        let hashes = self.hasher.hash_ids(&compressed, pos);
        let mut out = Vec::with_capacity(self.tables.len());
        for (li, tbl) in self.tables.iter().enumerate() {
            let mut r = vec![0f32; ein];
            tbl.gather_position(st, &hashes[li], &mut r)?;
            out.push(r);
        }
        Ok(out)
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// KL(p || q) in nats over softmaxes of two logit rows (f64).
fn kld(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    fn lse(v: &[f32]) -> f64 {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    }
    let lp = lse(p_logits);
    let lq = lse(q_logits);
    p_logits
        .iter()
        .zip(q_logits)
        .map(|(&a, &b)| {
            let lpa = a as f64 - lp;
            lpa.exp() * (lpa - (b as f64 - lq))
        })
        .sum()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
#[ignore]
fn multistream_step_matches_alone_and_decode() -> eyre::Result<()> {
    install_panic_handler()?;
    if std::env::var("V41_PAGED_EXPERTS").is_err() {
        std::env::set_var("V41_PAGED_EXPERTS", "1");
    }
    if std::env::var("V41_PAGER_POOL_GB").is_err() {
        std::env::set_var("V41_PAGER_POOL_GB", "40");
    }
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| HF_DIR_DEFAULT.to_string());
    let engram_dir = std::env::var("V41_ENGRAM_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/v41/engram", std::env::var("HOME").unwrap_or_default())
    });
    let n_streams = env_usize("MS_STREAMS", 4);
    let n_steps = env_usize("MS_STEPS", 6);
    let lens: Vec<usize> = match std::env::var("MS_LENS") {
        Ok(v) => v.split(',').map(|s| s.trim().parse().unwrap()).collect(),
        // Below / at / past the SWA window, odd and even (ratio-2 parity).
        Err(_) => vec![5, 40, 131, 260],
    };
    let allow_inexact = std::env::var("MS_ALLOW_INEXACT").as_deref() == Ok("1");
    let kld_mean_bar = env_f64("MS_KLD_MEAN", 0.02);
    let kld_max_bar = env_f64("MS_KLD_MAX", 0.5);
    let prompts: Vec<Vec<i32>> = (0..n_streams).map(|s| synth_prompt(s as u64 + 1, lens[s % lens.len()])).collect();
    let max_len = prompts.iter().map(|p| p.len()).max().unwrap();
    eprintln!(
        "multistream harness: S={n_streams} T={n_steps} lens={:?}",
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );

    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    let darch = dgpu.properties()?.gcn_arch_name;
    let iarch = igpu.properties()?.gcn_arch_name;
    let hf = V41HfWeights::open(&dir, None)?;
    let src = WeightSrc::from(&hf);
    let te = src.tensor("token_embd.weight").ok_or_else(|| eyre!("missing token_embd.weight"))?;
    let te_dtype = te.dtype;
    let te_bytes = src.read_tensor(te)?;
    let embed = |tok: i32| -> eyre::Result<Vec<f32>> {
        let mut r = vec![0f32; HC_DIM as usize];
        embed_lookup(&te_bytes, te_dtype, tok, &mut r)?;
        Ok(r)
    };

    eprintln!("loading weights (paged experts)...");
    let t0 = std::time::Instant::now();
    let weights = HetModelWeights::load_all(src, dgpu, igpu, &rope_for_layer)?;
    eprintln!("weights loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let engine = HeterogeneousEngine::new(dgpu, &darch, igpu, &iarch, ExecMode::HetParallel)?;
    let mut ds = DgpuScratch::alloc(dgpu)?;
    let mut is = IgpuScratch::alloc(igpu)?;
    let n_kv_max: u32 = (max_len + n_steps + 64).next_power_of_two().max(1024) as u32;
    let lane_rows = v4flash_kernels::het::batch_scratch::B_MAX.div_ceil(2);
    let mut bd_a = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_a = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut sd = BatchDgpuShared::alloc_rows_ctx(dgpu, lane_rows, n_kv_max)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    let mut pg = ExpertPager::new(V41HfWeights::open(&dir, None)?, igpu, 0)?;
    let hasher = EngramHash::load(Path::new(&engram_dir))?;
    hasher.self_check()?;
    let mut tables = Vec::new();
    for &l in ENGRAM_LAYERS {
        tables.push(EngramTable::open(pg.raw(), l as usize)?);
    }
    let engram = Engram { hasher, tables };

    // Arenas: one slot per stream in both; the store cap covers every stream.
    let comp_rows_cap = (n_streams as u32) * n_kv_max;
    let mut arena_alone = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_batch = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut dev = RowTablesDev::alloc(dgpu, n_streams as u32, KV_SOURCE_LAYERS.len())?;

    // 1. Prefill each prompt on its own state; admit into both arenas.
    let mut states: Vec<HetModelState> = Vec::new();
    let mut first_tok: Vec<i32> = Vec::new();
    let mut slots_alone: Vec<u32> = Vec::new();
    let mut slots_batch: Vec<u32> = Vec::new();
    for (s, toks) in prompts.iter().enumerate() {
        let mut st = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let hcs: Vec<Vec<f32>> = toks.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
        let rows = engram.rows_for_prompt(pg.raw(), toks)?;
        let t = std::time::Instant::now();
        let logits = engine.forward_prefill_pipelined(
            &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights,
            &hcs, toks, 0, true, None, None, None, None, Some(&mut pg), Some(&rows),
        )?;
        st.restore_compressor_lending();
        eprintln!("stream {s}: prefill {} tokens in {:.2} s", toks.len(), t.elapsed().as_secs_f64());
        let pos = toks.len() as u32;
        let cap = pos + n_steps as u32 + 8;
        slots_alone.push(arena_alone.admit_from_state(&st, cap, pos, &engine.dgpu.compute)?);
        slots_batch.push(arena_batch.admit_from_state(&st, cap, pos, &engine.dgpu.compute)?);
        engine.dgpu.compute.synchronize()?;
        first_tok.push(argmax(&logits) as i32);
        states.push(st);
    }
    let st_n_raw: Vec<u32> = states.iter().map(|s| s.layers[0].n_raw).collect();
    eprintln!("admitted; raw windows {:?} (W={SWA_WINDOW})", st_n_raw);

    // 2. Decode oracle, greedy, per stream: forced continuation + logits.
    let mut cont: Vec<Vec<i32>> = vec![Vec::new(); n_streams];
    let mut logits_dec: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let nv = N_VOCAB as usize;
    for s in 0..n_streams {
        let mut seq = prompts[s].clone();
        let mut tok = first_tok[s];
        for _ in 0..n_steps {
            let pos = seq.len();
            seq.push(tok);
            let rows = engram.rows_at(pg.raw(), &seq, pos)?;
            let hc = embed(tok)?;
            engine.forward_token_paged(&mut ds, &mut is, &mut states[s], &weights, &hc, pos as u32, tok, &mut pg, Some(&rows))?;
            engine.dgpu.compute.synchronize()?;
            let mut l = vec![0f32; nv];
            ds.logits.slice_view(0, nv).copy_to_host(&mut l)?;
            cont[s].push(tok);
            tok = argmax(&l) as i32;
            logits_dec[s].push(l);
        }
        eprintln!("stream {s}: decode oracle tokens {:?}", cont[s]);
    }

    // 3. Arena, one row per step (alone).
    let mut logits_alone: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    for s in 0..n_streams {
        let mut seq = prompts[s].clone();
        for t in 0..n_steps {
            let tok = cont[s][t];
            let pos = seq.len();
            seq.push(tok);
            let rows = engram.rows_at(pg.raw(), &seq, pos)?;
            let rows_b: Vec<Vec<f32>> = rows;
            engine.forward_step_arena(
                &mut bd_a, &mut bi_a, &mut sd, &mut si, &mut arena_alone, &mut dev, &[slots_alone[s]], &weights,
                &[embed(tok)?], &[tok], Some(&rows_b), Some(&mut pg),
            )?;
            let l = engine.head_rows(&mut ds, &bd_a, 1, &weights)?;
            logits_alone[s].push(l);
        }
    }

    // 4. Arena, all rows co-batched.
    let mut logits_batch: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let mut seqs: Vec<Vec<i32>> = prompts.clone();
    let ein = ENGRAM_IN as usize;
    let mut step_ms = Vec::new();
    for t in 0..n_steps {
        let toks: Vec<i32> = (0..n_streams).map(|s| cont[s][t]).collect();
        let mut rows_b = vec![vec![0f32; n_streams * ein]; ENGRAM_LAYERS.len()];
        let mut hcs = Vec::with_capacity(n_streams);
        for s in 0..n_streams {
            let pos = seqs[s].len();
            seqs[s].push(toks[s]);
            let rows = engram.rows_at(pg.raw(), &seqs[s], pos)?;
            for (li, r) in rows.iter().enumerate() {
                rows_b[li][s * ein..(s + 1) * ein].copy_from_slice(r);
            }
            hcs.push(embed(toks[s])?);
        }
        let t0 = std::time::Instant::now();
        engine.forward_step_arena(
            &mut bd_a, &mut bi_a, &mut sd, &mut si, &mut arena_batch, &mut dev, &slots_batch, &weights,
            &hcs, &toks, Some(&rows_b), Some(&mut pg),
        )?;
        let all = engine.head_rows(&mut ds, &bd_a, n_streams, &weights)?;
        step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        for s in 0..n_streams {
            logits_batch[s].push(all[s * nv..(s + 1) * nv].to_vec());
        }
    }
    eprintln!("batched step wall (S={n_streams}, incl. head): {:?} ms", step_ms.iter().map(|x| (*x * 10.0).round() / 10.0).collect::<Vec<_>>());

    // 5. Compare.
    let mut g5a_fail = 0;
    let mut kls = Vec::new();
    let mut kls_alone = Vec::new();
    eprintln!(" s  t   |alone-batch|  argmax(dec/alone/batch)   KL(dec||batch)  KL(dec||alone)");
    for s in 0..n_streams {
        for t in 0..n_steps {
            let d = max_abs_diff(&logits_alone[s][t], &logits_batch[s][t]);
            let kb = kld(&logits_dec[s][t], &logits_batch[s][t]);
            let ka = kld(&logits_dec[s][t], &logits_alone[s][t]);
            let (ad, aa, ab) = (argmax(&logits_dec[s][t]), argmax(&logits_alone[s][t]), argmax(&logits_batch[s][t]));
            if d != 0.0 {
                g5a_fail += 1;
            }
            kls.push(kb);
            kls_alone.push(ka);
            eprintln!("{s:2} {t:2}   {d:10.3e}   {ad:6}/{aa:6}/{ab:6}          {kb:9.5}       {ka:9.5}");
        }
    }
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let max = kls.iter().cloned().fold(0.0, f64::max);
    let mean_a = kls_alone.iter().sum::<f64>() / kls_alone.len() as f64;
    eprintln!(
        "G5a: {} of {} (stream, step) rows differ between alone and batched (want 0)",
        g5a_fail,
        kls.len()
    );
    eprintln!("G5b: KL(dec||batch) mean {mean:.5} max {max:.5} nats (bars {kld_mean_bar} / {kld_max_bar}); KL(dec||alone) mean {mean_a:.5}");
    if g5a_fail > 0 && !allow_inexact {
        return Err(eyre!("G5a failed: {g5a_fail} rows not batch-invariant (MS_ALLOW_INEXACT=1 to report only)"));
    }
    if mean > kld_mean_bar || max > kld_max_bar {
        return Err(eyre!("G5b failed: KL mean {mean:.5} / max {max:.5} over bars {kld_mean_bar} / {kld_max_bar}"));
    }
    Ok(())
}
