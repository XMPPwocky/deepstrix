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
//!   pipe   — the arena step with all S rows on the TWO-LANE PIPELINED driver
//!            (`forward_step_arena_pipelined`), which production uses from 4
//!            rows up and which interleaves the lanes' pre-MoE phases.
//!
//! Gates:
//!   G5a  alone == batch bit-exactly (a row's output does not depend on its
//!        co-rows), unless MS_ALLOW_INEXACT=1 (then reported only);
//!   G5b  KLD(dec || batch) per token under MS_KLD_MEAN / MS_KLD_MAX (defaults
//!        0.02 / 0.5 nats; the decode-vs-verify baseline is ~1e-3).
//!        NOTE: with the default tiny prompts (5/40/131/260) and the two-box
//!        split attached, the BASELINE itself sits at mean ~0.08 / max ~0.43,
//!        so G5b fails on an unmodified tree; raise MS_KLD_MEAN for that
//!        configuration. G5a and G5c are the gates that discriminate.
//!   G5c  alone == pipe bit-exactly, same bars as G5b on KLD(dec || pipe). This
//!        is the gate on the LANE INTERLEAVING: shared scratch clobbered across
//!        lanes shows up as exactly one lane's rows differing (a `lane` column
//!        in the per-row table says which). It caught two such hazards when the
//!        phases were first split on 2026-09-22 -- `BatchIgpuShared` scratch and
//!        the per-layer `remap_dev` exclusion mask -- both of which the ordinary
//!        `alone`/`batch` gates are blind to because they are single-lane.
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

fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 { s << 31 } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 { (s << 31) | 0x7f80_0000 | (m << 13) } else { (s << 31) | ((e + 127 - 15) << 23) | (m << 13) };
    f32::from_bits(bits)
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
    // MS_FORCE_CONT="a,b,c": teacher-force these tokens through the decode
    // oracle (and everything else) instead of decode's own greedy picks;
    // overrides MS_STEPS with its length.
    let force_cont: Option<Vec<i32>> = std::env::var("MS_FORCE_CONT").ok().map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
    let n_steps = force_cont.as_ref().map(|c| c.len()).unwrap_or_else(|| env_usize("MS_STEPS", 6));
    let lens: Vec<usize> = match std::env::var("MS_LENS") {
        Ok(v) => v.split(',').map(|s| s.trim().parse().unwrap()).collect(),
        // Below / at / past the SWA window, odd and even (ratio-2 parity).
        Err(_) => vec![5, 40, 131, 260],
    };
    let allow_inexact = std::env::var("MS_ALLOW_INEXACT").as_deref() == Ok("1");
    let kld_mean_bar = env_f64("MS_KLD_MEAN", 0.02);
    let kld_max_bar = env_f64("MS_KLD_MAX", 0.5);
    // MS_PROMPT_IDS="a,b,c,...": stream 0 runs THIS sequence — prompt = all but
    // the last id, and the last id is the forced first decode token (so the
    // step-0 logits line up with an oracle dump of the full sequence).
    let forced: Option<Vec<i32>> = std::env::var("MS_PROMPT_IDS").ok().map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
    let mut prompts: Vec<Vec<i32>> = (0..n_streams).map(|s| synth_prompt(s as u64 + 1, lens[s % lens.len()])).collect();
    if let Some(f) = &forced {
        prompts[0] = f[..f.len() - 1].to_vec();
    }
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
    // MS_ENGINE2=1: the decode oracle runs on a SECOND engine instance (own
    // streams, events, graph caches, atomics), everything else shared.
    let engine_dec = if std::env::var("MS_ENGINE2").as_deref() == Ok("1") {
        Some(HeterogeneousEngine::new(dgpu, &darch, igpu, &iarch, ExecMode::HetParallel)?)
    } else {
        None
    };
    let eng_dec: &HeterogeneousEngine = engine_dec.as_ref().unwrap_or(&engine);
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
    let mut arena_pipe = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut dev_b = RowTablesDev::alloc(dgpu, n_streams as u32, KV_SOURCE_LAYERS.len())?;
    let mut dev = RowTablesDev::alloc(dgpu, n_streams as u32, KV_SOURCE_LAYERS.len())?;

    // 1. Prefill each prompt on its own state; admit into both arenas.
    let mut states: Vec<HetModelState> = Vec::new();
    let mut first_tok: Vec<i32> = Vec::new();
    let mut prefill_logits: Vec<Vec<f32>> = Vec::new();
    let mut slots_alone: Vec<u32> = Vec::new();
    let mut slots_batch: Vec<u32> = Vec::new();
    let mut slots_pipe: Vec<u32> = Vec::new();
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
        slots_pipe.push(arena_pipe.admit_from_state(&st, cap, pos, &engine.dgpu.compute)?);
        engine.dgpu.compute.synchronize()?;
        first_tok.push(if s == 0 { forced.as_ref().map(|f| *f.last().unwrap()) } else { None }.unwrap_or(argmax(&logits) as i32));
        prefill_logits.push(logits);
        states.push(st);
    }
    // MS_DIAG=kv: prefill each prompt a SECOND time and compare the two states'
    // KV bit for bit (raw windows per layer, comp rows / keys / accumulators per
    // store). Says whether two prefills of one prompt even agree before any
    // decode-vs-prefill question is asked.
    // MS_DIAG=job[:rows]: the second prefill runs through `PrefillJob` (chunk by
    // chunk, `rows` per chunk, default 3) — must be bit-identical to the one-shot
    // `forward_prefill_pipelined` in KV and in the last logits.
    let diag = std::env::var("MS_DIAG").unwrap_or_default();
    let job_rows: Option<usize> = diag.strip_prefix("job").map(|r| r.trim_start_matches(':').parse().unwrap_or(3));
    if diag == "kv" || job_rows.is_some() {
        let hd = v4flash_kernels::config::N_HEAD_DIM as usize;
        for (s, toks) in prompts.iter().enumerate() {
            let mut st2 = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
            let hcs: Vec<Vec<f32>> = toks.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
            let rows = engram.rows_for_prompt(pg.raw(), toks)?;
            let l2 = if let Some(cr) = job_rows {
                let mut job = v4flash_kernels::het::forward_prefill::PrefillJob::new(toks.clone(), hcs.clone(), Some(rows.clone()), None, 0, cr)?;
                let mut n_chunks = 0;
                while !job.chunks_done() {
                    engine.prefill_job_chunk(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st2, &weights, Some(&mut pg))?;
                    n_chunks += 1;
                }
                let l = engine.prefill_job_finish(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st2, &weights, Some(&mut pg))?;
                eprintln!("stream {s}: PrefillJob ran {n_chunks} chunks of <= {cr} rows");
                l
            } else {
                engine.forward_prefill_pipelined(
                    &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st2, &weights,
                    &hcs, toks, 0, true, None, None, None, None, Some(&mut pg), Some(&rows),
                )?
            };
            st2.restore_compressor_lending();
            engine.dgpu.compute.synchronize()?;
            let first_logits = &prefill_logits[s];
            let dl = max_abs_diff(first_logits, &l2);
            eprintln!("stream {s}: second prefill argmax {} (first {}); |logits diff| max {dl:.3e}; KL(first||second) {:.5}; per-layer raw-window compare:", argmax(&l2), argmax(first_logits), kld(first_logits, &l2));
            let a = &states[s];
            for l in 0..v4flash_kernels::config::N_LAYER as usize {
                let (la, lb) = (&a.layers[l], &st2.layers[l]);
                let n = la.n_raw.min(lb.n_raw) as usize;
                let mut x = vec![0u16; n * hd];
                let mut y = vec![0u16; n * hd];
                la.kv_cache.slice_view(la.raw_off as usize * hd, n * hd).copy_to_host(&mut x)?;
                lb.kv_cache.slice_view(lb.raw_off as usize * hd, n * hd).copy_to_host(&mut y)?;
                let rows_diff = (0..n).filter(|&r| x[r * hd..(r + 1) * hd] != y[r * hd..(r + 1) * hd]).count();
                let mut comp = String::new();
                if let (Some(ca), Some(cb)) = (la.compressor.as_ref(), lb.compressor.as_ref()) {
                    let nc = ca.n_comp.min(cb.n_comp) as usize;
                    let w = ca.width as usize;
                    let (Some(fa), Some(fb)) = (ca.comp_kv.f16(), cb.comp_kv.f16()) else { continue };
                    let mut cx = vec![0u16; nc * w];
                    let mut cy = vec![0u16; nc * w];
                    fa.slice_view(0, nc * w).copy_to_host(&mut cx)?;
                    fb.slice_view(0, nc * w).copy_to_host(&mut cy)?;
                    let cd = (0..nc).filter(|&r| cx[r * w..(r + 1) * w] != cy[r * w..(r + 1) * w]).count();
                    let mut sx = vec![0f32; ca.state_kv.len()];
                    let mut sy = vec![0f32; cb.state_kv.len()];
                    ca.state_kv.copy_to_host(&mut sx)?;
                    cb.state_kv.copy_to_host(&mut sy)?;
                    let sd_max = sx.iter().zip(&sy).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
                    comp = format!("  comp n={}/{} rows_diff={cd} state max|d|={sd_max:.3e}", ca.n_comp, cb.n_comp);
                }
                if rows_diff > 0 || !comp.is_empty() || l < 3 {
                    eprintln!("  L{l:2}: raw n={}/{} off={}/{} rows_diff={rows_diff}{comp}", la.n_raw, lb.n_raw, la.raw_off, lb.raw_off);
                }
            }
        }
        return Ok(());
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
            eng_dec.forward_token_paged(&mut ds, &mut is, &mut states[s], &weights, &hc, pos as u32, tok, &mut pg, Some(&rows))?;
            eng_dec.dgpu.compute.synchronize()?;
            let mut l = vec![0f32; nv];
            ds.logits.slice_view(0, nv).copy_to_host(&mut l)?;
            cont[s].push(tok);
            tok = match (&force_cont, s) {
                (Some(c), 0) if cont[s].len() < c.len() => c[cont[s].len()],
                _ => argmax(&l) as i32,
            };
            logits_dec[s].push(l);
        }
        eprintln!("stream {s}: decode oracle tokens {:?}", cont[s]);
    }
    if std::env::var("MS_DEVSYNC").as_deref() == Ok("1") {
        dgpu.set_current()?;
        dgpu.synchronize()?;
        igpu.set_current()?;
        igpu.synchronize()?;
        dgpu.set_current()?;
        engine.invalidate_device_cache();
        eprintln!("MS_DEVSYNC: both devices synchronized after the decode oracle");
    }

    // MS_SPEC=1: hold the DSpark verify's `SpeculativeAppend` scope over the
    // continuation steps (pf1 AND the arena). It flips the batched driver's
    // eviction (none), the sparse-residency predicate, and — the suspect — the
    // pager's "count this as prefill + scan window" mode.
    let spec_guard = if std::env::var("MS_SPEC").as_deref() == Ok("1") {
        Some(v4flash_kernels::het::forward_prefill::SpeculativeAppend::begin())
    } else {
        None
    };

    // 2b. The CONTIGUOUS batched path, one token per call on a fresh prefilled
    //     state (what the DSpark verify runs at B=1): separates "arena arm
    //     wrong" (pf1 != alone) from "prefill family vs decode" (pf1 == alone).
    let mut logits_pf1: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let skip_pf1 = std::env::var("MS_SKIP_PF1").as_deref() == Ok("1");
    if skip_pf1 {
        logits_pf1 = vec![vec![vec![0f32; nv]; n_steps]; n_streams];
    }
    for s in 0..n_streams {
        if skip_pf1 { break; }
        let toks = &prompts[s];
        let mut st = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let hcs: Vec<Vec<f32>> = toks.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
        let rows = engram.rows_for_prompt(pg.raw(), toks)?;
        let _ = engine.forward_prefill_pipelined(
            &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights,
            &hcs, toks, 0, true, None, None, None, None, Some(&mut pg), Some(&rows),
        )?;
        st.restore_compressor_lending();
        let mut seq = toks.clone();
        for t in 0..n_steps {
            let tok = cont[s][t];
            let pos = seq.len();
            seq.push(tok);
            let rows = engram.rows_at(pg.raw(), &seq, pos)?;
            engine.normalize_raw_windows(&mut ds, &mut st)?;
            // last_only=false: `ced_enabled() && last_only` would otherwise turn
            // CED on and replay the decoder over THIS call's rows only (the
            // one new token), wiping layers 20-39's window. The DSpark verify
            // passes false for the same reason.
            let l = engine.forward_prefill_pipelined(
                &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights,
                &[embed(tok)?], &[tok], pos as u32, false, None, None, None, None, Some(&mut pg), Some(&rows),
            )?;
            st.restore_compressor_lending();
            logits_pf1[s].push(l);
            // MS_DIAG=kvrow (after step 0): compare this state's raw windows with the
            // decode state's, row by row, per layer. Rows before the step must be
            // identical (same prefill); the step's own row shows whether the two
            // paths WRITE the same K/V.
            if t == 0 && std::env::var("MS_DIAG").as_deref() == Ok("kvrow") {
                let hd = v4flash_kernels::config::N_HEAD_DIM as usize;
                let f = |u: u16| -> f32 { half_to_f32(u) };
                eprintln!("kvrow: layer  n_raw(dec/pf1)  rows_identical/total  last-row relRMSE(pf1 vs dec)");
                for l in 0..v4flash_kernels::config::N_LAYER as usize {
                    let (la, lb) = (&states[s].layers[l], &st.layers[l]);
                    let n = la.n_raw.min(lb.n_raw) as usize;
                    let mut x = vec![0u16; n * hd];
                    let mut y = vec![0u16; n * hd];
                    la.kv_cache.slice_view(la.raw_off as usize * hd, n * hd).copy_to_host(&mut x)?;
                    lb.kv_cache.slice_view(lb.raw_off as usize * hd, n * hd).copy_to_host(&mut y)?;
                    let same = (0..n).filter(|&r| x[r * hd..(r + 1) * hd] == y[r * hd..(r + 1) * hd]).count();
                    let (mut num, mut den) = (0f64, 0f64);
                    for i in (n - 1) * hd..n * hd {
                        let (a, b) = (f(x[i]) as f64, f(y[i]) as f64);
                        num += (a - b) * (a - b);
                        den += a * a;
                    }
                    if l < 4 || same != n {
                        eprintln!("kvrow: L{l:2}  {}/{}  {same}/{n}  {:.3e}", la.n_raw, lb.n_raw, (num / den.max(1e-30)).sqrt());
                    }
                }
            }
        }
    }

    if let Ok(dir) = std::env::var("MS_SAVE_LOGITS") {
        std::fs::create_dir_all(&dir)?;
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for s in 0..n_streams {
            let mut seq = prompts[s].clone();
            seq.extend_from_slice(&cont[s]);
            std::fs::write(format!("{dir}/s{s}_tokens.json"), serde_json::to_string(&seq)?)?;
            for t in 0..n_steps {
                std::fs::write(format!("{dir}/s{s}_t{t}_dec.bin"), f32s(&logits_dec[s][t]))?;
                std::fs::write(format!("{dir}/s{s}_t{t}_pf1.bin"), f32s(&logits_pf1[s][t]))?;
            }
        }
        eprintln!("MS_SAVE_LOGITS: wrote {dir}");
    }
    if std::env::var("MS_STOP_AFTER").as_deref() == Ok("pf1") {
        let mut kls = Vec::new();
        for s in 0..n_streams {
            for t in 0..n_steps {
                let kp = kld(&logits_dec[s][t], &logits_pf1[s][t]);
                eprintln!("{s:2} {t:2}  argmax dec/pf1 {:6}/{:6}  KL(dec||pf1) {kp:.5}", argmax(&logits_dec[s][t]), argmax(&logits_pf1[s][t]));
                kls.push(kp);
            }
        }
        eprintln!("MS_STOP_AFTER=pf1: KL(dec||pf1) mean {:.5}", kls.iter().sum::<f64>() / kls.len() as f64);
        return Ok(());
    }

    // MS_FRESH_PAGER=1: throw the pager away and build a new one right before
    // the arena passes, so its pool/remap carry no history from the decode
    // oracle and the contiguous reference.
    if std::env::var("MS_FRESH_PAGER").as_deref() == Ok("1") {
        drop(pg);
        pg = ExpertPager::new(V41HfWeights::open(&dir, None)?, igpu, 0)?;
        eprintln!("MS_FRESH_PAGER: pager rebuilt");
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
                &[embed(tok)?], &[tok], &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b.clone())), Some(&mut pg),
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
            &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b.clone())), Some(&mut pg),
        )?;
        let all = engine.head_rows(&mut ds, &bd_a, n_streams, &weights)?;
        step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        for s in 0..n_streams {
            logits_batch[s].push(all[s * nv..(s + 1) * nv].to_vec());
        }
    }
    eprintln!("batched step wall (S={n_streams}, incl. head): {:?} ms", step_ms.iter().map(|x| (*x * 10.0).round() / 10.0).collect::<Vec<_>>());

    // 4b. Arena, all rows co-batched, TWO-LANE PIPELINED driver (the production
    // step at >= 6 rows). Since 2026-09-22 this driver runs the pre-MoE phases
    // in dependency-graph order (chain A, chain B, route A, route B, prep A,
    // prep B, launch A, launch B) with the cross-lane iGPU drain removed; the
    // invariant it relies on -- no MoE in flight during `ensure` -- is checked
    // inside the driver, and THIS comparison is what says the reorder kept the
    // numerics: same rows, same tokens, must match `alone` like `batch` does.
    let mut logits_pipe: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let mut seqs_p: Vec<Vec<i32>> = prompts.clone();
    let mut step_ms_p = Vec::new();
    let b_a = n_streams.div_ceil(2);
    for t in 0..n_steps {
        let toks: Vec<i32> = (0..n_streams).map(|s| cont[s][t]).collect();
        let mut rows_b = vec![vec![0f32; n_streams * ein]; ENGRAM_LAYERS.len()];
        let mut hcs = Vec::with_capacity(n_streams);
        for s in 0..n_streams {
            let pos = seqs_p[s].len();
            seqs_p[s].push(toks[s]);
            let rows = engram.rows_at(pg.raw(), &seqs_p[s], pos)?;
            for (li, r) in rows.iter().enumerate() {
                rows_b[li][s * ein..(s + 1) * ein].copy_from_slice(r);
            }
            hcs.push(embed(toks[s])?);
        }
        let t0 = std::time::Instant::now();
        engine.forward_step_arena_pipelined(
            &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut arena_pipe, &mut dev, &mut dev_b, &slots_pipe, &weights,
            &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b.clone())), Some(&mut pg),
        )?;
        let mut all = engine.head_rows(&mut ds, &bd_a, b_a, &weights)?;
        all.extend(engine.head_rows(&mut ds, &bd_b, n_streams - b_a, &weights)?);
        step_ms_p.push(t0.elapsed().as_secs_f64() * 1e3);
        for s in 0..n_streams {
            logits_pipe[s].push(all[s * nv..(s + 1) * nv].to_vec());
        }
    }
    eprintln!("pipelined step wall (S={n_streams}, 2 lanes, incl. head): {:?} ms", step_ms_p.iter().map(|x| (*x * 10.0).round() / 10.0).collect::<Vec<_>>());

    if let Ok(dir) = std::env::var("MS_SAVE_LOGITS") {
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for s in 0..n_streams {
            for t in 0..n_steps {
                std::fs::write(format!("{dir}/s{s}_t{t}_alone.bin"), f32s(&logits_alone[s][t]))?;
                std::fs::write(format!("{dir}/s{s}_t{t}_batch.bin"), f32s(&logits_batch[s][t]))?;
                std::fs::write(format!("{dir}/s{s}_t{t}_pipe.bin"), f32s(&logits_pipe[s][t]))?;
            }
        }
    }

    drop(spec_guard);

    // 5. Compare.
    let mut g5a_fail = 0;
    let mut g5c_fail = 0;
    let mut kls = Vec::new();
    let mut kls_alone = Vec::new();
    let mut kls_pipe = Vec::new();
    eprintln!(" s  t  lane  |alone-batch|  |alone-pipe|  argmax(dec/alone/batch/pipe)   KL(dec||batch)  KL(dec||pipe)");
    for s in 0..n_streams {
        for t in 0..n_steps {
            let d = max_abs_diff(&logits_alone[s][t], &logits_batch[s][t]);
            let dp = max_abs_diff(&logits_pf1[s][t], &logits_alone[s][t]);
            let kb = kld(&logits_dec[s][t], &logits_batch[s][t]);
            let ka = kld(&logits_dec[s][t], &logits_alone[s][t]);
            let kp = kld(&logits_dec[s][t], &logits_pf1[s][t]);
            let (ad, ap, aa, ab) = (argmax(&logits_dec[s][t]), argmax(&logits_pf1[s][t]), argmax(&logits_alone[s][t]), argmax(&logits_batch[s][t]));
            if d != 0.0 {
                g5a_fail += 1;
            }
            let dpipe = max_abs_diff(&logits_alone[s][t], &logits_pipe[s][t]);
            let kpipe = kld(&logits_dec[s][t], &logits_pipe[s][t]);
            if dpipe != 0.0 {
                g5c_fail += 1;
            }
            kls.push(kb);
            kls_alone.push(ka);
            kls_pipe.push(kpipe);
            let _ = (dp, ap, ka, kp);
            // `lane` is which pipeline lane the row belongs to (rows [0, b_a) =
            // A): a G5c failure that is entirely one lane means shared scratch
            // clobbered across lanes, which is how the 2026-09-22 regression
            // presented (12 of 24 rows = one lane).
            let lane = if s < b_a { "A" } else { "B" };
            eprintln!("{s:2} {t:2}   {lane}    {d:10.3e}    {dpipe:10.3e}   {ad:6}/{aa:6}/{ab:6}/{:6}     {kb:9.5}      {kpipe:9.5}", argmax(&logits_pipe[s][t]));
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
    let mean_p = kls_pipe.iter().sum::<f64>() / kls_pipe.len() as f64;
    let max_p = kls_pipe.iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "G5c: {} of {} (stream, step) rows differ between alone and PIPELINED (want 0); KL(dec||pipe) mean {mean_p:.5} max {max_p:.5}",
        g5c_fail,
        kls_pipe.len()
    );
    if g5a_fail > 0 && !allow_inexact {
        return Err(eyre!("G5a failed: {g5a_fail} rows not batch-invariant (MS_ALLOW_INEXACT=1 to report only)"));
    }
    if mean > kld_mean_bar || max > kld_max_bar {
        return Err(eyre!("G5b failed: KL mean {mean:.5} / max {max:.5} over bars {kld_mean_bar} / {kld_max_bar}"));
    }
    if g5c_fail > 0 && !allow_inexact {
        return Err(eyre!("G5c failed: {g5c_fail} rows not invariant between alone and the pipelined driver (MS_ALLOW_INEXACT=1 to report only)"));
    }
    if mean_p > kld_mean_bar || max_p > kld_max_bar {
        return Err(eyre!("G5c failed: pipelined KL mean {mean_p:.5} / max {max_p:.5} over bars {kld_mean_bar} / {kld_max_bar}"));
    }
    Ok(())
}
