//! Layer-major CED prefill (`V41_LM_PREFILL`, `PrefillJob::set_layer_major_rows`)
//! against chunked CED prefill: the same prompt, the same chunk size, in ONE
//! process on ONE weight load, must leave
//!
//!   * BIT-IDENTICAL KV state -- every layer's raw window ring (`kv_cache`,
//!     `n_raw`, `raw_off`) and every compressor (`comp_kv`, `index_k`,
//!     `state_kv`, `state_score`, `n_comp`, `n_index_comp`), and
//!   * BIT-IDENTICAL next-token logits from `prefill_job_finish` (the CED replay),
//!     plus the logits of a few greedy decode steps after it.
//!
//! Layer-major runs the chunks `plan_chunk` cuts, one layer GROUP at a time over
//! a window of them; each layer sees the same rows in the same order with the
//! same inputs, so anything but byte equality is a bug (a group boundary that
//! leaks per-call state, a wrong device seed, a row-offset slip...).
//!
//! Every case also runs chunked prefill TWICE (the determinism null) and fails
//! with "no verdict" if those two disagree; and fails if the layer-major run
//! ran no window at all (a silent fallback would compare chunked with chunked).
//!
//! Cases (`LM_CASES` to pick, default all):
//!   fresh     -- a 5,000-token prompt from position 0 at `LM_ROWS` (default 4096):
//!                one window, then a 904-row tail that is too short for a window
//!                (chunked), exercising the switch between the two;
//!   multi     -- the same prompt at 2048-row windows: several windows back to back;
//!   restored  -- a 1,500-token prefix prefilled chunked, then a 3,000-token suffix
//!                (pos0 = 1500) layer-major vs chunked on copies of that state;
//!   lazy      -- 5,000 tokens at 2048-row windows with the SERVER's input path
//!                (lazy per-unit inputs via next_chunk_range / set_chunk_inputs,
//!                incl. the window's kept Engram rows for layer 14) and a foreign
//!                one-token forward on the same lanes between every two units.
//!
//! Needs the model loaded, i.e. the server DOWN. Run:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 V41_PAGED_EXPERTS=1 V41_INDEX_K=1 V41_CANDIDATE_POOL=1 \
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels \
//!   --features v41 --release --test lm_prefill_bitexact -- --ignored --nocapture
//! ```

use std::path::Path;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{EngramHash, EngramTable, V41HfWeights, WeightSrc};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::config::{COMPRESS_RATIOS, ENGRAM_IN, ENGRAM_LAYERS, HC_DIM, N_LAYER};
use v4flash_kernels::embed::embed_lookup;
use v4flash_kernels::het::forward_prefill::PrefillJob;
use v4flash_kernels::het::state::CompKvStore;
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, DgpuScratch, ExecMode,
    ExpertPager, HetModelState, HetModelWeights, HeterogeneousEngine,
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

fn synth_prompt(seed: u64, len: usize) -> Vec<i32> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            1000 + ((x >> 33) % 29000) as i32
        })
        .collect()
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

fn host<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

/// Zero every buffer `diff_states` compares, so bytes neither run wrote compare equal.
fn zero_state(st: &mut HetModelState) -> eyre::Result<()> {
    for ls in st.layers.iter_mut() {
        ls.kv_cache.fill_zero()?;
        for cs in [ls.compressor.as_mut(), ls.indexer_compressor.as_mut()].into_iter().flatten() {
            cs.state_kv.fill_zero()?;
            match &mut cs.comp_kv {
                CompKvStore::F16(b) => b.fill_zero()?,
                CompKvStore::Fp8 { rows, head } => {
                    rows.fill_zero()?;
                    head.fill_zero()?;
                }
                CompKvStore::E2m1(b) => b.fill_zero()?,
            }
            if let Some(k) = cs.index_k.as_mut() {
                k.fill_zero()?;
            }
        }
    }
    Ok(())
}

/// Every difference between two states, as readable lines (empty = identical).
fn diff_states(a: &HetModelState, b: &HetModelState) -> eyre::Result<Vec<String>> {
    let mut out = Vec::new();
    for l in 0..N_LAYER as usize {
        let (x, y) = (&a.layers[l], &b.layers[l]);
        if (x.n_raw, x.raw_off) != (y.n_raw, y.raw_off) {
            out.push(format!("L{l}: n_raw/raw_off {}/{} vs {}/{}", x.n_raw, x.raw_off, y.n_raw, y.raw_off));
        }
        if host(&x.kv_cache)? != host(&y.kv_cache)? {
            out.push(format!("L{l}: kv_cache bytes differ"));
        }
        for (name, cx, cy) in [
            ("compressor", x.compressor.as_ref(), y.compressor.as_ref()),
            ("indexer_compressor", x.indexer_compressor.as_ref(), y.indexer_compressor.as_ref()),
        ] {
            match (cx, cy) {
                (None, None) => {}
                (Some(p), Some(q)) => {
                    if (p.n_comp, p.n_index_comp) != (q.n_comp, q.n_index_comp) {
                        out.push(format!("L{l} {name}: n_comp/n_index_comp {}/{} vs {}/{}", p.n_comp, p.n_index_comp, q.n_comp, q.n_index_comp));
                    }
                    if host(&p.state_kv)? != host(&q.state_kv)? || host(&p.state_score)? != host(&q.state_score)? {
                        out.push(format!("L{l} {name}: accumulator state differs"));
                    }
                    let same = match (&p.comp_kv, &q.comp_kv) {
                        (CompKvStore::F16(u), CompKvStore::F16(v)) => host(u)? == host(v)?,
                        (CompKvStore::E2m1(u), CompKvStore::E2m1(v)) => host(u)? == host(v)?,
                        (CompKvStore::Fp8 { rows: u, head: uh }, CompKvStore::Fp8 { rows: v, head: vh }) => {
                            host(u)? == host(v)? && host(uh)? == host(vh)?
                        }
                        _ => false,
                    };
                    if !same {
                        out.push(format!("L{l} {name}: comp_kv differs"));
                    }
                    match (p.index_k.as_ref(), q.index_k.as_ref()) {
                        (Some(u), Some(v)) if host(u)? != host(v)? => out.push(format!("L{l} {name}: index_k differs")),
                        (Some(_), None) | (None, Some(_)) => out.push(format!("L{l} {name}: index_k presence differs")),
                        _ => {}
                    }
                }
                _ => out.push(format!("L{l}: {name} presence differs")),
            }
        }
    }
    Ok(out)
}

#[test]
#[ignore]
fn layer_major_prefill_is_bit_identical_to_chunked() -> eyre::Result<()> {
    install_panic_handler()?;
    if std::env::var("V41_PAGED_EXPERTS").is_err() {
        std::env::set_var("V41_PAGED_EXPERTS", "1");
    }
    if std::env::var("V41_PAGER_POOL_GB").is_err() {
        std::env::set_var("V41_PAGER_POOL_GB", "40");
    }
    if std::env::var("V41_CED").as_deref() == Ok("0") {
        return Err(eyre!("layer-major prefill is a CED driver; unset V41_CED=0"));
    }
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| HF_DIR_DEFAULT.to_string());
    let engram_dir = std::env::var("V41_ENGRAM_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/v41/engram", std::env::var("HOME").unwrap_or_default())
    });
    let lm_rows = env_usize("LM_ROWS", 4096);
    let decode_steps = env_usize("LM_DECODE_STEPS", 3);
    let cases: Vec<String> = std::env::var("LM_CASES")
        .unwrap_or_else(|_| "fresh,multi,restored,lazy".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

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
    let weights = HetModelWeights::load_all(src, dgpu, igpu, &rope_for_layer)?;
    let engine = HeterogeneousEngine::new(dgpu, &darch, igpu, &iarch, ExecMode::HetParallel)?;
    let mut ds = DgpuScratch::alloc(dgpu)?;
    let n_kv_max: u32 = 8192;
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
    let ein = ENGRAM_IN as usize;
    // Engram rows for `toks[from..]` (hashed over the whole sequence).
    let rows_for = |raw: &v4flash_core::SafetensorsDir, toks: &[i32], from: usize| -> eyre::Result<Vec<Vec<f32>>> {
        let hashes = hasher.hash_sequence(toks);
        let n = toks.len() - from;
        let mut out = vec![vec![0f32; n * ein]; tables.len()];
        for (li, tbl) in tables.iter().enumerate() {
            for t in from..toks.len() {
                let k = t - from;
                tbl.gather_position(raw, &hashes[t][li], &mut out[li][k * ein..(k + 1) * ein])?;
            }
        }
        Ok(out)
    };

    // A second sequence advanced one token between layer-major units (`foreign`):
    // stands in for the decode steps the scheduler runs on the same lanes, pager
    // and streams between two units of a prefill job.
    let mut x_state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
    // Starts empty: its first forward is position 0 on an empty state.
    let mut x_seq: Vec<i32> = Vec::new();

    // One prefill job over `toks[pos0..]` on `st` (which holds `toks[..pos0]`);
    // `rows` = the Engram rows of `toks[pos0..]`. `lazy`: inputs handed over per
    // unit through `next_chunk_range` / `set_chunk_inputs`, as the server does
    // (`multistream::chunk_inputs`). Returns the logits and the layer-major
    // windows the job ran.
    let mut run = |toks: &[i32], pos0: usize, rows: Vec<Vec<f32>>, st: &mut HetModelState, lm: usize,
                   lazy: bool, foreign: bool, pg: &mut ExpertPager| -> eyre::Result<(Vec<f32>, usize)> {
        let sfx = &toks[pos0..];
        let hcs: Vec<Vec<f32>> = sfx.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
        let mut job = if lazy {
            PrefillJob::new(sfx.to_vec(), Vec::new(), None, None, pos0 as u32, 1024)?
        } else {
            PrefillJob::new(sfx.to_vec(), hcs.clone(), Some(rows.clone()), None, pos0 as u32, 1024)?
        };
        job.set_layer_major_rows(lm)?;
        let mut units = 0usize;
        while !job.chunks_done() {
            if lazy {
                let (a, z) = job.next_chunk_range((bd_a.rows, bd_b.rows))?;
                let eng: Vec<Vec<f32>> = rows.iter().map(|r| r[a * ein..z * ein].to_vec()).collect();
                job.set_chunk_inputs(hcs[a..z].to_vec(), Some(eng));
            }
            engine.prefill_job_chunk(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, st, &weights, Some(&mut *pg))?;
            units += 1;
            if foreign && !job.checkpoint_ok() {
                // Mid-window: another sequence's one-token forward on the same lanes.
                let t = synth_prompt(1000 + x_seq.len() as u64, 1)[0];
                x_seq.push(t);
                let pos = x_seq.len() - 1;
                let xr = rows_for(pg.raw(), &x_seq, pos)?;
                let mut xj = PrefillJob::new(vec![t], vec![embed(t)?], Some(xr), None, pos as u32, 1024)?;
                xj.set_layer_major_rows(0)?;
                while !xj.chunks_done() {
                    engine.prefill_job_chunk(&mut xj, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut x_state, &weights, Some(&mut *pg))?;
                }
                engine.prefill_job_finish(&mut xj, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut x_state, &weights, Some(&mut *pg))?;
                x_state.restore_compressor_lending();
            }
        }
        let windows = job.lm_windows_run();
        eprintln!("    job pos0={pos0} rows={} lm_rows={lm} lazy={lazy} foreign={foreign}: {units} units, {windows} layer-major windows", sfx.len());
        let l = engine.prefill_job_finish(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, st, &weights, Some(&mut *pg))?;
        st.restore_compressor_lending();
        Ok((l, windows))
    };
    let fresh_state = || -> eyre::Result<HetModelState> {
        let mut st = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        zero_state(&mut st)?;
        // Production's initial compressor state (score accumulators at -inf).
        st.reset_in_place(dgpu, igpu)?;
        Ok(st)
    };

    let mut failures = Vec::new();
    for case in &cases {
        // (prompt, pos0, window, lazy + foreign units on the layer-major run)
        let (prompt, pos0, rows, lazy) = match case.as_str() {
            "fresh" => (synth_prompt(11, 5000), 0usize, lm_rows, false),
            "multi" => (synth_prompt(12, 5000), 0, 2048, false),
            "restored" => (synth_prompt(13, 4500), 1500, lm_rows, false),
            "lazy" => (synth_prompt(14, 5000), 0, 2048, true),
            other => return Err(eyre!("unknown LM_CASES entry {other}")),
        };
        eprintln!("case {case}: {} tokens, pos0 {pos0}, window {rows}, lazy+foreign {lazy}", prompt.len());
        let rows_all = rows_for(pg.raw(), &prompt, 0)?;
        let slice = |a: usize, z: usize| -> Vec<Vec<f32>> { rows_all.iter().map(|r| r[a * ein..z * ein].to_vec()).collect() };
        // c = chunked, n = chunked again (the determinism NULL), l = layer-major.
        let mut st_c = fresh_state()?;
        let mut st_n = fresh_state()?;
        let mut st_l = fresh_state()?;
        if pos0 > 0 {
            // The shared prefix, chunked, on all three states.
            let pre = &prompt[..pos0];
            for st in [&mut st_c, &mut st_n, &mut st_l] {
                run(pre, 0, slice(0, pos0), st, 0, false, false, &mut pg)?;
            }
        }
        let (l_c, _) = run(&prompt, pos0, slice(pos0, prompt.len()), &mut st_c, 0, false, false, &mut pg)?;
        let (l_n, _) = run(&prompt, pos0, slice(pos0, prompt.len()), &mut st_n, 0, false, false, &mut pg)?;
        let (l_l, windows) = run(&prompt, pos0, slice(pos0, prompt.len()), &mut st_l, rows, lazy, lazy, &mut pg)?;
        // The NULL first: chunked twice must agree, or no verdict on layer-major
        // is possible (a non-deterministic reduction would be blamed on it).
        let mut null = diff_states(&st_c, &st_n)?;
        let bad_null = l_c.iter().zip(&l_n).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        if bad_null > 0 {
            null.push(format!("prefill logits: {bad_null} differ"));
        }
        if !null.is_empty() {
            for d in null.iter().take(10) {
                eprintln!("  {case}: NULL (chunked vs chunked): {d}");
            }
            failures.push(format!("{case}: chunked prefill is not deterministic ({} differences) -- no verdict", null.len()));
            continue;
        }
        let mut diffs = Vec::new();
        if lazy {
            // Separate "lazy/foreign" from "layer-major": the same window, eager
            // inputs, no foreign forwards.
            let mut st_e = fresh_state()?;
            let (l_e, _) = run(&prompt, pos0, slice(pos0, prompt.len()), &mut st_e, rows, false, false, &mut pg)?;
            for d in diff_states(&st_c, &st_e)? {
                diffs.push(format!("eager layer-major: {d}"));
            }
            let n = l_c.iter().zip(&l_e).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            if n > 0 {
                diffs.push(format!("eager layer-major: prefill logits: {n} differ"));
            }
        }
        if windows == 0 {
            diffs.push("layer-major never ran (0 windows): single-lane shortcut or no device store".to_string());
        }
        diffs.extend(diff_states(&st_c, &st_l)?);
        let bad_logits = l_c.iter().zip(&l_l).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        if bad_logits > 0 {
            diffs.push(format!("prefill logits: {bad_logits} of {} differ", l_c.len()));
        }
        // A few greedy decode steps on each state: the replayed decoder rings and
        // the encoder KV together.
        let mut tok = argmax(&l_c) as i32;
        let mut seq = prompt.clone();
        for step in 0..decode_steps {
            seq.push(tok);
            let pos = seq.len() - 1;
            let r_new = rows_for(pg.raw(), &seq, pos)?;
            let (d_c, _) = run(&seq, pos, r_new.clone(), &mut st_c, 0, false, false, &mut pg)?;
            let (d_l, _) = run(&seq, pos, r_new, &mut st_l, 0, false, false, &mut pg)?;
            let n = d_c.iter().zip(&d_l).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            if n > 0 {
                diffs.push(format!("decode step {step}: {n} logits differ"));
            }
            tok = argmax(&d_c) as i32;
        }
        if diffs.is_empty() {
            eprintln!("  {case}: BIT-IDENTICAL over {windows} windows (state + logits + {decode_steps} decode steps; null clean)");
        } else {
            for d in diffs.iter().take(20) {
                eprintln!("  {case}: {d}");
            }
            failures.push(format!("{case}: {} differences", diffs.len()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(eyre!("layer-major != chunked: {}", failures.join("; ")))
    }
}
