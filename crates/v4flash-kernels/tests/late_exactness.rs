//! EXACTNESS GATE for box-1 LATE prefill reads (docs/v41/PREFILL_COLD_ASYNC_DESIGN.md rev 3, section 3; slice 4).
//!
//! One fresh synthetic prompt through the LAYER-OUTER prefill with box 2 attached, in ONE process on ONE weight load:
//!   A  -- `V41_PREFILL_COLD_FORCE_LATE=0` (no late picks)                          -- the reference;
//!   A2 -- the same again                                                            -- the NULL (run-to-run noise);
//!   B  -- `V41_PREFILL_COLD_FORCE_LATE=N`: box-1-home picks with id % N == 0 go through the LATE pass (resident: no
//!         read; else the background readers) instead of the main pass. The box-1 / box-2 assignment is unchanged, so
//!         B may differ from A only by f32 reassociation on box 1;
//!   X  -- B with `V41_PREFILL_COLD_LATE_SKIP=1` (the late pass dropped)           -- the NEGATIVE CONTROL.
//! Compared on the prefill logits and a few greedy decode steps (chunked: they read the KV the prefill left).
//! PASS: B's argmax == A's on every compared step, rel-L2(A, B) <= max(GATE_K x rel-L2(A, A2), GATE_EPS), the late path
//! really ran (late picks counted), and rel-L2(A, X) >= GATE_NEG x max(rel-L2(A, B), GATE_EPS).
//!
//! Needs the hub DOWN (model loaded here), box 2 UP, and the production env (the gate script sources the hub env file;
//! `V41_REMOTE_ADDR`, the T2 partition and `V41_B1_PAGE_MISSES=1` must be set). Run:
//! ```text
//! (prod env) V41_LM_LAYER_OUTER=1 V41_PREFILL_PIN_INFLIGHT=1 V41_PREFILL_COLD_ASYNC=1 V41_PREFILL_COLD_B1=off \
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels --features v41 --release \
//!   --test late_exactness -- --ignored --nocapture --exact late_pass_matches_main_pass
//! ```
//!
//! SECOND GATE, `late_pass_matches_main_pass_range` (`V41_PREFILL_COLD_ASYNC_RANGE`, 10-14): the same late pass in the
//! RANGE passes outside layer-outer windows -- a short job's chunked encoder chunks (layers 0-20, `KvSourceOnly`) and
//! the CED replay (layers 20-39). One ~1,500-token job (no window: `set_layer_major_rows(0)`), six arms on one load:
//!   A, A2 (null) -- no late picks anywhere;
//!   BE / XE -- FORCE_LATE=N (XE: + LATE_SKIP) in the ENCODER chunks only, verdict on the encoder's first MoE layer
//!              (`GATE_ENC_LAYER`, default 0: identical inputs in every arm);
//!   BR / XR -- FORCE_LATE=N (XR: + LATE_SKIP) in the REPLAY only, verdict on the replay's first MoE layer
//!              (`GATE_REP_LAYER`, default 20). The encoder of BR / XR runs as A's: its replay seed rows (layer 19's
//!              output + carry, compared before the replay), its KV (layers 0-20) and its layer-0 MoE must be
//!              BIT-IDENTICAL to A's (else no verdict), so the replay's inputs are A's.
//! PASS per path: null bit-identical, late vs main <= GATE_LAYER_TOL (1e-4), dropped >= GATE_NEG x late, late picks
//! counted, no window ran, no slot left pinned. Run it ALONE (one model load per process; the capture layers are read
//! once per process):
//! ```text
//! (prod env) V41_PREFILL_PIN_INFLIGHT=1 V41_PREFILL_COLD_ASYNC_RANGE=1 V41_PREFILL_COLD_B1=off \
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels --features v41 --release \
//!   --test late_exactness -- --ignored --nocapture --exact late_pass_matches_main_pass_range
//! ```
//!
//! SOAK, `late_range_soak_prod_selection`: the range path under the PRODUCTION selection (`V41_PREFILL_COLD_B1=short`,
//! FRAC 0.5, MAX 8 unless the env says otherwise; no forced picks): `GATE_SOAK_JOBS` (6) fresh short jobs must finish
//! without error, with late picks > 0 in the encoder chunks AND in the replays, finite logits, and every pin and stage
//! back after each job. Run it ALONE:
//! ```text
//! (prod env) V41_PREFILL_PIN_INFLIGHT=1 V41_PREFILL_COLD_ASYNC_RANGE=1 \
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels --features v41 --release \
//!   --test late_exactness -- --ignored --nocapture --exact late_range_soak_prod_selection
//! ```
#![cfg(feature = "v41")]

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
/// `(first row, last row, rows that differ)` of two row-major buffers with `row` elements per row, if any differ.
fn row_span<T: PartialEq>(a: &[T], b: &[T], row: usize) -> Option<(usize, usize, usize)> {
    let mut first = None;
    let (mut last, mut n) = (0usize, 0usize);
    for (r, (x, y)) in a.chunks(row).zip(b.chunks(row)).enumerate() {
        if x != y {
            first.get_or_insert(r);
            last = r;
            n += 1;
        }
    }
    first.map(|f| (f, last, n))
}

fn diff_states(a: &HetModelState, b: &HetModelState) -> eyre::Result<Vec<String>> {
    diff_states_in(a, b, 0..N_LAYER as usize)
}

/// [`diff_states`] over `layers` only.
fn diff_states_in(a: &HetModelState, b: &HetModelState, layers: std::ops::Range<usize>) -> eyre::Result<Vec<String>> {
    let mut out = Vec::new();
    for l in layers {
        let (x, y) = (&a.layers[l], &b.layers[l]);
        if (x.n_raw, x.raw_off) != (y.n_raw, y.raw_off) {
            out.push(format!("L{l}: n_raw/raw_off {}/{} vs {}/{}", x.n_raw, x.raw_off, y.n_raw, y.raw_off));
        }
        if let Some((f, z, n)) = row_span(&host(&x.kv_cache)?, &host(&y.kv_cache)?, v4flash_kernels::config::N_HEAD_DIM as usize) {
            out.push(format!("L{l}: kv_cache bytes differ: rows {f}..={z} ({n} rows; raw_off {} n_raw {}; ring {} rows)",
                x.raw_off, x.n_raw, x.kv_cache.len() / v4flash_kernels::config::N_HEAD_DIM as usize));
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
                        let span = match (&p.comp_kv, &q.comp_kv) {
                            (CompKvStore::F16(u), CompKvStore::F16(v)) => row_span(&host(u)?, &host(v)?, p.width as usize),
                            _ => None,
                        };
                        out.push(format!("L{l} {name}: comp_kv differs: rows {span:?} (first, last, count; n_comp {})", p.n_comp));
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
fn late_pass_matches_main_pass() -> eyre::Result<()> {
    install_panic_handler()?;
    if std::env::var("V41_PAGED_EXPERTS").is_err() {
        std::env::set_var("V41_PAGED_EXPERTS", "1");
    }
    if std::env::var("V41_PAGER_POOL_GB").is_err() {
        std::env::set_var("V41_PAGER_POOL_GB", "40");
    }
    if std::env::var("V41_REMOTE_ADDR").is_err() {
        return Err(eyre!("the late pass exists only under the box-2 partition: set V41_REMOTE_ADDR (box 2 up)"));
    }
    // The verdict compares layer-0 MoE outputs (identical inputs in every arm): end-to-end logits compound f32
    // reassociation through discrete router / indexer near-ties and saturate (gate runs 2-3, 10-12).
    std::env::set_var("V41_PREFILL_CAPTURE_MOE", std::env::var("GATE_LAYER").unwrap_or_else(|_| "0".into()));
    // `V41_PREFILL_COLD_ASYNC_RANGE` off: this gate is the WINDOW path (its decode steps run plain range passes, which
    // must not capture); the range path has its own gate below.
    for (k, v) in [("V41_LM_LAYER_OUTER", "1"), ("V41_PREFILL_PIN_INFLIGHT", "1"), ("V41_PREFILL_COLD_ASYNC", "1"), ("V41_PREFILL_COLD_B1", "off"),
        ("V41_PREFILL_COLD_ASYNC_RANGE", "0")] {
        std::env::set_var(k, v);
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
    let lane_rows = v4flash_kernels::het::batch_scratch::prefill_lane_rows();
    let mut bd_a = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_a = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut sd = BatchDgpuShared::alloc_rows_ctx(dgpu, lane_rows, n_kv_max)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    let mut pg = ExpertPager::new(V41HfWeights::open(&dir, None)?, igpu, 0)?;
    pg.start_late_reader(Path::new(&dir), env_usize("V41_PREFILL_COLD_READERS", 8), env_usize("V41_PREFILL_COLD_STAGES", 16).max(1))?;
    if !pg.late_on() {
        return Err(eyre!("late readers did not start (GPU repack off?)"));
    }
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
            PrefillJob::new(sfx.to_vec(), Vec::new(), None, None, pos0 as u32, v4flash_kernels::het::batch_scratch::chunk_rows_cap())?
        } else {
            PrefillJob::new(sfx.to_vec(), hcs.clone(), Some(rows.clone()), None, pos0 as u32, v4flash_kernels::het::batch_scratch::chunk_rows_cap())?
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
                let mut xj = PrefillJob::new(vec![t], vec![embed(t)?], Some(xr), None, pos as u32, v4flash_kernels::het::batch_scratch::chunk_rows_cap())?;
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

    let force_n = env_usize("GATE_FORCE_N", 7).to_string();
    let gate_k = std::env::var("GATE_K").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(4.0);
    let gate_eps = std::env::var("GATE_EPS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1e-4);
    let gate_neg = std::env::var("GATE_NEG").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(20.0);
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        let (mut d, mut n) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            d += ((*x - *y) as f64).powi(2);
            n += (*x as f64).powi(2);
        }
        (d / n.max(1e-30)).sqrt()
    };
    let prompt = synth_prompt(21, env_usize("GATE_TOKENS", 5000));
    let rows_all = rows_for(pg.raw(), &prompt, 0)?;
    // One arm: (prefill logits, decode-step logits, windows, late picks counted in the arm).
    let mut arm = |name: &str, force: &str, skip: &str, pg: &mut ExpertPager| -> eyre::Result<(Vec<f32>, Vec<Vec<f32>>, usize, u64, Vec<Vec<f32>>)> {
        v4flash_kernels::knobs::PREFILL_COLD_FORCE_LATE.set(force);
        v4flash_kernels::knobs::PREFILL_COLD_LATE_SKIP.set(skip);
        let landed0 = pg.late_stats().map_or(0, |x| x.2 + x.3);
        let mut st = fresh_state()?;
        let _ = v4flash_kernels::het::forward_prefill::take_moe_captures();
        let (l, windows) = run(&prompt, 0, rows_all.clone(), &mut st, lm_rows, false, false, pg)?;
        let caps: Vec<Vec<f32>> = v4flash_kernels::het::forward_prefill::take_moe_captures().into_iter().map(|c| c.2).collect();
        let landed1 = pg.late_stats().map_or(0, |x| x.2 + x.3);
        // Greedy decode steps, chunked (no late pass), off this arm's KV.
        let mut steps = Vec::new();
        let mut seq = prompt.clone();
        let mut tok = argmax(&l) as i32;
        for _ in 0..decode_steps {
            seq.push(tok);
            let pos = seq.len() - 1;
            let r_new = rows_for(pg.raw(), &seq, pos)?;
            let (d, _) = run(&seq, pos, r_new, &mut st, 0, false, false, pg)?;
            tok = argmax(&d) as i32;
            steps.push(d);
        }
        // Late picks joined in this arm (landed by a read + already resident at the join), from the pager.
        let late = landed1 - landed0;
        eprintln!("  arm {name}: force {force} skip {skip}: {windows} layer-outer windows, late picks {late}, argmax {}, {} captured units", argmax(&l), caps.len());
        Ok((l, steps, windows, late, caps))
    };
    let (la, sa, wa, _, ca) = arm("A", "0", "0", &mut pg)?;
    let (la2, sa2, _, _, ca2) = arm("A2 (null)", "0", "0", &mut pg)?;
    let (lb, sb, _, late_b, cb) = arm("B (late)", &force_n, "0", &mut pg)?;
    let (lx, sx, _, _, cx) = arm("X (late dropped)", &force_n, "1", &mut pg)?;
    // THE VERDICT: per-unit layer-0 MoE outputs (identical inputs in every arm).
    let caprel = |x: &[Vec<f32>], y: &[Vec<f32>]| -> f64 { x.iter().zip(y).map(|(a, b)| rel(a, b)).fold(0.0, f64::max) };
    let units_ok = !ca.is_empty() && ca.len() == cb.len() && ca.len() == cx.len() && ca.len() == ca2.len();
    let (m_null, m_late, m_neg) = (caprel(&ca, &ca2), caprel(&ca, &cb), caprel(&ca, &cx));
    let layer_tol = std::env::var("GATE_LAYER_TOL").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1e-4);
    eprintln!("LAYER gate ({} units): max rel-L2 null {m_null:.3e}, late {m_late:.3e}, dropped {m_neg:.3e} (tol {layer_tol:.0e})", ca.len());
    v4flash_kernels::knobs::PREFILL_COLD_FORCE_LATE.set("0");
    v4flash_kernels::knobs::PREFILL_COLD_LATE_SKIP.set("0");
    let worst = |l: &[f32], s: &[Vec<f32>], l2: &[f32], s2: &[Vec<f32>]| -> f64 {
        s.iter().zip(s2).map(|(x, y)| rel(x, y)).fold(rel(l, l2), f64::max)
    };
    let d_null = worst(&la, &sa, &la2, &sa2);
    let d_late = worst(&la, &sa, &lb, &sb);
    let d_neg = worst(&la, &sa, &lx, &sx);
    let argmax_ok = argmax(&la) == argmax(&lb) && sa.iter().zip(&sb).all(|(x, y)| argmax(x) == argmax(y));
    eprintln!("late gate: rel-L2 null {d_null:.3e}, late {d_late:.3e}, negative control {d_neg:.3e}; argmax equal {argmax_ok}; late picks {late_b}; windows {wa}");
    if let Some((q, rf, ld, res, fb, oe, wms)) = pg.late_stats() {
        eprintln!("late reads: queued {q}, refused {rf}, landed {ld}, resident at join {res}, fallback {fb}, owned evictions {oe}, join wait {wms} ms");
    }
    let mut fail = Vec::new();
    if wa == 0 {
        fail.push("no layer-outer window ran".to_string());
    }
    if late_b == 0 {
        fail.push("arm B counted no late picks: the late path never ran".to_string());
    }
    let _ = (argmax_ok, gate_k, gate_eps, d_null, d_late, d_neg); // end-to-end: information only (compounds, see above)
    if !units_ok {
        fail.push(format!("layer captures missing or unequal: A {} A2 {} B {} X {}", ca.len(), ca2.len(), cb.len(), cx.len()));
    }
    if m_null != 0.0 {
        fail.push(format!("layer null not bit-identical ({m_null:.3e}): no verdict"));
    }
    if m_late > layer_tol {
        fail.push(format!("layer-0 MoE late vs main {m_late:.3e} > {layer_tol:.0e}"));
    }
    if m_neg < gate_neg * m_late.max(gate_eps * 1e-2) {
        fail.push(format!("negative control too close at layer 0: dropped {m_neg:.3e} vs late {m_late:.3e}"));
    }
    if fail.is_empty() {
        eprintln!("LATE GATE PASS");
        Ok(())
    } else {
        Err(eyre!("LATE GATE FAIL: {}", fail.join("; ")))
    }
}

/// The RANGE-path gate (`V41_PREFILL_COLD_ASYNC_RANGE`): see the module doc.
#[test]
#[ignore]
fn late_pass_matches_main_pass_range() -> eyre::Result<()> {
    install_panic_handler()?;
    if std::env::var("V41_PAGED_EXPERTS").is_err() {
        std::env::set_var("V41_PAGED_EXPERTS", "1");
    }
    if std::env::var("V41_PAGER_POOL_GB").is_err() {
        std::env::set_var("V41_PAGER_POOL_GB", "40");
    }
    if std::env::var("V41_REMOTE_ADDR").is_err() {
        return Err(eyre!("the late pass exists only under the box-2 partition: set V41_REMOTE_ADDR (box 2 up)"));
    }
    let split = v4flash_kernels::config::CED_DECODER_START;
    let enc_layer = env_usize("GATE_ENC_LAYER", 0) as i32;
    let rep_layer = env_usize("GATE_REP_LAYER", split) as i32;
    if enc_layer as usize >= split || (rep_layer as usize) < split || rep_layer >= N_LAYER as i32 {
        return Err(eyre!("GATE_ENC_LAYER must be an encoder layer (< {split}), GATE_REP_LAYER a replay layer ({split}..{N_LAYER})"));
    }
    std::env::set_var("V41_PREFILL_CAPTURE_MOE", format!("{enc_layer},{rep_layer}"));
    for (k, v) in [("V41_PREFILL_PIN_INFLIGHT", "1"), ("V41_PREFILL_COLD_ASYNC_RANGE", "1"), ("V41_PREFILL_COLD_ASYNC", "0"), ("V41_PREFILL_COLD_B1", "off")] {
        std::env::set_var(k, v);
    }
    if std::env::var("V41_CED").as_deref() == Ok("0") {
        return Err(eyre!("the range path under test is the CED encoder + replay; unset V41_CED=0"));
    }
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| HF_DIR_DEFAULT.to_string());
    let engram_dir = std::env::var("V41_ENGRAM_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/v41/engram", std::env::var("HOME").unwrap_or_default())
    });

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
    let lane_rows = v4flash_kernels::het::batch_scratch::prefill_lane_rows();
    let mut bd_a = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_a = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut sd = BatchDgpuShared::alloc_rows_ctx(dgpu, lane_rows, n_kv_max)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    let mut pg = ExpertPager::new(V41HfWeights::open(&dir, None)?, igpu, 0)?;
    pg.start_late_reader(Path::new(&dir), env_usize("V41_PREFILL_COLD_READERS", 8), env_usize("V41_PREFILL_COLD_STAGES", 16).max(1))?;
    if !pg.late_on() {
        return Err(eyre!("late readers did not start (GPU repack off?)"));
    }
    let hasher = EngramHash::load(Path::new(&engram_dir))?;
    hasher.self_check()?;
    let mut tables = Vec::new();
    for &l in ENGRAM_LAYERS {
        tables.push(EngramTable::open(pg.raw(), l as usize)?);
    }
    let ein = ENGRAM_IN as usize;
    let prompt = synth_prompt(22, env_usize("GATE_TOKENS", 1500));
    let rows_all = {
        let hashes = hasher.hash_sequence(&prompt);
        let mut out = vec![vec![0f32; prompt.len() * ein]; tables.len()];
        for (li, tbl) in tables.iter().enumerate() {
            for t in 0..prompt.len() {
                tbl.gather_position(pg.raw(), &hashes[t][li], &mut out[li][t * ein..(t + 1) * ein])?;
            }
        }
        out
    };
    let hcs: Vec<Vec<f32>> = prompt.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;

    struct Arm {
        logits: Vec<f32>,
        enc: Vec<Vec<f32>>,
        rep: Vec<Vec<f32>>,
        late_enc: u64,
        late_rep: u64,
        windows: usize,
        /// The replay's input rows (token, residual, carry) as the encoder left them, before `prefill_job_finish`.
        replay_rows: Vec<(i32, Vec<f32>, Vec<f32>)>,
        state: HetModelState,
    }
    let force_n = env_usize("GATE_FORCE_N", 7).to_string();
    let late_count = |pg: &ExpertPager| pg.late_stats().map_or(0, |x| x.2 + x.3);
    // One job: the encoder chunks under (force_e, skip_e), the replay under (force_r, skip_r).
    let mut arm = |name: &str, (force_e, skip_e): (&str, &str), (force_r, skip_r): (&str, &str), pg: &mut ExpertPager| -> eyre::Result<Arm> {
        let mut st = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        zero_state(&mut st)?;
        st.reset_in_place(dgpu, igpu)?;
        let mut job = PrefillJob::new(prompt.clone(), hcs.clone(), Some(rows_all.clone()), None, 0, v4flash_kernels::het::batch_scratch::chunk_rows_cap())?;
        job.set_layer_major_rows(0)?;
        let _ = v4flash_kernels::het::forward_prefill::take_moe_captures();
        v4flash_kernels::knobs::PREFILL_COLD_FORCE_LATE.set(force_e);
        v4flash_kernels::knobs::PREFILL_COLD_LATE_SKIP.set(skip_e);
        let l0 = late_count(pg);
        let mut chunks = 0usize;
        while !job.chunks_done() {
            engine.prefill_job_chunk(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights, Some(&mut *pg))?;
            chunks += 1;
        }
        let l1 = late_count(pg);
        let caps_e = v4flash_kernels::het::forward_prefill::take_moe_captures();
        let replay_rows = job.replay_rows();
        v4flash_kernels::knobs::PREFILL_COLD_FORCE_LATE.set(force_r);
        v4flash_kernels::knobs::PREFILL_COLD_LATE_SKIP.set(skip_r);
        let logits = engine.prefill_job_finish(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights, Some(&mut *pg))?;
        st.restore_compressor_lending();
        let l2 = late_count(pg);
        let caps_r = v4flash_kernels::het::forward_prefill::take_moe_captures();
        v4flash_kernels::knobs::PREFILL_COLD_FORCE_LATE.set("0");
        v4flash_kernels::knobs::PREFILL_COLD_LATE_SKIP.set("0");
        let pick_layer = |caps: Vec<(i32, u32, Vec<f32>)>, layer: i32| caps.into_iter().filter(|c| c.0 == layer).map(|c| c.2).collect::<Vec<_>>();
        let a = Arm {
            logits,
            enc: pick_layer(caps_e, enc_layer),
            rep: pick_layer(caps_r, rep_layer),
            late_enc: l1 - l0,
            late_rep: l2 - l1,
            windows: job.lm_windows_run(),
            replay_rows,
            state: st,
        };
        eprintln!("  arm {name}: enc force {force_e} skip {skip_e}, replay force {force_r} skip {skip_r}: {chunks} chunks, {} windows, \
            late picks enc {} replay {}, captures L{enc_layer} {} L{rep_layer} {}, argmax {}",
            a.windows, a.late_enc, a.late_rep, a.enc.len(), a.rep.len(), argmax(&a.logits));
        Ok(a)
    };
    let f = force_n.as_str();
    let a = arm("A", ("0", "0"), ("0", "0"), &mut pg)?;
    let a2 = arm("A2 (null)", ("0", "0"), ("0", "0"), &mut pg)?;
    let be = arm("BE (late in the encoder)", (f, "0"), ("0", "0"), &mut pg)?;
    let xe = arm("XE (encoder late dropped)", (f, "1"), ("0", "0"), &mut pg)?;
    let br = arm("BR (late in the replay)", ("0", "0"), (f, "0"), &mut pg)?;
    let xr = arm("XR (replay late dropped)", ("0", "0"), (f, "1"), &mut pg)?;

    let rel = |x: &[f32], y: &[f32]| -> f64 {
        let (mut d, mut n) = (0f64, 0f64);
        for (p, q) in x.iter().zip(y) {
            d += ((*p - *q) as f64).powi(2);
            n += (*p as f64).powi(2);
        }
        (d / n.max(1e-30)).sqrt()
    };
    let caprel = |x: &[Vec<f32>], y: &[Vec<f32>]| -> f64 { x.iter().zip(y).map(|(p, q)| rel(p, q)).fold(0.0, f64::max) };
    let gate_eps = std::env::var("GATE_EPS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1e-4);
    let gate_neg = std::env::var("GATE_NEG").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(20.0);
    let layer_tol = std::env::var("GATE_LAYER_TOL").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1e-4);
    let mut fail = Vec::new();
    for x in [&a, &a2, &be, &xe, &br, &xr] {
        if x.windows != 0 {
            fail.push(format!("a layer-outer window ran ({}): the range path was not the one under test", x.windows));
        }
    }
    // One verdict: (path, reference, null, late arm, dropped arm, late picks in the late arm).
    let mut verdict = |path: &str, r: &[Vec<f32>], n: &[Vec<f32>], b: &[Vec<f32>], x: &[Vec<f32>], late: u64| {
        let ok = !r.is_empty() && r.len() == n.len() && r.len() == b.len() && r.len() == x.len();
        let (m_null, m_late, m_neg) = (caprel(r, n), caprel(r, b), caprel(r, x));
        eprintln!("{path} gate ({} units): max rel-L2 null {m_null:.3e}, late {m_late:.3e}, dropped {m_neg:.3e} (tol {layer_tol:.0e}); late picks {late}", r.len());
        if !ok {
            fail.push(format!("{path}: captures missing or unequal: {} / {} / {} / {}", r.len(), n.len(), b.len(), x.len()));
        }
        if late == 0 {
            fail.push(format!("{path}: the late arm counted no late picks: the late path never ran"));
        }
        if m_null != 0.0 {
            fail.push(format!("{path}: null not bit-identical ({m_null:.3e}): no verdict"));
        }
        if m_late > layer_tol {
            fail.push(format!("{path}: late vs main {m_late:.3e} > {layer_tol:.0e}"));
        }
        if m_neg < gate_neg * m_late.max(gate_eps * 1e-2) {
            fail.push(format!("{path}: negative control too close: dropped {m_neg:.3e} vs late {m_late:.3e}"));
        }
    };
    verdict(&format!("ENCODER L{enc_layer}"), &a.enc, &a2.enc, &be.enc, &xe.enc, be.late_enc);
    verdict(&format!("REPLAY L{rep_layer}"), &a.rep, &a2.rep, &br.rep, &xr.rep, br.late_rep);
    // The replay arms' inputs are A's: their encoders ran as A's (no late picks), bit for bit.
    for (name, x) in [("BR", &br), ("XR", &xr)] {
        let er = caprel(&a.enc, &x.enc);
        if x.late_enc != 0 || er != 0.0 {
            fail.push(format!("{name}: its encoder differs from A's (late picks {}, L{enc_layer} rel {er:.3e}): no replay verdict", x.late_enc));
        }
        // The replay's seed rows (layer 19's output entering layer 20, and the mHC carry) bit for bit: they, with the
        // KV below, are everything the replay reads from the encoder.
        if x.replay_rows.is_empty() || x.replay_rows != a.replay_rows {
            let first = x.replay_rows.iter().zip(&a.replay_rows).position(|(p, q)| p != q);
            fail.push(format!("{name}: replay seed rows differ from A's ({} vs {} rows, first differing {first:?}): no replay verdict",
                x.replay_rows.len(), a.replay_rows.len()));
        }
        // Encoder KV (layers 0-19) and layer 20's (written by the encoder's KvSourceOnly pass; its compressor is not
        // touched by the replay, its raw window is rewritten from the identical seed rows before any MoE).
        let d = diff_states_in(&a.state, &x.state, 0..split + 1)?;
        if !d.is_empty() {
            fail.push(format!("{name}: encoder / layer-{split} KV differs from A's: {}", d.join("; ")));
        }
    }
    // End-to-end logits: information only (f32 reassociation compounds through router / indexer near-ties).
    eprintln!("end-to-end logits rel-L2 vs A (information only): null {:.3e}, BE {:.3e}, XE {:.3e}, BR {:.3e}, XR {:.3e}",
        rel(&a.logits, &a2.logits), rel(&a.logits, &be.logits), rel(&a.logits, &xe.logits), rel(&a.logits, &br.logits), rel(&a.logits, &xr.logits));
    if let Some((q, rf, ld, res, fb, oe, wms)) = pg.late_stats() {
        eprintln!("late reads: queued {q}, refused {rf}, landed {ld}, resident at join {res}, fallback {fb}, owned evictions {oe}, join wait {wms} ms");
    }
    if pg.pinned_slots() != 0 {
        fail.push(format!("{} slots still pinned after the passes", pg.pinned_slots()));
    }
    if fail.is_empty() {
        eprintln!("LATE RANGE GATE PASS");
        Ok(())
    } else {
        Err(eyre!("LATE RANGE GATE FAIL: {}", fail.join("; ")))
    }
}

/// SOAK of the range path under the PRODUCTION selection (`V41_PREFILL_COLD_B1=short`, FRAC 0.5, MAX 8,
/// `V41_PREFILL_COLD_ASYNC_RANGE=1`; env values win if set): several short jobs (chunked encoder + replay, no window),
/// each must finish without error (three-way exactly-once, late join never ran, exclusion: all fail the job), with
/// real late reads on BOTH the encoder chunks and the replay, finite logits, and every pin / stage back after each job.
/// Not an exactness gate (cold picks move between the boxes: no bit-identical null exists); see the module doc.
#[test]
#[ignore]
fn late_range_soak_prod_selection() -> eyre::Result<()> {
    install_panic_handler()?;
    for (k, v) in [("V41_PAGED_EXPERTS", "1"), ("V41_PAGER_POOL_GB", "40"), ("V41_PREFILL_COLD_B1", "short"),
        ("V41_PREFILL_COLD_B1_FRAC", "0.5"), ("V41_PREFILL_COLD_B1_MAX", "8")] {
        if std::env::var(k).is_err() {
            std::env::set_var(k, v);
        }
    }
    if std::env::var("V41_REMOTE_ADDR").is_err() {
        return Err(eyre!("cold picks exist only under the box-2 partition: set V41_REMOTE_ADDR (box 2 up)"));
    }
    for (k, v) in [("V41_PREFILL_PIN_INFLIGHT", "1"), ("V41_PREFILL_COLD_ASYNC_RANGE", "1"), ("V41_PREFILL_COLD_ASYNC", "0"),
        ("V41_PREFILL_COLD_FORCE_LATE", "0"), ("V41_PREFILL_COLD_LATE_SKIP", "0")] {
        std::env::set_var(k, v);
    }
    if std::env::var("V41_CED").as_deref() == Ok("0") {
        return Err(eyre!("the range path under test is the CED encoder + replay; unset V41_CED=0"));
    }
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| HF_DIR_DEFAULT.to_string());
    let engram_dir = std::env::var("V41_ENGRAM_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/v41/engram", std::env::var("HOME").unwrap_or_default())
    });
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
    let lane_rows = v4flash_kernels::het::batch_scratch::prefill_lane_rows();
    let mut bd_a = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_a = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut sd = BatchDgpuShared::alloc_rows_ctx(dgpu, lane_rows, n_kv_max)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    let mut pg = ExpertPager::new(V41HfWeights::open(&dir, None)?, igpu, 0)?;
    let stages = env_usize("V41_PREFILL_COLD_STAGES", 16).max(1);
    pg.start_late_reader(Path::new(&dir), env_usize("V41_PREFILL_COLD_READERS", 8), stages)?;
    if !pg.late_on() {
        return Err(eyre!("late readers did not start (GPU repack off?)"));
    }
    let hasher = EngramHash::load(Path::new(&engram_dir))?;
    hasher.self_check()?;
    let mut tables = Vec::new();
    for &l in ENGRAM_LAYERS {
        tables.push(EngramTable::open(pg.raw(), l as usize)?);
    }
    let ein = ENGRAM_IN as usize;
    use v4flash_kernels::het::prefill_split::{cold_totals, late_totals};
    let late_now = |pg: &ExpertPager| {
        let (sh, new, refd) = late_totals();
        (sh + new, refd, pg.late_stats().map_or(0, |x| x.0))
    };
    let jobs = env_usize("GATE_SOAK_JOBS", 6);
    let base = env_usize("GATE_TOKENS", 1500);
    let (mut enc_late, mut rep_late, mut cold) = (0u64, 0u64, 0u64);
    let mut fail = Vec::new();
    for j in 0..jobs {
        // 0.5x .. 1.33x the base length, a fresh prompt each (different experts per job).
        let len = (base / 2 + (j * base) / (jobs.max(1) + 1)).max(200);
        let prompt = synth_prompt(100 + j as u64, len);
        let rows = {
            let hashes = hasher.hash_sequence(&prompt);
            let mut out = vec![vec![0f32; prompt.len() * ein]; tables.len()];
            for (li, tbl) in tables.iter().enumerate() {
                for t in 0..prompt.len() {
                    tbl.gather_position(pg.raw(), &hashes[t][li], &mut out[li][t * ein..(t + 1) * ein])?;
                }
            }
            out
        };
        let hcs: Vec<Vec<f32>> = prompt.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
        let mut st = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        zero_state(&mut st)?;
        st.reset_in_place(dgpu, igpu)?;
        let mut job = PrefillJob::new(prompt.clone(), hcs, Some(rows), None, 0, v4flash_kernels::het::batch_scratch::chunk_rows_cap())?;
        job.set_layer_major_rows(0)?;
        let c0 = cold_totals();
        let l0 = late_now(&pg);
        let mut run = || -> eyre::Result<(u64, u64, Vec<f32>)> {
            while !job.chunks_done() {
                engine.prefill_job_chunk(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights, Some(&mut pg))?;
            }
            let e = late_now(&pg).0 - l0.0;
            let l1 = late_now(&pg);
            let logits = engine.prefill_job_finish(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights, Some(&mut pg))?;
            st.restore_compressor_lending();
            Ok((e, late_now(&pg).0 - l1.0, logits))
        };
        match run() {
            Ok((e, r, logits)) => {
                let c1 = cold_totals();
                let c = (c1.1 - c0.1) + (c1.2 - c0.2) + (late_now(&pg).0 - l0.0);
                enc_late += e;
                rep_late += r;
                cold += c;
                let finite = logits.iter().all(|x| x.is_finite());
                eprintln!("  job {j}: {len} tokens, {} windows: late picks encoder {e} replay {r}, cold picks {c}, stage refusals {}, argmax {}, finite {finite}",
                    job.lm_windows_run(), late_now(&pg).1 - l0.1, argmax(&logits));
                if !finite {
                    fail.push(format!("job {j}: non-finite logits"));
                }
                if job.lm_windows_run() != 0 {
                    fail.push(format!("job {j}: a layer-outer window ran: the range path was not the one under test"));
                }
            }
            Err(e) => fail.push(format!("job {j} ({len} tokens) FAILED: {e:#}")),
        }
        if pg.pinned_slots() != 0 {
            fail.push(format!("job {j}: {} slots still pinned", pg.pinned_slots()));
        }
        if pg.late_free_stages() != stages {
            fail.push(format!("job {j}: {} of {stages} late stages free after the job", pg.late_free_stages()));
        }
    }
    if let Some((q, rf, ld, res, fb, oe, wms)) = pg.late_stats() {
        eprintln!("late reads: queued {q}, refused {rf}, landed {ld}, resident at join {res}, fallback {fb}, owned evictions {oe}, join wait {wms} ms");
    }
    if cold == 0 {
        fail.push("no cold picks at all (box 2 held every pick): inconclusive, rerun after box 2 has paged / with more jobs".into());
    } else {
        if enc_late == 0 {
            fail.push(format!("no late picks in the encoder chunks ({cold} cold picks)"));
        }
        if rep_late == 0 {
            fail.push(format!("no late picks in the replays ({cold} cold picks)"));
        }
    }
    eprintln!("soak: {jobs} jobs, late picks encoder {enc_late} replay {rep_late}, cold picks {cold}");
    if fail.is_empty() {
        eprintln!("LATE RANGE SOAK PASS");
        Ok(())
    } else {
        Err(eyre!("LATE RANGE SOAK FAIL: {}", fail.join("; ")))
    }
}
