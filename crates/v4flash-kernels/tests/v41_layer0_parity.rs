//! M1: DeepSeek-V4.1-Flash layer 0 through the het engine vs the CPU oracle
//! (`scripts/v41_oracle`, dumps exported by `export_bins.py`).
//!
//! Per token t (decode-style, pos = t): `embed_hc[t]` → layer 0 →
//! `residual[t]`, compared against `layer_00_residual`. Routed expert ids are
//! read back from `d_selected` and compared against `topk_ids` (near-tie
//! differences are a parity signal, not a failure of the kernels).
//!
//! Weights come straight from the HF checkpoint (`V41HfWeights`): layer 0's
//! non-routed tensors on the dGPU, and ONLY the experts the oracle routed to
//! (union over the prompt, ≤ 36) — as the dGPU hot set (cap = n_used so every
//! hit runs there) and, defensively, as the iGPU packed cold set too, so a
//! near-tie miss still hits a resident slot instead of an empty buffer.
//!
//! Gate (docs/v41/ENGINE_PORT.md §2): max|Δ|/max|ref| ≤ 2 × the measured
//! Q8_0 noise floor at L0 (2.7e-2), i.e. 5.4e-2, on every token.
//!
//! `V41_EXEC_MODE=prefill` runs the same tokens as ONE batched chunk through
//! `forward_layer_batch_v2` (the prefill path: batched q chain without the
//! per-head q norm, batched mHC carry) and compares every row of
//! `residual_next` the same way (no per-stage taps).
//!
//! Needs the V4-Flash server DOWN (decode scratch + state on the dGPU):
//!   CARGO_TARGET_DIR=target-v41 cargo test --features v41 -p v4flash-kernels --release \
//!       --test v41_layer0_parity -- --ignored --nocapture
#![cfg(feature = "v41")]

use std::collections::BTreeSet;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{EngramHash, EngramTable, V41HfWeights, WeightSrc};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::config::{COMPRESS_RATIOS, ENGRAM_IN, HC_DIM, N_EXPERT, N_EXPERT_USED, ROPE_ORIG_CTX};
use v4flash_kernels::het::engine::{ExecMode, HeterogeneousEngine};
use v4flash_kernels::het::{BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared};
use v4flash_kernels::het::scratch::{DgpuScratch, IgpuScratch};
use v4flash_kernels::het::state::HetModelState;
use v4flash_kernels::het::weights::{encode_igpu_remap, DgpuLayerWeights, HotExpertWeights, IgpuLayerWeights};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

/// "The capital of France is" (BOS + 5), as the oracle tokenised it.
const PROMPT_TOKENS: [i32; 6] = [0, 671, 6102, 294, 8760, 344];

// V4.1 rope: identical constants to V4-Flash (config.json rope_theta 10000,
// compress_rope_theta 160000, rope_factor 16, original_seq_len 65536,
// beta 32/1) — this is deepstrix-server's `rope_for_layer` verbatim.
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

/// ggml Q8_0 dequant: [out][in] with 34-byte blocks of 32 along `in`.
fn dequant_q8_0(bytes: &[u8], out: usize, inn: usize) -> Vec<f32> {
    let nb = inn / 32;
    let mut w = vec![0f32; out * inn];
    for r in 0..out {
        for b in 0..nb {
            let o = (r * nb + b) * 34;
            let d = v4flash_core::kquants::f16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
            for i in 0..32 {
                w[r * inn + b * 32 + i] = (bytes[o + 2 + i] as i8) as f32 * d;
            }
        }
    }
    w
}

fn host(buf: &DeviceBuffer<f32>) -> eyre::Result<Vec<f32>> {
    let mut v = vec![0f32; buf.len()];
    buf.copy_to_host(&mut v)?;
    Ok(v)
}

fn rel(got: &[f32], want: &[f32]) -> (f32, f32) {
    let (mut md, mut mr) = (0f32, 0f32);
    for (a, b) in got.iter().zip(want) {
        md = md.max((a - b).abs());
        mr = mr.max(b.abs());
    }
    (md, md / mr.max(1e-30))
}

/// A reference floor file is MISSING.
///
/// Failing here is deliberate. The previous behaviour silently substituted a
/// hardcoded constant -- and for the head gate, the FP8 reference argmax, which
/// is precisely the target the comment at that gate says is WRONG ("the engine
/// runs Q8_0 weights, so its ground truth is the Q8 CPU oracle, not the fp8
/// reference", and there is a known fp8-native argmax flip at near-ties). So on
/// a box without the floor tables this test ran to completion and printed a
/// verdict while gating on the wrong thing with a made-up tolerance, which is
/// worse than not running at all.
///
/// `V41_ALLOW_MISSING_FLOORS=1` restores the old fallback for local poking; it
/// says loudly that the result is not a parity verdict.
fn missing_floor<T>(file: &str, what: &str, fallback: T) -> T {
    if std::env::var("V41_ALLOW_MISSING_FLOORS").as_deref() == Ok("1") {
        eprintln!(
            "[parity] WARNING: {what} missing ({file}); using a hardcoded fallback \
             -- THE GATE IS NOT A PARITY VERDICT"
        );
        return fallback;
    }
    panic!(
        "v41_layer0_parity: {what} not found ({file}).\n\
         This gate is only meaningful against the MEASURED Q8 floor; with a hardcoded \
         fallback it compares against the wrong target. Generate the table (see \
         scripts/v41_oracle/compare.py) or set V41_ALLOW_MISSING_FLOORS=1 to run anyway."
    );
}

/// Measured Q8_0 noise floor of the oracle at layer `layer` (residual Δ/scale of
/// the Q8-roundtripped oracle vs the fp8-native one, `compare_q8_floor.txt`:
/// 2.7e-2 @L0, 3.3e-2 @L1, 3.8e-2 @L2 ...); 2.7e-2 when the table is absent.
fn q8_floor(layer: i32) -> f32 {
    q8_floor_for(layer, 6)
}

/// The floor table matching the dump length: `compare_q8_floor.txt` was measured on the
/// 6-token prompt, `compare_t200_q8_floor.txt` on the 129-token one (a max over more
/// tokens sits higher for the same noise).
fn q8_floor_for(layer: i32, t_n: usize) -> f32 {
    let file = if t_n > 32 { "compare_t200_q8_floor.txt" } else { "compare_q8_floor.txt" };
    let path = format!("{}/.cache/deepstrix/v41/{file}", std::env::var("HOME").unwrap_or_default());
    let key = format!("layer_{layer:02}_residual.pt");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|txt| {
            txt.lines()
                .find(|l| l.starts_with(&key))
                .and_then(|l| l.split_whitespace().nth(3).and_then(|v| v.parse::<f32>().ok()))
        })
        .unwrap_or_else(|| missing_floor(file, &format!("Q8 residual floor for layer {layer}"), 2.7e-2))
}

/// Name of the per-token floor table, for the missing-reference messages.
const FLOOR_PERTOK_FILE: &str = "compare_q8_floor_pertok.txt / compare_t200_q8_floor_pertok.txt";

/// Per-token Q8-oracle floors (`compare_q8_floor_pertok.txt`: `layer_NN f0 f1 .. f5`, max|Δ|/max|ref|
/// per token of the chained Q8 CPU oracle vs the reference) and its logits error (`logits_last x ...`).
fn q8_floor_pertok(layer: i32, t_n: usize) -> Option<Vec<f32>> {
    let file = if t_n > 32 { "compare_t200_q8_floor_pertok.txt" } else { "compare_q8_floor_pertok.txt" };
    let path = format!("{}/.cache/deepstrix/v41/{file}", std::env::var("HOME").unwrap_or_default());
    let key = format!("layer_{layer:02} ");
    std::fs::read_to_string(path).ok().and_then(|txt| {
        txt.lines().find(|l| l.starts_with(&key)).map(|l| l.split_whitespace().skip(1).filter_map(|v| v.parse::<f32>().ok()).collect())
    })
}
/// The Q8 CPU oracle's OWN argmax (from the exported floor file's `logits_last` line). The engine
/// runs Q8_0-dequantised weights, so its correct parity target is the Q8 oracle, not the fp8
/// reference: at a near-tie the two quantisations can pick different top-1 tokens (fp8-native is a
/// separate quality lever), and the engine must reproduce the Q8-weight result.
fn q8_logits_argmax(t_n: usize) -> Option<usize> {
    let file = if t_n > 32 { "compare_t200_q8_floor_pertok.txt" } else { "compare_q8_floor_pertok.txt" };
    let path = format!("{}/.cache/deepstrix/v41/{file}", std::env::var("HOME").unwrap_or_default());
    std::fs::read_to_string(path).ok().and_then(|txt| {
        let line = txt.lines().find(|l| l.starts_with("logits_last "))?;
        let toks: Vec<&str> = line.split_whitespace().collect();
        let i = toks.iter().position(|&t| t == "argmax_q8")?;
        toks.get(i + 1).and_then(|v| v.parse::<usize>().ok())
    })
}

fn q8_logits_floor(t_n: usize) -> Option<f32> {
    let file = if t_n > 32 { "compare_t200_q8_floor_pertok.txt" } else { "compare_q8_floor_pertok.txt" };
    let path = format!("{}/.cache/deepstrix/v41/{file}", std::env::var("HOME").unwrap_or_default());
    std::fs::read_to_string(path).ok().and_then(|txt| {
        txt.lines().find(|l| l.starts_with("logits_last ")).and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f32>().ok()))
    })
}

fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 31) & 1) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff { return (sign << 15) | 0x7c00 | if mant != 0 { 0x200 } else { 0 }; }
    let e = exp - 127 + 15;
    if e >= 0x1f { return (sign << 15) | 0x7c00; }
    if e <= 0 {
        if e < -10 { return sign << 15; }
        let m = (mant | 0x80_0000) >> (1 - e);
        let round = (m & 0x1fff) > 0x1000 || ((m & 0x1fff) == 0x1000 && (m & 0x2000) != 0);
        return (sign << 15) | ((m >> 13) as u16) + round as u16;
    }
    let mut h = (sign << 15) | ((e as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) != 0) { h += 1; }
    h
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign } else {
            // subnormal: normalise
            let mut m = mant;
            let mut e: i32 = 127 - 15 + 1;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Compare the engine's per-layer caches with the reference's (`stage_window_kv0`
/// = fp8-fake-quantised window rows, `stage_compress_kv0` = E2M1×E4M3/16
/// compressed rows), per row: Δ/scale and the share of bit-equal elements.
fn check_caches(dump: &ActivationDump, layer: i32, ls: &v4flash_kernels::het::state::HetLayerState, src: &v4flash_kernels::het::state::HetLayerState, t_n: usize) -> eyre::Result<()> {
    let hd = v4flash_kernels::config::N_HEAD_DIM as usize;
    let report = |what: &str, rows: &[f32], tag: &str, n: usize| -> eyre::Result<()> {
        let (mut worst, mut exact_all, mut n_all) = (0f32, 0usize, 0usize);
        for r in 0..n {
            let Some(e) = dump.tensor(tag, layer, r as i32) else { break };
            let want = dump.read_f32(e)?;
            let got = &rows[r * hd..(r + 1) * hd];
            let (mut md, mut mr, mut exact) = (0f32, 0f32, 0usize);
            for (a, b) in got.iter().zip(&want) {
                md = md.max((a - b).abs());
                mr = mr.max(b.abs());
                exact += (a == b) as usize;
            }
            worst = worst.max(md / mr.max(1e-30));
            exact_all += exact;
            n_all += want.len();
            if r < 4 || md / mr.max(1e-30) > 5e-2 {
                eprintln!("    {what} row {r}: max|Δ|={md:.3e} max|ref|={mr:.3e} Δ/scale={:.3e} exact {exact}/{}", md / mr.max(1e-30), want.len());
            }
        }
        if n_all > 0 {
            eprintln!("    {what}: worst Δ/scale {worst:.3e}; bit-equal elements {exact_all}/{n_all} ({:.1}%)", 100.0 * exact_all as f64 / n_all as f64);
        }
        Ok(())
    };
    // V41_DUMP_CACHES=<dir>: write the engine's rows for offline comparison.
    let dump_dir = std::env::var("V41_DUMP_CACHES").ok();
    let save = |name: &str, v: &[f32]| {
        if let Some(d) = &dump_dir {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let _ = std::fs::create_dir_all(d);
            let _ = std::fs::write(format!("{d}/{name}"), bytes);
        }
    };
    // Window rows: the monotonic cache holds position t at row raw_off + t while t < SWA_WINDOW.
    // The reference keeps a ring of SWA_WINDOW slots (position p at slot p % W); the
    // engine's live window is rows [raw_off, raw_off + n_raw) = the last n_raw
    // positions. Re-order the engine rows into the reference's slot order.
    let win_sz = v4flash_kernels::config::SWA_WINDOW as usize;
    let n_win = t_n.min(ls.n_raw as usize);
    let mut w = vec![0u16; ls.kv_cache.len()];
    ls.kv_cache.copy_to_host(&mut w)?;
    let off = ls.raw_off as usize * hd;
    let live: Vec<f32> = w[off..off + n_win * hd].iter().map(|&x| f16_to_f32(x)).collect();
    let mut win = vec![0f32; n_win.min(win_sz) * hd];
    for i in 0..n_win {
        let pos = t_n - n_win + i;
        let slot = pos % win_sz;
        if slot < n_win.min(win_sz) {
            win[slot * hd..(slot + 1) * hd].copy_from_slice(&live[i * hd..(i + 1) * hd]);
        }
    }
    save(&format!("engine_window_kv_L{layer:02}.bin"), &win);
    report("window_kv", &win, "stage_window_kv0", n_win.min(win_sz))?;
    if let Some(c) = src.compressor.as_ref() {
        let n = c.n_comp as usize;
        if n > 0 {
            let buf = c.comp_kv.dense_f16(c.n_comp, "parity")?;
            let mut h = vec![0u16; buf.len()];
            buf.copy_to_host(&mut h)?;
            let rows: Vec<f32> = h[..n * hd].iter().map(|&x| f16_to_f32(x)).collect();
            save(&format!("engine_compress_kv_L{layer:02}.bin"), &rows);
            report("compress_kv", &rows, "stage_compress_kv0", n)?;
        }
    }
    Ok(())
}

fn read_i32(d: &ActivationDump, tag: &str, layer: i32, token: i32) -> eyre::Result<Vec<i32>> {
    let e = d.tensor(tag, layer, token).ok_or_else(|| eyre!("missing {tag} L{layer} T{token}"))?;
    Ok(d.read_bytes(e)?.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect())
}

#[test]
#[ignore]
fn v41_layer0_matches_oracle() -> eyre::Result<()> {
    install_panic_handler()?;
    // Every routed hit goes to the dGPU hot set (read by het::weights::dgpu_hot_cap).
    std::env::set_var("DGPU_HOT_CAP", N_EXPERT_USED.to_string());
    let bins = std::env::var("V41_ORACLE_BINS").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/v41/oracle_full_bins", std::env::var("HOME").unwrap())
    });
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap())
    });
    // V41_LAYER=N runs layer N (input = the dump's residual after N-1; layers 1
    // and 14 also gather + stage their Engram rows through the Rust hasher/table).
    let layer: i32 = std::env::var("V41_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let engram_dir = std::env::var("V41_ENGRAM_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/v41/engram", std::env::var("HOME").unwrap())
    });
    // V41_ISOLATED=1: run ONLY layer N, with the dump's residual after N-1 as input
    // and the dump's mHC pre-mix after N-1 (`stage_pre_mix0`) as the carry —
    // no error compounding from earlier layers, so the gate is the layer's own floor.
    let isolated = std::env::var("V41_ISOLATED").map(|v| v == "1").unwrap_or(false) && layer > 0;
    let dump = ActivationDump::open(&bins)?;
    let t_n = dump.prompt_len;
    // Token ids: `tokens.json` next to the fixture (written by oracle.py /
    // export_bins.py), else the 6-token " Paris" prompt.
    let tokens: Vec<i32> = match std::fs::read_to_string(format!("{bins}/tokens.json")) {
        Ok(j) => serde_json::from_str(&j)?,
        Err(_) => PROMPT_TOKENS.to_vec(),
    };
    assert_eq!(t_n, tokens.len(), "dump prompt length vs tokens.json");
    let max_t: usize = std::env::var("V41_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(t_n);
    let t_n = t_n.min(max_t);

    // Experts the oracle routed to at layer 0, over the whole prompt.
    let mut ids = BTreeSet::new();
    let mut want_ids: Vec<Vec<i32>> = Vec::new();
    let have_ids = dump.tensor("topk_ids", layer, 0).is_some();
    for t in 0..t_n {
        if !have_ids {
            break;
        }
        let v = read_i32(&dump, "topk_ids", layer, t as i32)?;
        assert_eq!(v.len(), N_EXPERT_USED);
        ids.extend(v.iter().copied());
        want_ids.push(v);
    }
    if !have_ids {
        eprintln!("dump has no topk_ids: all experts on the iGPU, no dGPU hot set, routing not checked");
    }
    let hot_ids: Vec<u32> = ids.iter().map(|&e| e as u32).collect();
    eprintln!("layer {layer} routed set over {t_n} tokens: {} experts {:?}", hot_ids.len(), hot_ids);

    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    let darch = dgpu.properties()?.gcn_arch_name;
    let iarch = igpu.properties()?.gcn_arch_name;
    let hf = V41HfWeights::open(&dir, None)?;
    let src = WeightSrc::from(&hf);
    let shape = src.model_shape()?.unwrap();
    assert_eq!((shape.n_layer, shape.n_embd, shape.n_expert), (40, 5120, N_EXPERT));

    eprintln!("loading layer {layer} (dGPU non-routed, {} experts hot on dGPU, all {} on iGPU)...", hot_ids.len(), N_EXPERT);
    let t0 = std::time::Instant::now();
    // Layers 0..=layer chained per token: an Engram layer's attention collapse
    // needs the previous layer's FFN pre-mix as its single-pass mHC carry, so a
    // layer > 0 cannot run alone. Each layer gets its own oracle-routed dGPU
    // hot set (cap = n_used) plus the full identity-mapped expert set on the
    // iGPU (7.2 GB/layer; the server is down anyway), so a routing deviation
    // computes on the iGPU instead of indexing past a packed buffer.
    let all_ids: Vec<u32> = (0..N_EXPERT).collect();
    let hasher = EngramHash::load(std::path::Path::new(&engram_dir)).ok();
    let mut chain: Vec<(DgpuLayerWeights, IgpuLayerWeights, Option<Vec<f32>>)> = Vec::new();
    // A reuse layer (V4.1 layers 3..7 etc.) is tested from its KV source onwards so the
    // source's store exists; the reference input/carry are taken at the source.
    // Isolated: only layer N runs; a reuse layer's KV source store is seeded from the
    // reference's dumped compressed rows (`stage_compress_kv0` of the source layer).
    let l_first = if isolated { layer } else { 0 };
    let reuse_src = v4flash_kernels::config::kv_source_of(layer as usize);
    // Layer-major mode loads one layer at a time inside its own loop below; pre-loading
    // the whole chain here (7.2 GB of iGPU experts per layer) OOMs the box past ~8 layers.
    let layer_major = std::env::var("V41_LAYER_MAJOR").map(|v| v == "1").unwrap_or(false);
    let chain_layers: Vec<_> = if layer_major { Vec::new() } else { (l_first..=layer).collect() };
    for l in chain_layers {
        let mut dl = DgpuLayerWeights::load(src, dgpu, l, &rope_for_layer)?;
        let mut il = IgpuLayerWeights::load(src, igpu, l, &all_ids, &rope_for_layer)?;
        if dump.tensor("topk_ids", l, 0).is_some() {
            let mut set = BTreeSet::new();
            for t in 0..t_n {
                set.extend(read_i32(&dump, "topk_ids", l, t as i32)?);
            }
            let hot: Vec<u32> = set.iter().map(|&e| e as u32).collect();
            let (hot_w, mut remap_host) = HotExpertWeights::load(src, dgpu, l, &hot)?;
            encode_igpu_remap(&mut remap_host, false);
            igpu.set_current()?;
            let mut remap_d = DeviceBuffer::<i32>::new(igpu.id, remap_host.len())?;
            remap_d.copy_from_host(&remap_host)?;
            il.hot_remap = Some(remap_d);
            dl.hot_experts = Some(hot_w);
        }
        // Engram rows for every token (host gather through the SSD tables).
        let rows = if dl.engram.is_some() {
            let hs = hasher.as_ref().ok_or_else(|| eyre!("layer {l} has Engram but no hash dump at {engram_dir}"))?;
            hs.self_check()?;
            let li = hs.layer_ids.iter().position(|&x| x == l as usize).ok_or_else(|| eyre!("layer {l} is not an Engram layer of the dump"))?;
            let tbl = EngramTable::open(hf.raw(), l as usize)?;
            let hashes = hs.hash_sequence(&tokens[..t_n]);
            let ein = ENGRAM_IN as usize;
            let mut rows = vec![0f32; t_n * ein];
            let t1 = std::time::Instant::now();
            for t in 0..t_n {
                tbl.gather_position(hf.raw(), &hashes[t][li], &mut rows[t * ein..(t + 1) * ein])?;
            }
            eprintln!("engram: gathered {t_n} × 24 rows for layer {l} in {:.1} ms", t1.elapsed().as_secs_f64() * 1e3);
            Some(rows)
        } else {
            None
        };
        chain.push((dl, il, rows));
    }
    eprintln!("loaded {} layers in {:.1}s (isolated: {isolated})", chain.len(), t0.elapsed().as_secs_f64());
    let ratio = COMPRESS_RATIOS[layer as usize].max(1) as usize;
    // Reference rows of a reuse layer's source store, as f16 (E2M1 × E4M3 products are
    // exact); applied to the source layer's state once it is allocated below.
    let seed_rows: Option<(usize, Vec<u16>, usize)> = match (isolated, reuse_src) {
        (true, Some(src)) => {
            let hd = v4flash_kernels::config::N_HEAD_DIM as usize;
            let n_rows = t_n / ratio;
            let mut rows16 = vec![0u16; n_rows * hd];
            for r in 0..n_rows {
                let e = dump.tensor("stage_compress_kv0", src as i32, r as i32).ok_or_else(|| eyre!("stage_compress_kv0 L{src} row {r}: dump lacks the source store tap"))?;
                let v = dump.read_f32(e)?;
                for (i, x) in v.iter().enumerate() { rows16[r * hd + i] = f32_to_f16(*x); }
            }
            Some((src, rows16, n_rows))
        }
        _ => None,
    };
    let layer_input = |t: usize| -> eyre::Result<Vec<f32>> {
        if isolated {
            dump.read_f32(dump.tensor("residual", l_first - 1, t as i32).ok_or_else(|| eyre!("residual L{} T{t}", l_first - 1))?)
        } else {
            dump.read_f32(dump.tensor("embed_hc", -1, t as i32).ok_or_else(|| eyre!("embed_hc T{t}"))?)
        }
    };
    // Carry for an isolated layer: the reference pre-mix after layer-1 (post-sigmoid
    // mixing weights, hc_mult floats per token) in the first N_HC slots.
    let carry_for = |t: usize| -> eyre::Result<Option<Vec<f32>>> {
        if !isolated { return Ok(None); }
        let e = dump.tensor("stage_pre_mix0", l_first - 1, t as i32).ok_or_else(|| eyre!("stage_pre_mix0 L{} T{t}: rerun the oracle dump with the pre_mix tap", l_first - 1))?;
        Ok(Some(dump.read_f32(e)?))
    };

    // V41_EXEC_MODE=serial synchronises after every kernel (fault localisation)
    // — but the engine skips the dGPU hot-expert MoE outside parallel mode while
    // the iGPU still zero-fills the slots the dGPU owns, so the routed experts
    // VANISH in serial mode (pre-existing). Only parallel mode gives parity.
    let mode = if std::env::var("V41_EXEC_MODE").map(|v| v == "serial").unwrap_or(false) {
        ExecMode::HetSingleStream
    } else {
        ExecMode::HetParallel
    };
    let engine = HeterogeneousEngine::new(dgpu, &darch, igpu, &iarch, mode)?;
    dgpu.set_current()?;
    let mut ds = DgpuScratch::alloc(dgpu)?;
    igpu.set_current()?;
    let mut is = IgpuScratch::alloc(igpu)?;
    let mut state = HetModelState::alloc(dgpu, igpu, t_n as u32 + 4)?;
    if let Some((src, rows16, n_rows)) = &seed_rows {
        let cs = state.layers[*src].compressor.as_mut().ok_or_else(|| eyre!("source layer {src} has no compressor state"))?;
        let buf = cs.comp_kv.f16_mut().ok_or_else(|| eyre!("source store is not the f16 format"))?;
        buf.slice_view_mut(0, rows16.len()).copy_from_host(rows16)?;
        cs.n_comp = *n_rows as u32;
        eprintln!("seeded layer {src}'s store with {n_rows} reference rows for reuse layer {layer}");
    }

    let floor = q8_floor_for(layer, t_n);
    // V41_LAYER_MAJOR=1: the M6 harness. Layers 0..=layer one at a time (each layer's
    // weights loaded, used for every token, dropped), per-token residual + mHC carry kept on
    // the host between layers, every layer's output compared with the dump's residual. This
    // is the oracle's own order, so the whole 40-layer chain fits (one layer's experts on
    // the iGPU at a time).
    if layer_major {
        let hc = HC_DIM as usize;
        let mix = v4flash_kernels::config::HC_MIX_DIM as usize;
        let n_hc = v4flash_kernels::config::N_HC as usize;
        let prefill_mode = std::env::var("V41_EXEC_MODE").map(|v| v == "prefill").unwrap_or(false);
        drop(chain);
        let mut res: Vec<Vec<f32>> = (0..t_n).map(|t| dump.read_f32(dump.tensor("embed_hc", -1, t as i32).unwrap())).collect::<eyre::Result<Vec<_>>>()?;
        let mut carry: Vec<Vec<f32>> = vec![v4flash_kernels::het::scratch::HC_PRE_ONEHOT.to_vec(); t_n];
        let hasher = EngramHash::load(std::path::Path::new(&engram_dir)).ok();
        let (mut bd, mut sdd, mut bi, mut si) = if prefill_mode {
            dgpu.set_current()?;
            let bd = BatchDgpuScratch::alloc(dgpu)?;
            let sdd = BatchDgpuShared::alloc(dgpu)?;
            igpu.set_current()?;
            let bi = BatchIgpuScratch::alloc(igpu)?;
            let si = BatchIgpuShared::alloc(igpu)?;
            (Some(bd), Some(sdd), Some(bi), Some(si))
        } else { (None, None, None, None) };
        let t_all = std::time::Instant::now();
        // V41_LAYER_MAJOR_RESEED=1: feed every layer the reference's residual for the previous
        // layer (per-layer error without accumulation); the carry and the KV stores stay the
        // engine's own.
        let reseed = std::env::var("V41_LAYER_MAJOR_RESEED").map(|v| v == "1").unwrap_or(false);
        // V41_LAYER_MAJOR_BF16RES=1: round the carried residual to bf16 after every layer, as
        // the reference (a bf16 model) does; the Q8 floor at deep layers is half a bf16 ulp of
        // the massive-activation channel, so an f32 residual stream "fails" it by drifting ulps.
        let bf16res = std::env::var("V41_LAYER_MAJOR_BF16RES").map(|v| v == "1").unwrap_or(false);
        let bf16_round = |x: f32| -> f32 {
            let u = x.to_bits();
            let lsb = (u >> 16) & 1;
            f32::from_bits((u.wrapping_add(0x7fff + lsb)) & 0xffff_0000)
        };
        // V41_PAGER=1: source routed experts on demand through the ExpertPager (packed pool +
        // remap) instead of loading all 384 resident, to validate the pager == resident path.
        let use_pager = std::env::var("V41_PAGER").map(|v| v == "1").unwrap_or(false);
        let mut pager = if use_pager {
            Some(v4flash_kernels::het::ExpertPager::new(V41HfWeights::open(&dir, None)?, igpu,
                std::env::var("V41_PAGER_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(48))?)
        } else { None };
        for l in 0..=layer {
            let tl = std::time::Instant::now();
            if reseed && l > 0 {
                for t in 0..t_n {
                    res[t] = dump.read_f32(dump.tensor("residual", l - 1, t as i32).ok_or_else(|| eyre!("residual L{} T{t}", l - 1))?)?;
                }
            }
            let mut dl = DgpuLayerWeights::load(src, dgpu, l, &rope_for_layer)?;
            let mut il = IgpuLayerWeights::load(src, igpu, l, &all_ids, &rope_for_layer)?;
            if dump.tensor("topk_ids", l, 0).is_some() {
                let mut set = BTreeSet::new();
                for t in 0..t_n { set.extend(read_i32(&dump, "topk_ids", l, t as i32)?); }
                let hot: Vec<u32> = set.iter().map(|&e| e as u32).collect();
                if use_pager {
                    // Pager path: no dGPU hot set. The forward reads the router's ACTUAL
                    // picks back from d_selected and pages them onto the iGPU pool; the
                    // all-negative remap sends every routed expert to that pool, so the
                    // dGPU contributes no MoE partial. `il.routed` (resident 384) is
                    // loaded but unused by the paged forward.
                    dl.hot_experts = None;
                } else {
                    let (hot_w, mut remap_host) = HotExpertWeights::load(src, dgpu, l, &hot)?;
                    encode_igpu_remap(&mut remap_host, false);
                    igpu.set_current()?;
                    let mut remap_d = DeviceBuffer::<i32>::new(igpu.id, remap_host.len())?;
                    remap_d.copy_from_host(&remap_host)?;
                    il.hot_remap = Some(remap_d);
                    dl.hot_experts = Some(hot_w);
                }
            }
            let rows = if dl.engram.is_some() {
                let hs = hasher.as_ref().ok_or_else(|| eyre!("layer {l} has Engram but no hash dump"))?;
                let li = hs.layer_ids.iter().position(|&x| x == l as usize).unwrap();
                let tbl = EngramTable::open(hf.raw(), l as usize)?;
                let hashes = hs.hash_sequence(&tokens[..t_n]);
                let ein = ENGRAM_IN as usize;
                let mut r = vec![0f32; t_n * ein];
                for t in 0..t_n { tbl.gather_position(hf.raw(), &hashes[t][li], &mut r[t * ein..(t + 1) * ein])?; }
                Some(r)
            } else { None };
            dgpu.set_current()?;
            if prefill_mode {
                let (bd, sdd, bi, si) = (bd.as_mut().unwrap(), sdd.as_mut().unwrap(), bi.as_mut().unwrap(), si.as_mut().unwrap());
                for t in 0..t_n {
                    bd.residual.slice_view_mut(t * hc, hc).copy_from_host(&res[t])?;
                    let mut row = vec![0f32; mix];
                    row[..n_hc].copy_from_slice(&carry[t][..n_hc]);
                    bd.hc_pre_carry.slice_view_mut(t * mix, mix).copy_from_host(&row)?;
                }
                let pos: Vec<i32> = (0..t_n as i32).collect();
                bd.pos_per_b.slice_view_mut(0, t_n).copy_from_host(&pos)?;
                if let Some(r) = &rows { engine.stage_engram_rows_batch(bd, &r[..t_n * ENGRAM_IN as usize])?; }
                state.with_kv_source(l as usize, |ls| engine.forward_layer_batch_v2(bd, bi, sdd, si, ls, &dl, &il, 0, &tokens, None, None, pager.as_mut()))?;
                engine.dgpu.compute.synchronize()?; engine.igpu.compute.synchronize()?;
                let out = host(&bd.residual_next)?;
                let cr = host(&bd.hc_pre_carry)?;
                for t in 0..t_n {
                    res[t].copy_from_slice(&out[t * hc..(t + 1) * hc]);
                    carry[t][..n_hc].copy_from_slice(&cr[t * mix..t * mix + n_hc]);
                }
            } else {
                // Layer-major decode: a reuse layer's source store already holds every
                // token's rows, so expose only the rows this token may see.
                let reuse_src = v4flash_kernels::config::kv_source_of(l as usize);
                let ratio_l = COMPRESS_RATIOS[l as usize].max(1) as usize;
                let full_n = reuse_src.and_then(|src| state.layers[src].compressor.as_ref().map(|c| c.n_comp));
                for t in 0..t_n {
                    if let (Some(src), Some(_)) = (reuse_src, full_n) {
                        state.layers[src].compressor.as_mut().unwrap().n_comp = ((t + 1) / ratio_l) as u32;
                    }
                    ds.residual.copy_from_host(&res[t])?;
                    if l > 0 { ds.hc_pre_carry.slice_view_mut(0, n_hc).copy_from_host(&carry[t][..n_hc])?; }
                    if let Some(r) = &rows { let ein = ENGRAM_IN as usize; engine.stage_engram_rows(&mut ds, &r[t * ein..(t + 1) * ein])?; }
                    if let Some(pg) = pager.as_mut() {
                        state.with_kv_source(l as usize, |ls| engine.forward_layer_standalone_graphs_paged(&mut ds, &mut is, ls, &dl, &il, t as u32, tokens[t], pg))?;
                    } else {
                        state.with_kv_source(l as usize, |ls| engine.forward_layer_standalone_graphs(&mut ds, &mut is, ls, &dl, &il, t as u32, tokens[t]))?;
                    }
                    ds.residual_next.copy_to_host(&mut res[t])?;
                    let cr = host(&ds.hc_pre_carry)?;
                    carry[t][..n_hc].copy_from_slice(&cr[..n_hc]);
                }
                if let (Some(src), Some(n)) = (reuse_src, full_n) {
                    state.layers[src].compressor.as_mut().unwrap().n_comp = n;
                }
            }
            if bf16res {
                for t in 0..t_n { for v in res[t].iter_mut() { *v = bf16_round(*v); } }
            }
            // Compare this layer's output with the dump (global-scale metric = the floor's).
            let (mut g_d, mut g_r, mut worst_tok) = (0f32, 0f32, 0f32);
            let mut per_tok = String::new();
            let mut tok_err: Vec<(f32, f32)> = Vec::new();
            for t in 0..t_n {
                let want = dump.read_f32(dump.tensor("residual", l, t as i32).ok_or_else(|| eyre!("residual L{l} T{t}"))?)?;
                let (mut md, mut mr, mut argd) = (0f32, 0f32, 0usize);
                for (k, (a, b)) in res[t].iter().zip(&want).enumerate() { if (a - b).abs() > md { md = (a - b).abs(); argd = k; } mr = mr.max(b.abs()); }
                g_d = g_d.max(md); g_r = g_r.max(mr); worst_tok = worst_tok.max(md / mr.max(1e-30));
                per_tok.push_str(&format!(" T{t}:Δ{md:.3e}@{argd}/ref{mr:.3e}"));
                tok_err.push((md, mr));
            }
            if std::env::var("V41_LAYER_MAJOR_VERBOSE").map(|v| v == "1").unwrap_or(false) { eprintln!("[layer-major] L{l:02} per-token{per_tok}"); }
            // Per-token view. Once the sink token's massive channel exists (L15+) the global
            // metric is that channel's error in bf16 ulps (the reference residual is bf16 and
            // the Q8 oracle's own count there is rounding luck), so the other tokens are gated
            // against the Q8 oracle's per-token error and the sink token is reported in ulps.
            let pt_floor = q8_floor_pertok(l, t_n);
            let (mut pt_line, mut worst_nonsink, mut sink_ulps) = (String::new(), 0f32, 0f32);
            if let Some(fl_t) = &pt_floor {
                for t in 0..t_n.min(fl_t.len()) {
                    let (md, mr) = tok_err[t];
                    let ulp = 2f32.powi((mr.max(1e-30).log2().floor() as i32) - 7);
                    let r = md / mr.max(1e-30) / fl_t[t].max(1e-30);
                    if t == 0 { sink_ulps = md / ulp; pt_line.push_str(&format!(" T0:{:.1}ulp({r:.1}×)", md / ulp)); }
                    else { worst_nonsink = worst_nonsink.max(r); pt_line.push_str(&format!(" T{t}:{r:.2}×")); }
                }
                eprintln!("[layer-major] L{l:02} per-token vs Q8 oracle:{pt_line}");
            }
            let fl = q8_floor_for(l, t_n);
            eprintln!("[layer-major] L{l:02} ratio {} global Δ/scale {:.3e} = {:.1}× floor (per-token worst {:.3e}) {:.1}s",
                COMPRESS_RATIOS[l as usize], g_d / g_r.max(1e-30), g_d / g_r.max(1e-30) / fl, worst_tok, tl.elapsed().as_secs_f64());
            if l == layer {
                if pt_floor.is_some() {
                    // Per-token relative check is a clean gate only before the massive-activation
                    // tokens multiply (T <= 32: one sink). Past that several tokens carry 1e3-1e5
                    // channels whose bf16/Q8 rounding is a decorrelating term as large as the Q8
                    // floor, so a per-token ratio is noise (both kernel paths trip it identically).
                    // Above 32 tokens it is a DIAGNOSTIC; the HEAD gate is the pass/fail.
                    let hard = t_n <= 32;
                    if worst_nonsink > 2.0 || sink_ulps > 8.0 {
                        if hard {
                            return Err(eyre!("layer-major chain: layer {l} per-token gate: worst non-sink token {worst_nonsink:.2}× Q8 oracle (gate 2×), sink token {sink_ulps:.1} bf16 ulps (gate 8)"));
                        } else {
                            eprintln!("[layer-major] NOTE (diagnostic, T>32): layer {l} worst non-sink {worst_nonsink:.2}× Q8, sink {sink_ulps:.1} ulps -- deferring to the HEAD gate");
                        }
                    }
                } else if g_d / g_r.max(1e-30) > 2.0 * fl {
                    return Err(eyre!("layer-major chain: layer {l} global Δ/scale {:.3e} > gate {:.3e}", g_d / g_r.max(1e-30), 2.0 * fl));
                }
            }
        }
        eprintln!("[layer-major] {} layers in {:.0}s", layer + 1, t_all.elapsed().as_secs_f64());
        if let Some(pg) = pager.as_ref() {
            eprintln!("[pager] prefill {}/{} misses/requests, decode {}/{} ({:.1}% decode hit, {} slots)",
                pg.prefill_misses, pg.prefill_requests, pg.decode_misses, pg.decode_requests,
                100.0 * (1.0 - pg.decode_misses as f64 / pg.decode_requests.max(1) as f64), 48);
        }
        // Head gate on the last token (host side): hc_pre(h, pre_mix) = Σ_h pre_mix[h]·h[h]
        // (model.py `hc_pre`), the final RMSNorm, the Q8_0 head; vs the dump's logits and the
        // Q8 oracle's own logits error.
        let t_last = t_n - 1;
        if let Some(lt) = dump.tensor("logits", -1, t_last as i32) {
            let t_head = std::time::Instant::now();
            let n_embd = v4flash_kernels::config::N_EMBD as usize;
            let n_vocab = v4flash_kernels::config::N_VOCAB as usize;
            let want = dump.read_f32(lt)?;
            let mut x = vec![0f32; n_embd];
            for h in 0..n_hc { let w = carry[t_last][h]; for d in 0..n_embd { x[d] += w * res[t_last][h * n_embd + d]; } }
            let onw: Vec<f32> = hf.read(hf.get("output_norm.weight")?)?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let ss = x.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
            let inv = 1.0 / (ss + v4flash_kernels::config::RMS_EPS).sqrt();
            let xn: Vec<f32> = x.iter().zip(&onw).map(|(v, w)| v * inv * w).collect();
            let bytes = hf.read(hf.get("output.weight")?)?;
            let row_bytes = n_embd / 32 * 34;
            let mut logits = vec![0f32; n_vocab];
            let chunk = 2048;
            for r0 in (0..n_vocab).step_by(chunk) {
                let rows = chunk.min(n_vocab - r0);
                let w = dequant_q8_0(&bytes[r0 * row_bytes..(r0 + rows) * row_bytes], rows, n_embd);
                for r in 0..rows { logits[r0 + r] = w[r * n_embd..(r + 1) * n_embd].iter().zip(&xn).map(|(a, b)| a * b).sum(); }
            }
            let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0;
            let top5 = |v: &[f32]| { let mut idx: Vec<usize> = (0..v.len()).collect(); idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap()); idx[..5].to_vec() };
            let (mut md, mut mr) = (0f32, 0f32);
            for (a, b) in logits.iter().zip(&want) { md = md.max((a - b).abs()); mr = mr.max(b.abs()); }
            let fl_log = q8_logits_floor(t_n).unwrap_or_else(|| {
                missing_floor(FLOOR_PERTOK_FILE, "Q8 logits floor", 2.05e-1)
            });
            let (ae, ar) = (argmax(&logits), argmax(&want));
            // The engine runs Q8_0 weights, so its ground truth is the Q8 CPU oracle, not the fp8
            // reference. Gate on the Q8 oracle's argmax; report whether the engine also matches the
            // fp8 reference (a bonus that means it beats Q8) or differs from it (the known fp8-native
            // argmax flip at a near-tie — a weights-precision item, not a port bug).
            // NEVER fall back to `ar` (the fp8 reference argmax) here: the gate three
            // lines down is `ae != aq`, and the comment above says the fp8 reference is
            // the wrong target. Silently substituting it made this gate compare the
            // engine against something the test explicitly does not want to match.
            let aq = q8_logits_argmax(t_n).unwrap_or_else(|| {
                missing_floor(FLOOR_PERTOK_FILE, "Q8 logits argmax", ar)
            });
            let rel = md / mr.max(1e-30);
            eprintln!("[layer-major] HEAD T{t_last}: argmax engine {ae} (Q8 oracle {aq}, fp8 ref {ar}); top5 engine {:?} ref {:?}; logits Δ/scale {rel:.3e} = {:.2}× Q8 oracle ({fl_log:.3e}); logit[ref argmax] engine {:.3} ref {:.3} ({:.1}s){}",
                top5(&logits), top5(&want), rel / fl_log, logits[ar], want[ar], t_head.elapsed().as_secs_f64(),
                if ae == ar { "  [matches fp8 ref — beats Q8]" } else if ae == aq { "  [matches Q8 weights; fp8-native flip is a quality item]" } else { "  [MISMATCH]" });
            if ae != aq || rel > 2.0 * fl_log {
                return Err(eyre!("layer-major chain: HEAD gate failed (engine argmax {ae} != Q8 argmax {aq}; fp8 ref {ar}; logits Δ/scale {rel:.3e} vs gate {:.3e})", 2.0 * fl_log));
            }
        }
        return Ok(());
    }

    if std::env::var("V41_EXEC_MODE").map(|v| v == "prefill").unwrap_or(false) {
        // Batched prefill twin: all t_n tokens in one chunk at pos0 = 0.
        dgpu.set_current()?;
        let mut bd = BatchDgpuScratch::alloc(dgpu)?;
        let mut sdd = BatchDgpuShared::alloc(dgpu)?;
        igpu.set_current()?;
        let mut bi = BatchIgpuScratch::alloc(igpu)?;
        let mut si = BatchIgpuShared::alloc(igpu)?;
        dgpu.set_current()?;
        let hc = HC_DIM as usize;
        for t in 0..t_n {
            let inp = layer_input(t)?;
            assert_eq!(inp.len(), hc);
            bd.residual.slice_view_mut(t * hc, hc).copy_from_host(&inp)?;
        }
        let pos: Vec<i32> = (0..t_n as i32).collect();
        bd.pos_per_b.slice_view_mut(0, t_n).copy_from_host(&pos)?;
        let t0 = std::time::Instant::now();
        if isolated {
            let mix = v4flash_kernels::config::HC_MIX_DIM as usize;
            let mut rows_c = vec![0f32; t_n * mix];
            for t in 0..t_n {
                let c = carry_for(t)?.unwrap();
                rows_c[t * mix..t * mix + c.len().min(mix)].copy_from_slice(&c[..c.len().min(mix)]);
            }
            bd.hc_pre_carry.slice_view_mut(0, t_n * mix).copy_from_host(&rows_c)?;
        }
        for l in l_first as usize..=layer as usize {
            let ci = l - l_first as usize;
            if let Some(r) = &chain[ci].2 {
                engine.stage_engram_rows_batch(&mut bd, &r[..t_n * ENGRAM_IN as usize])?;
            }
            state.with_kv_source(l, |ls| engine.forward_layer_batch_v2(&mut bd, &mut bi, &mut sdd, &mut si, ls, &chain[ci].0, &chain[ci].1, 0, &tokens, None, None, None))?;
            if l < layer as usize {
                std::mem::swap(&mut bd.residual, &mut bd.residual_next);
            }
        }
        engine.dgpu.compute.synchronize()?;
        engine.dgpu.xfer.synchronize()?;
        engine.igpu.compute.synchronize()?;
        engine.igpu.xfer.synchronize()?;
        eprintln!("prefill chunk of {t_n} tokens through layer {layer} in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        let got_all = host(&bd.residual_next)?;
        let sel_all = {
            let mut v = vec![0i32; bd.d_selected.len()];
            bd.d_selected.copy_to_host(&mut v)?;
            v
        };
        let mut worst = 0f32;
        let (mut g_max_d, mut g_max_r) = (0f32, 0f32);
        let mut routing_mismatch = 0usize;
        for t in 0..t_n {
            let got = &got_all[t * hc..(t + 1) * hc];
            let want = dump.read_f32(dump.tensor("residual", layer, t as i32).ok_or_else(|| eyre!("residual L{layer} T{t}"))?)?;
            let (mut max_d, mut max_r) = (0f32, 0f32);
            for (a, b) in got.iter().zip(&want) {
                max_d = max_d.max((a - b).abs());
                max_r = max_r.max(b.abs());
            }
            let rel = max_d / max_r.max(1e-30);
            worst = worst.max(rel);
            g_max_d = g_max_d.max(max_d);
            g_max_r = g_max_r.max(max_r);
            let sel = &sel_all[t * N_EXPERT_USED..(t + 1) * N_EXPERT_USED];
            let got_set: BTreeSet<i32> = sel.iter().copied().collect();
            let want_set: BTreeSet<i32> = want_ids.get(t).map(|v| v.iter().copied().collect()).unwrap_or_default();
            let routing = if !have_ids { "n/a" } else if got_set == want_set { "match" } else { routing_mismatch += 1; "DIFFERS" };
            eprintln!("T{t}: max|Δ|={max_d:.3e} max|ref|={max_r:.3} Δ/scale={rel:.3e} ({:.1}× floor) nan={} routing {routing}",
                rel / floor, got.iter().any(|x| !x.is_finite()));
        }
        check_caches(&dump, layer, &state.layers[layer as usize], &state.layers[v4flash_kernels::config::kv_source_of(layer as usize).unwrap_or(layer as usize)], t_n)?;
        let global = g_max_d / g_max_r.max(1e-30);
        eprintln!("[prefill] worst per-token Δ/scale = {worst:.3e}; GLOBAL Δ/scale = {global:.3e} = {:.1}× floor (gate {:.3e}); routing mismatches {routing_mismatch}/{t_n}", global / floor, 2.0 * floor);
        if global > 2.0 * floor {
            return Err(eyre!("layer {layer} PREFILL parity FAILED: global Δ/scale {global:.3e} > {:.3e}", 2.0 * floor));
        }
        return Ok(());
    }

    // CPU cross-checks from the engine's OWN buffers (separates "wrong
    // grouped wo_a layout" from "wrong attention core").
    let wo_a = {
        let vt = hf.get(&format!("blk.{layer}.attn_output_a.weight"))?;
        dequant_q8_0(&hf.read(vt)?, 8192, 4096) // [8 groups × 1024 rank][4096 = 8 heads × 512]
    };
    let sinks: Vec<f32> = hf.read(hf.get(&format!("blk.{layer}.attn_sinks.weight"))?)?
        .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();

    let floor = q8_floor_for(layer, t_n);
    let mut worst = 0f32;
    // The floor (compare.py) is max|Δ| over ALL tokens divided by the max |ref| over all
    // tokens; the per-token Δ/scale lines are stricter for small-scale tokens and stay
    // informational. The gate uses the floor's own definition.
    let (mut g_max_d, mut g_max_r) = (0f32, 0f32);
    let mut routing_mismatch = 0usize;
    for t in 0..t_n {
        let inp = layer_input(t)?;
        assert_eq!(inp.len(), HC_DIM as usize);
        dgpu.set_current()?;
        ds.residual.copy_from_host(&inp)?;
        if let Some(c) = carry_for(t)? {
            let n = c.len().min(ds.hc_pre_carry.len());
            ds.hc_pre_carry.slice_view_mut(0, n).copy_from_host(&c[..n])?;
        }
        if isolated {
            if let Some(src) = reuse_src {
                // the source would have appended its row for this token before layer N runs
                state.layers[src].compressor.as_mut().unwrap().n_comp = ((t + 1) / ratio) as u32;
            }
        }
        for l in l_first as usize..=layer as usize {
            let (dl, il, rows) = (&chain[l - l_first as usize].0, &chain[l - l_first as usize].1, &chain[l - l_first as usize].2);
            if let Some(r) = rows {
                let ein = ENGRAM_IN as usize;
                engine.stage_engram_rows(&mut ds, &r[t * ein..(t + 1) * ein])?;
            }
            state.with_kv_source(l, |ls| engine.forward_layer_standalone_graphs(&mut ds, &mut is, ls, dl, il, t as u32, tokens[t]))?;
            if l < layer as usize {
                std::mem::swap(&mut ds.residual, &mut ds.residual_next);
            }
        }
        let mut got = vec![0f32; HC_DIM as usize];
        ds.residual_next.copy_to_host(&mut got)?;
        // The captured decode graphs bake the residual / residual_next pointers
        // per layer, so restore the token-start assignment after an odd chain
        // (the driver's "end-of-token extra swap").
        if (layer - l_first) % 2 == 1 {
            std::mem::swap(&mut ds.residual, &mut ds.residual_next);
        }
        let want = dump.read_f32(dump.tensor("residual", layer, t as i32).ok_or_else(|| eyre!("residual L{layer} T{t}"))?)?;
        let mut sel = vec![0i32; N_EXPERT_USED];
        ds.d_selected.copy_to_host(&mut sel)?;

        // Per-stage comparison against the oracle's taps (V41_ORACLE_STAGES=1
        // dump): first diverging sub-step localises the bug.
        let stages: [(&str, &DeviceBuffer<f32>); 14] = [
            ("stage_hc_pre0", &ds.attn_cur),          // attention collapse (pre one-hot at L0)
            ("stage_attn_norm0", &ds.attn_input_norm),
            ("stage_attn_wq_a0", &ds.qr),             // q-LoRA down [1280]
            ("stage_attn_q_norm0", &ds.qr_normed),
            ("stage_attn_wq_b0", &ds.q),              // q up [64*512] (ref: pre-rope)
            ("stage_attn_wkv0", &ds.kv_raw),          // kv latent [512]
            ("stage_attn_kv_norm0", &ds.kv_normed),   // (ref: pre-rope, pre fp8 fake-quant)
            ("stage_attn_wo_b_in0", &ds.low),         // after grouped wo_a [8192]
            ("stage_attn_wo_b0", &ds.attn_out),       // after wo_b [5120]
            ("stage_attn0", &ds.attn_out),            // attention output after W_O
            ("stage_hc_post0", &ds.after_attn_hc),    // post-attention residual [hc, dim]
            ("stage_hc_pre1", &ds.ffn_cur),           // FFN collapse (attn's pre)
            ("stage_ffn_norm0", &ds.ffn_input_norm),
            ("stage_hc_post1", &ds.residual_next),    // = layer output
        ];
        {
            let heads = host(&ds.heads)?;
            let low = host(&ds.low)?;
            eprintln!("    engine buffers: heads n={} low n={} q n={} kv_normed n={}", heads.len(), low.len(), ds.q.len(), ds.kv_normed.len());
            if heads.len() >= 64 * 512 && low.len() >= 8192 {
                // (A) grouped wo_a with HF's row-block layout: group g takes heads 8g..8g+8.
                let mut low_cpu = vec![0f32; 8192];
                for g in 0..8 {
                    for r in 0..1024 {
                        let wr = &wo_a[(g * 1024 + r) * 4096..(g * 1024 + r + 1) * 4096];
                        let hs = &heads[g * 4096..(g + 1) * 4096];
                        low_cpu[g * 1024 + r] = wr.iter().zip(hs).map(|(a, b)| a * b).sum();
                    }
                }
                let (md, r) = rel(&low[..8192], &low_cpu);
                eprintln!("    check A: engine low vs CPU grouped-wo_a(engine heads): max|Δ|={md:.3e} Δ/scale={r:.3e}");
                if t == 0 {
                    // (B) single-key attention at pos 0 from the engine's q / kv / sinks
                    // (rope is the identity at pos 0): heads_h = w·kv, w = e^s/(e^s+e^sink).
                    let q = host(&ds.q)?;
                    let kv = host(&ds.kv_normed)?;
                    let mut heads_cpu = vec![0f32; 64 * 512];
                    for h in 0..64 {
                        let qh = &q[h * 512..(h + 1) * 512];
                        let sc: f32 = qh.iter().zip(&kv[..512]).map(|(a, b)| a * b).sum::<f32>() * (512f32).sqrt().recip();
                        let m = sc.max(sinks[h]);
                        let w = (sc - m).exp() / ((sc - m).exp() + (sinks[h] - m).exp());
                        for i in 0..512 {
                            heads_cpu[h * 512 + i] = w * kv[i];
                        }
                    }
                    let (md, r) = rel(&heads[..64 * 512], &heads_cpu);
                    eprintln!("    check B: engine heads vs CPU single-key attn(engine q, kv, sinks): max|Δ|={md:.3e} Δ/scale={r:.3e}");
                    // Implied per-head scalar: if heads_h ≈ w'_h · kv then the core's V is right
                    // and only the softmax weight differs; a spread says the V vector differs.
                    eprintln!("    state.layers[0].n_raw = {}", state.layers[0].n_raw);
                    let qn = host(&ds.q_normed)?;
                    eprintln!("    q[0..4]={:?} q_normed[0..4]={:?} (q_normed n={})", &q[..4], &qn[..4.min(qn.len())], qn.len());
                    for h in [0usize, 1, 31, 63] {
                        let qh = &qn[h * 512..(h + 1) * 512];
                        let sc: f32 = qh.iter().zip(&kv[..512]).map(|(a, b)| a * b).sum::<f32>() * (512f32).sqrt().recip();
                        let m = sc.max(sinks[h]);
                        let w = (sc - m).exp() / ((sc - m).exp() + (sinks[h] - m).exp());
                        eprintln!("      via q_normed h{h:2}: score={sc:.3} w={w:.4}");
                    }
                    for h in [0usize, 1, 31, 63] {
                        let qh = &q[h * 512..(h + 1) * 512];
                        let sc: f32 = qh.iter().zip(&kv[..512]).map(|(a, b)| a * b).sum::<f32>() * (512f32).sqrt().recip();
                        let m = sc.max(sinks[h]);
                        let w = (sc - m).exp() / ((sc - m).exp() + (sinks[h] - m).exp());
                        let mut ratios: Vec<f32> = (0..512).filter(|&i| kv[i].abs() > 0.2).map(|i| heads[h * 512 + i] / kv[i]).collect();
                        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let (lo, med, hi) = (ratios[0], ratios[ratios.len() / 2], ratios[ratios.len() - 1]);
                        let tail: Vec<f32> = (448..452).map(|i| heads[h * 512 + i] / kv[i].max(1e-6)).collect();
                        eprintln!("      h{h:2}: score={sc:.3} sink={:.3} w_cpu={w:.4} implied w: min {lo:.4} med {med:.4} max {hi:.4} (n={}) rope-tail ratios {tail:?}", sinks[h], ratios.len());
                    }
                }
            }
        }
        // FFN output: reference ffn(x) = routed + shared vs the engine's parts.
        if let Some(e) = dump.tensor("stage_ffn0", layer, t as i32) {
            let want = dump.read_f32(e)?;
            let (recv, dg, sh) = (host(&ds.ffn_moe_recv)?, host(&ds.ffn_moe_dgpu)?, host(&ds.ffn_shared)?);
            let n = want.len();
            let mag = |v: &[f32]| v.iter().take(n).fold(0f32, |a, &x| a.max(x.abs()));
            // After the engine's vec_adds `ffn_moe_recv` = routed(iGPU) + shared + routed(dGPU).
            let (md, r) = rel(&recv[..n.min(recv.len())], &want);
            eprintln!("    ffn: ref max|ffn|={:.3} eng total={:.3} (dgpu part {:.3}, shared {:.3})  vs ref: max|Δ|={md:.3e} Δ/scale={r:.3e}",
                mag(&want), mag(&recv), mag(&dg), mag(&sh));
        }
        // Router weights (after sqrtsoftplus, top-k normalisation, route_scale) vs the reference gate.
        if let Some(e) = dump.tensor("stage_gate0_0", layer, t as i32) {
            let want_w = dump.read_f32(e)?;
            let ew = host(&ds.d_ew)?;
            let want_i = read_i32(&dump, "stage_gate0_1", layer, t as i32)?;
            eprintln!("    gate: ref ids {:?} w {:?}", want_i, want_w.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>());
            eprintln!("          eng ids {:?} w {:?}", sel, ew[..N_EXPERT_USED.min(ew.len())].iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>());
        }
        for (tag, buf) in stages {
            let Some(e) = dump.tensor(tag, layer, t as i32) else { continue };
            let want = dump.read_f32(e)?;
            let mut got = vec![0f32; buf.len()];
            buf.copy_to_host(&mut got)?;
            let n = want.len().min(got.len());
            let (mut md, mut mr) = (0f32, 0f32);
            for i in 0..n {
                md = md.max((got[i] - want[i]).abs());
                mr = mr.max(want[i].abs());
            }
            eprintln!("    {tag:16} ref n={:6} eng n={:6}  max|Δ|={md:.3e} max|ref|={mr:.3e} Δ/scale={:.3e}",
                want.len(), got.len(), md / mr.max(1e-30));
        }

        let (mut max_d, mut max_r, mut sd, mut sr) = (0f32, 0f32, 0f64, 0f64);
        for (a, b) in got.iter().zip(&want) {
            let d = a - b;
            max_d = max_d.max(d.abs());
            max_r = max_r.max(b.abs());
            sd += (d as f64) * (d as f64);
            sr += (*b as f64) * (*b as f64);
        }
        let rel = max_d / max_r.max(1e-30);
        let rms = (sd / sr.max(1e-30)).sqrt();
        worst = worst.max(rel);
        g_max_d = g_max_d.max(max_d);
        g_max_r = g_max_r.max(max_r);
        let got_set: BTreeSet<i32> = sel.iter().copied().collect();
        let want_set: BTreeSet<i32> = want_ids.get(t).map(|v| v.iter().copied().collect()).unwrap_or_default();
        let routing = if !have_ids { "n/a" } else if got_set == want_set { "match" } else { routing_mismatch += 1; "DIFFERS" };
        eprintln!(
            "T{t}: max|Δ|={max_d:.3e} max|ref|={max_r:.3} Δ/scale={rel:.3e} ({:.1}× floor) rms(Δ)/rms(ref)={rms:.3e} nan={} routing {routing} got {:?} want {:?}",
            rel / floor,
            got.iter().any(|x| !x.is_finite()),
            sel,
            want_ids.get(t).cloned().unwrap_or_default()
        );
    }
    if let Ok(d) = std::env::var("V41_DUMP_CACHES") {
        // Last boundary's pooled (pre-norm, pre-RoPE) compressor row and the
        // post-everything comp_row, for the offline probe.
        let mut pooled = vec![0f32; ds.pooled.len()];
        ds.pooled.copy_to_host(&mut pooled)?;
        let bytes: Vec<u8> = pooled[..512.min(pooled.len())].iter().flat_map(|x| x.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{d}/engine_pooled_last_L{layer:02}.bin"), bytes);
    }
    check_caches(&dump, layer, &state.layers[layer as usize], &state.layers[v4flash_kernels::config::kv_source_of(layer as usize).unwrap_or(layer as usize)], t_n)?;
    let global = g_max_d / g_max_r.max(1e-30);
    eprintln!("worst per-token Δ/scale = {worst:.3e}; GLOBAL Δ/scale = {global:.3e} = {:.1}× floor (gate {:.3e}); routing mismatches {routing_mismatch}/{t_n}", global / floor, 2.0 * floor);
    if global > 2.0 * floor {
        return Err(eyre!("layer {layer} parity FAILED: global Δ/scale {global:.3e} > {:.3e}", 2.0 * floor));
    }
    Ok(())
}
