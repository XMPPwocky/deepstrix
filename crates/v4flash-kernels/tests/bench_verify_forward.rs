//! Production `forward_verify` driver: correctness (batched B=k ≡ sequential
//! B=1 decode) + cost C(B) = t(B=k)/t(B=1) on the real verify path.
//!
//! Unlike `bench_verify_decode` (which routed the B>=2 step through
//! `forward_prefill` as a shape proxy), this drives the real
//! `HeterogeneousEngine::forward_verify` for BOTH the B=1 anchor and the
//! batched steps, so the anchor is the genuine decode critical path and
//! the C(B) is the real verify cost.
//!
//! ## Run
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 \
//! DGPU_HOT_EXPERTS=8 \
//! DGPU_HOT_EXPERTS_FILE=/home/claude-code/deepstrix/reference/decode_hot_experts.txt \
//! VERIFY_B=1,2,3,4 FAKE_POS=4096,98304 VERIFY_CORRECT_P=64 \
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_verify_forward -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};

use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{COMPRESS_RATIOS, HC_DIM, N_LAYER, N_VOCAB, SWA_WINDOW};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, DgpuScratch, ExecMode, HetModelState, HetModelWeights,
    HeterogeneousEngine, IgpuScratch,
};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 (9070 XT) device found"))
}
fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 (Strix iGPU) device found"))
}

fn stamp_fake_depth(state: &mut HetModelState, pos: u32) {
    for layer in 0..N_LAYER as usize {
        let ls = &mut state.layers[layer];
        ls.n_raw = SWA_WINDOW.min(pos);
        ls.raw_off = 0;
        let ratio = COMPRESS_RATIOS[layer];
        if ratio > 0 {
            if let Some(cs) = ls.compressor.as_mut() {
                cs.n_comp = pos / ratio;
            }
            if let Some(ics) = ls.indexer_compressor.as_mut() {
                ics.n_comp = pos / ratio;
            }
        }
    }
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn bench_verify_forward() -> eyre::Result<()> {
    install_panic_handler()?;

    let bs_list: Vec<usize> = std::env::var("VERIFY_B")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .map(|b| b.clamp(1, 32))
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2, 3, 4]);
    let b_max = *bs_list.iter().max().unwrap();

    let depths: Vec<u32> = std::env::var("FAKE_POS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<u32>| !v.is_empty())
        .unwrap_or_else(|| vec![4096, 98304]);

    // Correctness prime depth (real sequential decode up to P, then compare
    // batched B=k vs sequential at P..P+k). Small — every position is a real
    // forward, so keep it modest (>= 32 exercises compressor boundaries).
    let correct_p: u32 = std::env::var("VERIFY_CORRECT_P")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let n_warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let n_iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let perfetto_base = std::env::var("PERFETTO_DEVICE_OUT").ok();

    eprintln!("bench_verify_forward: B={bs_list:?} depths={depths:?} correct_P={correct_p} warmup={n_warmup} iters={n_iters}");
    eprintln!(
        "het-split: DGPU_HOT_EXPERTS={:?} FILE={:?}",
        std::env::var("DGPU_HOT_EXPERTS").ok(),
        std::env::var("DGPU_HOT_EXPERTS_FILE").ok(),
    );

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    eprintln!("dGPU={dgpu_arch} iGPU={igpu_arch}");

    let rope_for_layer = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing rope_params L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };

    eprintln!("loading het weights (slow, ~1-2 min)...");
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope_for_layer)?;
    eprintln!("loaded.");

    let mut engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;

    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;
    let mut head_scratch = DgpuScratch::alloc(dgpu)?;

    // Inputs: real dump residuals for the first 7 positions, cycled.
    let n_real = PROMPT_TOKENS.len();
    let n_inp = b_max.max(n_real).max(correct_p as usize + b_max + 1);
    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(n_inp);
    let mut tokens: Vec<i32> = Vec::with_capacity(n_inp);
    for i in 0..n_inp {
        let src = i % n_real;
        let entry = dump
            .tensor("layer_input_residual", 0, src as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{src}"))?;
        let hc = dump.read_f32(entry)?;
        assert_eq!(hc.len(), HC_DIM as usize);
        input_hcs.push(hc);
        tokens.push(PROMPT_TOKENS[src]);
    }

    // ================= ROCPROFV3 KERNEL-TRACE MODE =================
    // VERIFY_KT=1: run exactly ONE verify forward (B=bs_list[0],
    // depth=depths[0]) at a graph-warmed, faked depth, then exit — a
    // minimal kernel stream for `rocprofv3 --kernel-trace` so per-kernel
    // GPU durations (graph-equivalent device compute; no host-launch idle)
    // can be summed per device.
    if std::env::var("VERIFY_KT").is_ok() {
        let b = bs_list[0];
        let fake_pos = depths[0];
        let n_kv_max = fake_pos + b as u32 + 16;
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let inp = &input_hcs[0..b];
        let toks = &tokens[0..b];
        // Exactly ONE verify forward → the kernel trace is one forward's
        // worth of dispatches per device. Graph capture (B=1 decode path)
        // records + executes each kernel once, so it stays comparable.
        stamp_fake_depth(&mut state, fake_pos);
        engine.forward_verify(
            &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
            &mut state, &weights, inp, toks, fake_pos, false,
        )?;
        engine.shutdown()?;
        return Ok(());
    }

    // ================= DEVICE-BUSY (in-process, artifact-free) =============
    // VERIFY_BUSY=1: enable HIP-event stage recording, run ONE forward per
    // requested B at a warmed fake depth, and compute per-device BUSY time
    // from real GPU HW timestamps (Event::elapsed_ms), NOT perfetto stage
    // spans. Reports union-of-all-intervals, union-of-leaf-intervals (true
    // busy, parent spans excluded), and total span. Compare to Instant wall.
    if std::env::var("VERIFY_BUSY").is_ok() {
        // union of leaf intervals: exclude any interval strictly containing
        // another (parent/wait spans include intra-stage idle).
        let busy = |mut iv: Vec<(&'static str, f32, f32)>| -> (f32, f32, f32) {
            if iv.is_empty() {
                return (0.0, 0.0, 0.0);
            }
            // leaf = not a strict superset of any other interval.
            let n = iv.len();
            let mut is_leaf = vec![true; n];
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let contains = iv[i].1 <= iv[j].1
                        && iv[j].2 <= iv[i].2
                        && (iv[i].1 < iv[j].1 || iv[j].2 < iv[i].2);
                    if contains {
                        is_leaf[i] = false;
                        break;
                    }
                }
            }
            let span = iv.iter().map(|x| x.2).fold(0.0f32, f32::max)
                - iv.iter().map(|x| x.1).fold(f32::INFINITY, f32::min);
            let union = |sel: &dyn Fn(usize) -> bool| -> f32 {
                let mut ints: Vec<(f32, f32)> = (0..iv.len())
                    .filter(|&i| sel(i))
                    .map(|i| (iv[i].1, iv[i].2))
                    .collect();
                ints.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let mut total = 0.0f32;
                let mut cur: Option<(f32, f32)> = None;
                for (s, e) in ints {
                    match cur {
                        None => cur = Some((s, e)),
                        Some((cs, ce)) => {
                            if s <= ce {
                                cur = Some((cs, ce.max(e)));
                            } else {
                                total += ce - cs;
                                cur = Some((s, e));
                            }
                        }
                    }
                }
                if let Some((cs, ce)) = cur {
                    total += ce - cs;
                }
                total
            };
            let union_all = union(&|_| true);
            let union_leaf = union(&|i| is_leaf[i]);
            iv.clear();
            (union_all, union_leaf, span)
        };

        let busy_bs: Vec<usize> = bs_list.clone();
        for &b in &busy_bs {
            let fake_pos = depths[0];
            let n_kv_max = fake_pos + b as u32 + 16;
            let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
            let inp = &input_hcs[0..b];
            let toks = &tokens[0..b];
            // warm (esp. B=1 graph capture)
            for _ in 0..3 {
                stamp_fake_depth(&mut state, fake_pos);
                engine.forward_verify(
                    &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
                    &mut state, &weights, inp, toks, fake_pos, false,
                )?;
            }
            engine.dgpu.events.set_enabled(true);
            engine.igpu.events.set_enabled(true);
            stamp_fake_depth(&mut state, fake_pos);
            let t = Instant::now();
            engine.forward_verify(
                &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
                &mut state, &weights, inp, toks, fake_pos, false,
            )?;
            let wall = t.elapsed().as_secs_f64() * 1000.0;
            engine.dgpu.events.set_enabled(false);
            engine.igpu.events.set_enabled(false);
            let dg = engine.dgpu.events.harvest_intervals()?;
            let ig = engine.igpu.events.harvest_intervals()?;
            let (da, dl, ds) = busy(dg);
            let (ia, il, is_) = busy(ig);
            eprintln!("\n### VERIFY_BUSY B={b} depth={fake_pos}: wall(instr)={wall:.2} ms");
            eprintln!("  dGPU: leaf_busy={dl:.2}  all_busy={da:.2}  span={ds:.2} ms");
            eprintln!("  iGPU: leaf_busy={il:.2}  all_busy={ia:.2}  span={is_:.2} ms");
            eprintln!("  => wall - dGPU_leaf_busy = {:.2} ms (gap not covered by dGPU compute)", wall - dl as f64);
            eprintln!("  => iGPU_leaf_busy that could hide under dGPU = {il:.2} ms");
        }
        engine.shutdown()?;
        return Ok(());
    }

    // ================= CORRECTNESS =================
    // Reference: fresh state primed 0..P via forward_verify(B=1), then the
    // k verify positions run sequentially (still B=1 = the decode path).
    // Batched: identically-primed fresh state, one forward_verify(B=k).
    {
        let k = *bs_list.iter().filter(|&&b| b >= 2).max().unwrap_or(&2);
        eprintln!("\n########## CORRECTNESS (prime P={correct_p}, verify k={k}) ##########");
        let n_kv_max = correct_p + k as u32 + 8;

        // ---- sequential reference ----
        let mut seq_state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        for pos in 0..correct_p {
            let idx = pos as usize;
            engine.forward_verify(
                &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
                &mut seq_state, &weights, &input_hcs[idx..idx + 1], &tokens[idx..idx + 1],
                pos, false,
            )?;
        }
        let mut seq_argmax = Vec::with_capacity(k);
        let mut seq_hc: Vec<Vec<f32>> = Vec::with_capacity(k);
        for j in 0..k {
            let pos = correct_p + j as u32;
            let idx = correct_p as usize + j;
            let out = engine.forward_verify(
                &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
                &mut seq_state, &weights, &input_hcs[idx..idx + 1], &tokens[idx..idx + 1],
                pos, false,
            )?;
            seq_argmax.push(argmax(&out.logits));
            seq_hc.push(out.hc.clone());
        }

        // ---- batched ----
        // VERIFY_PRIME=decode (default): prime via the decode path (B=1).
        //          =prefill: prime via forward_prefill (prefill KV discipline).
        // Isolates whether continuation breaks due to a decode/prefill state
        // discipline mismatch vs a genuine batched pos0>0 bug.
        let prime_mode = std::env::var("VERIFY_PRIME").unwrap_or_else(|_| "decode".into());
        // Validate EACH batch size B∈{2,3,4} vs the sequential decode reference.
        let mut test_ks: Vec<usize> = bs_list.iter().copied().filter(|&b| b >= 2).collect();
        test_ks.sort_unstable();
        test_ks.dedup();
        let mut all_pass = true;
      for &k in &test_ks {
        let mut bat_state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        // Decode graphs bake the previous state's kv_cache pointers; clear
        // so bat_state's decode priming captures against ITS OWN buffers.
        engine.clear_graphs();
        if prime_mode == "prefill" && correct_p > 0 {
            let inp: Vec<Vec<f32>> = (0..correct_p as usize).map(|i| input_hcs[i].clone()).collect();
            let toks: Vec<i32> = tokens[0..correct_p as usize].to_vec();
            engine.forward_prefill(
                &mut bd, &mut bi, &mut head_scratch, &mut bat_state, &weights,
                &inp, &toks, 0, true, None,
            )?;
        } else {
            for pos in 0..correct_p {
                let idx = pos as usize;
                engine.forward_verify(
                    &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
                    &mut bat_state, &weights, &input_hcs[idx..idx + 1], &tokens[idx..idx + 1],
                    pos, false,
                )?;
            }
        }
        if k == *test_ks.last().unwrap() && std::env::var("VERIFY_DIAG").is_ok() {
            // Prime a SECOND state the OTHER way and compare per-layer counters
            // + KV/comp checksums, to locate the decode-vs-prefill state diff.
            let mut other = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
            {
                let inp: Vec<Vec<f32>> = (0..correct_p as usize).map(|i| input_hcs[i].clone()).collect();
                let toks: Vec<i32> = tokens[0..correct_p as usize].to_vec();
                engine.forward_prefill(
                    &mut bd, &mut bi, &mut head_scratch, &mut other, &weights,
                    &inp, &toks, 0, true, None,
                )?;
            }
            let sum_u16 = |buf: &v4flash_hip::DeviceBuffer<u16>, n: usize| -> f64 {
                let mut h = vec![0u16; n];
                buf.slice_view(0, n).copy_to_host(&mut h).unwrap();
                h.iter().map(|&x| x as f64).sum()
            };
            for l in [0usize, 1, 2, 4, 8] {
                let d = &bat_state.layers[l];
                let p = &other.layers[l];
                let hd = HC_DIM as usize; // placeholder
                let _ = hd;
                let n = (d.n_raw as usize) * (v4flash_kernels::config::N_HEAD_DIM as usize);
                let dk = sum_u16(&d.kv_cache, n);
                let pk = sum_u16(&p.kv_cache, n);
                eprintln!("  L{l}: decode(n_raw={},raw_off={}) prefill(n_raw={},raw_off={}) kvsum d={:.3} p={:.3} d-p={:.3e}",
                    d.n_raw, d.raw_off, p.n_raw, p.raw_off, dk, pk, dk-pk);
                if let (Some(dc), Some(pc)) = (d.compressor.as_ref(), p.compressor.as_ref()) {
                    let cn = (dc.n_comp as usize) * (v4flash_kernels::config::N_HEAD_DIM as usize);
                    let dcs = sum_u16(&dc.comp_kv, cn);
                    let pcs = sum_u16(&pc.comp_kv, cn);
                    eprintln!("       comp d.n_comp={} p.n_comp={} compsum d={:.3} p={:.3} d-p={:.3e}",
                        dc.n_comp, pc.n_comp, dcs, pcs, dcs-pcs);
                }
            }
        }
        let idx0 = correct_p as usize;
        let out = engine.forward_verify(
            &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
            &mut bat_state, &weights, &input_hcs[idx0..idx0 + k], &tokens[idx0..idx0 + k],
            correct_p, false,
        )?;

        let vocab = N_VOCAB as usize;
        let hc_dim = HC_DIM as usize;
        let mut match_argmax = 0usize;
        let mut worst_hc = 0.0f32;
        for j in 0..k {
            let bat_logits = &out.logits[j * vocab..(j + 1) * vocab];
            let bat_hc = &out.hc[j * hc_dim..(j + 1) * hc_dim];
            let bam = argmax(bat_logits);
            let ok = bam == seq_argmax[j];
            if ok {
                match_argmax += 1;
            }
            let dh = max_abs(&seq_hc[j], bat_hc);
            worst_hc = worst_hc.max(dh);
            eprintln!(
                "  pos {:>3}: seq_argmax={:>6} bat_argmax={:>6} {}  hc_max_abs={:.3e}",
                correct_p + j as u32,
                seq_argmax[j],
                bam,
                if ok { "MATCH" } else { "MISMATCH" },
                dh,
            );
        }
        eprintln!(
            "  B={k}: argmax match {match_argmax}/{k}   worst HC max_abs = {worst_hc:.3e}"
        );
        if match_argmax != k {
            eprintln!("  !!! CORRECTNESS GATE FAILED (B={k}): {match_argmax}/{k} positions match");
            all_pass = false;
        }
      }
        if all_pass {
            eprintln!("  >>> CORRECTNESS GATE PASS (all B in {test_ks:?})");
        } else {
            eprintln!("  !!! CORRECTNESS GATE FAILED");
        }
    }

    // ================= TIMING (anchor + C(B)) =================
    let graph_warm = 12u32;

    // Anchor: graph-captured B=1 forward_verify at faked depth.
    let anchor_b1 = |engine: &mut HeterogeneousEngine,
                     dgpu_scratch: &mut DgpuScratch,
                     igpu_scratch: &mut IgpuScratch,
                     bd: &mut BatchDgpuScratch,
                     bi: &mut BatchIgpuScratch,
                     head_scratch: &mut DgpuScratch,
                     fake_pos: u32|
     -> eyre::Result<f64> {
        let n_kv_max = fake_pos + graph_warm + n_iters as u32 + 64;
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        for pos in 0..graph_warm {
            let idx = pos as usize % n_real;
            engine.forward_verify(
                bd, bi, dgpu_scratch, igpu_scratch, head_scratch,
                &mut state, &weights, &input_hcs[idx..idx + 1], &tokens[idx..idx + 1],
                pos, true,
            )?;
        }
        stamp_fake_depth(&mut state, fake_pos);
        let mut ms: Vec<f64> = Vec::with_capacity(n_iters);
        for it in 0..(n_warmup + n_iters) {
            let pos = fake_pos + it as u32;
            let t = Instant::now();
            engine.forward_verify(
                bd, bi, dgpu_scratch, igpu_scratch, head_scratch,
                &mut state, &weights, &input_hcs[0..1], &tokens[0..1], pos, true,
            )?;
            let dt = t.elapsed().as_secs_f64() * 1000.0;
            if it >= n_warmup {
                ms.push(dt);
            }
        }
        Ok(median(&mut ms))
    };

    let verify_step = |engine: &mut HeterogeneousEngine,
                       bd: &mut BatchDgpuScratch,
                       bi: &mut BatchIgpuScratch,
                       dgpu_scratch: &mut DgpuScratch,
                       igpu_scratch: &mut IgpuScratch,
                       head_scratch: &mut DgpuScratch,
                       b: usize,
                       fake_pos: u32,
                       trace: Option<&str>|
     -> eyre::Result<f64> {
        let inp = &input_hcs[0..b];
        let toks = &tokens[0..b];
        let n_kv_max = fake_pos + b as u32 + 8;
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        for _ in 0..n_warmup {
            stamp_fake_depth(&mut state, fake_pos);
            engine.forward_verify(
                bd, bi, dgpu_scratch, igpu_scratch, head_scratch,
                &mut state, &weights, inp, toks, fake_pos, false,
            )?;
        }
        if let Some(p) = trace {
            engine.attach_perfetto(p)?;
        }
        let mut ms: Vec<f64> = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            stamp_fake_depth(&mut state, fake_pos);
            let t = Instant::now();
            engine.forward_verify(
                bd, bi, dgpu_scratch, igpu_scratch, head_scratch,
                &mut state, &weights, inp, toks, fake_pos, false,
            )?;
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        // Drain the last forward's device-time slices to the trace file
        // (the verify paths don't auto-emit like forward_token does).
        if trace.is_some() {
            engine.flush_perfetto()?;
        }
        Ok(median(&mut ms))
    };

    for &fake_pos in &depths {
        eprintln!("\n########## FAKE_POS = {fake_pos} ##########");
        let anchor = anchor_b1(
            &mut engine, &mut dgpu_scratch, &mut igpu_scratch, &mut bd, &mut bi,
            &mut head_scratch, fake_pos,
        )?;
        eprintln!(
            ">>> ANCHOR forward_verify B=1: {anchor:.2} ms  ({:.2} tok/s)",
            1000.0 / anchor
        );
        if fake_pos <= 8192 && (anchor < 25.0 || anchor > 55.0) {
            eprintln!("!!! WARNING: anchor {anchor:.1} ms outside the ~34 ms decode band.");
        }

        let mut walls: Vec<(usize, f64)> = Vec::with_capacity(bs_list.len());
        for &b in &bs_list {
            let w = verify_step(
                &mut engine, &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch,
                &mut head_scratch, b, fake_pos, None,
            )?;
            eprintln!("  verify B={b}: wall={w:.2} ms  ({:.2} ms/tok)", w / b as f64);
            walls.push((b, w));
        }

        let b1_wall = walls.iter().find(|(b, _)| *b == 1).map(|(_, w)| *w);
        eprintln!("\n=== C(B) @ depth {fake_pos} ===");
        eprintln!("   B | wall ms | C_wall | ms/tok | F(accept-breakeven)");
        for &(b, w) in &walls {
            let c = w / walls[0].1;
            let f = if b > walls[0].0 {
                (b as f64 - c) / (b as f64 - 1.0)
            } else {
                f64::NAN
            };
            eprintln!("  {b:>2} | {w:>7.2} | {c:>6.3} | {:>6.2} | {f:>6.3}", w / b as f64);
        }
        if let Some(w1) = b1_wall {
            eprintln!("  (C_wall(B) = wall(B)/wall(1); MTP wins iff top-1 acceptance > C_wall(k). anchor={anchor:.2} b1_wall={w1:.2})");
        }

        if let Some(base) = &perfetto_base {
            for &b in &bs_list {
                let path = format!("{base}_d{fake_pos}_b{b}.pftrace");
                eprintln!("  [perfetto] verify B={b} depth {fake_pos} -> {path}");
                let _ = verify_step(
                    &mut engine, &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch,
                    &mut head_scratch, b, fake_pos, Some(&path),
                )?;
                eprintln!("    analyze: python3 ~/scripts/analyze_pftrace_gaps.py {path}");
            }
        }
    }

    engine.shutdown()?;
    Ok(())
}
