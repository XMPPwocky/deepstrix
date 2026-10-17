//! Embed-phase GPU gates (docs/v41/EMBED_PHASE_DESIGN.md §10), dGPU only.
//!
//! `#[ignore]`: they drive the dGPU, so they need a GPU window (the hub
//! down -- it leaves the dGPU ~170 MiB free). Run with
//!
//! ```text
//! CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release -p v4flash-kernels \
//!     --features v41 --test qwen3_embed_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! - `tiny_gpu_matches_cpu`: a synthetic tiny Qwen3 (no weights needed), GPU
//!   forward vs the CPU oracle, sub-batches of 8 rows so inputs straddle
//!   sub-batches and sub-batches hold several inputs. Gate E2's machinery.
//! - `real_gpu_matches_cpu` (gate E2 proper): `QWEN3_EMBED_GGUF=<Q8_0 gguf>`;
//!   min cosine >= 0.9999 vs the CPU oracle on a few short inputs.
//! - `loan_round_trip`: a plain device buffer as the donor; lend it, let the
//!   forward clobber it, give it back, compare every byte.
//! - `qe_attn_g4` (design §19.11): `qe_attn_g4_producer_bit_identity` (G1b),
//!   `tiny_gpu_g4_matches_cpu` (G3), `real_gpu_g4_layer0_attention` (G3, layer 0
//!   vs the f64 oracle); `QWEN3_EMBED_ATTN=fa2|packed|g4|g4n8|g4wgp` picks the
//!   attention of `real_gpu_matches_cpu` (g4*: plus a 48-row-sub-batch run).

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::direct_io::DirectFiles;
use v4flash_core::qwen3_embed::{cosine, cpu_forward, testing, Qwen3EmbedConfig, Qwen3EmbedModel};
use v4flash_core::MappedGguf;
use v4flash_hip::{Device, DeviceBuffer, PinnedBuffer, Stream};
use v4flash_kernels::dgpu_loan::{Donor, Loan, LoanAlloc};
use v4flash_kernels::q8_0::Q8_0MatvecWmma;
use v4flash_kernels::qwen3_embed::{
    g4_unpack_kv, host_bytes, run, AttnPath, EmbedBuffers, EmbedSizing, G4Cell, KvLayout, Layer0Capture, Qwen3EmbedKernels,
};

fn dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 device"))
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("qwen3-embed-gpu-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `QWEN3_EMBED_ATTN=fa2|packed|g4|g4n8|g4wgp` (the `V41_EMBED_ATTN` names;
/// default fa2): the gates' forward attention kernel.
fn attn_from_env() -> AttnPath {
    match std::env::var("QWEN3_EMBED_ATTN") {
        Ok(s) => AttnPath::from_name(&s).unwrap_or_else(|| panic!("QWEN3_EMBED_ATTN={s}: not fa2 / packed / g4 / g4n8 / g4wgp")),
        Err(_) => AttnPath::Fa2,
    }
}

/// GPU embeddings of `inputs` (finished, full dims) with memory from one
/// plain allocation; attention per `QWEN3_EMBED_ATTN`.
fn gpu_embed(dev: Device, model: &Qwen3EmbedModel, file: &MappedGguf, inputs: &[Vec<u32>], sizing: EmbedSizing) -> eyre::Result<Vec<Vec<f32>>> {
    gpu_embed_with(dev, model, file, inputs, sizing, attn_from_env())
}

fn gpu_embed_with(
    dev: Device,
    model: &Qwen3EmbedModel,
    file: &MappedGguf,
    inputs: &[Vec<u32>],
    sizing: EmbedSizing,
    path: AttnPath,
) -> eyre::Result<Vec<Vec<f32>>> {
    Ok(gpu_embed_capture(dev, model, file, inputs, sizing, path, false)?.0)
}

/// [`gpu_embed_with`], optionally with layer 0's attention capture (design §19.11 G3).
fn gpu_embed_capture(
    dev: Device,
    model: &Qwen3EmbedModel,
    file: &MappedGguf,
    inputs: &[Vec<u32>],
    sizing: EmbedSizing,
    path: AttnPath,
    capture: bool,
) -> eyre::Result<(Vec<Vec<f32>>, Option<Layer0Capture>)> {
    dev.set_current()?;
    let mut k = Qwen3EmbedKernels::for_arch("gfx1201")?;
    k.set_attention(path)?;
    eprintln!("attention: {}", k.attention_path(&model.cfg)?.name());
    // One plain allocation per planned buffer (the hub's come from the loan).
    let backing: Vec<DeviceBuffer<u8>> = sizing
        .buffer_sizes(&model.cfg, &model.layout)
        .iter()
        .map(|(_, b)| DeviceBuffer::<u8>::new(dev.id, *b))
        .collect::<eyre::Result<_>>()?;
    let mut alloc = LoanAlloc::over(backing.iter().map(|b| b.slice_view(0, b.len())).collect());
    let mut bufs = EmbedBuffers::carve(&mut alloc, sizing, &model.cfg, &model.layout)?;
    if capture {
        bufs.capture = Some(Layer0Capture::default());
    }
    let hb = host_bytes(model)?;
    let mut host = [PinnedBuffer::<u8>::new(hb)?, PinnedBuffer::<u8>::new(hb)?];
    let (compute, copy) = (Stream::new(dev.id)?, Stream::new(dev.id)?);
    let refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let g = Q8_0MatvecWmma::for_arch("gfx1201")?;
    let direct = DirectFiles::open(&[file.path().to_path_buf()])?;
    let (last, tm) = run(&k, &g, model, file, &direct, 8, &mut bufs, &mut host, &compute, &copy, &refs, &mut |_| Ok(()))?;
    eprintln!("gpu forward: {tm:?}");
    let cap = bufs.capture.take();
    drop(bufs);
    drop(backing);
    Ok((last.iter().map(|r| model.finish(r, None)).collect(), cap))
}

fn compare(gpu: &[Vec<f32>], cpu: &[Vec<f32>]) -> f64 {
    let mut min = f64::INFINITY;
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        let cs = cosine(g, c);
        eprintln!("input {i}: cos {cs:.7}");
        min = min.min(cs);
    }
    min
}

#[test]
#[ignore]
fn tiny_gpu_matches_cpu() -> eyre::Result<()> {
    let dev = dgpu()?;
    let p = tmp("tiny").join("tiny.gguf");
    testing::write_synthetic(&p, &testing::tiny_config(), 23)?;
    let file = MappedGguf::open(&p)?;
    let model = Qwen3EmbedModel::from_gguf(&file)?;
    let eos = model.eos_id;
    let inputs: Vec<Vec<u32>> = vec![
        (1..20).chain([eos]).collect(),
        vec![7, eos],
        vec![eos],
        (30..43).chain([eos]).collect(),
        (100..125).chain([eos]).collect(),
    ];
    let cpu = cpu_forward(&model, &file, &inputs)?;
    let gpu = gpu_embed(dev, &model, &file, &inputs, EmbedSizing { phase_tokens: 128, sub_rows: 8 })?;
    let min = compare(&gpu, &cpu);
    assert!(min >= 0.9999, "min cosine {min}");
    // Packing invariance on the GPU: alone == packed.
    let alone = gpu_embed(dev, &model, &file, &inputs[3..4], EmbedSizing { phase_tokens: 128, sub_rows: 8 })?;
    let cs = cosine(&alone[0], &gpu[3]);
    assert!(cs >= 0.99999, "alone vs packed {cs}");
    Ok(())
}

/// The tiny model at kv_group 4 (8 query heads over 2 KV heads, as Qwen3-4B's
/// 32 / 8), so the full-group packed attention kernel runs (design §18, G3):
/// packed vs the CPU oracle, and packed vs `fa2` on the GPU.
#[test]
#[ignore]
fn tiny_gpu_packed_matches_cpu() -> eyre::Result<()> {
    let dev = dgpu()?;
    let p = tmp("tiny-g4").join("tiny-g4.gguf");
    let mut cfg = testing::tiny_config();
    cfg.n_head = 8;
    testing::write_synthetic(&p, &cfg, 29)?;
    let file = MappedGguf::open(&p)?;
    let model = Qwen3EmbedModel::from_gguf(&file)?;
    assert_eq!(model.cfg.n_head / model.cfg.n_kv_head, 4);
    let eos = model.eos_id;
    let inputs: Vec<Vec<u32>> = vec![
        (1..20).chain([eos]).collect(),
        vec![7, eos],
        vec![eos],
        (30..43).chain([eos]).collect(),
        (100..125).chain([eos]).collect(),
    ];
    let sizing = EmbedSizing { phase_tokens: 128, sub_rows: 8 };
    let cpu = cpu_forward(&model, &file, &inputs)?;
    let packed = gpu_embed_with(dev, &model, &file, &inputs, sizing, AttnPath::Packed)?;
    let min = compare(&packed, &cpu);
    assert!(min >= 0.9999, "packed vs cpu: min cosine {min}");
    let fa2 = gpu_embed_with(dev, &model, &file, &inputs, sizing, AttnPath::Fa2)?;
    let min = compare(&packed, &fa2);
    assert!(min >= 0.99999, "packed vs fa2: min cosine {min}");
    Ok(())
}

/// Gate E2 (design §10): the GPU forward on the REAL Q8_0 GGUF vs the CPU
/// oracle, min cosine >= 0.9999, every output finite. Covers all four Qwen
/// GEMM shapes and kv_group 4 by construction; inputs: the model card's
/// texts, code, CJK, a ~4.5K-token input (so at the production sizing one
/// input spans several 1024-row sub-batches: q_offset >= 1024), and
/// adversarial ones (a ~2K-token repetition, a punctuation run). Two runs:
/// the production sizing (T 16384, R 1024), and 64-row sub-batches (inputs
/// straddle sub-batches, sub-batches hold several inputs). The CPU oracle
/// takes minutes on the long inputs (run it in the GPU window: all cores).
#[test]
#[ignore]
fn real_gpu_matches_cpu() -> eyre::Result<()> {
    let path = std::env::var("QWEN3_EMBED_GGUF").map_err(|_| eyre!("gate E2 needs QWEN3_EMBED_GGUF (the Q8_0 GGUF)"))?;
    let dev = dgpu()?;
    let file = MappedGguf::open(&path)?;
    let model = Qwen3EmbedModel::from_gguf(&file)?;
    let inputs = real_inputs(&file, &model)?;
    let lens: Vec<usize> = inputs.iter().map(Vec::len).collect();
    eprintln!("input tokens: {lens:?}");
    assert!(lens.iter().any(|&n| n > 4096), "E2 needs a 4K+ input");
    let t0 = std::time::Instant::now();
    let cpu = cpu_forward(&model, &file, &inputs)?;
    eprintln!("cpu oracle: {:.1} s", t0.elapsed().as_secs_f64());
    let mut runs = vec![
        ("production sizing", EmbedSizing { phase_tokens: 16384, sub_rows: 1024 }, (0..inputs.len()).collect::<Vec<_>>()),
        ("64-row sub-batches", EmbedSizing { phase_tokens: 2048, sub_rows: 64 }, (0..6).collect::<Vec<_>>()),
    ];
    if matches!(attn_from_env(), AttnPath::G4(_)) {
        // G4 (design §19.11 G3): sub-batches off the 32-key grid -> qe_v_frag's
        // straddled-tile read-modify-write on every sub-batch.
        runs.push(("48-row sub-batches", EmbedSizing { phase_tokens: 2048, sub_rows: 48 }, (0..6).collect::<Vec<_>>()));
    }
    for (name, sizing, picked) in runs {
        let ins: Vec<Vec<u32>> = picked.iter().map(|&i| inputs[i].clone()).collect();
        let want: Vec<Vec<f32>> = picked.iter().map(|&i| cpu[i].clone()).collect();
        let gpu = gpu_embed(dev, &model, &file, &ins, sizing)?;
        assert!(gpu.iter().all(|e| e.iter().all(|x| x.is_finite())), "{name}: non-finite output");
        let min = compare(&gpu, &want);
        eprintln!("{name}: min cosine {min:.7}");
        assert!(min >= 0.9999, "gate E2 ({name}): min cosine {min} < 0.9999");
    }
    Ok(())
}

/// The E2 inputs: the model card's texts, code, CJK, adversarial runs, and a ~4.5K-token input.
fn real_inputs(file: &MappedGguf, model: &Qwen3EmbedModel) -> eyre::Result<Vec<Vec<u32>>> {
    let vocab = v4flash_core::tokenizer::BpeVocab::from_gguf(file.gguf())?;
    let task = "Given a web search query, retrieve relevant passages that answer the query";
    let texts: Vec<String> = vec![
        format!("Instruct: {task}\nQuery:What is the capital of China?"),
        format!("Instruct: {task}\nQuery:Explain gravity"),
        "The capital of China is Beijing.".into(),
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.".into(),
        "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n".into(),
        "日本の首都は東京です。北京是中国的首都。서울은 한국의 수도입니다.".into(),
        "!!!???...,,,;;;:::---___===+++***&&&%%%$$$###@@@".repeat(8),
        "the ".repeat(2000),
        "Retrieval-augmented generation pairs a search index with a language model; the index returns passages and the model reads them. ".repeat(220),
    ];
    Ok(texts
        .iter()
        .map(|t| vocab.encode_qwen2(t).into_iter().map(|i| i as u32).chain([model.eos_id]).collect())
        .collect())
}

/// f64 causal attention per input segment of a packed phase: query row r (of the
/// segment starting at s0) over keys s0..=r. q `[t][n_head][128]`, k / v
/// `[t][n_kv][128]` (f32 from f16 bits). Rows split over threads.
fn oracle_attention(q: &[f32], k: &[f32], v: &[f32], starts: &[usize], t: usize, n_head: usize, n_kv: usize) -> Vec<f64> {
    let (hd, g) = (128usize, n_head / n_kv);
    let scale = 1.0 / (hd as f64).sqrt();
    let row = n_head * hd;
    let mut out = vec![0f64; t * row];
    let seg_of = |r: usize| starts[starts.partition_point(|&s| s <= r) - 1];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(32);
    let per = t.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * row).enumerate() {
            s.spawn(move || {
                for (li, orow) in chunk.chunks_mut(row).enumerate() {
                    let r = ci * per + li;
                    let s0 = seg_of(r);
                    for h in 0..n_head {
                        let kh = h / g;
                        let qh = &q[(r * n_head + h) * hd..][..hd];
                        let sc: Vec<f64> = (s0..=r)
                            .map(|j| (0..hd).map(|d| f64::from(qh[d]) * f64::from(k[(j * n_kv + kh) * hd + d])).sum::<f64>() * scale)
                            .collect();
                        let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let w: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                        let den: f64 = w.iter().sum();
                        let oh = &mut orow[h * hd..][..hd];
                        for (j, wj) in (s0..=r).zip(&w) {
                            for d in 0..hd {
                                oh[d] += wj / den * f64::from(v[(j * n_kv + kh) * hd + d]);
                            }
                        }
                    }
                }
            });
        }
    });
    out
}

/// G3, the first-affected-layer rule (design §19.11): layer 0's attention OUTPUT
/// (f16, as the O-projection reads it) on the real weights and the E2 inputs,
/// against the f64 oracle computed from that run's own q16 / K / V, for fa2 (the
/// null) and every G4 cell. Layer 0's inputs do not depend on the attention kernel,
/// so q16 and the (un-permuted) K / V must also be BIT-identical across paths: the
/// producers on real activations. Bar per element: |got - oracle| <= 2e-3 + 2^-10
/// |oracle| (f16 output rounding); every output finite.
///   QWEN3_EMBED_GGUF=<Q8_0 gguf> ... --test qwen3_embed_gpu -- --ignored --nocapture real_gpu_g4_layer0_attention
#[test]
#[ignore]
fn real_gpu_g4_layer0_attention() -> eyre::Result<()> {
    let path = std::env::var("QWEN3_EMBED_GGUF").map_err(|_| eyre!("needs QWEN3_EMBED_GGUF (the Q8_0 GGUF)"))?;
    let dev = dgpu()?;
    let file = MappedGguf::open(&path)?;
    let model = Qwen3EmbedModel::from_gguf(&file)?;
    let inputs = real_inputs(&file, &model)?;
    let (nh, nkv) = (model.cfg.n_head, model.cfg.n_kv_head);
    let mut starts = Vec::new();
    let mut t = 0;
    for inp in &inputs {
        starts.push(t);
        t += inp.len();
    }
    let sizing = EmbedSizing { phase_tokens: 16384, sub_rows: 1024 };
    let n_tiles = sizing.phase_tokens.div_ceil(32);
    let f = |b: &[u16]| -> Vec<f32> { b.iter().map(|&x| v4flash_core::kquants::f16_to_f32(x)).collect() };
    let mut first: Option<(Vec<u16>, Vec<u16>, Vec<u16>)> = None;
    for path in [AttnPath::Fa2, AttnPath::G4(G4Cell::Cu16), AttnPath::G4(G4Cell::Cu8), AttnPath::G4(G4Cell::Wgp16)] {
        let (_, cap) = gpu_embed_capture(dev, &model, &file, &inputs, sizing, path, true)?;
        let cap = cap.ok_or_else(|| eyre!("no layer-0 capture"))?;
        assert_eq!(cap.path, Some(path));
        let (k, v) = match path {
            AttnPath::G4(_) => g4_unpack_kv(&cap.kc, &cap.vc, t, nkv, n_tiles),
            _ => (cap.kc[..t * nkv * 128].to_vec(), cap.vc[..t * nkv * 128].to_vec()),
        };
        match &first {
            None => first = Some((cap.q16.clone(), k.clone(), v.clone())),
            Some((q0, k0, v0)) => {
                assert!(&cap.q16 == q0, "{}: layer-0 q16 differs from fa2's", path.name());
                assert!(&k == k0 && &v == v0, "{}: layer-0 K / V (un-permuted) differ from fa2's row-major", path.name());
            }
        }
        let want = oracle_attention(&f(&cap.q16), &f(&k), &f(&v), &starts, t, nh, nkv);
        let got = f(&cap.attn);
        assert!(got.iter().all(|x| x.is_finite()), "{}: non-finite layer-0 attention", path.name());
        let (mut worst, mut max_abs) = (0f64, 0f64);
        for (g, w) in got.iter().zip(&want) {
            let e = (f64::from(*g) - w).abs();
            max_abs = max_abs.max(e);
            worst = worst.max(e / (2e-3 + w.abs() / 1024.0));
        }
        eprintln!("layer-0 attention {}: max abs {max_abs:.3e} vs f64, worst / bar {worst:.3}", path.name());
        assert!(worst <= 1.0, "{}: layer-0 attention misses the bar ({worst:.3} x)", path.name());
    }
    Ok(())
}

/// G1b (design §19.11): from the same fused qkv GEMM output, `qk_norm_rope(G4Tiles)`
/// + `qe_v_frag`, un-permuted on the host, equal `qk_norm_rope(RowMajor)`'s kc / vc
/// bit for bit, with q16 identical, on a 1000-row phase (a partial last tile) at
/// sub-batches of 1024 (one), 64, 48 and 8 rows (48 and 8 start off the 32-key grid:
/// qe_v_frag's straddled-tile read-modify-write). The V pad rows past the phase are 0.
#[test]
#[ignore]
fn qe_attn_g4_producer_bit_identity() -> eyre::Result<()> {
    let dev = dgpu()?;
    dev.set_current()?;
    let k = Qwen3EmbedKernels::for_arch("gfx1201")?;
    let stream = Stream::new(dev.id)?;
    let cfg = Qwen3EmbedConfig {
        n_layer: 36, n_embd: 2560, n_ff: 9728, n_head: 32, n_kv_head: 8, head_dim: 128,
        n_vocab: 151_665, rope_theta: 1e6, eps: 1e-6, n_ctx_train: 40960,
    };
    let t = 1000usize;
    let n_tiles = t.div_ceil(32);
    let (qw, kvw, w) = (cfg.q_width(), cfg.kv_width(), cfg.qkv_rows());
    let hash = |i: u64| {
        let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((x ^ (x >> 31)) >> 40) as f32 / (1u64 << 24) as f32 * 6.0 - 3.0
    };
    let qkv: Vec<f32> = (0..t * w).map(|i| hash(i as u64)).collect();
    let qn: Vec<f32> = (0..128).map(|i| 1.0 + hash(1 << 40 | i) / 10.0).collect();
    let kn: Vec<f32> = (0..128).map(|i| 1.0 + hash(2 << 40 | i) / 10.0).collect();
    let pos: Vec<u32> = (0..t as u32).collect();
    let upf = |b: &[f32]| -> eyre::Result<DeviceBuffer<f32>> {
        let mut d = DeviceBuffer::<f32>::new(dev.id, b.len())?;
        d.copy_from_host(b)?;
        Ok(d)
    };
    let (d_qkv, d_qn, d_kn) = (upf(&qkv)?, upf(&qn)?, upf(&kn)?);
    let mut d_pos = DeviceBuffer::<u32>::new(dev.id, t)?;
    d_pos.copy_from_host(&pos)?;
    let fill = |n: usize, x: u16| -> eyre::Result<DeviceBuffer<u16>> {
        let mut d = DeviceBuffer::<u16>::new(dev.id, n)?;
        d.copy_from_host(&vec![x; n])?;
        Ok(d)
    };
    // Row-major reference, one call.
    let (mut q_rm, mut kc_rm, mut vc_rm) = (fill(t * qw, 0)?, fill(t * kvw, 0)?, fill(t * kvw, 0)?);
    k.qk_norm_rope(&stream, &cfg, &d_qkv, &mut q_rm, &mut kc_rm, &mut vc_rm, &d_qn, &d_kn, &d_pos, KvLayout::RowMajor, 0, t as u32)?;
    stream.synchronize()?;
    let dl = |d: &DeviceBuffer<u16>| -> eyre::Result<Vec<u16>> {
        let mut h = vec![0u16; d.len()];
        d.copy_to_host(&mut h)?;
        Ok(h)
    };
    let (q_ref, k_ref, v_ref) = (dl(&q_rm)?, dl(&kc_rm)?, dl(&vc_rm)?);
    let tile_len = cfg.n_kv_head * n_tiles * 4096;
    for sub in [1024usize, 64, 48, 8] {
        let (mut q16, mut kc, mut vc) = (fill(t * qw, 0)?, fill(tile_len, 0xFFFF)?, fill(tile_len, 0xFFFF)?);
        let mut r0 = 0;
        while r0 < t {
            let rows = sub.min(t - r0);
            let x = d_qkv.slice_view(r0 * w, rows * w);
            let p = d_pos.slice_view(r0, rows);
            let mut qv = q16.slice_view_mut(r0 * qw, rows * qw);
            k.qk_norm_rope(&stream, &cfg, &x, &mut qv, &mut kc, &mut vc, &d_qn, &d_kn, &p, KvLayout::G4Tiles { n_tiles: n_tiles as u32 }, r0 as u32, rows as u32)?;
            k.v_frag(&stream, &cfg, &x, &mut vc, n_tiles as u32, r0 as u32, rows as u32)?;
            r0 += rows;
        }
        stream.synchronize()?;
        let (qh, kch, vch) = (dl(&q16)?, dl(&kc)?, dl(&vc)?);
        assert!(qh == q_ref, "sub {sub}: q16 differs between the layouts");
        let (ku, vu) = g4_unpack_kv(&kch, &vch, t, cfg.n_kv_head, n_tiles);
        assert!(ku == k_ref, "sub {sub}: G4 K (un-permuted) != row-major K");
        assert!(vu == v_ref, "sub {sub}: G4 V (un-permuted) != row-major V");
        // Phase-end pad (rows t .. n_tiles * 32) of V: 0.
        let (_, vpad) = g4_unpack_kv(&kch, &vch, n_tiles * 32, cfg.n_kv_head, n_tiles);
        assert!(vpad[t * kvw..].iter().all(|&x| x == 0), "sub {sub}: V pad rows not 0");
        eprintln!("producer bit identity: sub-batches of {sub}: q16 / K / V identical");
    }
    Ok(())
}

/// G3 (design §19.11): the tiny model at kv_group 4 (8 query heads over 2 KV heads)
/// with 8-row sub-batches (inputs straddle sub-batches; every sub-batch starts off
/// the 32-key grid, so qe_v_frag's read-modify-write runs): each G4 cell vs the CPU
/// oracle (>= 0.9999) and vs fa2 on the GPU (>= 0.99999).
#[test]
#[ignore]
fn tiny_gpu_g4_matches_cpu() -> eyre::Result<()> {
    let dev = dgpu()?;
    let p = tmp("tiny-g4attn").join("tiny-g4.gguf");
    let mut cfg = testing::tiny_config();
    cfg.n_head = 8;
    testing::write_synthetic(&p, &cfg, 31)?;
    let file = MappedGguf::open(&p)?;
    let model = Qwen3EmbedModel::from_gguf(&file)?;
    assert_eq!(model.cfg.n_head / model.cfg.n_kv_head, 4);
    let eos = model.eos_id;
    let inputs: Vec<Vec<u32>> = vec![
        (1..20).chain([eos]).collect(),
        vec![7, eos],
        vec![eos],
        (30..43).chain([eos]).collect(),
        (100..125).chain([eos]).collect(),
        (3..80).chain([eos]).collect(),
    ];
    let sizing = EmbedSizing { phase_tokens: 150, sub_rows: 8 };
    let cpu = cpu_forward(&model, &file, &inputs)?;
    let fa2 = gpu_embed_with(dev, &model, &file, &inputs, sizing, AttnPath::Fa2)?;
    for cell in [G4Cell::Cu16, G4Cell::Cu8, G4Cell::Wgp16] {
        let g4 = gpu_embed_with(dev, &model, &file, &inputs, sizing, AttnPath::G4(cell))?;
        assert!(g4.iter().all(|e| e.iter().all(|x| x.is_finite())), "{cell:?}: non-finite output");
        let min = compare(&g4, &cpu);
        assert!(min >= 0.9999, "{cell:?} vs cpu: min cosine {min}");
        let min = compare(&g4, &fa2);
        assert!(min >= 0.99999, "{cell:?} vs fa2: min cosine {min}");
    }
    Ok(())
}

#[test]
#[ignore]
fn loan_round_trip() -> eyre::Result<()> {
    let dev = dgpu()?;
    dev.set_current()?;
    let p = tmp("loan").join("tiny.gguf");
    testing::write_synthetic(&p, &testing::tiny_config(), 29)?;
    let file = MappedGguf::open(&p)?;
    let model = Qwen3EmbedModel::from_gguf(&file)?;
    let sizing = EmbedSizing { phase_tokens: 128, sub_rows: 16 };
    let sizes: Vec<usize> = sizing.buffer_sizes(&model.cfg, &model.layout).iter().map(|(_, b)| *b).collect();
    // Two donors with a recognizable pattern. The first (512 KiB) is too small
    // for the layer buffers but holds others (resid, pos, idx), so both are lent.
    let mk = |n: usize, salt: u8| -> eyre::Result<(DeviceBuffer<u8>, Vec<u8>)> {
        let mut b = DeviceBuffer::<u8>::new(dev.id, n)?;
        let pat: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt)).collect();
        b.copy_from_host(&pat)?;
        Ok((b, pat))
    };
    let (d0, p0) = mk(512 * 1024, 3)?;
    let (d1, p1) = mk(sizing.total_bytes(&model.cfg, &model.layout) + (1 << 20), 5)?;
    let donors = vec![
        Donor { name: "d0".into(), view: d0.slice_view(0, d0.len()) },
        Donor { name: "d1".into(), view: d1.slice_view(0, d1.len()) },
    ];
    let image = tmp("loan").join("embed-loan.img");
    // Two replicas of the image (the hub puts one on each drive).
    let image2 = tmp("loan").join("embed-loan-2.img");
    let mut loan = Loan::new(donors, &sizes, vec![image, image2], 1 << 20, 64 * 1024)?;
    eprintln!("donors {:?}", loan.donor_summary());
    let stream = Stream::new(dev.id)?;
    loan.write_image(&stream)?;
    let mut alloc = loan.allocator()?;
    let mut bufs = EmbedBuffers::carve(&mut alloc, sizing, &model.cfg, &model.layout)?;
    let k = Qwen3EmbedKernels::for_arch("gfx1201")?;
    let hb = host_bytes(&model)?.max(1 << 20);
    let mut host = [PinnedBuffer::<u8>::new(hb)?, PinnedBuffer::<u8>::new(hb)?];
    let (compute, copy) = (Stream::new(dev.id)?, Stream::new(dev.id)?);
    let inputs: Vec<Vec<u32>> = vec![(1..40).chain([model.eos_id]).collect()];
    let refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let g = Q8_0MatvecWmma::for_arch("gfx1201")?;
    let direct = DirectFiles::open(&[file.path().to_path_buf()])?;
    run(&k, &g, &model, &file, &direct, 8, &mut bufs, &mut host, &compute, &copy, &refs, &mut |_| Ok(()))?;
    // The forward clobbered the donors ...
    let mut now1 = vec![0u8; d1.len()];
    d1.copy_to_host(&mut now1)?;
    assert_ne!(now1, p1, "the forward never touched the donor");
    // ... and the return puts back every byte.
    let st = loan.give_back(&stream, &mut host, true, 4)?;
    eprintln!("return: {st:?}");
    let mut back0 = vec![0u8; d0.len()];
    let mut back1 = vec![0u8; d1.len()];
    d0.copy_to_host(&mut back0)?;
    d1.copy_to_host(&mut back1)?;
    assert_eq!(back0, p0);
    assert_eq!(back1, p1);
    assert_eq!(st.retried, 0);
    Ok(())
}

#[test]
#[ignore]
fn repack_matches_host() -> eyre::Result<()> {
    let dev = dgpu()?;
    dev.set_current()?;
    let k = Qwen3EmbedKernels::for_arch("gfx1201")?;
    let stream = Stream::new(dev.id)?;
    // Qwen3-Embedding-4B's row widths: K 2560 / 4096 / 9728.
    for (rows, blocks) in [(7usize, 80usize), (3, 128), (5, 304)] {
        let n = rows * blocks * 34;
        let src: Vec<u8> = (0..n).map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_left(7) as u8).collect();
        let want = v4flash_kernels::weights::repack_q8_0(&src, rows, blocks);
        let mut d = DeviceBuffer::<u8>::new(dev.id, n)?;
        d.copy_from_host(&src)?;
        k.repack_q8_0_rows(&stream, &mut d, rows, blocks)?;
        stream.synchronize()?;
        let mut got = vec![0u8; n];
        d.copy_to_host(&mut got)?;
        assert_eq!(got, want, "rows {rows} blocks {blocks}");
    }
    Ok(())
}
