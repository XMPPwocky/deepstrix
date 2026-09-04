//! GPU oracle + roofline benchmarks for the ViT tower. ALL IGNORED by
//! default — they need a device and ~2 GiB of host RAM. Run ONE at a time
//! (the production server owns the dGPU; device 1 = gfx1151 iGPU only):
//!
//! ```text
//! export DEEPSTRIX_MMPROJ=/persist/lumi/models/dsv4f-exp-q2-k-xl/mmproj-F16.gguf
//! export DEEPSTRIX_VISION_DEVICE=1
//! cargo test --release -p v4flash-vision --test tower_encode -- --ignored --test-threads=1 --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_vision::kernels::VitKernels;
use v4flash_vision::mmproj::MmprojHost;
use v4flash_vision::preprocess::PreprocessedImage;
use v4flash_vision::{reference, Tower, PATCH_ELEMS, TEXT_DIM};

fn device() -> Device {
    let id: i32 = std::env::var("DEEPSTRIX_VISION_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    assert_ne!(id, 0, "refusing to touch the dGPU (device 0) while the server is live");
    Device::new(id)
}

fn mmproj() -> PathBuf {
    PathBuf::from(std::env::var("DEEPSTRIX_MMPROJ").expect("DEEPSTRIX_MMPROJ"))
}

/// Deterministic pseudo-random patches in roughly the normalised range.
fn synth_image(n_h: u32, n_w: u32) -> PreprocessedImage {
    let n = (n_h * n_w) as usize;
    let mut st = 0x12345678u32;
    let mut patches = vec![0f32; n * PATCH_ELEMS];
    for v in patches.iter_mut() {
        st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = ((st >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
    }
    PreprocessedImage { patches, n_vit_h: n_h, n_vit_w: n_w, content_hash: [0u8; 32] }
}

/// (max_abs, max/rms(ref), rms(err)/rms(ref), 1 - cosine similarity).
fn err(got: &[f32], want: &[f32]) -> (f32, f32, f32, f32) {
    assert_eq!(got.len(), want.len());
    let n = want.len() as f64;
    let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n).sqrt();
    let mx = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let erms = (got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / n).sqrt();
    let (mut dot, mut ga, mut wa) = (0f64, 0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        dot += *a as f64 * *b as f64;
        ga += (*a as f64).powi(2);
        wa += (*b as f64).powi(2);
    }
    let cos = dot / (ga.sqrt() * wa.sqrt()).max(1e-300);
    (mx, (mx as f64 / rms.max(1e-20)) as f32, (erms / rms.max(1e-20)) as f32, (1.0 - cos) as f32)
}

/// Build a smooth synthetic RGB image with natural-ish statistics and run
/// it through the real preprocessing path.
fn smooth_image(px_h: u32, px_w: u32) -> PreprocessedImage {
    let mut data = vec![0u8; (px_h * px_w * 3) as usize];
    for y in 0..px_h {
        for x in 0..px_w {
            let fy = y as f32 / px_h as f32;
            let fx = x as f32 / px_w as f32;
            let r = 0.5 + 0.35 * (6.0 * fx).sin() * (4.0 * fy).cos();
            let g = 0.5 + 0.30 * ((fx - 0.4).powi(2) + (fy - 0.6).powi(2)).sqrt();
            let b = 0.5 + 0.25 * (9.0 * (fx + fy)).sin();
            let o = ((y * px_w + x) * 3) as usize;
            data[o] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            data[o + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            data[o + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    let rgb = v4flash_vision::resize::Rgb::new(px_w, px_h, data);
    v4flash_vision::preprocess::preprocess_rgb(&rgb).unwrap().0
}

// ------------------------------------------------------------------ probes

/// Confirm the RDNA3 wave32 WMMA C/D fragment map baked into `vit.hip`
/// (`WMMA_C_ROW(i,l) = 2i + l/16`, column `l%16`).
#[test]
#[ignore]
fn wmma_layout_probe() {
    let d = device();
    d.set_current().unwrap();
    let arch = d.properties().unwrap().gcn_arch_name;
    let kk = VitKernels::for_arch(&arch).unwrap();
    let st = Stream::new(d.id).unwrap();
    let mut out = DeviceBuffer::<f32>::new(d.id, 256).unwrap();
    kk.wmma_probe(&st, &mut out).unwrap();
    st.synchronize().unwrap();
    let mut h = vec![0f32; 256];
    out.copy_to_host(&mut h).unwrap();
    eprintln!("arch = {arch}, gemm path = {:?}", kk.gemm_path);
    for lane in 0..32usize {
        for i in 0..8usize {
            let row = 2 * i + (lane >> 4);
            let col = lane & 15;
            let want = (16 * row + col) as f32;
            assert_eq!(h[lane * 8 + i], want, "lane {lane} elem {i}: D[{row}][{col}]");
        }
    }
    eprintln!("WMMA C/D map confirmed: lane l, elem i -> D[2i + l/16][l%16]");
}

/// f16 WMMA GEMM peak on this device, at the tower's real K values.
/// Establishes the compute ceiling used by the roofline table.
#[test]
#[ignore]
fn gemm_peak_bench() {
    let d = device();
    d.set_current().unwrap();
    let arch = d.properties().unwrap().gcn_arch_name;
    let kk = VitKernels::for_arch(&arch).unwrap();
    let st = Stream::new(d.id).unwrap();
    eprintln!("device {} ({}), gemm path {:?}", d.id, d.properties().unwrap().name, kk.gemm_path);
    eprintln!("{:>8} {:>6} {:>6} {:>10} {:>10} {:>10}", "n_tok", "K", "M", "ms", "TFLOP/s", "tileGB/s");
    for &(n, k, m) in &[
        (3108u32, 1024u32, 3072u32), // qkv at the biggest real n
        (3108, 1024, 5632),          // gate|up
        (3108, 2816, 1024),          // down
        (3108, 1024, 1024),          // attn_out
        (350, 9216, 4096),           // mm.1
        (4096, 4096, 4096),          // square, operands 67 MB: exceeds MALL -> DRAM-bound
        (1024, 4096, 1024),          // operands 10 MB: MALL-resident
        (512, 8192, 512),            // operands 16 MB: MALL-resident, deep K
        (256, 16384, 256),           // operands 16 MB: max reuse per tile
    ] {
        let x = DeviceBuffer::<u16>::new(d.id, (n as usize) * (k as usize)).unwrap();
        let w = DeviceBuffer::<u16>::new(d.id, (m as usize) * (k as usize)).unwrap();
        let mut o = DeviceBuffer::<f32>::new(d.id, (n as usize) * (m as usize)).unwrap();
        for _ in 0..3 {
            kk.gemm(&st, Some(&mut o), None, &x, &w, None, n, k, m, 0).unwrap();
        }
        st.synchronize().unwrap();
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            kk.gemm(&st, Some(&mut o), None, &x, &w, None, n, k, m, 0).unwrap();
        }
        st.synchronize().unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tf = 2.0 * n as f64 * k as f64 * m as f64 / (ms * 1e-3) / 1e12;
        // Bytes the 64x64 tiling must pull per call (no cache): each of the
        // ceil(M/64) tile columns re-reads x, each of the ceil(n/64) tile
        // rows re-reads w.
        let gb = (m.div_ceil(64) as f64 * n as f64 * k as f64 * 2.0 + n.div_ceil(64) as f64 * m as f64 * k as f64 * 2.0) / 1e9;
        eprintln!("{n:>8} {k:>6} {m:>6} {ms:>10.3} {tf:>10.2} {:>10.1}", gb / (ms * 1e-3));
    }
}

/// Attention kernel throughput at the tower's real n. FLOPs = 2 matmuls of
/// n x n x head_dim per head.
#[test]
#[ignore]
fn attention_bench() {
    let d = device();
    d.set_current().unwrap();
    let arch = d.properties().unwrap().gcn_arch_name;
    let kk = VitKernels::for_arch(&arch).unwrap();
    let st = Stream::new(d.id).unwrap();
    eprintln!("{:>8} {:>10} {:>10} {:>12}", "n", "ms", "TFLOP/s", "ms x32 lyr");
    for &n in &[1024u32, 1610, 3108] {
        let sz = (n as usize) * 1024;
        let q = DeviceBuffer::<u16>::new(d.id, sz).unwrap();
        let k = DeviceBuffer::<u16>::new(d.id, sz).unwrap();
        let v = DeviceBuffer::<u16>::new(d.id, sz).unwrap();
        let mut o = DeviceBuffer::<u16>::new(d.id, sz).unwrap();
        for _ in 0..2 {
            kk.attention(&st, &mut o, &q, &k, &v, n, 0.125).unwrap();
        }
        st.synchronize().unwrap();
        let iters = 5;
        let t = Instant::now();
        for _ in 0..iters {
            kk.attention(&st, &mut o, &q, &k, &v, n, 0.125).unwrap();
        }
        st.synchronize().unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let fl = 2.0 * 2.0 * (n as f64).powi(2) * 64.0 * 16.0;
        eprintln!("{n:>8} {ms:>10.3} {:>10.2} {:>12.1}", fl / (ms * 1e-3) / 1e12, ms * 32.0);
    }
    drop(st);
    d.synchronize().unwrap();
}

// ------------------------------------------------------------- CPU oracles

fn check_against_cpu(img: PreprocessedImage, rel_tol: f32) {
    let d = device();
    let (n_h, n_w) = (img.n_vit_h, img.n_vit_w);
    let host = MmprojHost::load(&mmproj()).unwrap();

    let (nh, nw) = (n_h as usize, n_w as usize);
    let t = Instant::now();
    // The GPU's exact CPU twin: f32 weights + f32 accumulation, activations
    // rounded to f16 at the same dtype boundaries. This is the correctness bar.
    let twin = reference::tower_forward_prec(&host, &img.patches, nh, nw, reference::ActPrec::F16);
    // Context: the f32 oracle and the bf16 path `vision.py` actually runs.
    let want = reference::tower_forward(&host, &img.patches, nh, nw);
    let canon = reference::tower_forward_prec(&host, &img.patches, nh, nw, reference::ActPrec::Bf16);
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut tower = Tower::from_host(host, d).unwrap();
    let got = tower.encode_rows(&img).unwrap();
    // second run: warm (weights already touched)
    let got2 = tower.encode_rows(&img).unwrap();
    assert_eq!(got, got2, "encode is not deterministic");

    let (lh, lw) = ((n_h as usize).div_ceil(3), (n_w as usize).div_ceil(3));
    assert_eq!(got.len(), lh * lw * TEXT_DIM);
    let (mx_t, _, erms_t, cos_t) = err(&got, &twin);
    let (mx, _, erms, cosd) = err(&got, &want);
    let (_, _, erms_b, cos_b) = err(&canon, &want);
    eprintln!(
        "grid {n_h}x{n_w} = {} patches -> {lh}x{lw} = {} rows | gpu {:.1} ms (3x cpu ref {:.0} ms)",
        n_h * n_w,
        lh * lw,
        tower.last_encode_ms,
        cpu_ms
    );
    eprintln!("  GPU vs CPU-f16 twin : max_abs {mx_t:.3e}  rms_err/rms {erms_t:.3e}  1-cos {cos_t:.3e}   <- correctness bar");
    eprintln!("  GPU vs CPU-f32      : max_abs {mx:.3e}  rms_err/rms {erms:.3e}  1-cos {cosd:.3e}");
    eprintln!("  vision.py bf16 vs f32:                 rms_err/rms {erms_b:.3e}  1-cos {cos_b:.3e}   <- canonical path is worse");
    if !tower.stage_ms.is_empty() {
        let tot: f64 = tower.stage_ms.iter().map(|(_, v)| v).sum();
        for (k, v) in &tower.stage_ms {
            eprintln!("    {k:<16} {v:8.2} ms  {:5.1}%", 100.0 * v / tot);
        }
        eprintln!("    {:<16} {tot:8.2} ms", "TOTAL");
    }
    assert!(erms_t < rel_tol, "GPU vs f16 twin: rms_err/rms {erms_t:.3e} >= tol {rel_tol:.1e}");
    assert!(erms < erms_b, "GPU f16 ({erms:.3e}) should beat the canonical bf16 path ({erms_b:.3e})");
    drop(tower);
    d.synchronize().unwrap();
}

/// 56x84 px = 4x6 patches, 2x2 LLM grid.
#[test]
#[ignore]
fn encode_matches_cpu_small() {
    check_against_cpu(synth_image(4, 6), 1e-2);
}

/// Same 4x6 grid but with real (smooth) image statistics.
#[test]
#[ignore]
fn encode_matches_cpu_small_real() {
    let img = smooth_image(56, 84);
    eprintln!("smooth 56x84 -> grid {}x{}", img.n_vit_h, img.n_vit_w);
    check_against_cpu(img, 1e-2);
}

/// Layer-bisect: rms(err)/rms after each ViT block, GPU vs CPU reference.
/// Distinguishes a kernel bug (a step) from f16 error growth (a curve).
#[test]
#[ignore]
fn trunk_error_by_layer() {
    let d = device();
    let img = smooth_image(448, 448);
    let host = MmprojHost::load(&mmproj()).unwrap();
    let (n_h, n_w) = (img.n_vit_h as usize, img.n_vit_w as usize);
    eprintln!("grid {n_h}x{n_w} = {} patches", n_h * n_w);
    let mut tower = Tower::from_host(host.clone(), d).unwrap();
    eprintln!(
        "{:>6} {:>10} | {:>11} {:>10} | {:>11} | {:>11}",
        "layers", "rms(ref)", "GPU f16", "1-cos", "CPU f16", "CPU bf16"
    );
    eprintln!("{:>6} {:>10} | {:>11} {:>10} | {:>11} | {:>11}", "", "", "rms_e/rms", "", "rms_e/rms", "rms_e/rms");
    for l in [0usize, 1, 2, 4, 8, 16, 24, 32] {
        let want = reference::vit_trunk_x(&host, &img.patches, n_h, n_w, l);
        let got = tower.trunk_x(&img, l).unwrap();
        let (_, _, erms, cosd) = err(&got, &want);
        let cpu16 = reference::vit_trunk_x_prec(&host, &img.patches, n_h, n_w, l, reference::ActPrec::F16);
        let cpubf = reference::vit_trunk_x_prec(&host, &img.patches, n_h, n_w, l, reference::ActPrec::Bf16);
        let (_, _, e16, _) = err(&cpu16, &want);
        let (_, _, ebf, _) = err(&cpubf, &want);
        let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
        eprintln!("{l:>6} {rms:>10.4} | {erms:>11.3e} {cosd:>10.3e} | {e16:>11.3e} | {ebf:>11.3e}");
    }
    drop(tower);
    d.synchronize().unwrap();
}

/// 448x448 px = 32x32 patches, 11x11 LLM grid (the mid-size case).
#[test]
#[ignore]
fn encode_matches_cpu_mid() {
    check_against_cpu(smooth_image(448, 448), 1e-2);
}

/// End-to-end encode timing at the real image shapes (no CPU reference).
#[test]
#[ignore]
fn encode_bench() {
    let d = device();
    let mut tower = Tower::load(&mmproj(), d).unwrap();
    tower.profile = true;
    eprintln!("weights on device: {:.1} MiB", tower.device_bytes() as f64 / (1u64 << 20) as f64);
    for &(n_h, n_w, label) in &[
        (32u32, 32u32, "448x448"),
        (35, 46, "640x480"),
        (42, 74, "1920x1080 (max-ish n)"),
    ] {
        let img = synth_image(n_h, n_w);
        let _ = tower.encode_rows(&img).unwrap(); // warm
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let _ = tower.encode_rows(&img).unwrap();
            best = best.min(tower.last_encode_ms);
        }
        eprintln!("\n=== {label}: {n_h}x{n_w} = {} patches | best {best:.1} ms ===", n_h * n_w);
        let tot: f64 = tower.stage_ms.iter().map(|(_, v)| v).sum();
        for (k, v) in &tower.stage_ms {
            eprintln!("    {k:<16} {v:8.2} ms  {:5.1}%", 100.0 * v / tot);
        }
        eprintln!("    {:<16} {tot:8.2} ms (profiled, incl. per-launch syncs)", "TOTAL");
        eprintln!("    workspace: {:.1} MiB", tower.workspace_bytes() as f64 / (1u64 << 20) as f64);
    }
    drop(tower);
    d.synchronize().unwrap();
}
