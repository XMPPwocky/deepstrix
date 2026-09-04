//! Tower (iGPU) vs the CANONICAL PyTorch reference (`inference/vision.py`).
//!
//! The Python side (`scripts/gen_canonical_vision_vectors.py`) loads `mmproj-F16.gguf` into
//! the reference `ViT` / `Aligner` modules, runs `image_processor.load_image`
//! + `build_image_block`, and dumps raw little-endian f32:
//!     <tag>.json     grid dims, block types, perm
//!     <tag>.patches.f32   [n][588]   (input; replayed here verbatim)
//!     <tag>.hidden.f32    [n][1024]  post-`v.post_ln`
//!     <tag>.aligner.f32   [n_llm][4096]
//!     <tag>.block.f32     [n_block][4096]
//! This test replays the SAME patches through `Tower::encode_rows` on the
//! iGPU and writes `<tag>.gpu_aligner.f32` / `<tag>.gpu_block.f32`; the
//! error metrics are computed here and again by `compare.py`.
//!
//! Run (generate the vectors first):
//!   nix-shell -p python3Packages.torch python3Packages.numpy python3Packages.pillow \
//!     --run "python3 scripts/gen_canonical_vision_vectors.py --out /tmp/canon --ref-dir <ref>"
//!   DEEPSTRIX_MMPROJ=/persist/lumi/models/dsv4f-exp-q2-k-xl/mmproj-F16.gguf \
//!   DEEPSTRIX_VISION_DEVICE=1 CANON_DIR=/tmp/canon CANON_PNG=/tmp/canon/test640x480.png \
//!   cargo test --release -p v4flash-vision --test canonical_vs_python \
//!       -- --ignored --test-threads=1 --nocapture

use std::fs;
use std::path::PathBuf;

use v4flash_hip::Device;
use v4flash_vision::preprocess::PreprocessedImage;
use v4flash_vision::{layout_for, Tower, PATCH_ELEMS, TEXT_DIM};

fn canon_dir() -> PathBuf {
    PathBuf::from(std::env::var("CANON_DIR").expect("CANON_DIR"))
}

fn device() -> Device {
    let id: i32 = std::env::var("DEEPSTRIX_VISION_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    assert_ne!(id, 0, "refusing to touch the dGPU (device 0) while the server is live");
    Device::new(id)
}

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    let b = fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn write_f32(p: &std::path::Path, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(p, b).unwrap();
}

/// `gtt_used` of the iGPU node, bytes. `DEEPSTRIX_GTT_NODE` overrides the card.
fn gtt_used() -> u64 {
    let card = std::env::var("DEEPSTRIX_GTT_NODE").unwrap_or_else(|_| "card2".into());
    fs::read_to_string(format!("/sys/class/drm/{card}/device/mem_info_gtt_used"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

fn mib(a: u64, b: u64) -> f64 {
    (a as f64 - b as f64) / (1024.0 * 1024.0)
}

/// (max_abs, rms(err)/rms(ref), 1 - cosine, max |rel| over elements ≥ 1% of rms)
fn err(got: &[f32], want: &[f32]) -> (f32, f32, f32, f32) {
    assert_eq!(got.len(), want.len());
    let n = want.len() as f64;
    let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n).sqrt();
    let mut mx = 0f32;
    let mut mrel = 0f32;
    let mut esq = 0f64;
    let (mut dot, mut ga, mut wa) = (0f64, 0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        let d = (a - b).abs();
        mx = mx.max(d);
        if (*b as f64).abs() >= 0.01 * rms {
            mrel = mrel.max(d / b.abs());
        }
        esq += (d as f64).powi(2);
        dot += *a as f64 * *b as f64;
        ga += (*a as f64).powi(2);
        wa += (*b as f64).powi(2);
    }
    let erms = (esq / n).sqrt();
    let cos = dot / (ga.sqrt() * wa.sqrt()).max(1e-300);
    (mx, (erms / rms.max(1e-20)) as f32, (1.0 - cos) as f32, mrel)
}

fn argmax_agreement(got: &[f32], want: &[f32], dim: usize) -> (usize, usize) {
    let rows = want.len() / dim;
    let am = |s: &[f32]| s.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| if *v > bv { (i, *v) } else { (bi, bv) }).0;
    let agree = (0..rows)
        .filter(|r| am(&got[r * dim..(r + 1) * dim]) == am(&want[r * dim..(r + 1) * dim]))
        .count();
    (agree, rows)
}

#[derive(serde::Deserialize)]
struct Meta {
    n_vit_h: u32,
    n_vit_w: u32,
    n_llm_h: u32,
    n_llm_w: u32,
    start_pos: u32,
    n_block: usize,
    types: Vec<u8>,
    perm: Vec<u32>,
}

fn run(tag: &str) {
    let dir = canon_dir();
    let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join(format!("{tag}.json"))).unwrap()).unwrap();
    let patches = read_f32(&dir.join(format!("{tag}.patches.f32")));
    let n = (meta.n_vit_h * meta.n_vit_w) as usize;
    assert_eq!(patches.len(), n * PATCH_ELEMS);
    let want_al = read_f32(&dir.join(format!("{tag}.aligner.f32")));
    let want_bl = read_f32(&dir.join(format!("{tag}.block.f32")));

    // Layout must agree with the Python `build_image_block` before anything else.
    let img = PreprocessedImage { patches, n_vit_h: meta.n_vit_h, n_vit_w: meta.n_vit_w, content_hash: [0u8; 32] };
    let layout = layout_for(&img, meta.start_pos);
    assert_eq!((layout.n_llm_h, layout.n_llm_w), (meta.n_llm_h, meta.n_llm_w), "{tag}: llm grid");
    assert_eq!(layout.types, meta.types, "{tag}: block types");
    assert_eq!(layout.perm, meta.perm, "{tag}: block perm");
    assert_eq!(layout.types.len(), meta.n_block);

    let g0 = gtt_used();
    let d = device();
    d.set_current().unwrap();
    let g_ctx = gtt_used();
    let mut tower = Tower::load(&PathBuf::from(std::env::var("DEEPSTRIX_MMPROJ").expect("DEEPSTRIX_MMPROJ")), d).unwrap();
    tower.drop_host();
    let g_load = gtt_used();
    let got_al = tower.encode_rows(&img).unwrap();
    let g_enc = gtt_used();
    let enc1 = tower.last_encode_ms;
    let got_al2 = tower.encode_rows(&img).unwrap();
    assert_eq!(got_al, got_al2, "{tag}: encode not deterministic");
    let enc2 = tower.last_encode_ms;
    let got_bl = tower.place_rows(&layout, &got_al).unwrap();

    assert_eq!(got_al.len(), want_al.len(), "{tag}: aligner shape");
    assert_eq!(got_bl.len(), want_bl.len(), "{tag}: block shape");
    write_f32(&dir.join(format!("{tag}.gpu_aligner.f32")), &got_al);
    write_f32(&dir.join(format!("{tag}.gpu_block.f32")), &got_bl);

    let (mx, erms, cosd, mrel) = err(&got_al, &want_al);
    let (bmx, berms, bcos, _) = err(&got_bl, &want_bl);
    let (ag, rows) = argmax_agreement(&got_al, &want_al, TEXT_DIM);
    let (bag, brows) = argmax_agreement(&got_bl, &want_bl, TEXT_DIM);
    eprintln!("=== {tag}: grid {}x{} = {n} patches -> {}x{} = {rows} aligner rows, block {} tokens",
        meta.n_vit_h, meta.n_vit_w, meta.n_llm_h, meta.n_llm_w, meta.n_block);
    eprintln!("  encode wall: {enc1:.1} ms (cold) / {enc2:.1} ms (warm)");
    eprintln!("  GTT: ctx +{:.1} MiB | weights +{:.1} MiB | encode +{:.1} MiB | total +{:.1} MiB",
        mib(g_ctx, g0), mib(g_load, g_ctx), mib(g_enc, g_load), mib(g_enc, g0));
    eprintln!("  device_bytes {:.1} MiB, workspace {:.1} MiB",
        tower.device_bytes() as f64 / 1048576.0, tower.workspace_bytes() as f64 / 1048576.0);
    eprintln!("  ALIGNER vs canonical-f32: max_abs {mx:.4e}  rms_err/rms {erms:.4e}  1-cos {cosd:.3e}  max_rel {mrel:.3e}");
    eprintln!("  BLOCK   vs canonical-f32: max_abs {bmx:.4e}  rms_err/rms {berms:.4e}  1-cos {bcos:.3e}");
    eprintln!("  argmax agreement: aligner {ag}/{rows}   block {bag}/{brows}");
    if !tower.stage_ms.is_empty() {
        let tot: f64 = tower.stage_ms.iter().map(|(_, v)| v).sum();
        for (k, v) in &tower.stage_ms {
            eprintln!("    {k:<16} {v:8.2} ms  {:5.1}%", 100.0 * v / tot);
        }
    }
    drop(tower);
    d.synchronize().unwrap();
    eprintln!("  GTT after drop: {:.1} MiB above start", mib(gtt_used(), g0));
    assert!(erms < 5e-2, "{tag}: aligner rms_err/rms {erms:.3e}");
}

#[test]
#[ignore]
fn canonical_synth4x6() {
    run("synth4x6");
}

#[test]
#[ignore]
fn canonical_real640x480() {
    run("real640x480");
}

/// Our Rust preprocessing of the SAME PNG must reproduce the canonical
/// `image_processor.load_image` patch tensor bit-for-bit (both bf16-rounded).
#[test]
#[ignore]
fn preprocess_matches_python() {
    let dir = canon_dir();
    let png = std::env::var("CANON_PNG").expect("CANON_PNG");
    let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("real640x480.json")).unwrap()).unwrap();
    let want = read_f32(&dir.join("real640x480.patches.f32"));
    let img = v4flash_vision::preprocess(&fs::read(&png).unwrap()).unwrap();
    eprintln!("preprocess: grid {}x{} (python {}x{})", img.n_vit_h, img.n_vit_w, meta.n_vit_h, meta.n_vit_w);
    assert_eq!((img.n_vit_h, img.n_vit_w), (meta.n_vit_h, meta.n_vit_w));
    assert_eq!(img.patches.len(), want.len());
    // Our patches stay f32; `image_processor.load_image` casts to bf16 because
    // the reference model runs bf16 end to end. So the bar is: our f32 patches
    // ROUNDED to bf16 must equal the canonical tensor BIT-FOR-BIT — that pins
    // the decode / PIL resize / pad / normalise / patchify chain exactly, and
    // isolates the remaining delta as the deliberate extra precision.
    let (mx_raw, erms_raw, _, _) = err(&img.patches, &want);
    let ndiff_raw = img.patches.iter().zip(&want).filter(|(a, b)| a != b).count();
    let rounded: Vec<f32> = img.patches.iter().map(|v| v4flash_vision::preprocess::bf16_round(*v)).collect();
    let ndiff = rounded.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let (mx, _, _, _) = err(&rounded, &want);
    eprintln!("  f32 patches vs canonical bf16: {ndiff_raw}/{} differ, max_abs {mx_raw:.3e} (= half a bf16 ulp at |x|<1), rms_err/rms {erms_raw:.3e}", want.len());
    eprintln!("  bf16(our patches) vs canonical: {ndiff}/{} differ, max_abs {mx:.3e}", want.len());
    assert_eq!(ndiff, 0, "preprocess diverges from Pillow/PyTorch on {ndiff} elements (max_abs {mx:.3e})");
}
