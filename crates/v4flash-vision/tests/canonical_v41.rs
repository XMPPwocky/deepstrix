//! DeepSeek-V4.1 tower vs the CANONICAL PyTorch reference (`inference/vision.py`
//! `ViT` + `Aligner` fed straight from the HF safetensors, f32 on the CPU) —
//! the V4.1 twin of `canonical_vs_python.rs`.
//!
//! `scripts/gen_v41_vision_vectors.py` writes, per case <tag>:
//!     <tag>.json          grid dims, block types (V4.1 numbering), source file
//!     <tag>.patches.f32   [n][588]      (input; replayed here verbatim)
//!     <tag>.aligner.f32   [n_llm][5120]
//!     <tag>.block.f32     [n_block][5120]
//! This test loads the tower ONCE through the HF loader (`Tower::load_v41`),
//! replays every case's patches through `encode_rows` on the chosen device,
//! and reports max_abs / rms_err / 1-cos / per-row argmax agreement for the
//! aligner rows and the merged span. It never touches the language model:
//! ~0.9 GiB of f16 weights on the device + ~0.9 GiB host copy (dropped after
//! upload), < 2.5 GiB RSS.
//!
//! Run (vectors first, see the script's docstring):
//!   V41_MODEL=~/.cache/deepstrix/models/dsv4.1f CANON_DIR=~/.cache/deepstrix/v41/vision_canon \
//!   nix develop --profile ~/.cache/deepstrix/devshell --command \
//!   cargo test --release -p v4flash-vision --test canonical_v41 -- --ignored --test-threads=1 --nocapture
//!
//! Device: `DEEPSTRIX_VISION_DEVICE=<id>`, default = the first gfx1201 (the
//! dGPU; ~1 GiB of its VRAM for the duration).

use std::fs;
use std::path::{Path, PathBuf};

use v4flash_hip::Device;
use v4flash_vision::preprocess::PreprocessedImage;
use v4flash_vision::{layout_for_cfg, Tower, TokenType, VisionCfg, PATCH_ELEMS, V41_TEXT_DIM};

fn model_dir() -> PathBuf {
    std::env::var("V41_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/deepstrix/models/dsv4.1f"))
}

fn canon_dir() -> PathBuf {
    std::env::var("CANON_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/deepstrix/v41/vision_canon"))
}

fn device() -> Device {
    if let Some(id) = std::env::var("DEEPSTRIX_VISION_DEVICE").ok().and_then(|s| s.parse::<i32>().ok()) {
        return Device::new(id);
    }
    for d in Device::all().unwrap() {
        if d.properties().unwrap().gcn_arch_name.starts_with("gfx1201") {
            return d;
        }
    }
    panic!("no gfx1201 device; set DEEPSTRIX_VISION_DEVICE");
}

fn read_f32(p: &Path) -> Vec<f32> {
    let b = fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn write_f32(p: &Path, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(p, b).unwrap();
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
    let agree = (0..rows).filter(|r| am(&got[r * dim..(r + 1) * dim]) == am(&want[r * dim..(r + 1) * dim])).count();
    (agree, rows)
}

#[derive(serde::Deserialize)]
struct Meta {
    tag: String,
    source: Option<String>,
    n_vit_h: u32,
    n_vit_w: u32,
    n_llm_h: u32,
    n_llm_w: u32,
    n_block: usize,
    text_dim: usize,
    /// V4.1 numbering: START 0, IMAGE 1, NEW_LINE 2, END 3.
    types: Vec<u8>,
}

fn v41_types(t: &[u8]) -> Vec<u8> {
    t.iter()
        .map(|&x| match x {
            0 => TokenType::Start as u8,
            1 => TokenType::Image as u8,
            2 => TokenType::NewLine as u8,
            3 => TokenType::End as u8,
            o => panic!("bad V4.1 type {o}"),
        })
        .collect()
}

fn tags() -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(canon_dir())
        .expect("CANON_DIR")
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".json")).map(String::from))
        .filter(|n| !n.starts_with("layout_cases"))
        .collect();
    v.sort();
    v
}

/// One tower load, every case (memory discipline: never load twice).
#[test]
#[ignore]
fn canonical_v41_all() {
    let dir = canon_dir();
    let d = device();
    d.set_current().unwrap();
    let props = d.properties().unwrap();
    eprintln!("device {} ({}, {})", d.id, props.name, props.gcn_arch_name);
    let t0 = std::time::Instant::now();
    let mut tower = Tower::load_v41(&model_dir(), d).unwrap();
    eprintln!(
        "tower loaded in {:.1} s: {:.1} MiB on device, text_dim {}, gemm {:?}",
        t0.elapsed().as_secs_f64(),
        tower.device_bytes() as f64 / 1048576.0,
        tower.text_dim(),
        tower.kernels.gemm_path
    );
    assert_eq!(tower.text_dim(), V41_TEXT_DIM);
    tower.drop_host();
    let mut worst_erms = 0f32;
    let mut summary = Vec::new();
    for tag in tags() {
        let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join(format!("{tag}.json"))).unwrap()).unwrap();
        assert_eq!(meta.tag, tag);
        assert_eq!(meta.text_dim, V41_TEXT_DIM);
        let patches = read_f32(&dir.join(format!("{tag}.patches.f32")));
        let n = (meta.n_vit_h * meta.n_vit_w) as usize;
        assert_eq!(patches.len(), n * PATCH_ELEMS, "{tag}: patches");
        let want_al = read_f32(&dir.join(format!("{tag}.aligner.f32")));
        let want_bl = read_f32(&dir.join(format!("{tag}.block.f32")));

        let img = PreprocessedImage { patches, n_vit_h: meta.n_vit_h, n_vit_w: meta.n_vit_w, content_hash: [0u8; 32] };
        let layout = layout_for_cfg(&img, 0, &VisionCfg::V41);
        assert_eq!((layout.n_llm_h, layout.n_llm_w), (meta.n_llm_h, meta.n_llm_w), "{tag}: llm grid");
        assert_eq!(layout.types, v41_types(&meta.types), "{tag}: span types");
        assert_eq!(layout.types.len(), meta.n_block);

        let got_al = tower.encode_rows(&img).unwrap();
        let enc1 = tower.last_encode_ms;
        let got_al2 = tower.encode_rows(&img).unwrap();
        assert_eq!(got_al, got_al2, "{tag}: encode not deterministic");
        let enc2 = tower.last_encode_ms;
        let got_bl = tower.place_rows(&layout, &got_al).unwrap();
        assert_eq!(got_al.len(), want_al.len(), "{tag}: aligner shape");
        assert_eq!(got_bl.len(), want_bl.len(), "{tag}: block shape");
        write_f32(&dir.join(format!("{tag}.gpu_aligner.f32")), &got_al);
        write_f32(&dir.join(format!("{tag}.gpu_block.f32")), &got_bl);

        let td = V41_TEXT_DIM;
        let (mx, erms, cosd, mrel) = err(&got_al, &want_al);
        let (bmx, berms, bcos, _) = err(&got_bl, &want_bl);
        let (ag, rows) = argmax_agreement(&got_al, &want_al, td);
        let (bag, brows) = argmax_agreement(&got_bl, &want_bl, td);
        // Sentinel rows come from the same bf16 tensors on both sides: exact.
        let n_sent = layout.types.iter().filter(|&&t| t != TokenType::Image as u8).count();
        let sent_exact = layout
            .types
            .iter()
            .enumerate()
            .filter(|(_, &t)| t != TokenType::Image as u8)
            .all(|(i, _)| got_bl[i * td..(i + 1) * td] == want_bl[i * td..(i + 1) * td]);
        eprintln!(
            "=== {tag}: grid {}x{} = {n} patches -> {}x{} = {rows} aligner rows, span {} tokens",
            meta.n_vit_h, meta.n_vit_w, meta.n_llm_h, meta.n_llm_w, meta.n_block
        );
        eprintln!("  encode wall: {enc1:.1} ms (cold) / {enc2:.1} ms (warm); workspace {:.1} MiB", tower.workspace_bytes() as f64 / 1048576.0);
        eprintln!("  ALIGNER vs canonical-f32: max_abs {mx:.4e}  rms_err/rms {erms:.4e}  1-cos {cosd:.3e}  max_rel {mrel:.3e}");
        eprintln!("  SPAN    vs canonical-f32: max_abs {bmx:.4e}  rms_err/rms {berms:.4e}  1-cos {bcos:.3e}  ({n_sent} sentinel rows exact: {sent_exact})");
        eprintln!("  argmax agreement: aligner {ag}/{rows}   span {bag}/{brows}");
        if !tower.stage_ms.is_empty() {
            let tot: f64 = tower.stage_ms.iter().map(|(_, v)| v).sum();
            for (k, v) in &tower.stage_ms {
                eprintln!("    {k:<16} {v:8.2} ms  {:5.1}%", 100.0 * v / tot);
            }
        }
        assert!(sent_exact, "{tag}: sentinel rows differ");
        worst_erms = worst_erms.max(erms);
        // End to end from OUR decoder + preprocessing (the JPEG path uses a
        // different decoder than Pillow): what the server would actually feed
        // the text model for this file, vs the canonical rows.
        //
        // Two variants: (a) our f32 patches as the server feeds them (the
        // reference rounds its input to bf16, so this delta is mostly THAT
        // rounding — deliberate extra precision on our side, as for V4-Flash);
        // (b) our patches rounded to bf16 like the reference's, which isolates
        // the decoder: bit-exact for PNG, the libjpeg-vs-zune delta for JPEG.
        let mut e2e = String::new();
        if let Some(src) = meta.source.as_deref() {
            let mut ours = v4flash_vision::preprocess_cfg(&fs::read(src).unwrap(), &VisionCfg::V41).unwrap();
            assert_eq!((ours.n_vit_h, ours.n_vit_w), (meta.n_vit_h, meta.n_vit_w));
            let got = tower.encode_rows(&ours).unwrap();
            let (emx, eerms, ecos, _) = err(&got, &want_al);
            let (eag, _) = argmax_agreement(&got, &want_al, td);
            eprintln!("  E2E own decode, f32 input (server path)  vs canonical-f32: max_abs {emx:.4e}  rms_err/rms {eerms:.4e}  1-cos {ecos:.3e}  argmax {eag}/{rows}");
            for v in ours.patches.iter_mut() {
                *v = v4flash_vision::preprocess::bf16_round(*v);
            }
            let got_b = tower.encode_rows(&ours).unwrap();
            let (bmx2, berms2, bcos2, _) = err(&got_b, &want_al);
            let (bag2, _) = argmax_agreement(&got_b, &want_al, td);
            eprintln!("  E2E own decode, bf16 input (as reference) vs canonical-f32: max_abs {bmx2:.4e}  rms_err/rms {berms2:.4e}  1-cos {bcos2:.3e}  argmax {bag2}/{rows}");
            e2e = format!(" | own-decode f32-in rms {eerms:.2e} argmax {eag}/{rows}; bf16-in rms {berms2:.2e} argmax {bag2}/{rows}");
        }
        summary.push(format!("{tag}: n={n} rows={rows} rms_err/rms {erms:.2e} 1-cos {cosd:.1e} argmax {ag}/{rows} span {bag}/{brows} {enc2:.0} ms{e2e}"));
    }
    eprintln!("---- summary ({}):", props.gcn_arch_name);
    for s in &summary {
        eprintln!("  {s}");
    }
    drop(tower);
    d.synchronize().unwrap();
    assert!(worst_erms < 5e-2, "worst aligner rms_err/rms {worst_erms:.3e}");
}

/// Our Rust preprocessing (decode → PIL-semantics pad/resize → normalise →
/// patchify, V4.1 limits) vs the canonical `image_processor.load_image`
/// patch tensor, per source image. PNG must be bit-exact after bf16
/// rounding; JPEG goes through two different decoders (Pillow's libjpeg vs
/// the `image` crate), so its delta is reported, not asserted.
#[test]
#[ignore]
fn preprocess_v41_matches_python() {
    let dir = canon_dir();
    let mut checked = 0;
    for tag in tags() {
        let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join(format!("{tag}.json"))).unwrap()).unwrap();
        let Some(src) = meta.source.as_deref() else { continue };
        let want = read_f32(&dir.join(format!("{tag}.patches.f32")));
        let bytes = fs::read(src).unwrap_or_else(|e| panic!("{tag}: read {src}: {e}"));
        let img = v4flash_vision::preprocess_cfg(&bytes, &VisionCfg::V41).unwrap();
        assert_eq!((img.n_vit_h, img.n_vit_w), (meta.n_vit_h, meta.n_vit_w), "{tag}: grid");
        assert_eq!(img.patches.len(), want.len(), "{tag}: patch count");
        let rounded: Vec<f32> = img.patches.iter().map(|v| v4flash_vision::preprocess::bf16_round(*v)).collect();
        let ndiff = rounded.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let (mx, erms, _, _) = err(&rounded, &want);
        // One 8-bit level after normalisation is 2/255 = 7.84e-3.
        let n_levels = (mx / (2.0 / 255.0)).round();
        eprintln!(
            "{tag} ({src}): grid {}x{}; bf16(our patches) vs canonical: {ndiff}/{} differ, max_abs {mx:.3e} (~{n_levels} 8-bit level), rms_err/rms {erms:.2e}",
            img.n_vit_h, img.n_vit_w, want.len()
        );
        if src.ends_with(".png") {
            assert_eq!(ndiff, 0, "{tag}: PNG preprocessing diverges from Pillow/PyTorch on {ndiff} elements");
        }
        checked += 1;
    }
    assert!(checked > 0, "no cases with a source image in {}", dir.display());
}
