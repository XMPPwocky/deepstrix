//! Vision-Exp router bias (`bias_vl`) sidecar: format, path derivation and
//! — when the real file is present — its contents.
//!
//! WHY A SIDECAR. On the text side, DeepSeek's `Gate.forward` adds
//! `exp_probs_b` to the routing scores for text tokens and `bias_vl` for
//! IMAGE tokens (`input_ids >= vocab_size`). `exp_probs_b` ships in the
//! GGUF as `blk.N.exp_probs_b.bias`; the unsloth Vision-Exp upload (2026-09)
//! carries NO `bias_vl` tensor, so the engine loads it from
//!     ~/.cache/deepstrix/models/<gguf file stem>/bias_vl.bin
//! (override the full path with `DEEPSTRIX_BIAS_VL_FILE`).
//!
//! FORMAT: `N_LAYER * N_EXPERT` = 43 × 256 f32 little-endian, layer-major
//! (layer `l` at `[l*256 .. (l+1)*256)`), 44032 bytes. No header.
//!
//! PROVENANCE: `scripts/fetch_bias_vl.py` builds it from the HF safetensors
//! of `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. The tensor names in
//! `model.safetensors.index.json` (read 2026-09-03) are
//!     layers.{0..42}.ffn.gate.bias_vl    F32  [256]
//! with siblings `layers.N.ffn.gate.bias` ([256] F32 — that IS
//! `exp_probs_b`), `layers.N.ffn.gate.tid2eid` ([129280, 6] I64, layers
//! 0-2 only) and `layers.N.ffn.gate.weight` ([256, 4096] BF16). The 43
//! tensors live in 46 of the 48 shards, each fetched with one ranged GET.
//!
//! Regenerate:
//!   python3 scripts/fetch_bias_vl.py --gguf /persist/.../<model>-00001-of-*.gguf
//!
//! The `real_sidecar_*` test is `#[ignore]`d because it depends on that
//! file having been fetched; everything else is pure CPU and always runs.

use std::path::{Path, PathBuf};

use v4flash_kernels::config::{N_EXPERT, N_LAYER};
use v4flash_kernels::het::weights::{
    bias_vl_sidecar_path, read_bias_vl_sidecar, write_bias_vl_sidecar, BIAS_VL_FILE,
};

const N: usize = (N_LAYER as usize) * (N_EXPERT as usize);

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir()
        .join(format!("v4flash-biasvl-it-{}-{tag}", std::process::id()))
        .join(BIAS_VL_FILE)
}

#[test]
fn sidecar_roundtrip_and_rejects_corruption() {
    let p = tmp("rt");
    let vals: Vec<f32> = (0..N).map(|i| (i as f32) * 1e-4 - 2.0).collect();
    write_bias_vl_sidecar(&p, &vals).unwrap();
    assert_eq!(std::fs::metadata(&p).unwrap().len() as usize, N * 4);
    assert_eq!(read_bias_vl_sidecar(&p).unwrap(), vals);

    // Wrong length is rejected, not silently zero-padded.
    std::fs::write(&p, vec![0u8; N * 4 - 4]).unwrap();
    assert!(read_bias_vl_sidecar(&p).is_err());
    std::fs::write(&p, vec![0u8; N * 4 + 4]).unwrap();
    assert!(read_bias_vl_sidecar(&p).is_err());
    // NaN / inf is rejected: a bad bias would silently rewire routing.
    let mut bad = vals.clone();
    bad[7 * 256 + 3] = f32::NAN;
    let raw: Vec<u8> = bad.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&p, &raw).unwrap();
    assert!(read_bias_vl_sidecar(&p).is_err());
    // Wrong element count on write is rejected too.
    assert!(write_bias_vl_sidecar(&p, &vals[..N - 1]).is_err());

    std::fs::remove_dir_all(p.parent().unwrap()).ok();
}

#[test]
fn sidecar_path_is_per_model_and_overridable() {
    // The env override wins outright (used by tests / one-off models).
    std::env::set_var("DEEPSTRIX_BIAS_VL_FILE", "/tmp/somewhere/else.bin");
    assert_eq!(
        bias_vl_sidecar_path(Path::new("/persist/x/anything.gguf")).unwrap(),
        PathBuf::from("/tmp/somewhere/else.bin")
    );
    std::env::remove_var("DEEPSTRIX_BIAS_VL_FILE");

    // Otherwise: the GGUF's file stem under the shared model cache dir —
    // the same directory that holds expert_stats.json / hot_experts.txt,
    // so a sharded model keys off shard 1's name.
    let p = bias_vl_sidecar_path(Path::new(
        "/persist/lumi/models/dsv4f-exp-q2-k-xl/DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003.gguf",
    ))
    .unwrap();
    assert!(
        p.ends_with(
            ".cache/deepstrix/models/DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003/bias_vl.bin"
        ),
        "unexpected sidecar path {}",
        p.display()
    );
    // A different quant is a different sidecar (bias_vl is per-checkpoint).
    let q = bias_vl_sidecar_path(Path::new(
        "/persist/x/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004.gguf",
    ))
    .unwrap();
    assert_ne!(p, q);
}

/// The real file, as fetched by `scripts/fetch_bias_vl.py`. Ignored: it is
/// only present after that script has run.
#[test]
#[ignore]
fn real_sidecar_loads_and_is_plausible() {
    let mut found = 0usize;
    for stem in [
        "DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003",
        "DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004",
    ] {
        let p = PathBuf::from(std::env::var("HOME").unwrap())
            .join(".cache/deepstrix/models")
            .join(stem)
            .join(BIAS_VL_FILE);
        if !p.exists() {
            println!("skip (absent): {}", p.display());
            continue;
        }
        found += 1;
        let v = read_bias_vl_sidecar(&p).unwrap();
        assert_eq!(v.len(), N);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in &v {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        // A routing bias sits on the same scale as the sigmoid gate scores;
        // anything outside this would mean a dtype / stride mix-up in the
        // fetch, not a real bias.
        assert!(lo > -50.0 && hi < 50.0, "{stem}: bias_vl range [{lo}, {hi}] implausible");
        // Not all-zero, and every layer populated.
        assert!(v.iter().any(|&x| x != 0.0), "{stem}: all-zero bias_vl");
        for l in 0..N_LAYER as usize {
            let layer = &v[l * 256..(l + 1) * 256];
            assert!(
                layer.iter().any(|&x| x != 0.0),
                "{stem}: layer {l} bias_vl is all zero"
            );
        }
        println!(
            "{stem}: bias_vl ok — {} values, range [{lo:.4}, {hi:.4}], L0 mean {:.4}, L42 mean {:.4}",
            v.len(),
            v[..256].iter().sum::<f32>() / 256.0,
            v[42 * 256..].iter().sum::<f32>() / 256.0,
        );
    }
    assert!(found > 0, "no Vision-Exp bias_vl sidecar found — run scripts/fetch_bias_vl.py");
}
