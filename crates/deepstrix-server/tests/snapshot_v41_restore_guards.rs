//! V4.1 snapshot regressions from the 2026-09-23 review (no weights loaded;
//! a 4K-context state, a few tens of MiB on each GPU -- safe beside the live
//! server):
//!
//! 1. The raw window is saved from `raw_off`, not slot 0. The decode append is
//!    monotonic and slides `raw_off` past SWA_WINDOW, so a state saved after a
//!    long decode used to write rows that had already been evicted. Restore
//!    lands the saved rows at slot 0 with `raw_off = 0`.
//! 2. A short `index_k.bin` is REFUSED (a cache miss), not "degraded" to
//!    `n_index_comp = 0` with `n_comp` kept -- the next prefill then treated
//!    another request's leftover keys as valid.
//! 3. Index keys that do not cover every compressed row are refused.
//!
//! Needs the V4.1 checkpoint's `tokenizer.json` (`V41_MODEL`, default the
//! production snapshot dir) and both GPUs.
//!
//! `cargo test -p deepstrix-server --features v41 --release --test snapshot_v41_restore_guards -- --ignored --nocapture`

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use deepstrix_server::embed::build_gpt2_byte_decoder;
use deepstrix_server::snapshot::{self, ModelFingerprint, RestoreKernels};
use v4flash_core::tokenizer::BpeVocab;
use v4flash_hip::{install_panic_handler, Device, Stream};
use v4flash_kernels::config::{N_HEAD_DIM, N_LAYER};
use v4flash_kernels::het::state::HetModelState;
use v4flash_kernels::index_kv_e2m1::E2M1_KEY_ROW_BYTES;
use v4flash_kernels::CompKvFp8;

const DEFAULT_MODEL: &str =
    "/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277";
const N_KV_MAX: u32 = 4096;
const POS: u32 = 400;
const RAW_OFF: u32 = 37;
const N_RAW: u32 = 128;

fn devices() -> eyre::Result<(Device, Device)> {
    let (mut dgpu, mut igpu) = (None, None);
    for d in Device::all()? {
        let arch = d.properties()?.gcn_arch_name;
        if arch.starts_with("gfx1201") {
            dgpu = Some(d);
        } else if arch.starts_with("gfx1151") {
            igpu = Some(d);
        }
    }
    Ok((dgpu.ok_or_else(|| eyre!("no gfx1201"))?, igpu.ok_or_else(|| eyre!("no gfx1151"))?))
}

fn fp() -> ModelFingerprint {
    ModelFingerprint {
        n_layer: N_LAYER as u32,
        n_head_dim: N_HEAD_DIM,
        vocab_size: 1,
        token_embd_prefix_blake3: "v41-restore-guards".into(),
        tensor_directory_blake3: "test".into(),
    }
}

fn scratch_root() -> PathBuf {
    let base = std::env::var("CLAUDE_SCRATCHPAD").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    let p = base.join(format!("snapshot-v41-guards-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Raw windows at `RAW_OFF` with a per-(layer, row) pattern, junk before them;
/// every compressor at `POS / ratio` rows with patterned index keys.
fn fill(state: &mut HetModelState) -> eyre::Result<Vec<Vec<u16>>> {
    let hd = N_HEAD_DIM as usize;
    let mut windows = Vec::new();
    for (li, layer) in state.layers.iter_mut().enumerate() {
        let total = (RAW_OFF + N_RAW) as usize * hd;
        let all: Vec<u16> = (0..total)
            .map(|i| if i < RAW_OFF as usize * hd { 0xDEAD } else { (li as u16).wrapping_mul(977).wrapping_add((i % 30011) as u16) })
            .collect();
        layer.kv_cache.slice_view_mut(0, total).copy_from_host(&all)?;
        layer.raw_off = RAW_OFF;
        layer.n_raw = N_RAW;
        windows.push(all[RAW_OFF as usize * hd..].to_vec());
        if let Some(cs) = layer.compressor.as_mut() {
            let ratio = v4flash_kernels::config::COMPRESS_RATIOS[li].max(1);
            cs.n_comp = POS / ratio;
            if let Some(ik) = cs.index_k.as_mut() {
                let n = cs.n_comp as usize * E2M1_KEY_ROW_BYTES;
                let keys: Vec<u8> = (0..n).map(|i| (li * 13 + i) as u8).collect();
                ik.slice_view_mut(0, n).copy_from_host(&keys)?;
                cs.n_index_comp = cs.n_comp;
            }
        }
    }
    Ok(windows)
}

#[test]
#[ignore]
fn v41_snapshot_restore_guards() -> eyre::Result<()> {
    install_panic_handler()?;
    let (dgpu, igpu) = devices()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let packed = CompKvFp8::for_arch(&arch)?;
    let kernels = RestoreKernels { fp8: &packed, stream: &stream };
    let model = std::env::var("V41_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let vocab = BpeVocab::from_tokenizer_json(PathBuf::from(&model).join("tokenizer.json"), Some("joyai-llm".to_string()))?;
    let byte_decoder = build_gpt2_byte_decoder();
    let root = scratch_root();
    let fingerprint = fp();
    let tokens: Vec<i32> = (0..POS as i32).map(|i| 1000 + (i * 7) % 5000).collect();
    let hd = N_HEAD_DIM as usize;

    // 1. raw window saved from raw_off, restored at slot 0.
    let mut src = HetModelState::alloc(dgpu, igpu, N_KV_MAX)?;
    let windows = fill(&mut src)?;
    let entry = snapshot::save(&src, &tokens, &[], dgpu, igpu, &fingerprint, &root, &vocab, &byte_decoder, None)?;
    let dir = entry.dir.clone();
    let mut dst = HetModelState::alloc(dgpu, igpu, N_KV_MAX)?;
    snapshot::restore_vl(&mut dst, &dir, dgpu, igpu, &fingerprint, kernels)?;
    for (li, layer) in dst.layers.iter().enumerate() {
        assert_eq!(layer.raw_off, 0, "L{li}: restore must land the window at slot 0");
        assert_eq!(layer.n_raw, N_RAW, "L{li}: n_raw");
        let mut got = vec![0u16; N_RAW as usize * hd];
        layer.kv_cache.slice_view(0, got.len()).copy_to_host(&mut got)?;
        assert!(got == windows[li], "L{li}: restored raw window is not the one at raw_off (saved from slot 0?)");
        if let Some(cs) = layer.compressor.as_ref() {
            if cs.index_k.is_some() {
                assert_eq!(cs.n_index_comp, cs.n_comp, "L{li}: keys restored for every comp row");
            }
        }
    }
    println!("  1. raw window saved from raw_off={RAW_OFF}, restored at slot 0: OK");

    // 2. a short index_k.bin is refused.
    let ik = dir.join("index_k.bin");
    let len = std::fs::metadata(&ik)?.len();
    assert!(len > 0, "fixture must have index keys");
    std::fs::OpenOptions::new().write(true).open(&ik)?.set_len(len / 2)?;
    let mut dst2 = HetModelState::alloc(dgpu, igpu, N_KV_MAX)?;
    match snapshot::restore_vl(&mut dst2, &dir, dgpu, igpu, &fingerprint, kernels) {
        Ok(_) => return Err(eyre!("a short index_k.bin must be refused, not half-restored")),
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(msg.contains("index_k.bin is short"), "unexpected error: {msg}");
            println!("  2. short index_k.bin refused: {}", msg.lines().next().unwrap_or(""));
        }
    }

    // 3. keys that do not cover every comp row are refused.
    let mut src3 = HetModelState::alloc(dgpu, igpu, N_KV_MAX)?;
    fill(&mut src3)?;
    let li = src3.layers.iter().position(|l| l.compressor.as_ref().is_some_and(|c| c.index_k.is_some() && c.n_comp > 1))
        .ok_or_else(|| eyre!("no layer with index keys"))?;
    {
        let cs = src3.layers[li].compressor.as_mut().unwrap();
        cs.n_index_comp = cs.n_comp / 2;
    }
    let tokens3: Vec<i32> = tokens.iter().map(|t| t + 1).collect(); // distinct snapshot key
    let entry3 = snapshot::save(&src3, &tokens3, &[], dgpu, igpu, &fingerprint, &root, &vocab, &byte_decoder, None)?;
    let mut dst3 = HetModelState::alloc(dgpu, igpu, N_KV_MAX)?;
    match snapshot::restore_vl(&mut dst3, &entry3.dir, dgpu, igpu, &fingerprint, kernels) {
        Ok(_) => return Err(eyre!("keys short of n_comp must be refused")),
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(msg.contains("index keys for"), "unexpected error: {msg}");
            println!("  3. keys short of n_comp refused: {}", msg.lines().next().unwrap_or(""));
        }
    }
    let _ = std::fs::remove_dir_all(&root);
    Ok(())
}
