//! DeepSeek-V4.1 vision glue (`--features v41` only): the tower comes from the
//! HF snapshot directory instead of a mmproj GGUF, and the text-side `bias_vl`
//! routing bias comes from the checkpoint's own `layers.N.ffn.gate.bias_vl`
//! tensors (presented by `V41HfWeights` as `blk.N.exp_probs_b_vl.bias`)
//! instead of a hand-fetched file.
//!
//! What V4.1 changes on the text side of an image, relative to V4-Flash
//! Vision-Exp (`inference/model.py`, 2026-09-13 checkpoint):
//! * routing: `bias_vl` replaces the gate bias for every image-span row on
//!   every layer — SAME mechanism, handled by the engine off `id >= N_VOCAB`;
//! * attention: NONE. V4.1 has no `get_image_visible`; image tokens attend
//!   causally like text. So the engine is given NO `image_spans` (no raw
//!   window widening, no chunk / lane cut constraints) — see
//!   `engine_worker::prefill_suffix`;
//! * Engram: image positions are dead — zero contribution, and the n-gram
//!   look-back of later text tokens stops at them
//!   (`v4flash_core::engram_hash`, `EngramCtx`).

use std::path::Path;

use color_eyre::eyre::{self, eyre, WrapErr};
use v4flash_core::WeightSrc;
use v4flash_hip::Device;
use v4flash_kernels::config::{N_EMBD, N_EXPERT, N_LAYER};
use v4flash_kernels::het::weights::{bias_vl_sidecar_path, write_bias_vl_sidecar};
use v4flash_kernels::het::HetModelWeights;
use v4flash_vision::Tower;

/// Load the V4.1 tower from `model_dir` (the `--mmproj` argument, normally
/// the `--gguf` snapshot dir) onto `device`, and check that its aligner
/// width is the compiled text width.
pub fn load_tower(model_dir: &Path, device: Device) -> eyre::Result<Tower> {
    if !model_dir.is_dir() {
        return Err(eyre!(
            "--mmproj {} is not a directory; the V4.1 build reads the vision tower from the HF snapshot \
             directory (pass the same path as --gguf)",
            model_dir.display()
        ));
    }
    let tower = Tower::load_v41(model_dir, device).wrap_err_with(|| format!("loading V4.1 vision tower from {}", model_dir.display()))?;
    if tower.text_dim() != N_EMBD as usize {
        return Err(eyre!(
            "V4.1 vision tower projects to {} but this binary's text width is {N_EMBD}",
            tower.text_dim()
        ));
    }
    Ok(tower)
}

/// Make sure the router has `bias_vl` for every layer. The engine only reads
/// it from the `bias_vl.bin` sidecar (`bias_vl_sidecar_path(model_dir)`,
/// `DEEPSTRIX_BIAS_VL_FILE` override); when that file is missing it is
/// written here from the checkpoint's `blk.N.exp_probs_b_vl.bias` tensors
/// (F32 `[N_EXPERT]`, 40 × 384) and then loaded.
pub fn ensure_bias_vl(
    weights: &mut HetModelWeights,
    src: &WeightSrc,
    model_dir: &Path,
    dgpu: Device,
) -> eyre::Result<()> {
    if weights.has_bias_vl() {
        return Ok(());
    }
    let path = bias_vl_sidecar_path(model_dir).ok_or_else(|| eyre!("bias_vl sidecar path unresolvable (HOME unset?)"))?;
    if !path.exists() {
        let n = N_EXPERT as usize;
        let mut all = Vec::with_capacity(N_LAYER as usize * n);
        for l in 0..N_LAYER as usize {
            let name = format!("blk.{l}.exp_probs_b_vl.bias");
            let t = src
                .tensor(&name)
                .ok_or_else(|| eyre!("{name}: not presented by the V4.1 HF view (no vision routing bias in this checkpoint?)"))?;
            let bytes = src.read_tensor(t)?;
            if bytes.len() != n * 4 {
                return Err(eyre!("{name}: {} bytes, expected {} f32", bytes.len(), n));
            }
            all.extend(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        }
        write_bias_vl_sidecar(&path, &all)?;
        tracing::info!(path = %path.display(), layers = N_LAYER, experts = N_EXPERT, "wrote bias_vl sidecar from the HF checkpoint");
    }
    let loaded = weights.load_bias_vl_sidecar(model_dir, dgpu)?;
    if !weights.has_bias_vl() {
        return Err(eyre!("bias_vl sidecar {:?} did not attach to every layer", loaded));
    }
    Ok(())
}
