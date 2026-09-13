//! DeepSeek-V4.1 vision tower + aligner straight from the HF safetensors
//! checkpoint (no converted files): `vision.*`, `aligner.*` and the three
//! span-delimiter embeddings `image_{start,newline,end}`, all BF16, in
//! shards 1–2 of `model-000NN-of-00048.safetensors`.
//!
//! Presented as a [`MmprojHost`] so the tower / kernels / CPU reference are
//! shared with V4-Flash unchanged. Layout mapping (HF `[out, in]` row-major
//! is what the kernels want, so weights pass through):
//!
//! | HF (bf16)                                   | host field            |
//! |---------------------------------------------|-----------------------|
//! | `vision.patch_embed.proj.{weight,bias}` `[1024,588]` — `nn.Linear` over the `(c,y,x)`-flattened patch | `patch_embd_{w,b}` |
//! | `vision.blocks.N.norm{1,2}.weight`          | `ln{1,2}`             |
//! | `vision.blocks.N.attn.wqkv.{weight,bias}` `[3072,1024]` fused q‖k‖v | `attn_{q,k,v}_{w,b}` (split by rows) |
//! | `vision.blocks.N.attn.wo.{weight,bias}`     | `attn_out_{w,b}`      |
//! | `vision.blocks.N.mlp.w1.weight` `[5632,1024]` fused gate‖up (`chunk(2)`: gate first) | `ffn_gate_w`, `ffn_up_w` |
//! | `vision.blocks.N.mlp.w2.weight` `[1024,2816]` | `ffn_down_w`        |
//! | `vision.norm.weight`                        | `post_ln`             |
//! | `aligner.w1.{weight,bias}` `[5120,9216]`    | `mm1_{w,b}`           |
//! | `aligner.w2.{weight,bias}` `[5120,5120]`    | `mm2_{w,b}`           |
//! | `image_start` / `image_newline` / `image_end` `[5120]` | sentinels (`img_pad` = `None`) |
//!
//! Casts at read time: weights bf16 → f16 bits (what `vit_gemm` consumes;
//! lossless for every bf16 value inside f16's normal range — bf16 has 8
//! significant bits, f16 has 11 — so only |w| < 2⁻¹⁴ loses bits and
//! |w| > 65504 would overflow, which is refused), biases / norms /
//! sentinels bf16 → f32 (exact).
//!
//! Architecture numbers are cross-checked against `inference/config.json`
//! (`vision_n_layers` 32, `vision_dim` 1024, `vision_n_heads` 16,
//! `vision_inter_dim` 2816, `vision_patch_size` 14, `vision_rope_theta`
//! 1e4, `vision_downsample_ratio` 3) — the kernels bake those in — and
//! `dim` sets [`MmprojHost::text_dim`].

use std::path::Path;

use color_eyre::eyre::{self, eyre, WrapErr};
use v4flash_core::hf_v41::bf16_to_f32;
use v4flash_core::kquants::f32_to_f16_bits;
use v4flash_core::safetensors::{SafetensorsDir, StDtype, StTensor};

use crate::mmproj::{MmprojHost, MmprojMeta, VitBlockHost};
use crate::{
    VisionCfg, ALIGNER_IN, DOWNSAMPLE, PATCH, PATCH_ELEMS, VIT_DIM, VIT_FFN, VIT_N_HEADS, VIT_N_LAYERS, VIT_RMS_EPS,
    VIT_ROPE_THETA,
};

/// `inference/config.json` of a V4.1 snapshot directory.
pub fn read_config(model_dir: &Path) -> eyre::Result<serde_json::Value> {
    let p = model_dir.join("inference").join("config.json");
    let f = std::fs::File::open(&p).wrap_err_with(|| format!("open {}", p.display()))?;
    serde_json::from_reader(f).wrap_err_with(|| format!("parse {}", p.display()))
}

fn cfg_u64(cfg: &serde_json::Value, key: &str) -> eyre::Result<u64> {
    cfg.get(key).and_then(|v| v.as_u64()).ok_or_else(|| eyre!("inference/config.json: `{key}` missing or not an integer"))
}

/// The tower geometry the HF config declares, checked against what the
/// kernels are built for. Returns the text width (`dim`).
pub fn check_config(cfg: &serde_json::Value) -> eyre::Result<usize> {
    let checks: [(&str, u64); 6] = [
        ("vision_n_layers", VIT_N_LAYERS as u64),
        ("vision_dim", VIT_DIM as u64),
        ("vision_n_heads", VIT_N_HEADS as u64),
        ("vision_inter_dim", VIT_FFN as u64),
        ("vision_patch_size", PATCH as u64),
        ("vision_downsample_ratio", DOWNSAMPLE as u64),
    ];
    for (k, want) in checks {
        let got = cfg_u64(cfg, k)?;
        if got != want {
            return Err(eyre!("inference/config.json: {k} = {got}, this crate's kernels are built for {want}"));
        }
    }
    let theta = cfg.get("vision_rope_theta").and_then(|v| v.as_f64()).unwrap_or(f64::from(VIT_ROPE_THETA));
    if (theta - f64::from(VIT_ROPE_THETA)).abs() > 1e-6 {
        return Err(eyre!("inference/config.json: vision_rope_theta = {theta}, expected {VIT_ROPE_THETA}"));
    }
    let dim = cfg_u64(cfg, "dim")? as usize;
    // Preprocessing limits: the crate's V41 profile must agree with the checkpoint.
    let v41 = VisionCfg::V41;
    let (min_px, max_tok) = (cfg_u64(cfg, "vision_min_pixels")?, cfg_u64(cfg, "vision_max_n_token")?);
    if min_px != v41.min_pixels as u64 || max_tok != v41.max_n_token as u64 || dim != v41.text_dim {
        return Err(eyre!(
            "inference/config.json: vision_min_pixels {min_px} / vision_max_n_token {max_tok} / dim {dim} differ from \
             VisionCfg::V41 ({} / {} / {}); update the profile",
            v41.min_pixels, v41.max_n_token, v41.text_dim
        ));
    }
    if !cfg.get("vision_max_wh_ratio").map_or(true, |v| v.is_null()) {
        return Err(eyre!("inference/config.json: vision_max_wh_ratio is set; VisionCfg::V41 assumes none"));
    }
    Ok(dim)
}

struct Rd<'a> {
    st: &'a SafetensorsDir,
    /// bf16 → f16 casts that lost range: (overflowed to ±inf, flushed to 0).
    overflow: usize,
    underflow: usize,
    n_tensors: usize,
    bytes: usize,
}

impl<'a> Rd<'a> {
    fn tensor(&mut self, name: &str, shape: &[u64]) -> eyre::Result<&'a StTensor> {
        let t = self.st.get(name)?;
        if t.dtype != StDtype::BF16 {
            return Err(eyre!("{name}: dtype {:?}, expected BF16", t.dtype));
        }
        if t.shape != shape {
            return Err(eyre!("{name}: shape {:?}, expected {:?}", t.shape, shape));
        }
        self.n_tensors += 1;
        self.bytes += t.len as usize;
        Ok(t)
    }

    fn bf16_bits(&mut self, name: &str, shape: &[u64]) -> eyre::Result<Vec<u16>> {
        let t = self.tensor(name, shape)?;
        let bytes = self.st.read(t).wrap_err_with(|| format!("read {name}"))?;
        Ok(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    fn f32(&mut self, name: &str, shape: &[u64]) -> eyre::Result<Vec<f32>> {
        Ok(self.bf16_bits(name, shape)?.into_iter().map(bf16_to_f32).collect())
    }

    fn f16(&mut self, name: &str, shape: &[u64]) -> eyre::Result<Vec<u16>> {
        let bits = self.bf16_bits(name, shape)?;
        let mut out = Vec::with_capacity(bits.len());
        for b in bits {
            let x = bf16_to_f32(b);
            let h = f32_to_f16_bits(x);
            if x.is_finite() && (h & 0x7fff) == 0x7c00 {
                self.overflow += 1;
            } else if x != 0.0 && (h & 0x7fff) == 0 {
                self.underflow += 1;
            }
            out.push(h);
        }
        Ok(out)
    }
}

/// Load the V4.1 tower from an opened checkpoint. `config` is
/// `inference/config.json` ([`read_config`]).
pub fn load_host(st: &SafetensorsDir, config: &serde_json::Value) -> eyre::Result<MmprojHost> {
    let text_dim = check_config(config)?;
    let mut rd = Rd { st, overflow: 0, underflow: 0, n_tensors: 0, bytes: 0 };
    let d = VIT_DIM as u64;
    let f = VIT_FFN as u64;
    let td = text_dim as u64;

    let patch_embd_w = rd.f16("vision.patch_embed.proj.weight", &[d, PATCH_ELEMS as u64])?;
    let patch_embd_b = rd.f32("vision.patch_embed.proj.bias", &[d])?;
    let mut blocks = Vec::with_capacity(VIT_N_LAYERS);
    for l in 0..VIT_N_LAYERS {
        let n = |s: &str| format!("vision.blocks.{l}.{s}");
        let qkv_w = rd.f16(&n("attn.wqkv.weight"), &[3 * d, d])?;
        let qkv_b = rd.f32(&n("attn.wqkv.bias"), &[3 * d])?;
        let w1 = rd.f16(&n("mlp.w1.weight"), &[2 * f, d])?;
        let dd = VIT_DIM * VIT_DIM;
        let fd = VIT_FFN * VIT_DIM;
        blocks.push(VitBlockHost {
            ln1: rd.f32(&n("norm1.weight"), &[d])?,
            attn_q_w: qkv_w[..dd].to_vec(),
            attn_q_b: qkv_b[..VIT_DIM].to_vec(),
            attn_k_w: qkv_w[dd..2 * dd].to_vec(),
            attn_k_b: qkv_b[VIT_DIM..2 * VIT_DIM].to_vec(),
            attn_v_w: qkv_w[2 * dd..].to_vec(),
            attn_v_b: qkv_b[2 * VIT_DIM..].to_vec(),
            attn_out_w: rd.f16(&n("attn.wo.weight"), &[d, d])?,
            attn_out_b: rd.f32(&n("attn.wo.bias"), &[d])?,
            ln2: rd.f32(&n("norm2.weight"), &[d])?,
            // `gate, up = w1(x).chunk(2, dim=-1)`: rows [0, ffn) are gate, [ffn, 2ffn) up.
            ffn_gate_w: w1[..fd].to_vec(),
            ffn_up_w: w1[fd..].to_vec(),
            ffn_down_w: rd.f16(&n("mlp.w2.weight"), &[d, f])?,
        });
    }
    let post_ln = rd.f32("vision.norm.weight", &[d])?;
    let mm1_w = rd.f16("aligner.w1.weight", &[td, ALIGNER_IN as u64])?;
    let mm1_b = rd.f32("aligner.w1.bias", &[td])?;
    let mm2_w = rd.f16("aligner.w2.weight", &[td, td])?;
    let mm2_b = rd.f32("aligner.w2.bias", &[td])?;
    let img_start = rd.f32("image_start", &[td])?;
    let image_newline = rd.f32("image_newline", &[td])?;
    let img_end = rd.f32("image_end", &[td])?;

    if rd.overflow > 0 {
        return Err(eyre!(
            "V4.1 vision weights: {} bf16 values exceed the f16 range (|w| > 65504); refusing a lossy load",
            rd.overflow
        ));
    }
    tracing::info!(
        tensors = rd.n_tensors,
        mib = rd.bytes as f64 / (1u64 << 20) as f64,
        f16_underflow = rd.underflow,
        text_dim,
        "V4.1 vision tower read from HF safetensors (bf16 -> f16/f32 at read time)"
    );
    let meta = MmprojMeta {
        projector_type: "deepseek4.1-hf".to_string(),
        n_layers: VIT_N_LAYERS as u32,
        dim: VIT_DIM as u32,
        n_heads: VIT_N_HEADS as u32,
        ffn: VIT_FFN as u32,
        eps: VIT_RMS_EPS,
        patch: PATCH,
        scale_factor: DOWNSAMPLE,
        proj_dim: text_dim as u32,
        min_pixels: VisionCfg::V41.min_pixels,
        image_mean: [crate::IMAGE_MEAN; 3],
        image_std: [crate::IMAGE_STD; 3],
        use_silu: true,
    };
    Ok(MmprojHost {
        meta,
        text_dim,
        patch_embd_w,
        patch_embd_b,
        blocks,
        post_ln,
        mm1_w,
        mm1_b,
        mm2_w,
        mm2_b,
        img_start,
        img_pad: None,
        img_end,
        image_newline,
    })
}

/// [`load_host`] from a snapshot directory (opens the shard headers itself).
pub fn load_host_dir(model_dir: &Path) -> eyre::Result<MmprojHost> {
    let st = SafetensorsDir::open(model_dir).wrap_err_with(|| format!("open V4.1 checkpoint {}", model_dir.display()))?;
    let cfg = read_config(model_dir)?;
    load_host(&st, &cfg)
}
