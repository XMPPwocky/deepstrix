//! Typed host-side loader for `mmproj-F16.gguf` (arch `clip`,
//! `clip.projector_type = deepseek4v`, 427 tensors).
//!
//! Layout conventions (GGUF `dims` are `ne[0]` = innermost first):
//! * a linear `y = W x + b` with W stored as `[in, out]` in GGUF dims is
//!   row-major `[out][in]` in memory — `W[o*in + i]`;
//! * `v.patch_embd.weight` dims `[14,14,3,1024]` → row-major
//!   `[1024][3][14][14]`, i.e. `[out][c][y][x]`, matching the patch
//!   flattening order `(c, y, x)` (see `preprocess::patchify`) so it acts
//!   as a plain `[1024][588]` linear;
//! * `ffn_gate`/`ffn_up` are `[out=2816][in=1024]`, `ffn_down` `[1024][2816]`
//!   (SwiGLU, no biases; reference `w1` = cat(gate, up), `w2` = down);
//! * `mm.1` `[4096][9216]` (+bias), GELU(erf), `mm.2` `[4096][4096]` (+bias);
//! * `ln1`/`ln2`/`post_ln` are RMSNorm weights, eps 1e-6.
//! * `v.token_embd.img_{start,pad,end}`, `v.image_newline`: `[4096]` f32.
//!
//! f16 tensors are kept as raw `u16` bit patterns (`v4flash_core::kquants::
//! f16_to_f32` converts).

use std::path::Path;

use color_eyre::eyre::{self, eyre, WrapErr};
use v4flash_core::{GgufTensor, GgufType, MappedGguf};

use crate::{ALIGNER_IN, PATCH, PATCH_ELEMS, TEXT_DIM, VIT_DIM, VIT_FFN, VIT_N_HEADS, VIT_N_LAYERS, VIT_RMS_EPS};

/// Metadata read from the mmproj header (validated against the crate constants).
#[derive(Debug, Clone)]
pub struct MmprojMeta {
    pub projector_type: String,
    pub n_layers: u32,
    pub dim: u32,
    pub n_heads: u32,
    pub ffn: u32,
    pub eps: f32,
    pub patch: u32,
    pub scale_factor: u32,
    pub proj_dim: u32,
    pub min_pixels: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub use_silu: bool,
}

/// One ViT block. Weights `[out][in]` row-major as f16 bits; biases f32.
#[derive(Debug, Clone)]
pub struct VitBlockHost {
    pub ln1: Vec<f32>,        // [1024]
    pub attn_q_w: Vec<u16>,   // [1024][1024]
    pub attn_q_b: Vec<f32>,   // [1024]
    pub attn_k_w: Vec<u16>,
    pub attn_k_b: Vec<f32>,
    pub attn_v_w: Vec<u16>,
    pub attn_v_b: Vec<f32>,
    pub attn_out_w: Vec<u16>, // [1024][1024]
    pub attn_out_b: Vec<f32>,
    pub ln2: Vec<f32>,        // [1024]
    pub ffn_gate_w: Vec<u16>, // [2816][1024]
    pub ffn_up_w: Vec<u16>,   // [2816][1024]
    pub ffn_down_w: Vec<u16>, // [1024][2816]
}

/// Whole mmproj on the host.
#[derive(Debug, Clone)]
pub struct MmprojHost {
    pub meta: MmprojMeta,
    pub patch_embd_w: Vec<u16>, // [1024][588]
    pub patch_embd_b: Vec<f32>, // [1024]
    pub blocks: Vec<VitBlockHost>,
    pub post_ln: Vec<f32>, // [1024]
    pub mm1_w: Vec<u16>,   // [4096][9216]
    pub mm1_b: Vec<f32>,   // [4096]
    pub mm2_w: Vec<u16>,   // [4096][4096]
    pub mm2_b: Vec<f32>,   // [4096]
    pub img_start: Vec<f32>,     // [4096]
    pub img_pad: Vec<f32>,       // [4096]
    pub img_end: Vec<f32>,       // [4096]
    pub image_newline: Vec<f32>, // [4096]
}

struct Loader<'a> {
    m: &'a MappedGguf,
    seen: Vec<bool>,
    scratch: Vec<u8>,
}

impl<'a> Loader<'a> {
    fn tensor(&mut self, name: &str, dtype: GgufType, dims: &[u64]) -> eyre::Result<&GgufTensor> {
        let g = self.m.gguf();
        let idx = g
            .tensors()
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| eyre!("mmproj: tensor `{name}` missing"))?;
        let t = &g.tensors()[idx];
        if t.dtype != dtype {
            return Err(eyre!("mmproj: `{name}` dtype {:?}, expected {:?}", t.dtype, dtype));
        }
        if t.dims != dims {
            return Err(eyre!("mmproj: `{name}` dims {:?}, expected {:?}", t.dims, dims));
        }
        self.seen[idx] = true;
        Ok(t)
    }

    fn f16(&mut self, name: &str, dims: &[u64]) -> eyre::Result<Vec<u16>> {
        let t = self.tensor(name, GgufType::F16, dims)?.clone();
        self.m.read_tensor_into(&t, &mut self.scratch).wrap_err_with(|| format!("read {name}"))?;
        Ok(self.scratch.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    fn f32(&mut self, name: &str, dims: &[u64]) -> eyre::Result<Vec<f32>> {
        let t = self.tensor(name, GgufType::F32, dims)?.clone();
        self.m.read_tensor_into(&t, &mut self.scratch).wrap_err_with(|| format!("read {name}"))?;
        Ok(self.scratch.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

fn meta_u32(m: &MappedGguf, key: &str) -> eyre::Result<u32> {
    m.gguf().metadata(key).and_then(|v| v.as_u32()).ok_or_else(|| eyre!("mmproj: metadata `{key}` missing/not u32"))
}
fn meta_f32(m: &MappedGguf, key: &str) -> eyre::Result<f32> {
    m.gguf().metadata(key).and_then(|v| v.as_f32()).ok_or_else(|| eyre!("mmproj: metadata `{key}` missing/not f32"))
}
fn meta_f32x3(m: &MappedGguf, key: &str) -> eyre::Result<[f32; 3]> {
    match m.gguf().metadata(key) {
        Some(v4flash_core::GgufValue::Array(a)) => {
            let s = a.as_f32s().ok_or_else(|| eyre!("mmproj: `{key}` not f32 array"))?;
            if s.len() != 3 {
                return Err(eyre!("mmproj: `{key}` len {}, expected 3", s.len()));
            }
            Ok([s[0], s[1], s[2]])
        }
        _ => Err(eyre!("mmproj: metadata `{key}` missing")),
    }
}

/// Parse + validate the header only (cheap; no tensor bytes read).
pub fn read_meta(m: &MappedGguf) -> eyre::Result<MmprojMeta> {
    let g = m.gguf();
    let arch = g.architecture().unwrap_or("");
    if arch != "clip" {
        return Err(eyre!("mmproj: general.architecture = `{arch}`, expected `clip`"));
    }
    let projector_type = g
        .metadata("clip.projector_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| eyre!("mmproj: clip.projector_type missing"))?
        .to_string();
    if projector_type != "deepseek4v" {
        return Err(eyre!("mmproj: projector_type `{projector_type}`, expected `deepseek4v`"));
    }
    let meta = MmprojMeta {
        projector_type,
        n_layers: meta_u32(m, "clip.vision.block_count")?,
        dim: meta_u32(m, "clip.vision.embedding_length")?,
        n_heads: meta_u32(m, "clip.vision.attention.head_count")?,
        ffn: meta_u32(m, "clip.vision.feed_forward_length")?,
        eps: meta_f32(m, "clip.vision.attention.layer_norm_epsilon")?,
        patch: meta_u32(m, "clip.vision.patch_size")?,
        scale_factor: meta_u32(m, "clip.vision.projector.scale_factor")?,
        proj_dim: meta_u32(m, "clip.vision.projection_dim")?,
        min_pixels: meta_u32(m, "clip.vision.image_min_pixels")?,
        image_mean: meta_f32x3(m, "clip.vision.image_mean")?,
        image_std: meta_f32x3(m, "clip.vision.image_std")?,
        use_silu: g.metadata("clip.use_silu").and_then(|v| v.as_bool()).unwrap_or(false),
    };
    let checks: [(&str, u64, u64); 7] = [
        ("block_count", meta.n_layers as u64, VIT_N_LAYERS as u64),
        ("embedding_length", meta.dim as u64, VIT_DIM as u64),
        ("head_count", meta.n_heads as u64, VIT_N_HEADS as u64),
        ("feed_forward_length", meta.ffn as u64, VIT_FFN as u64),
        ("patch_size", meta.patch as u64, PATCH as u64),
        ("scale_factor", meta.scale_factor as u64, crate::DOWNSAMPLE as u64),
        ("projection_dim", meta.proj_dim as u64, TEXT_DIM as u64),
    ];
    for (k, got, want) in checks {
        if got != want {
            return Err(eyre!("mmproj: clip.vision.{k} = {got}, this crate is built for {want}"));
        }
    }
    if (meta.eps - VIT_RMS_EPS).abs() > 1e-12 {
        return Err(eyre!("mmproj: layer_norm_epsilon = {}, expected {VIT_RMS_EPS}", meta.eps));
    }
    if meta.min_pixels != crate::MIN_PIXELS {
        return Err(eyre!("mmproj: image_min_pixels = {}, expected {}", meta.min_pixels, crate::MIN_PIXELS));
    }
    if meta.image_mean != [crate::IMAGE_MEAN; 3] || meta.image_std != [crate::IMAGE_STD; 3] {
        return Err(eyre!("mmproj: image_mean/std {:?}/{:?} not 0.5", meta.image_mean, meta.image_std));
    }
    // `vit_swiglu_f16` hardcodes SiLU (`g/(1+exp(-g))*u`); a checkpoint
    // declaring a different FFN activation would be silently
    // mis-activated. Every other metadata field the crate depends on is
    // checked, so check this one too.
    if !meta.use_silu {
        return Err(eyre!(
            "mmproj: clip.use_silu is false (or absent); this crate's ViT FFN is SiLU-only"
        ));
    }
    Ok(meta)
}

impl MmprojHost {
    /// Load every tensor into host memory (~890 MiB f16 + ~1 MiB f32).
    /// Errors on any missing / mis-shaped / unexpected tensor.
    pub fn load(path: &Path) -> eyre::Result<MmprojHost> {
        let m = MappedGguf::open(path).wrap_err_with(|| format!("open mmproj {}", path.display()))?;
        Self::load_from(&m)
    }

    pub fn load_from(m: &MappedGguf) -> eyre::Result<MmprojHost> {
        let meta = read_meta(m)?;
        let n_t = m.gguf().tensors().len();
        let mut ld = Loader { m, seen: vec![false; n_t], scratch: Vec::new() };
        let d = VIT_DIM as u64;
        let f = VIT_FFN as u64;
        let p = PATCH as u64;
        let td = TEXT_DIM as u64;

        let patch_embd_w = ld.f16("v.patch_embd.weight", &[p, p, 3, d])?;
        debug_assert_eq!(patch_embd_w.len(), VIT_DIM * PATCH_ELEMS);
        let patch_embd_b = ld.f32("v.patch_embd.bias", &[d])?;
        let mut blocks = Vec::with_capacity(VIT_N_LAYERS);
        for l in 0..VIT_N_LAYERS {
            let n = |s: &str| format!("v.blk.{l}.{s}");
            blocks.push(VitBlockHost {
                ln1: ld.f32(&n("ln1.weight"), &[d])?,
                attn_q_w: ld.f16(&n("attn_q.weight"), &[d, d])?,
                attn_q_b: ld.f32(&n("attn_q.bias"), &[d])?,
                attn_k_w: ld.f16(&n("attn_k.weight"), &[d, d])?,
                attn_k_b: ld.f32(&n("attn_k.bias"), &[d])?,
                attn_v_w: ld.f16(&n("attn_v.weight"), &[d, d])?,
                attn_v_b: ld.f32(&n("attn_v.bias"), &[d])?,
                attn_out_w: ld.f16(&n("attn_out.weight"), &[d, d])?,
                attn_out_b: ld.f32(&n("attn_out.bias"), &[d])?,
                ln2: ld.f32(&n("ln2.weight"), &[d])?,
                ffn_gate_w: ld.f16(&n("ffn_gate.weight"), &[d, f])?,
                ffn_up_w: ld.f16(&n("ffn_up.weight"), &[d, f])?,
                ffn_down_w: ld.f16(&n("ffn_down.weight"), &[f, d])?,
            });
        }
        let post_ln = ld.f32("v.post_ln.weight", &[d])?;
        let mm1_w = ld.f16("mm.1.weight", &[ALIGNER_IN as u64, td])?;
        let mm1_b = ld.f32("mm.1.bias", &[td])?;
        let mm2_w = ld.f16("mm.2.weight", &[td, td])?;
        let mm2_b = ld.f32("mm.2.bias", &[td])?;
        let img_start = ld.f32("v.token_embd.img_start", &[td])?;
        let img_pad = ld.f32("v.token_embd.img_pad", &[td])?;
        let img_end = ld.f32("v.token_embd.img_end", &[td])?;
        let image_newline = ld.f32("v.image_newline", &[td])?;

        let unseen: Vec<&str> = ld
            .seen
            .iter()
            .enumerate()
            .filter(|(_, s)| !**s)
            .map(|(i, _)| m.gguf().tensors()[i].name.as_str())
            .collect();
        if !unseen.is_empty() {
            return Err(eyre!("mmproj: {} unexpected tensor(s): {:?}", unseen.len(), unseen));
        }
        Ok(MmprojHost {
            meta,
            patch_embd_w,
            patch_embd_b,
            blocks,
            post_ln,
            mm1_w,
            mm1_b,
            mm2_w,
            mm2_b,
            img_start,
            img_pad,
            img_end,
            image_newline,
        })
    }

    /// Bytes of f16 weights (what a straight f16 device upload costs).
    pub fn f16_bytes(&self) -> usize {
        let blk: usize = self
            .blocks
            .iter()
            .map(|b| {
                (b.attn_q_w.len() + b.attn_k_w.len() + b.attn_v_w.len() + b.attn_out_w.len() + b.ffn_gate_w.len() + b.ffn_up_w.len() + b.ffn_down_w.len()) * 2
            })
            .sum();
        blk + (self.patch_embd_w.len() + self.mm1_w.len() + self.mm2_w.len()) * 2
    }

    /// Sentinel vector for a block token type (`TokenType` as u8); `None` for IMAGE.
    pub fn sentinel(&self, ty: u8) -> Option<&[f32]> {
        match crate::TokenType::from_u8(ty)? {
            crate::TokenType::Start => Some(&self.img_start),
            crate::TokenType::Pad => Some(&self.img_pad),
            crate::TokenType::Image => None,
            crate::TokenType::NewLine => Some(&self.image_newline),
            crate::TokenType::End => Some(&self.img_end),
        }
    }
}
