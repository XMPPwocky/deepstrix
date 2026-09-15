//! The weight-source seam: the loaders in `v4flash-kernels` read tensors
//! through this enum instead of `&MappedGguf`, so the same code loads a GGUF
//! (V4-Flash, or a llama.cpp-quantised V4.1) or the HF checkpoint view
//! (`V41HfWeights`). Descriptors are `GgufTensor`s for both arms — the HF view
//! synthesises them (shard/offset fields are 0 and never read) — so the loaders'
//! `tensor.dims / dtype / byte_size` code is untouched.
//!
//! An enum rather than a trait object: two implementors, exhaustive matching,
//! and no vtable in the per-expert read loop. It is `Copy` so it moves into
//! `thread::scope` closures freely.

use std::path::Path;

use color_eyre::eyre::{self, eyre};

use crate::gguf::GgufTensor;
use crate::hf_v41::V41HfWeights;
use crate::mapped::MappedGguf;

#[derive(Clone, Copy)]
pub enum WeightSrc<'a> {
    Gguf(&'a MappedGguf),
    V41(&'a V41HfWeights),
}

/// The model geometry a source claims, for the compiled-vs-loaded check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_expert: u32,
}

impl<'a> From<&'a MappedGguf> for WeightSrc<'a> {
    fn from(g: &'a MappedGguf) -> Self {
        Self::Gguf(g)
    }
}

impl<'a> From<&'a V41HfWeights> for WeightSrc<'a> {
    fn from(v: &'a V41HfWeights) -> Self {
        Self::V41(v)
    }
}

impl<'a> WeightSrc<'a> {
    pub fn tensor(&self, name: &str) -> Option<&'a GgufTensor> {
        match *self {
            Self::Gguf(g) => g.gguf().tensor(name),
            Self::V41(v) => v.gguf_tensor(name),
        }
    }

    pub fn tensors(&self) -> &'a [GgufTensor] {
        match *self {
            Self::Gguf(g) => g.gguf().tensors(),
            Self::V41(v) => v.gguf_tensors(),
        }
    }

    pub fn read_tensor(&self, t: &GgufTensor) -> eyre::Result<Vec<u8>> {
        match *self {
            Self::Gguf(g) => g.read_tensor(t),
            Self::V41(v) => v.read(v.get(&t.name)?),
        }
    }

    pub fn read_tensor_into_slice_parallel(&self, t: &GgufTensor, dst: &mut [u8]) -> eyre::Result<()> {
        match *self {
            Self::Gguf(g) => g.read_tensor_into_slice_parallel(t, dst),
            Self::V41(v) => v.read_into(v.get(&t.name)?, dst),
        }
    }

    /// Expert `e` of a stacked `[n_expert, out, in]` tensor; `dst.len()` is the
    /// per-expert byte size (the GGUF arm reads `abs_offset + e * dst.len()`,
    /// exactly the arithmetic the loaders used to do inline).
    pub fn read_expert_into(&self, t: &GgufTensor, e: usize, dst: &mut [u8]) -> eyre::Result<()> {
        match *self {
            Self::Gguf(g) => g.read_range_into(t.shard, t.abs_offset + (e as u64) * (dst.len() as u64), dst),
            Self::V41(v) => v.read_expert_into(v.get(&t.name)?, e, dst),
        }
    }

    /// Read one expert leaving MXFP4 in the **HF** layout (packed nibbles then
    /// e8m0 scales) instead of ggml blocks, so the consumer can permute on the
    /// GPU. Same byte count either way. `Ok(false)` = this source has no such
    /// form (a GGUF is already in ggml layout); the caller must use
    /// [`Self::read_expert_into`] and leave `dst` alone.
    pub fn read_expert_hf_layout(&self, t: &GgufTensor, e: usize, dst: &mut [u8]) -> eyre::Result<bool> {
        match *self {
            Self::Gguf(_) => Ok(false),
            Self::V41(v) => {
                v.read_expert_hf_layout_into(v.get(&t.name)?, e, dst)?;
                Ok(true)
            }
        }
    }

    /// Zero-copy O_DIRECT HF-layout expert read into padded, 4096-aligned
    /// staging. `Ok(None)` = unavailable (GGUF source, or no O_DIRECT handle);
    /// the caller must fall back to [`Self::read_expert_hf_layout`].
    pub fn read_expert_hf_layout_direct(
        &self,
        t: &GgufTensor,
        e: usize,
        dst: &mut [u8],
    ) -> eyre::Result<Option<(usize, usize, u32, u32)>> {
        match *self {
            Self::Gguf(_) => Ok(None),
            Self::V41(v) => v.read_expert_hf_layout_direct(v.get(&t.name)?, e, dst),
        }
    }

    /// Zero-copy O_DIRECT read of ALL THREE roles of one expert in TWO preads.
    /// `Ok(None)` = unavailable (GGUF source, no O_DIRECT handle, or the
    /// checkpoint's expert planes are not contiguous); caller falls back to the
    /// per-role path. See `V41HfWeights::read_expert_runs_direct`.
    pub fn read_expert_runs_direct(
        &self,
        ts: [&GgufTensor; 3],
        e: usize,
        dst_w: &mut [u8],
        dst_s: &mut [u8],
    ) -> eyre::Result<Option<[(usize, usize, u32, u32); 3]>> {
        match *self {
            Self::Gguf(_) => Ok(None),
            Self::V41(v) => {
                let vts = [v.get(&ts[0].name)?, v.get(&ts[1].name)?, v.get(&ts[2].name)?];
                v.read_expert_runs_direct(vts, e, dst_w, dst_s)
            }
        }
    }

    /// Staging bytes [`Self::read_expert_hf_layout_direct`] needs for one role.
    pub fn hf_layout_direct_capacity(&self, t: &GgufTensor) -> eyre::Result<Option<usize>> {
        match *self {
            Self::Gguf(_) => Ok(None),
            Self::V41(v) => v.hf_layout_direct_capacity(v.get(&t.name)?).map(Some),
        }
    }

    /// The model's on-disk location: the GGUF file, or the HF snapshot dir.
    pub fn path(&self) -> &'a Path {
        match *self {
            Self::Gguf(g) => g.path(),
            Self::V41(v) => v.raw().dir(),
        }
    }

    pub fn as_gguf(&self) -> Option<&'a MappedGguf> {
        match *self {
            Self::Gguf(g) => Some(g),
            Self::V41(_) => None,
        }
    }

    /// Geometry the weights were produced for, when the source records it.
    pub fn model_shape(&self) -> eyre::Result<Option<ModelShape>> {
        match *self {
            Self::Gguf(g) => {
                let gg = g.gguf();
                let Some(arch) = gg.architecture() else { return Ok(None) };
                let key = |k: &str| gg.metadata(&format!("{arch}.{k}")).and_then(|v| v.as_u32());
                Ok(match (key("block_count"), key("embedding_length"), key("expert_count")) {
                    (Some(n_layer), Some(n_embd), Some(n_expert)) => Some(ModelShape { n_layer, n_embd, n_expert }),
                    _ => None,
                })
            }
            Self::V41(v) => {
                let c = v.config();
                let get = |k: &str| c[k].as_u64().map(|x| x as u32).ok_or_else(|| eyre!("config: {k} missing"));
                Ok(Some(ModelShape { n_layer: get("n_layers")?, n_embd: get("dim")?, n_expert: get("n_routed_experts")? }))
            }
        }
    }
}
