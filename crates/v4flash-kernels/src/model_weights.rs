//! Per-layer weight schemas + their GGUF loaders.
//!
//! These structs describe the *shape* of each per-layer weight bundle
//! (compressor / indexer / shared expert / routed experts). The dGPU/iGPU
//! split and the `HetModelWeights::load_all` orchestrator live in
//! [`super::het::weights`] — this file is just the data definitions plus
//! the small GGUF read helpers used by that loader.
//!
//! Distinct from [`super::weights`]: that module is the generic
//! "GGUF tensor → `DeviceBuffer<u8>`" loader (`load_to_device` /
//! `DeviceWeight`); this module is the V4-Flash-specific schema.

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::DeviceBuffer;

use crate::weights::DeviceWeight;

pub struct CompressorWeights {
    pub wkv: DeviceWeight,       // F16 [n_embd, comp_width]
    pub wgate: DeviceWeight,     // F16 [n_embd, comp_width]
    pub ape: DeviceWeight,       // F16 [comp_width, ratio]
    pub norm: DeviceBuffer<f32>, // F32 [head_dim]
    pub width: u32,
    pub head_dim: u32,
}

pub struct SharedExpertWeights {
    pub gate: DeviceWeight,
    pub up: DeviceWeight,
    pub down: DeviceWeight,
}

/// Per-layer routed-expert weights, iGPU-resident. ~1.2 GiB/layer × 43
/// layers = ~52 GiB total — fits the 88 GiB budget enabled by
/// `amdgpu.no_system_mem_limit=1`. The per-expert byte slice is
/// addressed via pointer-offset on the kernel launch (see
/// `Iq2XxsPairMatvec::launch_with_offsets` and
/// `Q2KAccumulateMatvec::launch_with_offset`); no per-token host→device
/// upload (that was too slow — see feedback memory).
pub struct RoutedExpertWeights {
    pub gate: DeviceWeight,
    pub up: DeviceWeight,
    pub down: DeviceWeight,
    pub gate_bytes_per_expert: usize,
    pub up_bytes_per_expert: usize,
    pub down_bytes_per_expert: usize,
}

/// Load a named F32 tensor from GGUF into a fresh per-device buffer.
/// Validates dtype + byte count against `expected_len` elements.
pub fn load_f32_weight(
    gguf: &MappedGguf,
    name: &str,
    device_id: i32,
    expected_len: usize,
) -> eyre::Result<DeviceBuffer<f32>> {
    let t = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor `{name}` missing"))?;
    if t.dtype != GgufType::F32 {
        return Err(eyre!("{name}: dtype {:?} != F32", t.dtype));
    }
    let bytes = gguf.read_tensor(t)?;
    if bytes.len() != expected_len * 4 {
        return Err(eyre!(
            "{name}: have {} bytes, expected {}",
            bytes.len(),
            expected_len * 4
        ));
    }
    let mut v = vec![0f32; expected_len];
    for (i, c) in bytes.chunks_exact(4).enumerate() {
        v[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(device_id, expected_len)?;
    buf.copy_from_host(&v)?;
    Ok(buf)
}

/// Load a named I32 tensor from GGUF into a host-side `Vec<i32>`.
pub fn load_i32_tensor(gguf: &MappedGguf, name: &str) -> eyre::Result<Vec<i32>> {
    let t = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor `{name}` missing"))?;
    if t.dtype != GgufType::I32 {
        return Err(eyre!("{name}: dtype {:?} != I32", t.dtype));
    }
    let bytes = gguf.read_tensor(t)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
