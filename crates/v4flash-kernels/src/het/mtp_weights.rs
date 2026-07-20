//! MTP (Multi-Token Prediction) weights — V4-Flash speculative draft.
//!
//! Restored (M40) and adapted to the current engine (2026-07). Loaded from
//! a SEPARATE GGUF file (antirez ships
//! `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`). The MTP block is
//! structurally one extra transformer layer (full attn + MoE + head)
//! plus two HC-combining projections (e_proj, h_proj) that fold the
//! last accepted token's embedding into the previous HC state.
//!
//! Differences vs a main-model layer:
//! * No compressor / indexer (MTP processes one token at a time).
//! * Routed experts are **Q4_K** (vs main's iq2_xxs gate/up + q2_k down).
//! * Head weights (hc_head_*) are F32 in the GGUF; we convert hc_head_fn
//!   to F16 at load time so the existing f16_matvec kernel can be reused.
//!
//! Device split:
//! * dGPU: all attention LoRAs, hc projections, shared expert, head
//!   weights, e_proj/h_proj, norms. ~120 MB.
//! * iGPU: routed-MoE experts (gate/up/down × 256 experts × Q4_K) ~1.2 GB.

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_HC, N_HEAD, N_HEAD_DIM, N_LORA_Q};
use crate::model_weights::load_f32_weight;
use crate::rope::RopeParams;
use crate::weights::{load_to_device, DeviceWeight};

/// Routed MoE experts for the MTP layer. Same per-expert dims as the
/// main model (gate/up = [N_EMBD, N_FF_EXP] each, down = [N_FF_EXP,
/// N_EMBD]) but **Q4_K** quantization throughout.
pub struct MtpRoutedExperts {
    pub gate_exps: DeviceWeight,
    pub up_exps: DeviceWeight,
    pub down_exps: DeviceWeight,
}

pub struct MtpWeights {
    // ---- HC combine (dGPU): fold prev HC + new token embedding into mtp_input_hc ----
    /// RMS norm scale for the embedding before e_proj.
    pub enorm: DeviceBuffer<f32>,
    /// Q8_0 [N_EMBD, N_EMBD] — projects normalized embedding.
    pub e_proj: DeviceWeight,
    /// RMS norm scale applied per HC row of prev_hc before h_proj.
    pub hnorm: DeviceBuffer<f32>,
    /// Q8_0 [N_EMBD, N_EMBD] — projects each normalized HC row.
    pub h_proj: DeviceWeight,
    /// RMS norm scale applied after the MTP layer, before the MTP head.
    pub norm: DeviceBuffer<f32>,

    // ---- MTP transformer-layer block (dGPU; same structure as a main layer minus compressor) ----
    pub hc_attn_fn: DeviceWeight,
    pub hc_attn_scale: DeviceBuffer<f32>,
    pub hc_attn_base: DeviceBuffer<f32>,
    pub hc_ffn_fn: DeviceWeight,
    pub hc_ffn_scale: DeviceBuffer<f32>,
    pub hc_ffn_base: DeviceBuffer<f32>,
    pub attn_norm: DeviceBuffer<f32>,
    pub attn_q_a: DeviceWeight,
    pub attn_q_b: DeviceWeight,
    pub q_a_norm: DeviceBuffer<f32>,
    pub attn_kv: DeviceWeight,
    pub kv_a_norm: DeviceBuffer<f32>,
    pub attn_sinks: DeviceBuffer<f32>,
    pub attn_output_a: DeviceWeight,
    pub attn_output_b: DeviceWeight,
    pub rope_params: RopeParams,
    pub ffn_norm: DeviceBuffer<f32>,

    // Shared expert (dGPU, Q8_0)
    pub ffn_gate_shexp: DeviceWeight,
    pub ffn_up_shexp: DeviceWeight,
    pub ffn_down_shexp: DeviceWeight,

    // Router (dGPU — MTP is always a learned router, not hash)
    pub ffn_gate_inp: DeviceWeight,
    pub router_bias_dev: DeviceBuffer<f32>,

    // Routed MoE (iGPU, Q4_K)
    pub routed: MtpRoutedExperts,

    // ---- MTP-specific head (dGPU). Reuses the main model's output /
    // token_embd (vocab projection) — only hc_head_* differ. ----
    /// F16 view of `mtp.0.hc_head_fn.weight` (converted from F32 at load).
    /// Shape [HC_DIM, N_HC]. Backs a F16 matvec for the per-token HC-blend.
    pub hc_head_fn: DeviceWeight,
    pub hc_head_scale: DeviceBuffer<f32>,
    pub hc_head_base: DeviceBuffer<f32>,
}

impl MtpWeights {
    /// Load all MTP weights. `mtp_gguf` must be the V4-Flash MTP GGUF
    /// (separate file from the main model). `rope_params` supplies the MTP
    /// layer's RoPE params — identical to the last main-model layer's,
    /// since MTP shares the same context.
    pub fn load(
        mtp_gguf: &MappedGguf,
        dgpu_device: Device,
        igpu_device: Device,
        rope_params: RopeParams,
    ) -> eyre::Result<Self> {
        let dgpu_id = dgpu_device.id;
        let igpu_id = igpu_device.id;

        dgpu_device.set_current()?;
        let enorm = load_f32_weight(mtp_gguf, "mtp.0.enorm.weight", dgpu_id, N_EMBD as usize)?;
        let e_proj = load_to_device(mtp_gguf, "mtp.0.e_proj.weight", dgpu_id)?;
        let hnorm = load_f32_weight(mtp_gguf, "mtp.0.hnorm.weight", dgpu_id, N_EMBD as usize)?;
        let h_proj = load_to_device(mtp_gguf, "mtp.0.h_proj.weight", dgpu_id)?;
        let norm = load_f32_weight(mtp_gguf, "mtp.0.norm.weight", dgpu_id, N_EMBD as usize)?;

        // hc_attn_fn / hc_ffn_fn are F32 in the MTP file (the main model uses
        // F16). Convert to F16 on load so the existing f16_matvec kernel
        // works. Shape [HC_DIM, HC_MIX_DIM] = [16384, 24].
        let hc_attn_fn = load_f32_as_f16_device_weight(
            mtp_gguf,
            "mtp.0.hc_attn_fn.weight",
            dgpu_id,
            (HC_DIM * HC_MIX_DIM) as usize,
        )?;
        let hc_attn_scale =
            load_f32_weight(mtp_gguf, "mtp.0.hc_attn_scale.weight", dgpu_id, 3)?;
        let hc_attn_base = load_f32_weight(
            mtp_gguf,
            "mtp.0.hc_attn_base.weight",
            dgpu_id,
            HC_MIX_DIM as usize,
        )?;
        let hc_ffn_fn = load_f32_as_f16_device_weight(
            mtp_gguf,
            "mtp.0.hc_ffn_fn.weight",
            dgpu_id,
            (HC_DIM * HC_MIX_DIM) as usize,
        )?;
        let hc_ffn_scale =
            load_f32_weight(mtp_gguf, "mtp.0.hc_ffn_scale.weight", dgpu_id, 3)?;
        let hc_ffn_base = load_f32_weight(
            mtp_gguf,
            "mtp.0.hc_ffn_base.weight",
            dgpu_id,
            HC_MIX_DIM as usize,
        )?;

        let attn_norm =
            load_f32_weight(mtp_gguf, "mtp.0.attn_norm.weight", dgpu_id, N_EMBD as usize)?;
        let attn_q_a = load_to_device(mtp_gguf, "mtp.0.attn_q_a.weight", dgpu_id)?;
        let attn_q_b = load_to_device(mtp_gguf, "mtp.0.attn_q_b.weight", dgpu_id)?;
        let q_a_norm = load_f32_weight(
            mtp_gguf,
            "mtp.0.attn_q_a_norm.weight",
            dgpu_id,
            N_LORA_Q as usize,
        )?;
        let attn_kv = load_to_device(mtp_gguf, "mtp.0.attn_kv.weight", dgpu_id)?;
        let kv_a_norm = load_f32_weight(
            mtp_gguf,
            "mtp.0.attn_kv_a_norm.weight",
            dgpu_id,
            N_HEAD_DIM as usize,
        )?;
        let attn_sinks =
            load_f32_weight(mtp_gguf, "mtp.0.attn_sinks.weight", dgpu_id, N_HEAD as usize)?;
        let attn_output_a = load_to_device(mtp_gguf, "mtp.0.attn_output_a.weight", dgpu_id)?;
        let attn_output_b = load_to_device(mtp_gguf, "mtp.0.attn_output_b.weight", dgpu_id)?;
        let ffn_norm =
            load_f32_weight(mtp_gguf, "mtp.0.ffn_norm.weight", dgpu_id, N_EMBD as usize)?;

        let ffn_gate_shexp = load_to_device(mtp_gguf, "mtp.0.ffn_gate_shexp.weight", dgpu_id)?;
        let ffn_up_shexp = load_to_device(mtp_gguf, "mtp.0.ffn_up_shexp.weight", dgpu_id)?;
        let ffn_down_shexp = load_to_device(mtp_gguf, "mtp.0.ffn_down_shexp.weight", dgpu_id)?;

        // MTP's ffn_gate_inp is F32 [N_EMBD, N_EXPERT] in the GGUF
        // (main model uses F16). Convert on load so the existing
        // f16.matvec can consume it.
        use crate::config::N_EXPERT;
        let ffn_gate_inp = load_f32_as_f16_device_weight(
            mtp_gguf,
            "mtp.0.ffn_gate_inp.weight",
            dgpu_id,
            (N_EMBD * N_EXPERT) as usize,
        )?;
        let router_bias_dev = {
            let t = mtp_gguf
                .gguf()
                .tensor("mtp.0.exp_probs_b.bias")
                .ok_or_else(|| eyre!("missing mtp.0.exp_probs_b.bias"))?;
            if t.dtype != GgufType::F32 {
                return Err(eyre!(
                    "mtp.0.exp_probs_b.bias dtype {:?} != F32",
                    t.dtype
                ));
            }
            let bytes = mtp_gguf.read_tensor(t)?;
            let host: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(dgpu_id, host.len())?;
            buf.copy_from_host(&host)?;
            buf
        };

        // Head weights — F32 in GGUF; convert hc_head_fn to F16 for the
        // existing matvec, keep scale/base as F32.
        let hc_head_fn = load_f32_as_f16_device_weight(
            mtp_gguf,
            "mtp.0.hc_head_fn.weight",
            dgpu_id,
            (HC_DIM * N_HC) as usize,
        )?;
        let hc_head_scale = load_f32_weight(mtp_gguf, "mtp.0.hc_head_scale.weight", dgpu_id, 1)?;
        let hc_head_base =
            load_f32_weight(mtp_gguf, "mtp.0.hc_head_base.weight", dgpu_id, N_HC as usize)?;

        // Routed MoE experts on iGPU (Q4_K).
        igpu_device.set_current()?;
        let routed = MtpRoutedExperts {
            gate_exps: load_to_device(mtp_gguf, "mtp.0.ffn_gate_exps.weight", igpu_id)?,
            up_exps: load_to_device(mtp_gguf, "mtp.0.ffn_up_exps.weight", igpu_id)?,
            down_exps: load_to_device(mtp_gguf, "mtp.0.ffn_down_exps.weight", igpu_id)?,
        };
        dgpu_device.set_current()?;

        Ok(MtpWeights {
            enorm,
            e_proj,
            hnorm,
            h_proj,
            norm,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            hc_ffn_fn,
            hc_ffn_scale,
            hc_ffn_base,
            attn_norm,
            attn_q_a,
            attn_q_b,
            q_a_norm,
            attn_kv,
            kv_a_norm,
            attn_sinks,
            attn_output_a,
            attn_output_b,
            rope_params,
            ffn_norm,
            ffn_gate_shexp,
            ffn_up_shexp,
            ffn_down_shexp,
            ffn_gate_inp,
            router_bias_dev,
            routed,
            hc_head_fn,
            hc_head_scale,
            hc_head_base,
        })
    }
}

/// Read an F32 tensor from the GGUF, convert to F16 (half-precision)
/// on the host, and upload to `device_id` as a raw byte buffer (so the
/// existing `f16_matvec` can consume it via `DeviceWeight`). `expected_len`
/// is the number of scalars (not bytes).
fn load_f32_as_f16_device_weight(
    gguf: &MappedGguf,
    name: &str,
    device_id: i32,
    expected_len: usize,
) -> eyre::Result<DeviceWeight> {
    let t = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("missing tensor {name}"))?;
    if t.dtype != GgufType::F32 {
        return Err(eyre!("{name} dtype {:?} != F32", t.dtype));
    }
    let bytes = gguf.read_tensor(t)?;
    let n_floats = bytes.len() / 4;
    if n_floats != expected_len {
        return Err(eyre!(
            "{name}: have {} f32, expected {}",
            n_floats,
            expected_len
        ));
    }
    let mut f16_bytes = vec![0u8; n_floats * 2];
    for i in 0..n_floats {
        let f = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
        let h = f32_to_f16_bits(f);
        f16_bytes[i * 2] = (h & 0xff) as u8;
        f16_bytes[i * 2 + 1] = (h >> 8) as u8;
    }
    let mut buf: DeviceBuffer<u8> = DeviceBuffer::new(device_id, f16_bytes.len())?;
    buf.copy_from_host(&f16_bytes)?;
    Ok(DeviceWeight {
        buffer: buf,
        n_elements: n_floats as u64,
        dtype: GgufType::F16,
        shape: t.dims.clone(),
    })
}

/// IEEE-754 round-to-nearest-even f32 → f16 (binary16). Handles
/// subnormals, NaN, and infinities. Not the fastest but bit-correct.
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;

    if exp32 == 0xff {
        let m = if frac != 0 { 0x200 } else { 0 };
        return (sign << 15) | 0x7c00 | m;
    }

    let exp16 = exp32 - 127 + 15;

    if exp16 >= 0x1f {
        return (sign << 15) | 0x7c00;
    }

    if exp16 <= 0 {
        if exp16 < -10 {
            return sign << 15;
        }
        let frac_with_hidden = frac | 0x800000;
        let shift = (14 - exp16) as u32;
        let mantissa = (frac_with_hidden >> shift) as u16;
        let round_bit = (frac_with_hidden >> (shift - 1)) & 1;
        let sticky = (frac_with_hidden & ((1u32 << (shift - 1)) - 1)) != 0;
        let mut m = mantissa;
        if round_bit != 0 && (sticky || (m & 1) != 0) {
            m = m.wrapping_add(1);
        }
        return (sign << 15) | m;
    }

    let mantissa = (frac >> 13) as u16;
    let round_bit = (frac >> 12) & 1;
    let sticky = (frac & 0xfff) != 0;
    let mut m = mantissa;
    let mut e = exp16 as u16;
    if round_bit != 0 && (sticky || (m & 1) != 0) {
        m = m.wrapping_add(1);
        if m == 0x400 {
            m = 0;
            e += 1;
            if e == 0x1f {
                return (sign << 15) | 0x7c00;
            }
        }
    }
    (sign << 15) | (e << 10) | m
}
