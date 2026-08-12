//! Per-tensor weight contract for the V4-Flash `het/` engine.
//!
//! The forward pass has no dtype branching at most call sites — each buffer
//! is fed to a kernel that decodes one specific quant format, so a tensor of
//! an unexpected type produces *silent garbage*, not an error (wrong
//! per-expert strides being the worst case; see `het/weights.rs`). This
//! module makes the implicit contract explicit:
//!
//! - [`expectation`] maps a tensor's role (name with `blk.<n>.` collapsed)
//!   to what the engine can execute: either raw quant bytes for a known
//!   kernel ([`Expect::Quant`]) or host conversion to F16 at load
//!   ([`Expect::ToF16`]).
//! - [`validate_model`] walks a GGUF and reports EVERY violation at once,
//!   so pointing the engine at a new quant mix fails with a complete list
//!   instead of the first mismatch (or worse, garbage output).
//! - [`bytes_per_expert`] derives MoE expert strides from the actual dtype
//!   instead of compile-time constants (the unsloth UD mix varies dtype
//!   per layer: blk.26 gate/up are IQ2_S, blk.26/42 down are MXFP4).
//!
//! Roles absent from the table (MTP GGUF tensors, Laguna, ad-hoc test
//! tensors) keep the legacy loader behavior.

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::{Gguf, GgufType};

use crate::config::{N_EMBD, N_FF_EXP, N_VOCAB};

/// What the engine expects of one tensor role.
#[derive(Debug, Clone, Copy)]
pub enum Expect {
    /// Raw bytes passed through to a kernel that decodes one of these
    /// types. (Q8_0 additionally gets the M18 scales/quants repack.)
    Quant(&'static [GgufType]),
    /// Host-converted to F16 at load; these on-disk types are accepted.
    ToF16(&'static [GgufType]),
}

use GgufType::*;

/// Collapse `blk.<n>.` to `blk.N.` so per-layer tensors share one role.
pub fn role_of(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            if rest[..dot].bytes().all(|b| b.is_ascii_digit()) {
                return format!("blk.N.{}", &rest[dot + 1..]);
            }
        }
    }
    name.to_string()
}

/// The engine's expectation for a role, or None if this module doesn't
/// govern it (unknown tensors keep legacy loader behavior; F32/I32 tensors
/// go through `load_f32_weight`/`load_i32_tensor` which self-validate).
///
/// The `Quant` lists enumerate the types the forward pass has *wired
/// kernels for today* — extend a list only together with the dispatch that
/// executes the new type. Phase 2 (unsloth-UD) wired: IQ3_XXS/MXFP4 down,
/// IQ2_S gate/up, Q5_K/Q6_K shexp+q_a, Q4_K head/embd. [`validate_model`]
/// on a file using anything else reports it as the missing-kernel list.
pub fn expectation(role: &str) -> Option<Expect> {
    Some(match role {
        // ---- dense attention / head: kernel-decoded quants ----
        "output.weight" => Expect::Quant(&[Q8_0, Q4_K]),
        "blk.N.attn_q_a.weight" => Expect::Quant(&[Q8_0, Q5_K, Q6_K]),
        "blk.N.attn_q_b.weight"
        | "blk.N.attn_kv.weight"
        | "blk.N.attn_output_a.weight"
        | "blk.N.attn_output_b.weight" => Expect::Quant(&[Q8_0]),

        // ---- MoE ----
        // IQ2_XXS/IQ2_S: antirez + unsloth UD-IQ2_XXS. IQ2_XS: unsloth
        // UD-Q2_K_XL's 42 non-exceptional layers. IQ3_XXS: that mix's
        // blk.26 — the only place IQ3_XXS appears at gate/up rather than
        // down, hence the separate `iq3_xxs_pair` kernel family.
        "blk.N.ffn_gate_exps.weight" | "blk.N.ffn_up_exps.weight" => {
            Expect::Quant(&[IQ2_XXS, IQ2_S, IQ2_XS, IQ3_XXS])
        }
        "blk.N.ffn_down_exps.weight" => Expect::Quant(&[Q2_K, IQ3_XXS, MXFP4]),
        "blk.N.ffn_gate_shexp.weight" | "blk.N.ffn_up_shexp.weight" => {
            Expect::Quant(&[Q8_0, Q5_K, Q6_K])
        }
        "blk.N.ffn_down_shexp.weight" => Expect::Quant(&[Q8_0, Q6_K]),

        // ---- consumed by F16 kernels: convert at load ----
        "blk.N.ffn_gate_inp.weight" => Expect::ToF16(&[F16, BF16, F32]),
        "blk.N.hc_attn_fn.weight" | "blk.N.hc_ffn_fn.weight" | "output_hc_fn.weight" => {
            Expect::ToF16(&[F16, F32])
        }
        "blk.N.attn_compressor_kv.weight"
        | "blk.N.attn_compressor_gate.weight"
        | "blk.N.indexer_compressor_kv.weight"
        | "blk.N.indexer_compressor_gate.weight"
        | "blk.N.indexer.attn_q_b.weight" => Expect::ToF16(&[F16, Q8_0]),
        "blk.N.attn_compressor_ape.weight"
        | "blk.N.indexer_compressor_ape.weight"
        | "blk.N.indexer.proj.weight" => Expect::ToF16(&[F16, F32]),

        _ => return None,
    })
}

/// Types accepted for `token_embd.weight` (host-side embed path, not
/// loaded through `load_to_device` — validated by its consumers; see
/// [`crate::embed::embed_lookup`]).
/// Host-side embedding lookup (deepstrix-server/src/embed.rs) — F16 raw,
/// Q4_K (unsloth UD-IQ2_XXS) and Q5_K (unsloth UD-Q2_K_XL) row-dequant.
pub const TOKEN_EMBD_ALLOWED: &[GgufType] = &[F16, Q4_K, Q5_K];

/// Expected dims for roles whose shapes are pinned by `config.rs`
/// constants (the ones a wrong shape would silently corrupt). Dims are in
/// GGUF order: [K, rows] / [K, rows, experts].
fn expected_dims(role: &str) -> Option<Vec<u64>> {
    let (e, f, v) = (N_EMBD as u64, N_FF_EXP as u64, N_VOCAB as u64);
    Some(match role {
        "output.weight" | "token_embd.weight" => vec![e, v],
        "blk.N.ffn_gate_exps.weight" | "blk.N.ffn_up_exps.weight" => vec![e, f, 256],
        "blk.N.ffn_down_exps.weight" => vec![f, e, 256],
        "blk.N.ffn_gate_shexp.weight" | "blk.N.ffn_up_shexp.weight" => vec![e, f],
        "blk.N.ffn_down_shexp.weight" => vec![f, e],
        "blk.N.attn_q_a.weight" => vec![e, 1024],
        "blk.N.attn_q_b.weight" => vec![1024, 32768],
        "blk.N.attn_kv.weight" => vec![e, 512],
        "blk.N.attn_output_a.weight" => vec![e, 8192],
        "blk.N.attn_output_b.weight" => vec![8192, e],
        "blk.N.ffn_gate_inp.weight" => vec![e, 256],
        _ => return None,
    })
}

/// Bytes of one expert's slice of a 3-D MoE tensor with input dim `k` and
/// `rows` output rows, in dtype `dt`. Replaces the compile-time
/// `BLOCKS_Q8K_* × BLOCK_*_BYTES` stride constants.
pub fn bytes_per_expert(dt: GgufType, k: u64, rows: u64) -> eyre::Result<usize> {
    let (block_elems, block_bytes) = dt
        .block_shape()
        .ok_or_else(|| eyre!("bytes_per_expert: unknown dtype {dt:?}"))?;
    if k % block_elems as u64 != 0 {
        return Err(eyre!(
            "bytes_per_expert: K={k} not a multiple of {}-elem {} blocks",
            block_elems,
            dt.name()
        ));
    }
    Ok((rows as usize) * (k as usize / block_elems as usize) * block_bytes as usize)
}

/// Validate every governed tensor of a model against the contract.
/// Returns Ok(()) or ONE error carrying the full list of violations —
/// the "clean enumerated error list" a new quant mix should fail with
/// until its kernels exist.
pub fn validate_model(gguf: &Gguf) -> eyre::Result<()> {
    let mut violations: Vec<String> = Vec::new();
    for t in gguf.tensors() {
        let role = role_of(&t.name);
        let allowed: &[GgufType] = if role == "token_embd.weight" {
            TOKEN_EMBD_ALLOWED
        } else {
            match expectation(&role) {
                Some(Expect::Quant(a)) | Some(Expect::ToF16(a)) => a,
                None => continue,
            }
        };
        if !allowed.contains(&t.dtype) {
            violations.push(format!(
                "{}: dtype {} not supported (allowed: {})",
                t.name,
                t.dtype.name(),
                allowed
                    .iter()
                    .map(|d| d.name())
                    .collect::<Vec<_>>()
                    .join("|")
            ));
        }
        if let Some(dims) = expected_dims(&role) {
            if t.dims != dims {
                violations.push(format!(
                    "{}: dims {:?} != expected {:?}",
                    t.name, t.dims, dims
                ));
            }
        }
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(eyre!(
            "weight contract: {} violation(s):\n  {}",
            violations.len(),
            violations.join("\n  ")
        ))
    }
}

// ---- host converters (load-time, small tensors only) ----

/// f32 -> f16 bits, round-to-nearest-even, denormal + inf/nan correct.
pub fn f32_to_f16_bits(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x007f_ffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half_mant = (m >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let mut result = sign | half_mant;
        if (m & round_bit) != 0 && ((m & (round_bit - 1)) != 0 || (half_mant & 1) != 0) {
            result += 1;
        }
        return result;
    }
    let half_mant = (mant >> 13) as u16;
    let mut result = sign | ((e as u16) << 10) | half_mant;
    if (mant & 0x0000_1000) != 0 && ((mant & 0x0000_0fff) != 0 || (half_mant & 1) != 0) {
        result += 1;
    }
    result
}

/// Convert an on-disk tensor payload of `dtype` to F16 little-endian bytes.
/// Supports the [`Expect::ToF16`] source types: F16 (passthrough), F32,
/// BF16, Q8_0 (dequant).
pub fn convert_to_f16(dtype: GgufType, src: &[u8], n_elements: u64) -> eyre::Result<Vec<u8>> {
    let n = n_elements as usize;
    match dtype {
        F16 => Ok(src.to_vec()),
        F32 => {
            debug_assert_eq!(src.len(), n * 4);
            let mut out = Vec::with_capacity(n * 2);
            for c in src.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            Ok(out)
        }
        BF16 => {
            debug_assert_eq!(src.len(), n * 2);
            let mut out = Vec::with_capacity(n * 2);
            for c in src.chunks_exact(2) {
                let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            Ok(out)
        }
        Q8_0 => {
            // block: f16 scale + 32 × i8
            debug_assert_eq!(src.len(), n / 32 * 34);
            let mut out = Vec::with_capacity(n * 2);
            for blk in src.chunks_exact(34) {
                let d = crate::iq2_xxs_tables::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &q in &blk[2..34] {
                    let v = d * (q as i8) as f32;
                    out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
                }
            }
            Ok(out)
        }
        other => Err(eyre!("convert_to_f16: unsupported source dtype {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_collapse() {
        assert_eq!(role_of("blk.26.ffn_down_exps.weight"), "blk.N.ffn_down_exps.weight");
        assert_eq!(role_of("blk.7.indexer.attn_q_b.weight"), "blk.N.indexer.attn_q_b.weight");
        assert_eq!(role_of("output.weight"), "output.weight");
        assert_eq!(role_of("blk.x.foo"), "blk.x.foo"); // non-numeric: untouched
    }

    #[test]
    fn f16_roundtrip_and_edges() {
        for v in [0.0f32, -0.0, 1.0, -1.5, 0.333251953125, 65504.0, 6.1e-5, 5.9e-8] {
            let bits = f32_to_f16_bits(v);
            let back = crate::iq2_xxs_tables::f16_to_f32(bits);
            let err = (back - v).abs();
            let tol = (v.abs() * 1e-3).max(1e-7);
            assert!(err <= tol, "{v} -> {bits:#x} -> {back} (err {err})");
        }
        assert_eq!(f32_to_f16_bits(1e6), 0x7c00); // overflow -> inf
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
    }

    #[test]
    fn q8_0_dequant_to_f16() {
        // One block: scale 0.5, quants 0..31
        let mut blk = vec![];
        blk.extend_from_slice(&f32_to_f16_bits(0.5).to_le_bytes());
        blk.extend((0..32u8).map(|i| i as u8));
        let out = convert_to_f16(Q8_0, &blk, 32).unwrap();
        assert_eq!(out.len(), 64);
        let v5 = crate::iq2_xxs_tables::f16_to_f32(u16::from_le_bytes([out[10], out[11]]));
        assert_eq!(v5, 2.5); // 0.5 * 5
    }

    #[test]
    fn bytes_per_expert_matches_legacy_constants() {
        // Legacy: gate/up = N_FF_EXP * (N_EMBD/256) * 66, down = N_EMBD * (N_FF_EXP/256) * 84
        assert_eq!(
            bytes_per_expert(IQ2_XXS, N_EMBD as u64, N_FF_EXP as u64).unwrap(),
            (N_FF_EXP as usize) * (N_EMBD as usize / 256) * 66
        );
        assert_eq!(
            bytes_per_expert(Q2_K, N_FF_EXP as u64, N_EMBD as u64).unwrap(),
            (N_EMBD as usize) * (N_FF_EXP as usize / 256) * 84
        );
        // New types on their actual shapes
        assert_eq!(
            bytes_per_expert(IQ3_XXS, N_FF_EXP as u64, N_EMBD as u64).unwrap(),
            (N_EMBD as usize) * (N_FF_EXP as usize / 256) * 98
        );
        assert_eq!(
            bytes_per_expert(MXFP4, N_FF_EXP as u64, N_EMBD as u64).unwrap(),
            (N_EMBD as usize) * (N_FF_EXP as usize / 32) * 17
        );
    }
}
