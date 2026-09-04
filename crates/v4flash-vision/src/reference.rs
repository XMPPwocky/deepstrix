//! Naive f32 CPU reference of the ViT + aligner forward (same math as
//! `inference/vision.py`), used by the GPU oracle tests. Weights are the
//! f16 mmproj tensors converted to f32; activations stay f32 throughout.
//! GEMMs are parallelised over output rows with `std::thread::scope`.

use v4flash_core::kquants::{f16_to_f32, f32_to_f16_bits};

use crate::mmproj::MmprojHost;
use crate::rope::{apply_rotary_host, vision_cos_sin};
use crate::{ALIGNER_IN, PATCH_ELEMS, TEXT_DIM, VIT_DIM, VIT_FFN, VIT_HEAD_DIM, VIT_N_HEADS, VIT_RMS_EPS};

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32)
}

/// `out[n][M] = x[n][K] · W[M][K]^T + bias`, W as f16 bits.
pub fn linear(x: &[f32], n: usize, k: usize, w: &[u16], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(x.len(), n * k);
    assert_eq!(w.len(), m * k);
    let wf: Vec<f32> = w.iter().map(|&b| f16_to_f32(b)).collect();
    let mut out = vec![0f32; n * m];
    let nt = threads().min(n.max(1));
    let rows_per = n.div_ceil(nt);
    std::thread::scope(|s| {
        for (ti, chunk) in out.chunks_mut(rows_per * m).enumerate() {
            let wf = &wf;
            let x = &x[ti * rows_per * k..];
            s.spawn(move || {
                for (r, orow) in chunk.chunks_mut(m).enumerate() {
                    let xr = &x[r * k..(r + 1) * k];
                    for (mi, o) in orow.iter_mut().enumerate() {
                        let wr = &wf[mi * k..(mi + 1) * k];
                        let mut acc = 0f32;
                        for i in 0..k {
                            acc += xr[i] * wr[i];
                        }
                        *o = acc + bias.map(|b| b[mi]).unwrap_or(0.0);
                    }
                }
            });
        }
    });
    out
}

pub fn rms_norm(x: &[f32], n: usize, dim: usize, w: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; n * dim];
    for t in 0..n {
        let xr = &x[t * dim..(t + 1) * dim];
        let ss: f32 = xr.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let s = 1.0 / (ss + VIT_RMS_EPS).sqrt();
        for i in 0..dim {
            out[t * dim + i] = w[i] * (xr[i] * s);
        }
    }
    out
}

/// Full bidirectional MHA on `[n][16][64]` q/k/v, scale 1/8.
pub fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize) -> Vec<f32> {
    let hd = VIT_HEAD_DIM;
    let nh = VIT_N_HEADS;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; n * nh * hd];
    let nt = threads().min(nh);
    std::thread::scope(|s| {
        let heads_per = nh.div_ceil(nt);
        let out_ptr = out.as_mut_ptr() as usize;
        for ti in 0..nt {
            s.spawn(move || {
                let mut scores = vec![0f32; n];
                for h in ti * heads_per..((ti + 1) * heads_per).min(nh) {
                    for i in 0..n {
                        let qi = &q[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                        let mut mx = f32::NEG_INFINITY;
                        for j in 0..n {
                            let kj = &k[(j * nh + h) * hd..(j * nh + h + 1) * hd];
                            let mut d = 0f32;
                            for e in 0..hd {
                                d += qi[e] * kj[e];
                            }
                            scores[j] = d * scale;
                            mx = mx.max(scores[j]);
                        }
                        let mut l = 0f32;
                        for sc in scores.iter_mut() {
                            *sc = (*sc - mx).exp();
                            l += *sc;
                        }
                        let mut o = [0f32; VIT_HEAD_DIM];
                        for j in 0..n {
                            let p = scores[j] / l;
                            let vj = &v[(j * nh + h) * hd..(j * nh + h + 1) * hd];
                            for e in 0..hd {
                                o[e] += p * vj[e];
                            }
                        }
                        // SAFETY: each (i, h) slot is written by exactly one thread.
                        unsafe {
                            let dst = (out_ptr as *mut f32).add((i * nh + h) * hd);
                            std::ptr::copy_nonoverlapping(o.as_ptr(), dst, hd);
                        }
                    }
                }
            });
        }
    });
    out
}

pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// erf via f64 series/continued fraction (|err| < 1e-7 rel to libm).
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

/// Precision of the *activations* handed between stages. The GPU tower is
/// [`ActPrec::F16`]; DeepSeek's `vision.py` runs the whole ViT in
/// [`ActPrec::BF16`]; [`ActPrec::F32`] is the oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActPrec {
    F32,
    F16,
    Bf16,
}

impl ActPrec {
    fn round(self, v: &mut [f32]) {
        match self {
            ActPrec::F32 => {}
            ActPrec::F16 => {
                for x in v.iter_mut() {
                    *x = f16_to_f32(f32_to_f16_bits(*x));
                }
            }
            ActPrec::Bf16 => {
                for x in v.iter_mut() {
                    *x = crate::preprocess::bf16_round(*x);
                }
            }
        }
    }
}

/// ViT trunk residual stream after `n_layers` blocks (no post_ln), with
/// activations rounded to `prec` at exactly the dtype boundaries the GPU
/// pipeline has (rmsnorm out, q/k/v after RoPE, attention out, swiglu out).
/// The residual stream itself stays f32, as on the GPU.
pub fn vit_trunk_x_prec(
    host: &MmprojHost,
    patches: &[f32],
    n_h: usize,
    n_w: usize,
    n_layers: usize,
    prec: ActPrec,
) -> Vec<f32> {
    let n = n_h * n_w;
    assert_eq!(patches.len(), n * PATCH_ELEMS);
    let mut p = patches.to_vec();
    prec.round(&mut p);
    let mut x = linear(&p, n, PATCH_ELEMS, &host.patch_embd_w, VIT_DIM, Some(&host.patch_embd_b));
    let (cos, sin) = vision_cos_sin(n_h as u32, n_w as u32);
    for blk in host.blocks.iter().take(n_layers) {
        let mut h = rms_norm(&x, n, VIT_DIM, &blk.ln1);
        prec.round(&mut h);
        let mut q = linear(&h, n, VIT_DIM, &blk.attn_q_w, VIT_DIM, Some(&blk.attn_q_b));
        let mut k = linear(&h, n, VIT_DIM, &blk.attn_k_w, VIT_DIM, Some(&blk.attn_k_b));
        let mut v = linear(&h, n, VIT_DIM, &blk.attn_v_w, VIT_DIM, Some(&blk.attn_v_b));
        apply_rotary_host(&mut q, &cos, &sin);
        apply_rotary_host(&mut k, &cos, &sin);
        prec.round(&mut q);
        prec.round(&mut k);
        prec.round(&mut v);
        let mut o = attention(&q, &k, &v, n);
        prec.round(&mut o);
        let o = linear(&o, n, VIT_DIM, &blk.attn_out_w, VIT_DIM, Some(&blk.attn_out_b));
        for (xi, oi) in x.iter_mut().zip(&o) {
            *xi += oi;
        }
        let mut h = rms_norm(&x, n, VIT_DIM, &blk.ln2);
        prec.round(&mut h);
        // UNPROVEN ASSUMPTION — the one thing this oracle cannot catch.
        //
        // The HF checkpoint has ONE fused `vision.blocks.N.mlp.w1.weight`
        // and the reference does `gate, up = self.w1(x).chunk(2, -1)`.
        // The converter split it into `ffn_gate` + `ffn_up`, and nothing
        // local records which half went where. This CPU twin makes the
        // same choice as the GPU path (`Tower::upload_block` concatenates
        // gate‖up, `vit_swiglu_f16` silu's the first half), so a shared
        // misreading passes `tests/tower_encode.rs` at 1e-3 — a swap is
        // SILENT under the whole test suite.
        //
        // Circumstantial support for the current orientation: in
        // mmproj-F16.gguf the two slices are byte-contiguous in emission
        // order (v.blk.0.ffn_gate at 117555200, ffn_up at +5767168 =
        // 2816*1024*2), as are attn_q/attn_k/attn_v at +2097152 each —
        // i.e. the writer emitted (first slice -> gate, second -> up).
        // Consistent, not proof.
        //
        // To settle it: diff `Tower::encode_rows` against llama.cpp's
        // `deepseek4v` clip graph on one image, or check the converter
        // script. An end-to-end caption A/B (gate/up swapped vs not) also
        // separates them; a swapped SwiGLU is not subtly wrong.
        let g = linear(&h, n, VIT_DIM, &blk.ffn_gate_w, VIT_FFN, None);
        let u = linear(&h, n, VIT_DIM, &blk.ffn_up_w, VIT_FFN, None);
        let mut a: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
        prec.round(&mut a);
        let d = linear(&a, n, VIT_FFN, &blk.ffn_down_w, VIT_DIM, None);
        for (xi, di) in x.iter_mut().zip(&d) {
            *xi += di;
        }
    }
    x
}

/// ViT trunk residual stream after `n_layers` blocks (no post_ln), f32
/// activations. Used by the layer-bisect oracle; `n_layers = 0` returns
/// the patch embedding.
pub fn vit_trunk_x(host: &MmprojHost, patches: &[f32], n_h: usize, n_w: usize, n_layers: usize) -> Vec<f32> {
    let n = n_h * n_w;
    assert_eq!(patches.len(), n * PATCH_ELEMS);
    let mut x = linear(patches, n, PATCH_ELEMS, &host.patch_embd_w, VIT_DIM, Some(&host.patch_embd_b));
    let (cos, sin) = vision_cos_sin(n_h as u32, n_w as u32);
    for blk in host.blocks.iter().take(n_layers) {
        let h = rms_norm(&x, n, VIT_DIM, &blk.ln1);
        let mut q = linear(&h, n, VIT_DIM, &blk.attn_q_w, VIT_DIM, Some(&blk.attn_q_b));
        let mut k = linear(&h, n, VIT_DIM, &blk.attn_k_w, VIT_DIM, Some(&blk.attn_k_b));
        let v = linear(&h, n, VIT_DIM, &blk.attn_v_w, VIT_DIM, Some(&blk.attn_v_b));
        apply_rotary_host(&mut q, &cos, &sin);
        apply_rotary_host(&mut k, &cos, &sin);
        let o = attention(&q, &k, &v, n);
        let o = linear(&o, n, VIT_DIM, &blk.attn_out_w, VIT_DIM, Some(&blk.attn_out_b));
        for (xi, oi) in x.iter_mut().zip(&o) {
            *xi += oi;
        }
        let h = rms_norm(&x, n, VIT_DIM, &blk.ln2);
        let g = linear(&h, n, VIT_DIM, &blk.ffn_gate_w, VIT_FFN, None);
        let u = linear(&h, n, VIT_DIM, &blk.ffn_up_w, VIT_FFN, None);
        let a: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
        let d = linear(&a, n, VIT_FFN, &blk.ffn_down_w, VIT_DIM, None);
        for (xi, di) in x.iter_mut().zip(&d) {
            *xi += di;
        }
    }
    x
}

/// ViT trunk: patches `[n][588]` → post-norm hidden `[n][1024]`.
pub fn vit_forward(host: &MmprojHost, patches: &[f32], n_h: usize, n_w: usize) -> Vec<f32> {
    let x = vit_trunk_x(host, patches, n_h, n_w, host.blocks.len());
    rms_norm(&x, n_h * n_w, VIT_DIM, &host.post_ln)
}

/// Aligner unfold: `[n_h*n_w][1024]` → `[n_llm][9216]` (feature = c*9 + ky*3 + kx).
pub fn unfold(x: &[f32], n_h: usize, n_w: usize) -> (Vec<f32>, usize, usize) {
    let lh = n_h.div_ceil(3);
    let lw = n_w.div_ceil(3);
    let mut out = vec![0f32; lh * lw * ALIGNER_IN];
    for ty in 0..lh {
        for tx in 0..lw {
            let o = &mut out[(ty * lw + tx) * ALIGNER_IN..(ty * lw + tx + 1) * ALIGNER_IN];
            for c in 0..VIT_DIM {
                for ky in 0..3 {
                    for kx in 0..3 {
                        let (y, xx) = (ty * 3 + ky, tx * 3 + kx);
                        if y < n_h && xx < n_w {
                            o[c * 9 + ky * 3 + kx] = x[(y * n_w + xx) * VIT_DIM + c];
                        }
                    }
                }
            }
        }
    }
    (out, lh, lw)
}

/// Aligner with activations rounded to `prec` at the GPU's dtype
/// boundaries (unfold input, GELU output).
pub fn aligner_forward_prec(host: &MmprojHost, hidden: &[f32], n_h: usize, n_w: usize, prec: ActPrec) -> Vec<f32> {
    let (mut u, lh, lw) = unfold(hidden, n_h, n_w);
    prec.round(&mut u);
    let n_llm = lh * lw;
    let mut a = linear(&u, n_llm, ALIGNER_IN, &host.mm1_w, TEXT_DIM, Some(&host.mm1_b));
    for v in a.iter_mut() {
        *v = gelu_erf(*v);
    }
    prec.round(&mut a);
    linear(&a, n_llm, TEXT_DIM, &host.mm2_w, TEXT_DIM, Some(&host.mm2_b))
}

/// Aligner: post-norm hidden → `[n_llm_h*n_llm_w][4096]`, f32 activations.
pub fn aligner_forward(host: &MmprojHost, hidden: &[f32], n_h: usize, n_w: usize) -> Vec<f32> {
    aligner_forward_prec(host, hidden, n_h, n_w, ActPrec::F32)
}

/// Whole tower with activations at `prec`. `ActPrec::F16` is the exact twin
/// of the GPU pipeline (same dtype boundaries) and is what the GPU oracle
/// asserts against; `ActPrec::Bf16` is what `vision.py` actually runs.
pub fn tower_forward_prec(host: &MmprojHost, patches: &[f32], n_h: usize, n_w: usize, prec: ActPrec) -> Vec<f32> {
    let x = vit_trunk_x_prec(host, patches, n_h, n_w, host.blocks.len(), prec);
    let mut hidden = rms_norm(&x, n_h * n_w, VIT_DIM, &host.post_ln);
    prec.round(&mut hidden);
    aligner_forward_prec(host, &hidden, n_h, n_w, prec)
}

/// Whole tower: patches → aligner rows `[n_llm][4096]`, f32 activations.
pub fn tower_forward(host: &MmprojHost, patches: &[f32], n_h: usize, n_w: usize) -> Vec<f32> {
    tower_forward_prec(host, patches, n_h, n_w, ActPrec::F32)
}
