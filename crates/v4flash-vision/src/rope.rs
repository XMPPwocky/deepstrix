//! ViT 2-D RoPE tables — port of `vision.get_vision_cos_sin` and a host
//! reference `apply_rotary` (for oracle tests of the tower kernels).
//!
//! `rope_dim = head_dim / 2 = 32`. For patch `(y, x)` the 32 angles are
//! `[y * inv_freq[0..16], x * inv_freq[0..16]]` with
//! `inv_freq[j] = theta^(-(2j)/32)`. `apply_rotary` splits each head's 64
//! dims into `x1 = [0..32)`, `x2 = [32..64)` and emits
//! `[x1*cos − x2*sin, x2*cos + x1*sin]` — i.e. rotation pairs `(i, i+32)`,
//! rows for `i < 16`, columns for `16 ≤ i < 32`.

use crate::{VIT_HEAD_DIM, VIT_N_HEADS, VIT_ROPE_DIM, VIT_ROPE_THETA};

/// `(cos, sin)`, each `[n_h * n_w][VIT_ROPE_DIM]` row-major over the patch grid.
pub fn vision_cos_sin(n_h: u32, n_w: u32) -> (Vec<f32>, Vec<f32>) {
    let half = VIT_ROPE_DIM / 2; // 16
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0 / VIT_ROPE_THETA.powf((2 * j) as f32 / VIT_ROPE_DIM as f32))
        .collect();
    let n = n_h as usize * n_w as usize;
    let mut cos = Vec::with_capacity(n * VIT_ROPE_DIM);
    let mut sin = Vec::with_capacity(n * VIT_ROPE_DIM);
    for y in 0..n_h {
        for x in 0..n_w {
            for &f in &inv_freq {
                let a = y as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
            for &f in &inv_freq {
                let a = x as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
    }
    (cos, sin)
}

/// Host reference of `apply_rotary` on `x: [n][VIT_N_HEADS][VIT_HEAD_DIM]`
/// (in place), with `cos`/`sin` from [`vision_cos_sin`].
pub fn apply_rotary_host(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let n = x.len() / (VIT_N_HEADS * VIT_HEAD_DIM);
    assert_eq!(x.len(), n * VIT_N_HEADS * VIT_HEAD_DIM);
    assert_eq!(cos.len(), n * VIT_ROPE_DIM);
    assert_eq!(sin.len(), n * VIT_ROPE_DIM);
    for t in 0..n {
        let c = &cos[t * VIT_ROPE_DIM..(t + 1) * VIT_ROPE_DIM];
        let s = &sin[t * VIT_ROPE_DIM..(t + 1) * VIT_ROPE_DIM];
        for h in 0..VIT_N_HEADS {
            let base = (t * VIT_N_HEADS + h) * VIT_HEAD_DIM;
            for i in 0..VIT_ROPE_DIM {
                let x1 = x[base + i];
                let x2 = x[base + VIT_ROPE_DIM + i];
                x[base + i] = x1 * c[i] - x2 * s[i];
                x[base + VIT_ROPE_DIM + i] = x2 * c[i] + x1 * s[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_shape_and_origin() {
        let (c, s) = vision_cos_sin(3, 5);
        assert_eq!(c.len(), 15 * 32);
        assert_eq!(s.len(), 15 * 32);
        // patch (0,0): all angles zero.
        assert!(c[..32].iter().all(|&v| v == 1.0));
        assert!(s[..32].iter().all(|&v| v == 0.0));
        // patch (0,1): rows angles zero, column angle for j=0 is 1 rad.
        let r = &c[32..64];
        assert!(r[..16].iter().all(|&v| v == 1.0));
        assert!((r[16] - 1f32.cos()).abs() < 1e-6);
        // patch (1,0) (row 1, col 0) = index n_w = 5.
        let r = &c[5 * 32..6 * 32];
        assert!((r[0] - 1f32.cos()).abs() < 1e-6);
        assert!(r[16..].iter().all(|&v| v == 1.0));
    }

    #[test]
    fn rotary_preserves_norm() {
        let (c, s) = vision_cos_sin(2, 2);
        let mut x: Vec<f32> = (0..4 * VIT_N_HEADS * VIT_HEAD_DIM).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
        let n0: f32 = x.iter().map(|v| v * v).sum();
        apply_rotary_host(&mut x, &c, &s);
        let n1: f32 = x.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() / n0 < 1e-5);
    }
}
