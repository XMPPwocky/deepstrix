//! `image_processor.load_image` port: decode → (plan) → PIL resize/pad →
//! normalise `(x/255 − 0.5)/0.5` → patchify `[n_vit_h*n_vit_w, 3*14*14]`.
//!
//! Patch flattening follows
//! `x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4)`: patches are
//! spatial-major (row `ph`, then column `pw`), and inside a patch the 588
//! values are ordered `(c, y, x)`.
//!
//! The reference casts the normalised tensor to bfloat16 before the ViT;
//! we keep f32 (the tower agent decides the on-device dtype). Use
//! [`bf16_round`] if a bit-faithful oracle comparison needs the rounding.

use std::io::Cursor;

use color_eyre::eyre::{self, eyre, WrapErr};

use crate::layout::{plan_resize, ResizePlan};
use crate::resize::{pad_contain, resize_bicubic, Rgb};
use crate::{IMAGE_MEAN, IMAGE_STD, PAD_GRAY, PATCH, PATCH_ELEMS};

/// One image, ready for the ViT.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// `[n_vit_h * n_vit_w, 3*14*14]` normalised, patch-major, (c, y, x) inside.
    pub patches: Vec<f32>,
    pub n_vit_h: u32,
    pub n_vit_w: u32,
    /// blake3 of `patches` as little-endian f32 bytes.
    pub content_hash: [u8; 32],
}

impl PreprocessedImage {
    pub fn n_patches(&self) -> usize {
        self.n_vit_h as usize * self.n_vit_w as usize
    }
}

/// Largest decoded side (pixels) accepted by [`decode_rgb`]. The
/// preprocessor caps the useful canvas at 384 LLM tokens, so anything
/// beyond this is thrown away by `plan_resize` regardless.
pub const MAX_DECODE_SIDE: u32 = 16_384;
/// Decoder allocation cap. 16384x16384 RGB8 would be 768 MiB, but a
/// realistic photo at MAX_DECODE_SIDE on one axis is far smaller; 256 MiB
/// bounds the transient cost on a host with single-digit GiB free.
pub const MAX_DECODE_ALLOC: u64 = 256 * 1024 * 1024;

/// Decode PNG/JPEG bytes to RGB8 (`Image.open(...).convert("RGB")`: alpha
/// is dropped, not composited; grayscale is expanded).
pub fn decode_rgb(bytes: &[u8]) -> eyre::Result<Rgb> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .wrap_err("image: format sniff")?;
    // `Limits::default()` sets NO dimension cap (only a 512 MiB alloc
    // cap), so a small, highly compressible PNG could force a ~512 MiB
    // decode plus an `into_rgb8` copy on whichever thread is decoding.
    // Nothing past MAX_DECODE_SIDE survives `plan_resize` anyway — the
    // pipeline's whole canvas is at most 384 LLM tokens.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODE_SIDE);
    limits.max_image_height = Some(MAX_DECODE_SIDE);
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    reader.limits(limits);
    let fmt = reader.format();
    let dyn_img = reader.decode().wrap_err_with(|| format!("image: decode ({fmt:?})"))?;
    let rgb = dyn_img.into_rgb8();
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return Err(eyre!("image: empty image {w}x{h}"));
    }
    Ok(Rgb::new(w, h, rgb.into_raw()))
}

/// `(x / 255 − mean) / std` for one 8-bit sample.
#[inline]
pub fn normalize(px: u8) -> f32 {
    (px as f32 / 255.0 - IMAGE_MEAN) / IMAGE_STD
}

/// Round an f32 to the nearest bfloat16 value (round-half-even), returned as f32.
#[inline]
pub fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 {
        return x; // inf / nan
    }
    let lsb = (b >> 16) & 1;
    let rounded = b.wrapping_add(0x7fff + lsb) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Patchify a canvas that is exactly `n_vit_w*14 × n_vit_h*14`.
pub fn patchify(canvas: &Rgb, n_vit_h: u32, n_vit_w: u32) -> Vec<f32> {
    assert_eq!(canvas.w, n_vit_w * PATCH);
    assert_eq!(canvas.h, n_vit_h * PATCH);
    let p = PATCH as usize;
    let mut out = vec![0f32; n_vit_h as usize * n_vit_w as usize * PATCH_ELEMS];
    for ph in 0..n_vit_h as usize {
        for pw in 0..n_vit_w as usize {
            let base = (ph * n_vit_w as usize + pw) * PATCH_ELEMS;
            for c in 0..3 {
                for y in 0..p {
                    for x in 0..p {
                        let v = canvas.px((pw * p + x) as u32, (ph * p + y) as u32, c);
                        out[base + c * p * p + y * p + x] = normalize(v);
                    }
                }
            }
        }
    }
    out
}

/// The pixel half of `load_image` on an already-decoded RGB image.
pub fn preprocess_rgb(img: &Rgb) -> eyre::Result<(PreprocessedImage, ResizePlan)> {
    let plan = plan_resize(img.h, img.w)?;
    let canvas = if plan.plain_resize {
        resize_bicubic(img, plan.best_w, plan.best_h)?
    } else {
        pad_contain(img, plan.best_w, plan.best_h, [PAD_GRAY; 3])?
    };
    let patches = patchify(&canvas, plan.n_vit_h, plan.n_vit_w);
    let content_hash = hash_patches(&patches);
    Ok((
        PreprocessedImage { patches, n_vit_h: plan.n_vit_h, n_vit_w: plan.n_vit_w, content_hash },
        plan,
    ))
}

/// blake3 over the patches' little-endian f32 bytes.
pub fn hash_patches(patches: &[f32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    // Chunked to avoid a full-size temporary.
    let mut buf = Vec::with_capacity(4096 * 4);
    for chunk in patches.chunks(4096) {
        buf.clear();
        for v in chunk {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        h.update(&buf);
    }
    *h.finalize().as_bytes()
}

/// `image_processor.load_image` for raw PNG/JPEG bytes.
pub fn preprocess(bytes: &[u8]) -> eyre::Result<PreprocessedImage> {
    let img = decode_rgb(bytes)?;
    let (pre, _plan) = preprocess_rgb(&img)?;
    Ok(pre)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_endpoints() {
        assert_eq!(normalize(0), -1.0);
        assert_eq!(normalize(255), 1.0);
        assert!((normalize(127) - (127.0 / 255.0 - 0.5) / 0.5).abs() < 1e-7);
    }

    #[test]
    fn bf16_round_basic() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(-1.0), -1.0);
        let v = bf16_round(0.1);
        assert!((v - 0.1).abs() < 1e-3);
        assert_eq!(v.to_bits() & 0xffff, 0);
    }
}
