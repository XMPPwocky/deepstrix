//! v4flash-vision — DeepSeek-V4-Flash-Vision-Exp image side.
//!
//! Pure-host pieces are tested against the Python reference
//! (`inference/image_processor.py`) and, for the PIL semantics, against
//! real Pillow 12.3.0 vectors (`tests/data/pillow_resize_cases.json`):
//!
//! * [`preprocess`] — decode (PNG/JPEG) → PIL-semantics resize/pad →
//!   normalise → patchify. See [`preprocess::preprocess`].
//! * [`layout`] — `grid_tokens` / `solve_resize_ratio` / `safe_resize` /
//!   `build_image_block` ports; [`layout_for`] builds the [`ImageLayout`].
//! * [`rope`] — the ViT's 2-D RoPE cos/sin tables (`get_vision_cos_sin`).
//! * [`mmproj`] — typed loader for the 427-tensor `mmproj-F16.gguf`.
//!
//! [`Tower`] loads the mmproj weights onto a HIP device and runs the
//! ViT + aligner forward ([`Tower::encode`]); see the roofline block at
//! the top of `tower.rs` for where the time goes.
//!
//! The text-side `ffn.gate.bias_vl` sidecar is NOT read here — the
//! engine owns that file end to end
//! (`v4flash_kernels::het::weights::{bias_vl_sidecar_path,
//! read_bias_vl_sidecar, write_bias_vl_sidecar}`), including the
//! `DEEPSTRIX_BIAS_VL_FILE` override. A second copy of the path
//! convention used to live in this crate, unused, and would have drifted.
//!
//! Token conventions shared with the engine / server (from the reference):
//! * synthetic ids at image slots = `VOCAB_SIZE + type`, types
//!   START,PAD,IMAGE,NEWLINE,END = 0..4 ([`TokenType`]);
//! * the placeholder `<｜deepseek_image｜>` is looked up BY NAME in the
//!   text GGUF vocab (id 129264 in the Vision-Exp GGUF) — see
//!   [`IMAGE_PLACEHOLDER`].

pub mod kernels;
pub mod layout;
pub mod mmproj;
pub mod preprocess;
pub mod reference;
pub mod resize;
pub mod rope;
pub mod tower;

pub use layout::{layout_for, ImageLayout, ResizePlan};
pub use preprocess::{preprocess, PreprocessedImage};
pub use tower::Tower;

// ---------------------------------------------------------------- constants

/// Vision patch size (pixels) — `clip.vision.patch_size`.
pub const PATCH: u32 = 14;
/// Elements per flattened patch: 3 × 14 × 14, order (c, y, x).
pub const PATCH_ELEMS: usize = 3 * (PATCH as usize) * (PATCH as usize);
/// Aligner downsample ratio (3×3 unfold) — `clip.vision.projector.scale_factor`.
pub const DOWNSAMPLE: u32 = 3;
/// `vision_max_n_token`: LLM-token budget per image (before the −3 compress pad).
pub const MAX_N_TOKEN: u32 = 384;
/// `vision_max_wh_ratio`: width capped at 8 × height.
pub const MAX_WH_RATIO: u32 = 8;
/// `clip.vision.image_min_pixels`: images smaller than this are upscaled.
pub const MIN_PIXELS: u32 = 147_456;
/// `COMPRESS_PAD_TO`: IMAGE_START is aligned so that `(start + pad) % 4 == 3`.
pub const COMPRESS_PAD_TO: u32 = 4;
/// Pad colour used by `ImageOps.pad(..., color=(127,127,127))`.
pub const PAD_GRAY: u8 = 127;
/// `image_mean` / `image_std` (both 0.5 on every channel): x/255 → (x−0.5)/0.5.
pub const IMAGE_MEAN: f32 = 0.5;
pub const IMAGE_STD: f32 = 0.5;

/// ViT hidden size — `clip.vision.embedding_length`.
pub const VIT_DIM: usize = 1024;
/// ViT attention heads — `clip.vision.attention.head_count`.
pub const VIT_N_HEADS: usize = 16;
pub const VIT_HEAD_DIM: usize = VIT_DIM / VIT_N_HEADS; // 64
/// 2-D RoPE rotates `head_dim / 2` = 32 dims: pairs (i, i+32); i<16 row
/// frequencies, 16≤i<32 column frequencies.
pub const VIT_ROPE_DIM: usize = VIT_HEAD_DIM / 2; // 32
pub const VIT_ROPE_THETA: f32 = 10_000.0;
/// SwiGLU intermediate — `clip.vision.feed_forward_length`.
pub const VIT_FFN: usize = 2816;
/// Transformer blocks — `clip.vision.block_count`.
pub const VIT_N_LAYERS: usize = 32;
/// RMSNorm epsilon — `clip.vision.attention.layer_norm_epsilon`.
pub const VIT_RMS_EPS: f32 = 1e-6;
/// Aligner input = VIT_DIM × 3 × 3.
pub const ALIGNER_IN: usize = VIT_DIM * (DOWNSAMPLE as usize) * (DOWNSAMPLE as usize); // 9216
/// Text-model hidden size = aligner output — `clip.vision.projection_dim`.
pub const TEXT_DIM: usize = 4096;

/// Text vocab size; synthetic image ids are `VOCAB_SIZE + type`.
pub const VOCAB_SIZE: u32 = 129_280;
/// Placeholder token the chat template emits per image part. Look it up
/// by name in the text GGUF's vocab at load (id 129264 in the Vision-Exp
/// GGUF; the 0731 vocab has `<｜image2｜>` there) — error if absent.
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";
/// Number of text layers carrying `ffn.gate.bias_vl` (43 × 256 f32).
pub const TEXT_N_LAYERS: usize = 43;
pub const N_ROUTED_EXPERTS: usize = 256;

/// Block token types (`IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Start = 0,
    Pad = 1,
    Image = 2,
    NewLine = 3,
    End = 4,
}

impl TokenType {
    pub fn from_u8(v: u8) -> Option<TokenType> {
        Some(match v {
            0 => TokenType::Start,
            1 => TokenType::Pad,
            2 => TokenType::Image,
            3 => TokenType::NewLine,
            4 => TokenType::End,
            _ => return None,
        })
    }
}

/// Synthetic token id placed at an image slot of the given type.
#[inline]
pub fn synthetic_token_id(ty: u8) -> u32 {
    debug_assert!(ty <= TokenType::End as u8);
    VOCAB_SIZE + ty as u32
}

/// Inverse of [`synthetic_token_id`]: `Some(type)` for image-slot ids.
#[inline]
pub fn image_token_type(id: u32) -> Option<u8> {
    if (VOCAB_SIZE..VOCAB_SIZE + 5).contains(&id) {
        Some((id - VOCAB_SIZE) as u8)
    } else {
        None
    }
}

/// `true` for any synthetic image-slot id (the engine's per-row `is_image`).
#[inline]
pub fn is_image_token(id: u32) -> bool {
    id >= VOCAB_SIZE
}
