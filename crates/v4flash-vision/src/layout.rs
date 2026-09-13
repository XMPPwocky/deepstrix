//! Line-for-line ports of the reference layout functions in
//! `inference/image_processor.py`, for both checkpoints ([`VisionCfg`]):
//!
//! * V4-Flash Vision-Exp: `grid_tokens`, `solve_resize_ratio`, `safe_resize`,
//!   the integer prologue of `load_image` ([`plan_resize`]) and
//!   `build_image_block` ([`build_image_block`] / [`layout_for`]). Verified
//!   against `tests/data/layout_cases.json` (`scripts/gen_vision_layout_vectors.py`).
//! * V4.1: `llm_grid`, `num_image_tokens`, `solve_resize_ratio`, `safe_resize`,
//!   `plan_image_grid` ([`plan_resize_cfg`]) and `image_token_types`
//!   ([`build_image_block_v41`]). Verified against
//!   `tests/data/layout_cases_v41.json`, produced by running the reference
//!   module itself (`scripts/gen_v41_vision_vectors.py`).

use color_eyre::eyre::{self, eyre};

use crate::preprocess::PreprocessedImage;
use crate::{LayoutKind, TokenType, VisionCfg, COMPRESS_PAD_TO, DOWNSAMPLE, PATCH};

/// Token-level layout of one image block, in FINAL (block) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageLayout {
    pub n_vit_h: u32,
    pub n_vit_w: u32,
    pub n_llm_h: u32,
    pub n_llm_w: u32,
    /// Per block token: `TokenType` as u8 (0..4). `types.len()` is the block
    /// length INCLUDING the leading compress pads; the block occupies
    /// positions `start_pos .. start_pos + types.len()` in the prompt.
    pub types: Vec<u8>,
    /// For each IMAGE slot in block order, the aligner output row
    /// (`0 .. n_llm_h*n_llm_w`, row-major over the LLM grid) it carries.
    pub perm: Vec<u32>,
    /// Prompt position of the first block token (the first compress pad, or
    /// IMAGE_START when there are none — always for V4.1).
    pub start_pos: u32,
    /// Number of leading IMAGE_PAD tokens before IMAGE_START (V4-Flash:
    /// `3 - start_pos % 4`; V4.1: 0).
    pub compress_pad: u32,
}

impl ImageLayout {
    /// Number of leading IMAGE_PAD tokens before IMAGE_START.
    pub fn compress_pad(&self) -> u32 {
        self.compress_pad
    }
    /// Prompt position of IMAGE_START.
    pub fn image_start_pos(&self) -> u32 {
        self.start_pos + self.compress_pad()
    }
    /// Prompt position of IMAGE_END (inclusive end of the bidirectional span).
    pub fn image_end_pos(&self) -> u32 {
        self.start_pos + self.types.len() as u32 - 1
    }
    /// `(start_pos_of_IMAGE_START, len)` of the inclusive `[START..END]`
    /// span, in the form the engine's `image_spans` argument expects.
    pub fn span(&self) -> (u32, u32) {
        let s = self.image_start_pos();
        (s, self.image_end_pos() - s + 1)
    }
    /// Synthetic token ids (`VOCAB_SIZE + type`) for the whole block.
    pub fn token_ids(&self) -> Vec<u32> {
        self.types.iter().map(|&t| crate::synthetic_token_id(t)).collect()
    }
}

/// Output of [`plan_resize`]: everything `load_image` decides before it
/// touches pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizePlan {
    /// Target canvas (multiple of 14) the ViT sees.
    pub best_h: u32,
    pub best_w: u32,
    pub n_vit_h: u32,
    pub n_vit_w: u32,
    pub n_llm_h: u32,
    pub n_llm_w: u32,
    /// `true` → `image.resize((best_w, best_h))` (aspect NOT kept; taken when
    /// `orig_w >= 8 * orig_h`); `false` → `ImageOps.pad` (contain + gray).
    pub plain_resize: bool,
}

/// `grid_tokens(best_height, best_width, patch_size, downsample_ratio)`.
pub fn grid_tokens(best_height: u32, best_width: u32) -> (u32, u32, u32) {
    let n_llm_h = (best_height / PATCH).div_ceil(DOWNSAMPLE);
    let n_llm_w = (best_width / PATCH).div_ceil(DOWNSAMPLE);
    let mut num_tokens = n_llm_h * (n_llm_w + 1) + 2;
    if n_llm_h % 2 == 1 {
        num_tokens += n_llm_w + 1;
    }
    num_tokens += n_llm_h.div_ceil(2) * (n_llm_w + 1) % 2 * 2; // (n_llm_h + 1) // 2 * ... in the reference
    (n_llm_h, n_llm_w, num_tokens)
}

/// `solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)`
/// → `(n_llm_h, n_llm_w, best_height, best_width, num_tokens)`.
///
/// Fallible rather than `assert!`-ing: this runs on a request-serving
/// thread, and the fn is `pub`, so a caller outside `plan_resize`'s
/// `width <= 8*height` cap must get an error instead of a panic.
pub fn solve_resize_ratio(
    height: u32,
    width: u32,
    max_n_token: u32,
) -> eyre::Result<(u32, u32, u32, u32, u32)> {
    if height == 0 || width == 0 {
        return Err(eyre!("solve_resize_ratio: empty image {width}x{height}"));
    }
    if max_n_token < 3 {
        return Err(eyre!("solve_resize_ratio: max_n_token {max_n_token} too small"));
    }
    let p = PATCH as f64;
    let ds = DOWNSAMPLE as f64;
    let r = height as f64 / width as f64;
    let max_w_float = ((max_n_token as f64 - 2.0) / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    let (best_width, best_height);
    if max_w_float < 1.0 {
        let max_w = 1u32;
        let mut max_h = (max_n_token - 2) / (max_w + 1);
        if max_h % 2 == 1 {
            max_h -= 1;
        }
        best_width = max_w * PATCH * DOWNSAMPLE;
        best_height = max_h * PATCH * DOWNSAMPLE;
    } else if max_h_float < 2.0 {
        let max_h = 2u32;
        let max_w = ((max_n_token - 2) / max_h).saturating_sub(1);
        if max_w <= 1 {
            return Err(eyre!(
                "solve_resize_ratio: max_n_token {max_n_token} leaves max_w {max_w} (need > 1)"
            ));
        }
        best_width = max_w * PATCH * DOWNSAMPLE;
        best_height = max_h * PATCH * DOWNSAMPLE;
    } else {
        let max_w = max_w_float.floor();
        let mut max_h = max_h_float.floor();
        if (max_h as i64) % 2 == 1 {
            max_h -= 1.0;
        }
        let beta = (max_w * p * ds / width as f64).min(max_h * p * ds / height as f64);
        best_width = ((width as f64 * beta / p).floor() as u32) * PATCH;
        best_height = ((height as f64 * beta / p).floor() as u32) * PATCH;
    }
    let (n_llm_h, n_llm_w, num_tokens) = grid_tokens(best_height, best_width);
    Ok((n_llm_h, n_llm_w, best_height, best_width, num_tokens))
}

/// `safe_resize(height, width, best_height, best_width, patch_size,
/// downsample_ratio, max_n_token)` → `(n_llm_h, n_llm_w, best_height, best_width)`.
/// Note the budget is `max_n_token - (COMPRESS_PAD_TO - 1)` = 381.
pub fn safe_resize(
    height: u32,
    width: u32,
    mut best_height: u32,
    mut best_width: u32,
    max_n_token: u32,
) -> eyre::Result<(u32, u32, u32, u32)> {
    let max_n_token = max_n_token
        .checked_sub(COMPRESS_PAD_TO - 1)
        .ok_or_else(|| eyre!("safe_resize: max_n_token {max_n_token} below the compress pad"))?;
    let (mut n_llm_h, mut n_llm_w, mut num_tokens) = grid_tokens(best_height, best_width);
    let mut budget = max_n_token;
    while num_tokens > max_n_token {
        (n_llm_h, n_llm_w, best_height, best_width, num_tokens) =
            solve_resize_ratio(height, width, budget)?;
        // `budget -= 1` on a u32: an unguarded decrement in a retry loop
        // reachable from the HTTP task.
        budget = budget
            .checked_sub(1)
            .ok_or_else(|| eyre!("safe_resize: exhausted the token budget for {width}x{height}"))?;
    }
    Ok((n_llm_h, n_llm_w, best_height, best_width))
}

/// The integer prologue of `load_image` for an `orig_h × orig_w` decoded
/// image (V4-Flash): w ≤ 8h cap, ≥ MIN_PIXELS upscale (`int()` truncation),
/// ceil to multiples of 14, then [`safe_resize`].
pub fn plan_resize(orig_h: u32, orig_w: u32) -> eyre::Result<ResizePlan> {
    plan_resize_cfg(orig_h, orig_w, &VisionCfg::V4_FLASH)
}

/// [`plan_resize`] for either checkpoint. Both references share the prologue
/// (`plan_image_grid` in V4.1: the wh-ratio cap only when configured, the
/// `min_pixels` upscale with `int()` truncation, ceil to patch multiples) and
/// differ in the token-budget fit that follows.
pub fn plan_resize_cfg(orig_h: u32, orig_w: u32, cfg: &VisionCfg) -> eyre::Result<ResizePlan> {
    if orig_h == 0 || orig_w == 0 {
        return Err(eyre!("plan_resize: empty image {orig_w}x{orig_h}"));
    }
    let mut height = orig_h;
    let mut width = orig_w;
    if let Some(r) = cfg.max_wh_ratio {
        if width > height * r {
            width = height * r;
        }
    }
    let px = width as u64 * height as u64;
    if px > 0 && px < cfg.min_pixels as u64 {
        // `(min_pixels / (w*h)) ** 0.5` — Python's `**` is C `pow`, as is `powf`.
        let ratio = (cfg.min_pixels as f64 / px as f64).powf(0.5);
        width = (width as f64 * ratio) as u32; // Python int(): truncation
        height = (height as f64 * ratio) as u32;
    }
    let best_width = width.div_ceil(PATCH) * PATCH;
    let best_height = height.div_ceil(PATCH) * PATCH;
    let (n_llm_h, n_llm_w, best_height, best_width) = match cfg.kind {
        LayoutKind::V4FlashInterleaved => safe_resize(height, width, best_height, best_width, cfg.max_n_token)?,
        LayoutKind::V41Flat => safe_resize_v41(height, width, best_height, best_width, cfg.max_n_token)?,
    };
    Ok(ResizePlan {
        best_h: best_height,
        best_w: best_width,
        n_vit_h: best_height / PATCH,
        n_vit_w: best_width / PATCH,
        n_llm_h,
        n_llm_w,
        plain_resize: cfg.max_wh_ratio.is_some_and(|r| orig_w >= r * orig_h),
    })
}

// ------------------------------------------------------------------ V4.1

/// V4.1 `llm_grid(best_height, best_width, patch_size, downsample_ratio)`.
pub fn llm_grid_v41(best_height: u32, best_width: u32) -> (u32, u32) {
    ((best_height / PATCH).div_ceil(DOWNSAMPLE), (best_width / PATCH).div_ceil(DOWNSAMPLE))
}

/// V4.1 `num_image_tokens(n_llm_h, n_llm_w)` = `h * (w + 1) + 2`
/// (START + rows of `w` IMAGE + NEWLINE + END).
pub fn num_image_tokens_v41(n_llm_h: u32, n_llm_w: u32) -> u32 {
    n_llm_h * (n_llm_w + 1) + 2
}

/// V4.1 `solve_resize_ratio(height, width, patch_size, downsample_ratio,
/// max_n_token)` → `(best_height, best_width)`: the largest aspect-preserving
/// pixel size whose token grid fits `max_n_token`; degenerate aspects
/// collapse to a single column / row.
pub fn solve_resize_ratio_v41(height: u32, width: u32, max_n_token: u32) -> eyre::Result<(u32, u32)> {
    if height == 0 || width == 0 {
        return Err(eyre!("solve_resize_ratio: empty image {width}x{height}"));
    }
    if max_n_token < 4 {
        return Err(eyre!("solve_resize_ratio: max_n_token {max_n_token} too small"));
    }
    let p = PATCH as f64;
    let cell = PATCH * DOWNSAMPLE; // 42
    let r = height as f64 / width as f64;
    let max_w_float = ((max_n_token as f64 - 2.0) / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    if max_w_float < 1.0 {
        // very tall: a single column
        return Ok(((max_n_token - 2) / 2 * cell, cell));
    }
    if max_h_float < 1.0 {
        // very wide: a single row
        return Ok((cell, (max_n_token - 3) * cell));
    }
    let beta = (max_w_float.floor() * cell as f64 / width as f64).min(max_h_float.floor() * cell as f64 / height as f64);
    Ok((
        ((height as f64 * beta / p).floor() as u32) * PATCH,
        ((width as f64 * beta / p).floor() as u32) * PATCH,
    ))
}

/// V4.1 `safe_resize`: shrink once (no retry loop, no compress-pad
/// reservation) when the grid costs more than `max_n_token` LLM tokens.
/// → `(n_llm_h, n_llm_w, best_height, best_width)`.
pub fn safe_resize_v41(
    height: u32,
    width: u32,
    mut best_height: u32,
    mut best_width: u32,
    max_n_token: u32,
) -> eyre::Result<(u32, u32, u32, u32)> {
    let (mut n_llm_h, mut n_llm_w) = llm_grid_v41(best_height, best_width);
    if num_image_tokens_v41(n_llm_h, n_llm_w) > max_n_token {
        (best_height, best_width) = solve_resize_ratio_v41(height, width, max_n_token)?;
        (n_llm_h, n_llm_w) = llm_grid_v41(best_height, best_width);
        let n = num_image_tokens_v41(n_llm_h, n_llm_w);
        if n > max_n_token {
            // The reference asserts here; it cannot trigger for max_n_token >= 4.
            return Err(eyre!("safe_resize: {width}x{height} still costs {n} > {max_n_token} tokens after the fit"));
        }
    }
    Ok((n_llm_h, n_llm_w, best_height, best_width))
}

/// V4.1 `image_token_types(n_llm_h, n_llm_w)` → `(types, perm)`:
/// `START, (IMAGE*n_llm_w, NEWLINE)*n_llm_h, END`; `perm` is the identity
/// (aligner rows are consumed in reading order by `merge_image_embeddings`).
pub fn build_image_block_v41(n_llm_h: u32, n_llm_w: u32) -> (Vec<u8>, Vec<u32>) {
    let n = num_image_tokens_v41(n_llm_h, n_llm_w) as usize;
    let mut types = Vec::with_capacity(n);
    types.push(TokenType::Start as u8);
    for _ in 0..n_llm_h {
        types.extend(std::iter::repeat_n(TokenType::Image as u8, n_llm_w as usize));
        types.push(TokenType::NewLine as u8);
    }
    types.push(TokenType::End as u8);
    debug_assert_eq!(types.len(), n);
    (types, (0..n_llm_h * n_llm_w).collect())
}

/// `build_image_block(n_llm_h, n_llm_w, start_pos)` → `(types, perm)`.
///
/// `types` is in final block order:
/// `[PAD]*compress_pad, START, rows pair-interleaved column-wise
/// (row 2p col c, row 2p+1 col c, ...), [PAD]*pad_last, END`, where each
/// source row is `IMAGE*n_llm_w, NEWLINE` and an all-PAD row is appended
/// when `n_llm_h` is odd. `perm[i]` is the aligner row for the i-th IMAGE
/// slot in block order.
pub fn build_image_block(n_llm_h: u32, n_llm_w: u32, start_pos: u32) -> (Vec<u8>, Vec<u32>) {
    let compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO;
    let pad_h = n_llm_h % 2;
    let rows = n_llm_h + pad_h;
    let row_len = n_llm_w + 1;
    let pad_last = rows / 2 * row_len % 2 * 2;

    // Source (row-major, N-layout before interleave).
    let mut src_types = Vec::with_capacity((rows * row_len) as usize);
    for _ in 0..n_llm_h {
        src_types.extend(std::iter::repeat_n(TokenType::Image as u8, n_llm_w as usize));
        src_types.push(TokenType::NewLine as u8);
    }
    src_types.extend(std::iter::repeat_n(TokenType::Pad as u8, (row_len * pad_h) as usize));
    debug_assert_eq!(src_types.len(), (rows * row_len) as usize);

    let mut types = Vec::with_capacity((compress_pad + 1 + rows * row_len + pad_last + 1) as usize);
    let mut perm = Vec::with_capacity((n_llm_h * n_llm_w) as usize);
    types.extend(std::iter::repeat_n(TokenType::Pad as u8, compress_pad as usize));
    types.push(TokenType::Start as u8);
    // order = arange(rows*row_len).view(rows//2, 2, row_len).transpose(1, 2).reshape(-1)
    for p in 0..rows / 2 {
        for c in 0..row_len {
            for r in 0..2 {
                let row = p * 2 + r;
                let o = row * row_len + c;
                types.push(src_types[o as usize]);
                if row < n_llm_h && c < n_llm_w {
                    perm.push(row * n_llm_w + c);
                }
            }
        }
    }
    types.extend(std::iter::repeat_n(TokenType::Pad as u8, pad_last as usize));
    types.push(TokenType::End as u8);
    (types, perm)
}

/// Build the V4-Flash block layout for a preprocessed image whose IMAGE
/// block starts at prompt position `start_pos` (the position the placeholder
/// token occupied, i.e. `len(tokens)` so far in `prepare_vl_inputs`).
pub fn layout_for(img: &PreprocessedImage, start_pos: u32) -> ImageLayout {
    layout_for_grid(img.n_vit_h, img.n_vit_w, start_pos)
}

/// [`layout_for`] for either checkpoint.
pub fn layout_for_cfg(img: &PreprocessedImage, start_pos: u32, cfg: &VisionCfg) -> ImageLayout {
    layout_for_grid_cfg(img.n_vit_h, img.n_vit_w, start_pos, cfg)
}

/// [`layout_for`] from the ViT grid alone (`n_llm = ceil(n_vit / 3)`), V4-Flash.
pub fn layout_for_grid(n_vit_h: u32, n_vit_w: u32, start_pos: u32) -> ImageLayout {
    layout_for_grid_cfg(n_vit_h, n_vit_w, start_pos, &VisionCfg::V4_FLASH)
}

/// [`layout_for_grid`] for either checkpoint.
pub fn layout_for_grid_cfg(n_vit_h: u32, n_vit_w: u32, start_pos: u32, cfg: &VisionCfg) -> ImageLayout {
    let n_llm_h = n_vit_h.div_ceil(DOWNSAMPLE);
    let n_llm_w = n_vit_w.div_ceil(DOWNSAMPLE);
    let (types, perm, compress_pad) = match cfg.kind {
        LayoutKind::V4FlashInterleaved => {
            let (t, p) = build_image_block(n_llm_h, n_llm_w, start_pos);
            (t, p, COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO)
        }
        LayoutKind::V41Flat => {
            let (t, p) = build_image_block_v41(n_llm_h, n_llm_w);
            (t, p, 0)
        }
    };
    debug_assert_eq!(perm.len() as u32, n_llm_h * n_llm_w);
    ImageLayout { n_vit_h, n_vit_w, n_llm_h, n_llm_w, types, perm, start_pos, compress_pad }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_examples() {
        let p = plan_resize(1080, 1920).unwrap();
        assert_eq!((p.best_h, p.best_w, p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w), (588, 1036, 42, 74, 14, 25));
        let l = layout_for_grid(42, 74, 3);
        assert_eq!(l.types.len(), 366);
        let p = plan_resize(768, 1024).unwrap();
        assert_eq!((p.best_h, p.best_w, p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w), (658, 882, 47, 63, 16, 21));
        assert_eq!(layout_for_grid(47, 63, 3).types.len(), 354);
        let p = plan_resize(512, 512).unwrap();
        assert_eq!((p.best_h, p.best_w, p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w), (518, 518, 37, 37, 13, 13));
        assert_eq!(layout_for_grid(37, 37, 3).types.len(), 198);
        let p = plan_resize(384, 2208).unwrap();
        assert_eq!((p.best_h, p.best_w, p.n_vit_h, p.n_vit_w), (336, 1932, 24, 138));
    }

    #[test]
    fn v41_flat_layout_and_budget() {
        // 640x480 → no upscale (307200 ≥ 295936), 644x490 → 35x46 → 12x16, 206 tokens.
        let p = plan_resize_cfg(480, 640, &VisionCfg::V41).unwrap();
        assert_eq!((p.best_h, p.best_w, p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w), (490, 644, 35, 46, 12, 16));
        assert!(!p.plain_resize);
        let l = layout_for_grid_cfg(35, 46, 9, &VisionCfg::V41);
        assert_eq!(l.types.len(), 206);
        assert_eq!(l.compress_pad(), 0);
        assert_eq!(l.image_start_pos(), 9);
        assert_eq!(l.span(), (9, 206));
        assert_eq!(l.types[0], TokenType::Start as u8);
        assert_eq!(l.types[1], TokenType::Image as u8);
        assert_eq!(l.types[17], TokenType::NewLine as u8);
        assert_eq!(*l.types.last().unwrap(), TokenType::End as u8);
        assert!(l.types.iter().all(|&t| t != TokenType::Pad as u8));
        assert_eq!(l.perm, (0..192).collect::<Vec<u32>>());
        // 1024x701 (carrots.jpeg): 1036x714 → 51x74 → 17x25, 444 tokens.
        let p = plan_resize_cfg(701, 1024, &VisionCfg::V41).unwrap();
        assert_eq!((p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w), (51, 74, 17, 25));
        assert_eq!(num_image_tokens_v41(17, 25), 444);
        // 450x308 (corn.jpeg): below min_pixels → ×1.461 → 657x450 → 658x462 → 33x47 → 11x16.
        let p = plan_resize_cfg(308, 450, &VisionCfg::V41).unwrap();
        assert_eq!((p.best_h, p.best_w, p.n_llm_h, p.n_llm_w), (462, 658, 11, 16));
        // Over budget: 4000x3000 must shrink to ≤ 1024 tokens; 1x100000 collapses to one column.
        let p = plan_resize_cfg(3000, 4000, &VisionCfg::V41).unwrap();
        assert!(num_image_tokens_v41(p.n_llm_h, p.n_llm_w) <= 1024);
        let p = plan_resize_cfg(100_000, 1, &VisionCfg::V41).unwrap();
        assert_eq!((p.best_h, p.best_w, p.n_llm_h, p.n_llm_w), (21462, 42, 511, 1));
        assert_eq!(num_image_tokens_v41(511, 1), 1024);
    }

    #[test]
    fn span_positions() {
        let l = layout_for_grid(37, 37, 5);
        assert_eq!(l.compress_pad(), 2);
        assert_eq!(l.image_start_pos(), 7);
        assert_eq!(l.types[2], TokenType::Start as u8);
        assert_eq!(*l.types.last().unwrap(), TokenType::End as u8);
        assert_eq!(l.span(), (7, 198));
        assert_eq!(l.image_end_pos(), 5 + 2 + 198 - 1);
        // IMAGE_START lands at a position ≡ 3 (mod 4).
        assert_eq!(l.image_start_pos() % 4, 3);
    }
}
