//! Vision-Exp text-side geometry for batched prefill: per-row image
//! visibility, the widened raw-attention window, and chunk / lane cut
//! planning so an `[IMAGE_START .. IMAGE_END]` span is always prefilled
//! inside ONE KV-visible unit.
//!
//! Reference: DeepSeek `model.py` `get_image_visible` /
//! `get_window_topk_idxs_visible` (start_pos == 0 only there; we
//! generalise to absolute positions — see [`raw_window`]) and `Gate.forward`
//! (`image_mask = input_ids >= vocab_size`).
//!
//! Everything in here is host-only integer arithmetic and is covered by
//! `tests/image_visibility.rs` against a Python-derived table.

use color_eyre::eyre::{self, eyre};

use crate::config::{N_VOCAB, SWA_WINDOW};

/// Synthetic image token ids are `N_VOCAB + type`, type ∈ 0..4
/// (IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END).
pub const IMAGE_TOKEN_BASE: i32 = N_VOCAB as i32;
pub const IMAGE_START: i32 = IMAGE_TOKEN_BASE;
pub const IMAGE_PAD: i32 = IMAGE_TOKEN_BASE + 1;
pub const IMAGE: i32 = IMAGE_TOKEN_BASE + 2;
pub const IMAGE_NEW_LINE: i32 = IMAGE_TOKEN_BASE + 3;
pub const IMAGE_END: i32 = IMAGE_TOKEN_BASE + 4;

/// `vision_max_n_token` — clamps `left` to `MAX-1` and `right` to `MAX`,
/// and caps the widened raw window at `SWA_WINDOW + MAX` keys.
pub const VISION_MAX_N_TOKEN: u32 = 384;

/// Widest raw window any prefill row can have (text rows: `SWA_WINDOW`).
/// `attention_swa_batched`'s LDS arrays and the `attn_scores` stride
/// budget are sized against this — see `ATTN_SWA_BATCHED_MAX_KV`.
pub const IMAGE_RAW_WINDOW_MAX: u32 = SWA_WINDOW + VISION_MAX_N_TOKEN;

/// Router-side image flag (`Gate.forward`: `input_ids >= vocab_size`).
/// Includes the compress-pad tokens that precede IMAGE_START.
#[inline]
pub fn is_image_token(id: i32) -> bool {
    id >= IMAGE_TOKEN_BASE
}

/// One image span: `(start_pos, len)` with `start_pos` the absolute
/// position of IMAGE_START and `len` counting through IMAGE_END inclusive.
pub type ImageSpan = (u32, u32);

/// Spans must be sorted, non-overlapping, at least 2 tokens (START+END)
/// and lie inside `[pos0, pos0 + t)`.
pub fn validate_spans(spans: &[ImageSpan], pos0: u32, t: usize) -> eyre::Result<()> {
    let end_excl = pos0 as u64 + t as u64;
    let mut prev_end: Option<u64> = None;
    for (k, &(s, len)) in spans.iter().enumerate() {
        if len < 2 {
            return Err(eyre!("image_spans[{k}]: len {len} < 2 (START+END)"));
        }
        let e = s as u64 + len as u64; // exclusive
        if (s as u64) < pos0 as u64 || e > end_excl {
            return Err(eyre!(
                "image_spans[{k}] = ({s}, {len}) not inside the prompt window [{pos0}, {end_excl})"
            ));
        }
        if let Some(pe) = prev_end {
            if (s as u64) < pe {
                return Err(eyre!("image_spans[{k}] = ({s}, {len}) overlaps / is unsorted"));
            }
        }
        prev_end = Some(e);
    }
    Ok(())
}

/// Per-row `(left, right)` visibility for the `b` rows at absolute
/// positions `[pos0, pos0 + b)`, clamped like `get_image_visible`
/// (`left ≤ MAX-1`, `right ≤ MAX`). Text rows are `(0, 0)`.
///
/// Errors if any span straddles the row range — an image span must be
/// prefilled inside one KV-visible unit (chunk / lane).
pub fn rows_visibility(pos0: u32, b: usize, spans: &[ImageSpan]) -> eyre::Result<Vec<(u32, u32)>> {
    let mut vis = vec![(0u32, 0u32); b];
    if b == 0 {
        return Ok(vis);
    }
    let end_excl = pos0 as u64 + b as u64;
    for &(s, len) in spans {
        let e_incl = s as u64 + len as u64 - 1;
        let inside_start = (s as u64) >= pos0 as u64 && (s as u64) < end_excl;
        let inside_end = e_incl >= pos0 as u64 && e_incl < end_excl;
        if !inside_start && !inside_end {
            // Either entirely outside the range, or covering all of it.
            if (s as u64) < pos0 as u64 && e_incl >= end_excl {
                return Err(eyre!(
                    "image span ({s}, {len}) covers the whole row range [{pos0}, {end_excl}) — \
                     image spans must be prefilled inside one chunk"
                ));
            }
            continue;
        }
        if inside_start != inside_end {
            return Err(eyre!(
                "image span ({s}, {len}) straddles the row range [{pos0}, {end_excl}) — \
                 image spans must be prefilled inside one chunk"
            ));
        }
        for p in s as u64..=e_incl {
            let i = (p - pos0 as u64) as usize;
            let left = ((p - s as u64) as u32).min(VISION_MAX_N_TOKEN - 1);
            let right = ((e_incl - p) as u32).min(VISION_MAX_N_TOKEN);
            vis[i] = (left, right);
        }
    }
    Ok(vis)
}

/// Raw-KV window of row `i` of a batch appended at cache slots
/// `[n_raw_before .. n_raw_before + b)`, as `(offset_slot, count)`.
///
/// Mirrors `get_window_topk_idxs_visible` with `W = SWA_WINDOW`:
///   start = max(idx - max(W-1, left), 0)      (absolute; slot-space here)
///   end   = idx + right                        (inclusive)
///   count = min(end - start + 1, W + MAX)      (the `width` cap)
/// Text rows (`left = right = 0`) reduce to the causal trailing window
/// `[max(0, idx-W+1) .. idx]`, bit-identical to the pre-vision code.
#[inline]
pub fn raw_window(n_raw_before: u32, i: u32, left: u32, right: u32) -> (u32, u32) {
    let causal_end = n_raw_before + i + 1; // exclusive slot of self
    let back = (SWA_WINDOW - 1).max(left) + 1; // rows incl. self
    let offset = causal_end.saturating_sub(back);
    let end_excl = causal_end + right;
    let count = (end_excl - offset).min(IMAGE_RAW_WINDOW_MAX);
    (offset, count)
}

/// Contiguous runs of image-token rows (router-side flag) as
/// `(first_row, n_rows)`.
pub fn image_runs(tokens: &[i32]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if is_image_token(tokens[i]) {
            let s = i;
            while i < tokens.len() && is_image_token(tokens[i]) {
                i += 1;
            }
            runs.push((s, i - s));
        } else {
            i += 1;
        }
    }
    runs
}

/// A cut between positions `p-1` and `p` is legal iff no span has
/// `start < p <= end` (cutting exactly at a span start or right after its
/// END is fine).
#[inline]
pub fn cut_ok(p: u64, spans: &[ImageSpan]) -> bool {
    spans
        .iter()
        .all(|&(s, len)| !((s as u64) < p && p < s as u64 + len as u64))
}

/// Largest legal cut `q` with `lo < q <= p` (`None` if there is none).
fn largest_cut_le(p: u64, lo: u64, spans: &[ImageSpan]) -> Option<u64> {
    if cut_ok(p, spans) {
        return if p > lo { Some(p) } else { None };
    }
    // p is strictly inside a span: the only candidate below it is that
    // span's start (everything between is inside the same span).
    let s = spans
        .iter()
        .find(|&&(s, len)| (s as u64) < p && p < s as u64 + len as u64)
        .map(|&(s, _)| s as u64)?;
    if s > lo {
        Some(s)
    } else {
        None
    }
}

/// Lane split for a chunk of `b` rows at `pos0`: returns `b_a` (lane A
/// rows). Without spans this is the historical `b.div_ceil(2)`; with
/// spans, the legal cut nearest the middle such that
/// `b_a <= lane_a_cap && b - b_a <= lane_b_cap`. `b >= 2` required.
pub fn lane_split(
    pos0: u32,
    b: usize,
    spans: &[ImageSpan],
    lane_a_cap: usize,
    lane_b_cap: usize,
) -> eyre::Result<usize> {
    let mid = b.div_ceil(2);
    if spans.is_empty() {
        return Ok(mid);
    }
    let lo = 1usize.max(b.saturating_sub(lane_b_cap));
    let hi = lane_a_cap.min(b - 1);
    if lo > hi {
        return Err(eyre!(
            "lane_split: chunk of {b} rows cannot be split into lanes of {lane_a_cap}/{lane_b_cap}"
        ));
    }
    // Search outwards from the middle.
    for d in 0..b {
        for p in [mid.checked_sub(d), mid.checked_add(d)].into_iter().flatten() {
            if p >= lo && p <= hi && cut_ok(pos0 as u64 + p as u64, spans) {
                return Ok(p);
            }
        }
    }
    Err(eyre!(
        "lane_split: no legal lane cut in chunk [{pos0}, {}) — an image span is longer than \
         the lane capacity",
        pos0 as u64 + b as u64
    ))
}

/// Plan the next prefill chunk starting at row `start` (absolute
/// `pos0 + start`) of a `t`-row prompt: returns `(chunk_end, b_a)` where
/// `b_a` is the lane-A row count (`== chunk_b` when `lane_caps` is `None`,
/// i.e. the single-lane driver). Chunk length ≤ `chunk_cap`, the chunk end
/// and the lane cut are legal cuts, lanes fit their caps. Without spans
/// this reproduces the historical fixed chunking exactly.
pub fn plan_chunk(
    pos0: u32,
    start: usize,
    t: usize,
    chunk_cap: usize,
    lane_caps: Option<(usize, usize)>,
    spans: &[ImageSpan],
) -> eyre::Result<(usize, usize)> {
    let abs = |r: usize| pos0 as u64 + r as u64;
    let mut chunk_end = (start + chunk_cap).min(t);
    if spans.is_empty() {
        let cb = chunk_end - start;
        let b_a = match lane_caps {
            Some(_) if cb >= 2 => cb.div_ceil(2),
            _ => cb,
        };
        return Ok((chunk_end, b_a));
    }
    loop {
        let ce = largest_cut_le(abs(chunk_end), abs(start), spans).ok_or_else(|| {
            eyre!(
                "plan_chunk: no legal chunk end in [{}, {}] — an image span is longer than the \
                 chunk capacity {chunk_cap}",
                abs(start) + 1,
                abs(chunk_end)
            )
        })?;
        chunk_end = (ce - pos0 as u64) as usize;
        let cb = chunk_end - start;
        let Some((cap_a, cap_b)) = lane_caps else {
            return Ok((chunk_end, cb));
        };
        if cb < 2 {
            return Ok((chunk_end, cb));
        }
        match lane_split(abs(start) as u32, cb, spans, cap_a, cap_b) {
            Ok(b_a) => return Ok((chunk_end, b_a)),
            Err(e) => {
                // No legal lane cut at this chunk length: shorten the chunk
                // to just before the span that blocks the middle and retry.
                if chunk_end - 1 <= start {
                    return Err(e);
                }
                chunk_end -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_rows_match_causal_window() {
        for n_raw_before in [0u32, 5, 128] {
            for i in 0..300u32 {
                let (off, cnt) = raw_window(n_raw_before, i, 0, 0);
                let causal_end = n_raw_before + i + 1;
                assert_eq!(cnt, causal_end.min(SWA_WINDOW));
                assert_eq!(off, causal_end.saturating_sub(SWA_WINDOW));
            }
        }
    }

    #[test]
    fn image_row_window_widens_both_ways() {
        // span at rows 10..=209 (200 tokens), n_raw_before = 128
        let (off, cnt) = raw_window(128, 100, 90, 109);
        // idx slot = 228; back = max(127, 90)+1 = 128 → off = 229-128 = 101
        assert_eq!(off, 101);
        assert_eq!(cnt, 229 + 109 - 101);
        let (off2, cnt2) = raw_window(128, 200, 190, 9);
        assert_eq!(off2, 329 - 191);
        assert_eq!(cnt2, 329 + 9 - (329 - 191));
        assert!(cnt2 <= IMAGE_RAW_WINDOW_MAX);
    }

    #[test]
    fn straddle_is_error() {
        assert!(rows_visibility(0, 100, &[(90, 20)]).is_err());
        assert!(rows_visibility(100, 100, &[(90, 20)]).is_err());
        assert!(rows_visibility(100, 10, &[(90, 200)]).is_err());
        assert!(rows_visibility(0, 200, &[(90, 20)]).is_ok());
        assert!(rows_visibility(0, 90, &[(90, 20)]).is_ok());
        assert!(rows_visibility(110, 5, &[(90, 20)]).is_ok());
    }

    #[test]
    fn plan_chunk_without_spans_is_historical() {
        assert_eq!(plan_chunk(7, 0, 3000, 1024, Some((512, 512)), &[]).unwrap(), (1024, 512));
        assert_eq!(plan_chunk(7, 2048, 3000, 1024, Some((512, 512)), &[]).unwrap(), (3000, 476));
        assert_eq!(plan_chunk(7, 2048, 3000, 1024, None, &[]).unwrap(), (3000, 952));
        assert_eq!(plan_chunk(0, 0, 1, 1024, Some((512, 512)), &[]).unwrap(), (1, 1));
    }

    #[test]
    fn plan_chunk_moves_cuts_off_spans() {
        // span covering the natural chunk end 1024 and one covering the lane mid.
        let spans = [(400u32, 300u32), (1000, 100)];
        let (ce, b_a) = plan_chunk(0, 0, 3000, 1024, Some((512, 512)), &spans).unwrap();
        assert!(cut_ok(ce as u64, &spans));
        assert!(cut_ok(b_a as u64, &spans));
        assert!(b_a <= 512 && ce - b_a <= 512);
        // continue from ce: next chunk must contain span 2 whole
        let (ce2, b_a2) = plan_chunk(0, ce, 3000, 1024, Some((512, 512)), &spans).unwrap();
        assert!(cut_ok(ce2 as u64, &spans));
        assert!(cut_ok((ce + b_a2) as u64, &spans));
        assert!(ce2 > 1099);
    }
}
