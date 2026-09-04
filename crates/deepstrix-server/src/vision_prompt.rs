//! Image side of a request, between the rendered prompt and the engine.
//!
//! Mirrors the reference `inference/image_processor.py::prepare_vl_inputs`:
//! walk the rendered token stream; at every `<｜deepseek_image｜>`
//! placeholder, preprocess the next image, build its block layout with
//! `start_pos = len(tokens)` (the EXPANDED count so far) and splice the
//! synthetic ids (`VOCAB_SIZE + type`) in place of the placeholder.
//!
//! Also owns the pieces the KV-cache machinery needs to treat images as
//! part of the prompt's identity:
//!
//! * [`ImageSpan`] — `(start, len, content_hash)` of one `[START..END]`
//!   span, in token-index space. The engine gets `(start, len)` (as absolute
//!   KV positions); the snapshot key / byte-aligned LCP mix `hash` into the
//!   byte stream at the START position so two prompts with different
//!   pixels but identical layouts never alias.
//! * [`synthetic_token_bytes`] — the byte-stream encoding of a synthetic
//!   id (they have no vocab text). Every type gets a distinct marker so
//!   different layouts diverge at the first differing slot; START carries
//!   the content hash.
//!
//! Loading rules: only `data:` base64 URLs and absolute local paths are
//! accepted — the server never fetches http(s) URLs.

use std::path::Path;

use color_eyre::eyre::{self, eyre, WrapErr};
use serde::{Deserialize, Serialize};
use v4flash_vision::{image_token_type, layout_for, ImageLayout, PreprocessedImage, TokenType, VOCAB_SIZE};

use crate::openai::types::{ImageInput, ImageSource};

/// Refuse to read image payloads larger than this (decoded bytes).
///
/// Sized so the HTTP body limit derived from it below stays sane on a
/// machine whose free host RAM is measured in single-digit GiB: one
/// 16 MiB JPEG/PNG is well past any phone camera.
pub const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;

/// Maximum `/v1/chat/completions` request body, applied as an axum
/// `DefaultBodyLimit` in `main`.
///
/// MUST be derived from [`MAX_IMAGE_BYTES`], not chosen independently:
/// axum's own default is 2 MiB, so before this existed every `data:` URL
/// over ~1.4 MiB (base64 inflates by 4/3) was rejected by the framework
/// with an opaque 413 that never mentioned images, and the 64 MiB check
/// in [`load_image_bytes`] was dead for data URLs. The slack term covers
/// base64 padding, the JSON envelope and the prompt text.
pub const MAX_REQUEST_BODY_BYTES: usize = MAX_IMAGE_BYTES * 4 / 3 + 8 * 1024 * 1024;

/// One image's `[IMAGE_START ..= IMAGE_END]` span in a token stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSpan {
    /// Index of IMAGE_START in the token stream (NOT the first compress pad).
    pub start: u32,
    /// Tokens from IMAGE_START through IMAGE_END inclusive.
    pub len: u32,
    /// `PreprocessedImage::content_hash` (blake3 of the normalised patches).
    pub hash: [u8; 32],
}

impl ImageSpan {
    #[inline]
    pub fn end_exclusive(&self) -> u32 {
        self.start + self.len
    }
    /// `(start, len)` in the form the engine's `image_spans` argument takes.
    #[inline]
    pub fn as_pair(&self) -> (u32, u32) {
        (self.start, self.len)
    }
}

/// A preprocessed image plus its block layout at a fixed prompt position.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub image: PreprocessedImage,
    pub layout: ImageLayout,
}

impl PreparedImage {
    pub fn span(&self) -> ImageSpan {
        let (start, len) = self.layout.span();
        ImageSpan { start, len, hash: self.image.content_hash }
    }
    /// Token index of the first block token (first compress pad).
    pub fn block_start(&self) -> usize {
        self.layout.start_pos as usize
    }
    pub fn block_len(&self) -> usize {
        self.layout.types.len()
    }
    pub fn block_end(&self) -> usize {
        self.block_start() + self.block_len()
    }
}

/// Output of [`expand_images`]: the expanded token stream plus per-image data.
#[derive(Debug, Clone, Default)]
pub struct VlPrompt {
    pub tokens: Vec<i32>,
    /// In stream order; `spans[i] == images[i].span()`.
    pub images: Vec<PreparedImage>,
    pub spans: Vec<ImageSpan>,
}

// ------------------------------------------------------------ loading

/// What the server is allowed to read as an image source.
///
/// Reading arbitrary absolute paths on request of an unauthenticated HTTP
/// client is a filesystem read primitive AND an existence/type oracle, so
/// it is OFF unless the operator opts in with `--allow-image-dir`. The
/// default bind is loopback, but nothing in the server authenticates, so
/// the blast radius scales with whatever `--addr` the operator picks.
#[derive(Debug, Clone, Default)]
pub struct ImagePolicy {
    /// Canonicalised roots under which absolute local image paths are
    /// accepted. Empty (the default) disables local paths entirely;
    /// `data:` base64 URLs always work.
    pub allow_local_dirs: Vec<std::path::PathBuf>,
}

impl ImagePolicy {
    /// Canonicalise each root, dropping (with a warning) any that does
    /// not resolve to an existing directory.
    pub fn from_dirs<I: IntoIterator<Item = std::path::PathBuf>>(dirs: I) -> ImagePolicy {
        let mut allow_local_dirs = Vec::new();
        for d in dirs {
            match std::fs::canonicalize(&d) {
                Ok(c) if c.is_dir() => allow_local_dirs.push(c),
                Ok(c) => tracing::warn!(dir = %c.display(), "--allow-image-dir is not a directory; ignored"),
                Err(e) => tracing::warn!(dir = %d.display(), error = %e, "--allow-image-dir cannot be resolved; ignored"),
            }
        }
        ImagePolicy { allow_local_dirs }
    }

    fn permits(&self, canonical: &Path) -> bool {
        self.allow_local_dirs.iter().any(|root| canonical.starts_with(root))
    }
}

/// Resolve an `image_url` part to raw file bytes. `data:` payloads are
/// base64-decoded; absolute local paths are read only when `policy`
/// allows them; everything else is an error with a message suitable for
/// an HTTP 400 body.
pub fn load_image_bytes(input: &ImageInput, policy: &ImagePolicy) -> eyre::Result<Vec<u8>> {
    match &input.source {
        ImageSource::DataBase64 { media_type, payload } => {
            let bytes = decode_base64(payload)
                .wrap_err_with(|| format!("image_url data: URL ({media_type}): invalid base64"))?;
            if bytes.len() > MAX_IMAGE_BYTES {
                return Err(eyre!(
                    "image_url data: payload is {} bytes, max {MAX_IMAGE_BYTES}",
                    bytes.len()
                ));
            }
            if bytes.is_empty() {
                return Err(eyre!("image_url data: URL has an empty payload"));
            }
            Ok(bytes)
        }
        ImageSource::LocalFile(path) => read_local_image(path, policy),
        ImageSource::Unsupported(url) => {
            let shown: String = url.chars().take(96).collect();
            Err(eyre!(
                "image_url {shown:?} is not supported: the server does not fetch remote URLs; \
                 send a `data:<type>;base64,...` URL or an absolute local file path"
            ))
        }
    }
}

/// One message for every rejection reason. Distinguishable errors here
/// ("no such file" vs "not a regular file" vs "outside the allowlist")
/// would turn the endpoint into a filesystem existence/type oracle.
fn local_path_denied() -> color_eyre::eyre::Report {
    eyre!(
        "local image paths are not readable by this server; send a \
         `data:<type>;base64,...` URL, or start the server with \
         --allow-image-dir <DIR> for a directory it may read"
    )
}

fn read_local_image(path: &Path, policy: &ImagePolicy) -> eyre::Result<Vec<u8>> {
    if policy.allow_local_dirs.is_empty() || !path.is_absolute() {
        return Err(local_path_denied());
    }
    // Canonicalise BEFORE the allowlist test: `/allowed/../etc/shadow`
    // is "absolute" and starts_with("/allowed") on the raw path.
    let canonical = std::fs::canonicalize(path).map_err(|_| local_path_denied())?;
    if !policy.permits(&canonical) {
        return Err(local_path_denied());
    }
    let meta = std::fs::metadata(&canonical).map_err(|_| local_path_denied())?;
    if !meta.is_file() {
        return Err(local_path_denied());
    }
    if meta.len() as usize > MAX_IMAGE_BYTES {
        return Err(eyre!(
            "image is {} bytes, max {MAX_IMAGE_BYTES}",
            meta.len()
        ));
    }
    std::fs::read(&canonical).map_err(|_| local_path_denied())
}

/// Base64 decoder (standard and URL-safe alphabets, whitespace ignored,
/// padding optional). Kept in-crate: no `base64` dependency is approved.
pub fn decode_base64(s: &str) -> eyre::Result<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 26,
            b'0'..=b'9' => (c - b'0') as u32 + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        })
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut nbits: u32 = 0;
    let mut seen_pad = false;
    for (i, &c) in s.as_bytes().iter().enumerate() {
        match c {
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            b'=' => {
                seen_pad = true;
                continue;
            }
            _ => {}
        }
        if seen_pad {
            return Err(eyre!("base64: data after padding at byte {i}"));
        }
        let v = val(c).ok_or_else(|| eyre!("base64: invalid character {:?} at byte {i}", c as char))?;
        acc = (acc << 6) | v;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push(((acc >> nbits) & 0xff) as u8);
        }
    }
    // A single leftover sextet is a malformed length (1 mod 4).
    if nbits >= 6 {
        return Err(eyre!("base64: invalid length"));
    }
    Ok(out)
}

// ------------------------------------------------------------ expansion

/// `prepare_vl_inputs`: load + preprocess every image (in order) and splice
/// their synthetic blocks over the placeholder ids in `tokens`.
///
/// `inputs.len()` must equal the number of placeholders in `tokens`.
pub fn expand_images(
    tokens: Vec<i32>,
    inputs: &[&ImageInput],
    placeholder: i32,
    policy: &ImagePolicy,
) -> eyre::Result<VlPrompt> {
    let n_ph = tokens.iter().filter(|&&t| t == placeholder).count();
    if n_ph != inputs.len() {
        return Err(eyre!(
            "found {n_ph} image placeholder tokens but {} images",
            inputs.len()
        ));
    }
    if inputs.is_empty() {
        return Ok(VlPrompt { tokens, images: Vec::new(), spans: Vec::new() });
    }
    let mut images = Vec::with_capacity(inputs.len());
    for (i, inp) in inputs.iter().enumerate() {
        let bytes = load_image_bytes(inp, policy).wrap_err_with(|| format!("image {i}"))?;
        let t0 = std::time::Instant::now();
        let img = v4flash_vision::preprocess(&bytes).wrap_err_with(|| format!("image {i}"))?;
        tracing::debug!(
            image = i,
            bytes = bytes.len(),
            n_vit = format!("{}x{}", img.n_vit_h, img.n_vit_w),
            ms = t0.elapsed().as_millis() as u64,
            "image preprocessed"
        );
        images.push(img);
    }
    expand_prepared(tokens, placeholder, images)
}

/// [`expand_images`] on already-preprocessed images (no I/O; unit-testable
/// with synthetic `PreprocessedImage`s).
pub fn expand_prepared(
    tokens: Vec<i32>,
    placeholder: i32,
    images: Vec<PreprocessedImage>,
) -> eyre::Result<VlPrompt> {
    let n_ph = tokens.iter().filter(|&&t| t == placeholder).count();
    if n_ph != images.len() {
        return Err(eyre!(
            "found {n_ph} image placeholder tokens but {} images",
            images.len()
        ));
    }
    let mut out = VlPrompt {
        tokens: Vec::with_capacity(tokens.len() + images.len() * 400),
        images: Vec::with_capacity(images.len()),
        spans: Vec::with_capacity(images.len()),
    };
    let mut it = images.into_iter();
    for tok in tokens {
        if tok != placeholder {
            out.tokens.push(tok);
            continue;
        }
        let image = it.next().expect("counted above");
        let layout = layout_for(&image, out.tokens.len() as u32);
        debug_assert_eq!(layout.start_pos as usize, out.tokens.len());
        out.tokens.extend(layout.token_ids().iter().map(|&id| id as i32));
        let prepared = PreparedImage { image, layout };
        out.spans.push(prepared.span());
        out.images.push(prepared);
    }
    Ok(out)
}

// ------------------------------------------------------------ span helpers

/// Content hash of the image whose IMAGE_START sits at token index `idx`.
pub fn span_hash_at(spans: &[ImageSpan], idx: usize) -> Option<&[u8; 32]> {
    spans.iter().find(|s| s.start as usize == idx).map(|s| &s.hash)
}

/// `true` for a synthetic image-slot id (`>= VOCAB_SIZE`).
#[inline]
pub fn is_image_token(id: i32) -> bool {
    id >= 0 && (id as u32) >= VOCAB_SIZE
}

/// `true` for the IMAGE_START synthetic id.
#[inline]
pub fn is_image_start(id: i32) -> bool {
    id == (VOCAB_SIZE + TokenType::Start as u32) as i32
}

/// Byte-stream encoding of a synthetic token for snapshot keys and the
/// byte-aligned LCP (synthetic ids have no vocab text). `None` for ordinary
/// ids and for ids outside the 5 synthetic types.
///
/// Layout: `0x00 "<IMG" <type byte>` (+ 32 hash bytes for START when the
/// span is known). The NUL prefix keeps it clear of ordinary prose, but
/// it is NOT a reserved or escaped marker: JSON permits U+0000 and the
/// GPT-2 byte decoder maps a real token to `0x00`, so a client can emit
/// these bytes. The property that actually matters is narrower — forging
/// a START marker additionally requires the target image's 32-byte
/// content hash, which is internal, so this is a cache-collision cost
/// bound by knowing a blake3 digest, not a forgeable identity.
pub fn synthetic_token_bytes(id: i32, hash: Option<&[u8; 32]>) -> Option<Vec<u8>> {
    if id < 0 {
        return None;
    }
    let ty = image_token_type(id as u32)?;
    let tag = match TokenType::from_u8(ty)? {
        TokenType::Start => b'S',
        TokenType::Pad => b'P',
        TokenType::Image => b'I',
        TokenType::NewLine => b'N',
        TokenType::End => b'E',
    };
    let mut v = Vec::with_capacity(6 + 32);
    v.extend_from_slice(b"\x00<IMG");
    v.push(tag);
    if ty == TokenType::Start as u8 {
        if let Some(h) = hash {
            v.extend_from_slice(h);
        }
    }
    Some(v)
}

/// The spans of `spans` that fall inside token-index range `[lo, hi)`,
/// re-based so `lo` → 0. Errors if any span straddles either edge (a
/// suffix boundary can never legitimately cut through an image block).
pub fn spans_in_range(spans: &[ImageSpan], lo: usize, hi: usize) -> eyre::Result<Vec<ImageSpan>> {
    let mut out = Vec::new();
    for s in spans {
        let (a, b) = (s.start as usize, s.end_exclusive() as usize);
        if b <= lo || a >= hi {
            continue;
        }
        if a < lo || b > hi {
            return Err(eyre!(
                "image span [{a}, {b}) straddles the token range [{lo}, {hi})"
            ));
        }
        out.push(ImageSpan { start: (a - lo) as u32, len: s.len, hash: s.hash });
    }
    Ok(out)
}

/// `true` if any span of `spans` is CUT by a split at token index `n` —
/// i.e. `n` falls strictly inside `[start, end)`. This is the test a
/// prefill / snapshot boundary needs: a cut with whole image blocks on
/// both sides of it is legal, only a cut through one is not.
/// (`het::image_spans::cut_ok` is the engine-side equivalent.)
pub fn spans_straddle(spans: &[ImageSpan], n: usize) -> bool {
    spans
        .iter()
        .any(|s| (s.start as usize) < n && (s.end_exclusive() as usize) > n)
}

/// `true` if any span of `spans` overlaps token-index range `[lo, hi)`.
pub fn any_span_in_range(spans: &[ImageSpan], lo: usize, hi: usize) -> bool {
    spans
        .iter()
        .any(|s| (s.start as usize) < hi && (s.end_exclusive() as usize) > lo)
}

/// Shift every span by `delta` token positions.
pub fn shift_spans(spans: &[ImageSpan], delta: i64) -> Vec<ImageSpan> {
    spans
        .iter()
        .map(|s| ImageSpan {
            start: (s.start as i64 + delta) as u32,
            len: s.len,
            hash: s.hash,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use v4flash_vision::{layout::layout_for_grid, synthetic_token_id};

    fn fake_image(n_vit_h: u32, n_vit_w: u32, seed: u8) -> PreprocessedImage {
        // Tiny patch buffer — only the grid + hash matter for expansion.
        let patches = vec![seed as f32; 8];
        let content_hash = *blake3::hash(&[seed, n_vit_h as u8, n_vit_w as u8]).as_bytes();
        PreprocessedImage { patches, n_vit_h, n_vit_w, content_hash }
    }

    const PH: i32 = 129264;

    #[test]
    fn base64_roundtrip_and_errors() {
        assert_eq!(decode_base64("").unwrap(), b"");
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64("aGVsbG8").unwrap(), b"hello"); // no padding
        assert_eq!(decode_base64("aGVs\nbG8=\n").unwrap(), b"hello"); // whitespace
        assert_eq!(decode_base64("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        assert_eq!(decode_base64("_-8=").unwrap(), decode_base64("/+8=").unwrap()); // url-safe
        assert_eq!(decode_base64("iVBORw0KGgo=").unwrap(), b"\x89PNG\r\n\x1a\n");
        assert!(decode_base64("a").is_err());
        assert!(decode_base64("ab$d").is_err());
        assert!(decode_base64("ab=cd").is_err());
    }

    fn input(url: &str) -> ImageInput {
        ImageInput { source: ImageSource::classify(url), detail: None }
    }

    #[test]
    fn load_rejects_http_and_relative_and_missing() {
        let deny = ImagePolicy::default();
        let e = load_image_bytes(&input("https://example.com/a.png"), &deny)
            .unwrap_err()
            .to_string();
        assert!(e.contains("does not fetch remote URLs"), "{e}");
        assert!(load_image_bytes(&input("a.png"), &deny).is_err());
        assert!(load_image_bytes(&input("/nonexistent/deepstrix-test-image.png"), &deny).is_err());
        assert_eq!(
            load_image_bytes(&input("data:image/png;base64,aGVsbG8="), &deny).unwrap(),
            b"hello"
        );
        assert!(load_image_bytes(&input("data:image/png;base64,a$"), &deny).is_err());
    }

    #[test]
    fn local_paths_are_denied_by_default_and_gated_by_the_allowlist() {
        let dir = std::env::temp_dir().join(format!("deepstrix-vp-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let inside = dir.join("sub").join("ok.png");
        std::fs::write(&inside, b"pixels").unwrap();
        let outside = dir.parent().unwrap().join(format!(
            "deepstrix-vp-outside-{}.png",
            std::process::id()
        ));
        std::fs::write(&outside, b"nope").unwrap();

        // Default policy: local paths off entirely, and the message must
        // not distinguish "missing" from "denied" (no existence oracle).
        let deny = ImagePolicy::default();
        let e_present = load_image_bytes(&input(inside.to_str().unwrap()), &deny)
            .unwrap_err()
            .to_string();
        let e_absent = load_image_bytes(&input("/definitely/not/here.png"), &deny)
            .unwrap_err()
            .to_string();
        assert_eq!(e_present, e_absent);

        let allow = ImagePolicy::from_dirs([dir.clone()]);
        assert_eq!(
            load_image_bytes(&input(inside.to_str().unwrap()), &allow).unwrap(),
            b"pixels"
        );
        // Outside the root, and traversal back out of it, are both denied.
        assert!(load_image_bytes(&input(outside.to_str().unwrap()), &allow).is_err());
        let escape = format!("{}/sub/../../{}", dir.display(), outside.file_name().unwrap().to_str().unwrap());
        assert!(load_image_bytes(&input(&escape), &allow).is_err());
        // Directories are not images.
        assert!(load_image_bytes(&input(dir.join("sub").to_str().unwrap()), &allow).is_err());

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&outside).ok();
    }

    #[test]
    fn straddle_test_permits_cuts_between_image_blocks() {
        let sp = |start, len| ImageSpan { start, len, hash: [0u8; 32] };
        let spans = vec![sp(10, 4), sp(40, 4)];
        // Inside a block => straddle.
        assert!(spans_straddle(&spans, 11));
        assert!(spans_straddle(&spans, 13));
        assert!(spans_straddle(&spans, 41));
        // Block edges and gaps are legal cut points — in particular a cut
        // with one image before it and another after it, which the old
        // `any_span_in_range` pair-test rejected and thereby disabled the
        // system-prefix snapshot for multi-image conversations.
        for n in [0usize, 5, 10, 14, 20, 40, 44, 60] {
            assert!(!spans_straddle(&spans, n), "n={n}");
        }
        // A legal cut is exactly one `spans_in_range` can split on.
        assert!(spans_in_range(&spans, 0, 20).is_ok());
        assert!(spans_in_range(&spans, 20, 60).is_ok());
        assert!(spans_in_range(&spans, 0, 12).is_err());
    }

    #[test]
    fn expand_splices_block_at_running_position() {
        // prompt: [0, 128803, 11, PH, 12, 128804]  (PH at index 3)
        let tokens = vec![0, 128803, 11, PH, 12, 128804];
        let img = fake_image(37, 37, 1); // 512x512 → 13x13 LLM grid, 198 span tokens
        let vl = expand_prepared(tokens, PH, vec![img.clone()]).unwrap();
        let layout = layout_for_grid(37, 37, 3);
        assert_eq!(vl.images.len(), 1);
        assert_eq!(vl.images[0].layout, layout);
        // start_pos 3 → compress_pad = 3 - 3%4 = 0 → START at 3.
        assert_eq!(layout.compress_pad(), 0);
        assert_eq!(vl.spans, vec![ImageSpan { start: 3, len: 198, hash: img.content_hash }]);
        assert_eq!(vl.tokens.len(), 5 + layout.types.len());
        assert_eq!(&vl.tokens[..3], &[0, 128803, 11]);
        assert_eq!(vl.tokens[3], synthetic_token_id(TokenType::Start as u8) as i32);
        assert_eq!(vl.tokens[3 + 198 - 1], synthetic_token_id(TokenType::End as u8) as i32);
        assert_eq!(&vl.tokens[3 + 198..], &[12, 128804]);
        // The engine's per-row is_image predicate holds exactly on the block.
        for (i, &t) in vl.tokens.iter().enumerate() {
            assert_eq!(is_image_token(t), (3..3 + 198).contains(&i), "idx {i}");
        }
    }

    #[test]
    fn expand_two_images_second_start_pos_uses_expanded_count() {
        // PH at 1 and 3; the second image's start_pos must account for the
        // first image's expansion (prepare_vl_inputs uses len(tokens)).
        let tokens = vec![0, PH, 5, PH, 6];
        let a = fake_image(37, 37, 1);
        let b = fake_image(42, 74, 2);
        let vl = expand_prepared(tokens, PH, vec![a, b]).unwrap();
        let la = layout_for_grid(37, 37, 1); // compress_pad = 3-1 = 2 → START at 3
        assert_eq!(la.compress_pad(), 2);
        assert_eq!(vl.images[0].layout, la);
        let second_start = 1 + la.types.len() as u32 + 1;
        let lb = layout_for_grid(42, 74, second_start);
        assert_eq!(vl.images[1].layout, lb);
        assert_eq!(vl.spans[0].as_pair(), la.span());
        assert_eq!(vl.spans[1].as_pair(), lb.span());
        assert_eq!(vl.spans[1].start % 4, 3);
        assert_eq!(vl.tokens.len(), 3 + la.types.len() + lb.types.len());
        assert_eq!(*vl.tokens.last().unwrap(), 6);
        assert_ne!(vl.spans[0].hash, vl.spans[1].hash);
    }

    #[test]
    fn expand_count_mismatch_is_error() {
        assert!(expand_prepared(vec![0, PH, PH], PH, vec![fake_image(37, 37, 1)]).is_err());
        assert!(expand_prepared(vec![0, 1], PH, vec![fake_image(37, 37, 1)]).is_err());
        let vl = expand_prepared(vec![0, 1], PH, vec![]).unwrap();
        assert_eq!(vl.tokens, vec![0, 1]);
        assert!(vl.spans.is_empty());
    }

    #[test]
    fn synthetic_bytes_distinguish_types_and_hashes() {
        let h1 = [1u8; 32];
        let h2 = [2u8; 32];
        let start = synthetic_token_id(0) as i32;
        let b1 = synthetic_token_bytes(start, Some(&h1)).unwrap();
        let b2 = synthetic_token_bytes(start, Some(&h2)).unwrap();
        assert_ne!(b1, b2);
        assert_eq!(b1.len(), 6 + 32);
        assert!(b1.starts_with(b"\x00<IMGS"));
        let types: Vec<Vec<u8>> = (0..5u8)
            .map(|t| synthetic_token_bytes(synthetic_token_id(t) as i32, None).unwrap())
            .collect();
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(types[i] == types[j], i == j);
            }
        }
        assert_eq!(synthetic_token_bytes(129263, None), None);
        assert_eq!(synthetic_token_bytes(129285, None), None);
        assert_eq!(synthetic_token_bytes(-1, None), None);
    }

    #[test]
    fn spans_in_range_rebases_and_rejects_straddle() {
        let s = |start: u32, len: u32| ImageSpan { start, len, hash: [0; 32] };
        let spans = vec![s(10, 5), s(40, 8)];
        assert_eq!(spans_in_range(&spans, 0, 100).unwrap(), spans);
        assert_eq!(spans_in_range(&spans, 20, 100).unwrap(), vec![s(20, 8)]);
        assert_eq!(spans_in_range(&spans, 15, 40).unwrap(), Vec::<ImageSpan>::new());
        assert!(spans_in_range(&spans, 12, 100).is_err());
        assert!(spans_in_range(&spans, 0, 44).is_err());
        assert!(any_span_in_range(&spans, 14, 15));
        assert!(!any_span_in_range(&spans, 15, 40));
        assert_eq!(shift_spans(&spans, -10), vec![s(0, 5), s(30, 8)]);
        assert_eq!(span_hash_at(&spans, 40), Some(&[0u8; 32]));
        assert_eq!(span_hash_at(&spans, 41), None);
    }
}
