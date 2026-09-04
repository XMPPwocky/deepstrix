//! Preprocess round-trips on synthetic PNGs (CPU only).

use std::io::Cursor;

use v4flash_vision::layout::plan_resize;
use v4flash_vision::preprocess::{decode_rgb, normalize, patchify, preprocess, preprocess_rgb};
use v4flash_vision::resize::Rgb;
use v4flash_vision::{layout_for, PATCH, PATCH_ELEMS};

/// Deterministic pattern; distinct across (x, y, c) within a 14x14 window.
fn pattern(x: u32, y: u32, c: usize) -> u8 {
    ((x * 7 + y * 13 + (c as u32) * 101) & 255) as u8
}

fn png_of(w: u32, h: u32, f: impl Fn(u32, u32, usize) -> u8) -> Vec<u8> {
    let mut img = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(x, y, image::Rgb([f(x, y, 0), f(x, y, 1), f(x, y, 2)]));
        }
    }
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    buf.into_inner()
}

#[test]
fn png_decodes_to_rgb() {
    let png = png_of(30, 20, pattern);
    let rgb = decode_rgb(&png).unwrap();
    assert_eq!((rgb.w, rgb.h), (30, 20));
    for y in 0..20 {
        for x in 0..30 {
            for c in 0..3 {
                assert_eq!(rgb.px(x, y, c), pattern(x, y, c));
            }
        }
    }
}

#[test]
fn identity_size_image_patch_order_is_c_y_x() {
    // 448x392 (w x h): ≥ MIN_PIXELS, multiples of 14, same aspect as its
    // canvas → PIL takes the copy path, so patches are exact pixels.
    let (w, h) = (448u32, 392u32);
    let png = png_of(w, h, pattern);
    let plan = plan_resize(h, w).unwrap();
    assert_eq!((plan.best_h, plan.best_w, plan.n_vit_h, plan.n_vit_w), (392, 448, 28, 32));
    assert_eq!((plan.n_llm_h, plan.n_llm_w), (10, 11));
    let pre = preprocess(&png).unwrap();
    assert_eq!((pre.n_vit_h, pre.n_vit_w), (28, 32));
    assert_eq!(pre.patches.len(), 28 * 32 * PATCH_ELEMS);
    let p = PATCH;
    for ph in 0..28u32 {
        for pw in 0..32u32 {
            let base = ((ph * 32 + pw) as usize) * PATCH_ELEMS;
            for c in 0..3usize {
                for y in 0..p {
                    for x in 0..p {
                        let want = normalize(pattern(pw * p + x, ph * p + y, c));
                        let got = pre.patches[base + c * 196 + (y * p + x) as usize];
                        assert_eq!(got, want, "patch ({ph},{pw}) c{c} y{y} x{x}");
                    }
                }
            }
        }
    }
    // Direct patchify on the decoded image agrees with the full pipeline.
    let rgb = decode_rgb(&png).unwrap();
    assert_eq!(patchify(&rgb, 28, 32), pre.patches);
    // Hash is stable and content-dependent.
    let pre2 = preprocess(&png).unwrap();
    assert_eq!(pre.content_hash, pre2.content_hash);
    let png3 = png_of(w, h, |x, y, c| pattern(x, y, c).wrapping_add(1));
    assert_ne!(preprocess(&png3).unwrap().content_hash, pre.content_hash);
    // Layout for it.
    let l = layout_for(&pre, 0);
    assert_eq!((l.n_llm_h, l.n_llm_w), (10, 11));
    assert_eq!(l.perm.len(), 110);
    // rows=10, row_len=12, pad_last = 5*12%2*2 = 0 → 3 + 1 + 120 + 0 + 1
    assert_eq!(l.types.len(), 125);
}

#[test]
fn small_image_is_upscaled_and_letterboxed() {
    // 28x42 (w x h): upscaled to ≥147456 px → canvas 322x476, contain
    // gives 317x476 pasted at x=2 → gray columns 0..2 and 319..322.
    let png = png_of(28, 42, |_, _, c| [200, 30, 90][c]);
    let plan = plan_resize(42, 28).unwrap();
    assert_eq!((plan.best_h, plan.best_w), (476, 322));
    assert!(!plan.plain_resize);
    let pre = preprocess(&png).unwrap();
    assert_eq!((pre.n_vit_h, pre.n_vit_w), (34, 23));
    let gray = normalize(127);
    // patch (0,0), channel 0, y=0: x=0,1 gray then 200-valued.
    let base = 0;
    assert_eq!(pre.patches[base], gray);
    assert_eq!(pre.patches[base + 1], gray);
    assert_eq!(pre.patches[base + 2], normalize(200));
    assert_eq!(pre.patches[base + 196 + 2], normalize(30)); // c=1
    assert_eq!(pre.patches[base + 392 + 2], normalize(90)); // c=2
    // last patch column (pw=22) x=13 → canvas x=321 → gray.
    let last = (22usize) * PATCH_ELEMS;
    assert_eq!(pre.patches[last + 13], gray);
    assert_eq!(pre.patches[last + 12], gray); // x=320
    assert_eq!(pre.patches[last + 11], gray); // x=319
    assert_eq!(pre.patches[last + 10], normalize(200)); // x=318 = last content column
}

#[test]
fn wide_image_takes_plain_resize() {
    // w >= 8h → squash-resize, no gray border.
    let png = png_of(800, 100, |_, _, c| [10, 20, 30][c]);
    let plan = plan_resize(100, 800).unwrap();
    assert!(plan.plain_resize);
    let pre = preprocess(&png).unwrap();
    assert_eq!((pre.n_vit_h, pre.n_vit_w), (plan.n_vit_h, plan.n_vit_w));
    assert!(pre.patches.chunks(196).enumerate().all(|(i, ch)| {
        let c = i % 3;
        ch.iter().all(|&v| v == normalize([10, 20, 30][c]))
    }));
}

#[test]
fn preprocess_rgb_reports_plan() {
    let img = Rgb::filled(512, 512, [1, 2, 3]);
    let (pre, plan) = preprocess_rgb(&img).unwrap();
    assert_eq!((plan.best_h, plan.best_w), (518, 518));
    assert_eq!((pre.n_vit_h, pre.n_vit_w), (37, 37));
    assert_eq!(layout_for(&pre, 3).types.len(), 198);
}

#[test]
fn rejects_garbage() {
    assert!(preprocess(b"not an image").is_err());
}
