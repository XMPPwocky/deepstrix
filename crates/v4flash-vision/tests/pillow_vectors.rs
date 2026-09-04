//! Byte-exactness of the PIL port against REAL Pillow.
//!
//! `crates/v4flash-vision/src/resize.rs` reimplements Pillow's
//! `Image.resize` (BICUBIC, antialiased on downscale) and
//! `ImageOps.pad`. Its own unit tests only check internal consistency,
//! so the port's faithfulness rested on a reading of `Resample.c`.
//!
//! These vectors close that: digests produced by Pillow 12.3.0 itself
//! (see `scripts/gen_pillow_resize_vectors.py`). The pad cases include
//! deltas ≡ 3 (mod 4), where Pillow's `round()` centring offset and the
//! `int()` truncation used by much older Pillow differ by one pixel —
//! a shift that would move every patch of a letterboxed image.

use serde::Deserialize;
use v4flash_vision::resize::{pad_contain, resize_bicubic, Rgb};

#[derive(Deserialize)]
struct Cases {
    pillow: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    op: String,
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
    seed: u32,
    res_w: u32,
    res_h: u32,
    len: usize,
    fnv1a64: String,
    head: String,
    tail: String,
}

fn pattern(x: u32, y: u32, c: u32) -> u8 {
    ((x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)).wrapping_add(c * 101)) & 255) as u8
}

/// Same generator as the Python side.
fn make(w: u32, h: u32, seed: u32) -> Rgb {
    let mut data = vec![0u8; (w as usize) * (h as usize) * 3];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) as usize) * 3;
            data[i] = pattern(x + seed, y, 0);
            data[i + 1] = pattern(x, y + seed, 1);
            data[i + 2] = pattern(x, y, 2);
        }
    }
    Rgb::new(w, h, data)
}

fn fnv1a64(b: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn pil_resize_and_pad_are_byte_exact_vs_real_pillow() {
    let raw = include_str!("data/pillow_resize_cases.json");
    let cases: Cases = serde_json::from_str(raw).expect("parse pillow_resize_cases.json");
    assert!(
        cases.pillow.starts_with("12."),
        "vectors were generated with Pillow {} — regenerate or update this assertion",
        cases.pillow
    );
    assert!(cases.cases.len() >= 20);
    for (i, c) in cases.cases.iter().enumerate() {
        let src = make(c.src_w, c.src_h, c.seed);
        let out = match c.op.as_str() {
            "resize" => resize_bicubic(&src, c.out_w, c.out_h).unwrap(),
            "pad" => pad_contain(&src, c.out_w, c.out_h, [127, 127, 127]).unwrap(),
            o => panic!("unknown op {o}"),
        };
        let label = format!(
            "case {i}: {} {}x{} -> {}x{}",
            c.op, c.src_w, c.src_h, c.out_w, c.out_h
        );
        assert_eq!((out.w, out.h), (c.res_w, c.res_h), "{label}: size");
        assert_eq!(out.data.len(), c.len, "{label}: byte length");
        assert_eq!(hex(&out.data[..24]), c.head, "{label}: first 8 pixels");
        assert_eq!(hex(&out.data[out.data.len() - 24..]), c.tail, "{label}: last 8 pixels");
        assert_eq!(fnv1a64(&out.data), c.fnv1a64, "{label}: full-image digest");
    }
}
