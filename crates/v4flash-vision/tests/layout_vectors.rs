//! Rust layout port vs the pure-Python reference vectors in
//! `tests/data/layout_cases.json` (`scripts/gen_vision_layout_vectors.py`).

use serde::Deserialize;
use v4flash_vision::layout::{build_image_block, layout_for_grid, plan_resize};
use v4flash_vision::{COMPRESS_PAD_TO, DOWNSAMPLE, MAX_N_TOKEN, MAX_WH_RATIO, MIN_PIXELS, PATCH};

#[derive(Deserialize)]
struct Block {
    start_pos: u32,
    len: usize,
    compress_pad: u32,
    #[serde(default)]
    types: Option<Vec<u8>>,
}

#[derive(Deserialize)]
struct Case {
    height: u32,
    width: u32,
    best_h: u32,
    best_w: u32,
    n_vit_h: u32,
    n_vit_w: u32,
    n_llm_h: u32,
    n_llm_w: u32,
    plain_resize: bool,
    perm: Vec<u32>,
    blocks: Vec<Block>,
}

#[derive(Deserialize)]
struct File {
    patch: u32,
    downsample: u32,
    max_n_token: u32,
    max_wh_ratio: u32,
    min_pixels: u32,
    cases: Vec<Case>,
}

fn load() -> File {
    let s = include_str!("data/layout_cases.json");
    serde_json::from_str(s).expect("layout_cases.json")
}

#[test]
fn constants_match_generator() {
    let f = load();
    assert_eq!(f.patch, PATCH);
    assert_eq!(f.downsample, DOWNSAMPLE);
    assert_eq!(f.max_n_token, MAX_N_TOKEN);
    assert_eq!(f.max_wh_ratio, MAX_WH_RATIO);
    assert_eq!(f.min_pixels, MIN_PIXELS);
    assert!(f.cases.len() >= 40);
}

#[test]
fn plan_resize_matches_reference() {
    for c in load().cases {
        let p = plan_resize(c.height, c.width).unwrap();
        assert_eq!(
            (p.best_h, p.best_w, p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w, p.plain_resize),
            (c.best_h, c.best_w, c.n_vit_h, c.n_vit_w, c.n_llm_h, c.n_llm_w, c.plain_resize),
            "{}x{}",
            c.height,
            c.width
        );
        // Budget invariant: block (without compress pads) ≤ 381.
        let l = layout_for_grid(p.n_vit_h, p.n_vit_w, 3);
        assert!(l.types.len() as u32 <= MAX_N_TOKEN - (COMPRESS_PAD_TO - 1), "{}x{}: {}", c.height, c.width, l.types.len());
    }
}

#[test]
fn build_image_block_matches_reference() {
    for c in load().cases {
        for b in &c.blocks {
            let (types, perm) = build_image_block(c.n_llm_h, c.n_llm_w, b.start_pos);
            assert_eq!(types.len(), b.len, "{}x{} sp{}", c.height, c.width, b.start_pos);
            assert_eq!(perm, c.perm, "{}x{} perm", c.height, c.width);
            let l = layout_for_grid(c.n_vit_h, c.n_vit_w, b.start_pos);
            assert_eq!(l.types, types);
            assert_eq!(l.perm, perm);
            assert_eq!(l.compress_pad(), b.compress_pad);
            assert_eq!(l.image_start_pos() % COMPRESS_PAD_TO, COMPRESS_PAD_TO - 1);
            if let Some(t) = &b.types {
                assert_eq!(&types, t, "{}x{} sp{} types", c.height, c.width, b.start_pos);
            }
        }
    }
}

#[test]
fn reference_examples_from_task() {
    // (h, w) -> best (h, w), vit, llm, block len at start_pos%4==3
    let ex = [
        ((1080, 1920), (588, 1036), (42, 74), (14, 25), 366),
        ((768, 1024), (658, 882), (47, 63), (16, 21), 354),
        ((512, 512), (518, 518), (37, 37), (13, 13), 198),
    ];
    for ((h, w), best, vit, llm, len) in ex {
        let p = plan_resize(h, w).unwrap();
        assert_eq!((p.best_h, p.best_w), best);
        assert_eq!((p.n_vit_h, p.n_vit_w), vit);
        assert_eq!((p.n_llm_h, p.n_llm_w), llm);
        assert_eq!(layout_for_grid(vit.0, vit.1, 3).types.len(), len);
    }
    let p = plan_resize(384, 2208).unwrap();
    assert_eq!((p.best_h, p.best_w, p.n_vit_h, p.n_vit_w), (336, 1932, 24, 138));
}
