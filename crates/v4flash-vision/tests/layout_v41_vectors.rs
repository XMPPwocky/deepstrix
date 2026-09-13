//! V4.1 layout port vs vectors produced by the REFERENCE module itself
//! (`inference/image_processor.py::{plan_image_grid, image_token_types,
//! num_image_tokens}` run by `scripts/gen_v41_vision_vectors.py`) —
//! `tests/data/layout_cases_v41.json`, 54 sizes from 1x1 to 100000x1.
//!
//! The reference numbers V4.1 token types `START, IMAGE, NEW_LINE, END =
//! range(4)`; this crate keeps the V4-Flash 5-type `TokenType` (with PAD)
//! for both checkpoints, so the vectors are mapped through [`v41_type`].

use serde::Deserialize;
use v4flash_vision::layout::{build_image_block_v41, layout_for_grid_cfg, num_image_tokens_v41, plan_resize_cfg};
use v4flash_vision::{TokenType, VisionCfg, PATCH, V41_IMAGE_TOKEN_ID};

#[derive(Deserialize)]
struct Case {
    width: u32,
    height: u32,
    best_h: u32,
    best_w: u32,
    n_vit_h: u32,
    n_vit_w: u32,
    n_llm_h: u32,
    n_llm_w: u32,
    n_tokens: u32,
    types: Vec<u8>,
}

#[derive(Deserialize)]
struct TypesEnum {
    #[serde(rename = "IMAGE_START")]
    start: u8,
    #[serde(rename = "IMAGE")]
    image: u8,
    #[serde(rename = "IMAGE_NEW_LINE")]
    newline: u8,
    #[serde(rename = "IMAGE_END")]
    end: u8,
}

#[derive(Deserialize)]
struct File {
    patch: u32,
    downsample: u32,
    max_n_token: u32,
    min_pixels: u32,
    max_wh_ratio: Option<u32>,
    image_token_id: i32,
    types_enum: TypesEnum,
    cases: Vec<Case>,
}

fn load() -> File {
    serde_json::from_str(include_str!("data/layout_cases_v41.json")).expect("layout_cases_v41.json")
}

/// Reference V4.1 type number → this crate's `TokenType`.
fn v41_type(e: &TypesEnum, t: u8) -> u8 {
    match t {
        x if x == e.start => TokenType::Start as u8,
        x if x == e.image => TokenType::Image as u8,
        x if x == e.newline => TokenType::NewLine as u8,
        x if x == e.end => TokenType::End as u8,
        other => panic!("unknown V4.1 type {other}"),
    }
}

#[test]
fn constants_match_reference_config() {
    let f = load();
    let c = VisionCfg::V41;
    assert_eq!(f.patch, PATCH);
    assert_eq!(f.downsample, v4flash_vision::DOWNSAMPLE);
    assert_eq!(f.max_n_token, c.max_n_token);
    assert_eq!(f.min_pixels, c.min_pixels);
    assert_eq!(f.max_wh_ratio, c.max_wh_ratio);
    assert_eq!(f.image_token_id, V41_IMAGE_TOKEN_ID);
    assert!(f.cases.len() >= 50);
}

#[test]
fn plan_resize_v41_matches_reference() {
    let f = load();
    for c in &f.cases {
        let p = plan_resize_cfg(c.height, c.width, &VisionCfg::V41)
            .unwrap_or_else(|e| panic!("{}x{}: {e:#}", c.width, c.height));
        assert_eq!(
            (p.best_h, p.best_w, p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w),
            (c.best_h, c.best_w, c.n_vit_h, c.n_vit_w, c.n_llm_h, c.n_llm_w),
            "{}x{}",
            c.width,
            c.height
        );
        assert!(!p.plain_resize, "{}x{}: V4.1 never takes the aspect-breaking resize", c.width, c.height);
        assert_eq!(num_image_tokens_v41(p.n_llm_h, p.n_llm_w), c.n_tokens, "{}x{}", c.width, c.height);
        assert!(c.n_tokens <= f.max_n_token);
    }
}

#[test]
fn image_token_types_v41_match_reference() {
    let f = load();
    for c in &f.cases {
        let want: Vec<u8> = c.types.iter().map(|&t| v41_type(&f.types_enum, t)).collect();
        let (types, perm) = build_image_block_v41(c.n_llm_h, c.n_llm_w);
        assert_eq!(types, want, "{}x{}", c.width, c.height);
        assert_eq!(perm.len() as u32, c.n_llm_h * c.n_llm_w);
        assert!(perm.iter().enumerate().all(|(i, &p)| p as usize == i), "perm is the identity");
        // The layout the server builds at some prompt position: same span, no pads.
        let l = layout_for_grid_cfg(c.n_vit_h, c.n_vit_w, 17, &VisionCfg::V41);
        assert_eq!(l.types, want);
        assert_eq!(l.compress_pad(), 0);
        assert_eq!(l.span(), (17, c.n_tokens));
        assert_eq!(l.image_end_pos(), 17 + c.n_tokens - 1);
    }
}
