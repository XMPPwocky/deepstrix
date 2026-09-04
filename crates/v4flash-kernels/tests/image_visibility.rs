//! Vision-Exp text-side visibility / raw-window arithmetic vs a Python
//! table derived from DeepSeek `model.py`.
//!
//! `scripts/gen_image_visibility_vectors.py` evaluates the reference
//! `get_image_visible` + `get_window_topk_idxs_visible` on whole prompts
//! and records, per row, `[left, right, first_key, last_key]` in ABSOLUTE
//! prompt positions. This test replays each prompt through
//! `het::image_spans` the way `forward_prompt_batch_v2` does — chunk by
//! chunk, in raw-cache SLOT space — and checks it reproduces the table.
//!
//! CPU only: no HIP context, no model, no device buffers.

// Rows are indexed alongside an absolute-position offset into a second
// slice, so `enumerate()` would not actually be clearer here.
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeSet;

use v4flash_kernels::config::SWA_WINDOW;
use v4flash_kernels::het::image_spans::{
    self, is_image_token, ImageSpan, IMAGE_RAW_WINDOW_MAX, VISION_MAX_N_TOKEN,
};

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    ids: Vec<i32>,
    spans: Vec<[u32; 2]>,
    /// per row: [left, right, first_key_abs, last_key_abs]
    rows: Vec<[i64; 4]>,
    legal_cuts: Vec<usize>,
    illegal_cuts: Vec<usize>,
}

#[derive(serde::Deserialize)]
struct Cases {
    vocab: i64,
    window: u32,
    max_image_tokens: u32,
    cases: Vec<Case>,
}

fn load() -> Cases {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/image_visibility_cases.json");
    let s = std::fs::read_to_string(p)
        .unwrap_or_else(|e| panic!("{p}: {e} — run scripts/gen_image_visibility_vectors.py"));
    serde_json::from_str(&s).expect("parse image_visibility_cases.json")
}

fn spans_of(c: &Case) -> Vec<ImageSpan> {
    c.spans.iter().map(|s| (s[0], s[1])).collect()
}

/// Constants must agree with the generator, or every comparison below is
/// silently checking the wrong model.
#[test]
fn constants_match_reference() {
    let cs = load();
    assert_eq!(cs.vocab, image_spans::IMAGE_TOKEN_BASE as i64);
    assert_eq!(cs.window, SWA_WINDOW);
    assert_eq!(cs.max_image_tokens, VISION_MAX_N_TOKEN);
    assert_eq!(IMAGE_RAW_WINDOW_MAX, SWA_WINDOW + VISION_MAX_N_TOKEN);
    assert!(!cs.cases.is_empty());
}

/// `rows_visibility` reproduces `get_image_visible`'s (left, right) for
/// every row, chunked exactly as the prefill driver chunks.
#[test]
fn left_right_matches_reference() {
    let cs = load();
    for c in &cs.cases {
        let spans = spans_of(c);
        image_spans::validate_spans(&spans, 0, c.ids.len()).unwrap_or_else(|e| {
            panic!("{}: validate_spans: {e}", c.name);
        });
        let mut start = 0usize;
        while start < c.ids.len() {
            let (end, _) =
                image_spans::plan_chunk(0, start, c.ids.len(), 1024, None, &spans)
                    .unwrap_or_else(|e| panic!("{}: plan_chunk at {start}: {e}", c.name));
            let b = end - start;
            let vis = image_spans::rows_visibility(start as u32, b, &spans)
                .unwrap_or_else(|e| panic!("{}: rows_visibility at {start}: {e}", c.name));
            for i in 0..b {
                let want = &c.rows[start + i];
                assert_eq!(
                    (vis[i].0 as i64, vis[i].1 as i64),
                    (want[0], want[1]),
                    "{}: row {} (chunk @{start}) left/right",
                    c.name,
                    start + i
                );
            }
            start = end;
        }
    }
}

/// The full pipeline: `plan_chunk` → `rows_visibility` → `raw_window`,
/// in raw-cache slot space, must land on exactly the reference's absolute
/// key range for every row of every chunk.
///
/// Slot model (mirrors `forward_layer_pre_moe_v2`): a chunk at absolute
/// `pos0` appends its `b` rows at slots `[n_raw_before .. n_raw_before+b)`,
/// so absolute position `p` of this chunk sits at slot
/// `p - pos0 + n_raw_before`. After the chunk the last `SWA_WINDOW` rows
/// are evicted down to `[0..W)`, i.e. `n_raw_before = min(pos0, W)`.
#[test]
fn raw_window_matches_reference() {
    let cs = load();
    for c in &cs.cases {
        let spans = spans_of(c);
        let t = c.ids.len();
        let mut start = 0usize;
        let mut checked = 0usize;
        while start < t {
            let (end, _) = image_spans::plan_chunk(0, start, t, 1024, None, &spans).unwrap();
            let b = end - start;
            // Post-eviction steady state: the cache holds min(pos0, W) rows
            // of history before this chunk's rows are appended.
            let n_raw_before = (start as u32).min(SWA_WINDOW);
            let vis = image_spans::rows_visibility(start as u32, b, &spans).unwrap();
            for i in 0..b {
                let (left, right) = vis[i];
                let (off, cnt) = image_spans::raw_window(n_raw_before, i as u32, left, right);
                assert!(cnt > 0, "{}: row {} empty window", c.name, start + i);
                assert!(
                    cnt <= IMAGE_RAW_WINDOW_MAX,
                    "{}: row {} count {cnt} over cap",
                    c.name,
                    start + i
                );
                // Slot -> absolute.
                let abs_first = off as i64 + start as i64 - n_raw_before as i64;
                let abs_last = abs_first + cnt as i64 - 1;
                let want = &c.rows[start + i];
                assert_eq!(
                    (abs_first, abs_last),
                    (want[2], want[3]),
                    "{}: row {} (chunk @{start}, n_raw_before {n_raw_before}) window",
                    c.name,
                    start + i
                );
                // The window must live inside the cache rows that actually
                // exist during this chunk — this is the invariant that makes
                // forward-looking keys legal.
                assert!(
                    off + cnt <= n_raw_before + b as u32,
                    "{}: row {} window [{off},{}) past appended K/V {}",
                    c.name,
                    start + i,
                    off + cnt,
                    n_raw_before + b as u32
                );
                checked += 1;
            }
            start = end;
        }
        assert_eq!(checked, t, "{}: not every row checked", c.name);
    }
}

/// Text rows must keep the exact pre-vision causal window, and image rows
/// must be the only rows that differ.
#[test]
fn text_rows_are_unchanged_by_vision() {
    let cs = load();
    for c in &cs.cases {
        let spans = spans_of(c);
        let t = c.ids.len();
        let mut start = 0usize;
        while start < t {
            let (end, _) = image_spans::plan_chunk(0, start, t, 1024, None, &spans).unwrap();
            let b = end - start;
            let n_raw_before = (start as u32).min(SWA_WINDOW);
            let vis = image_spans::rows_visibility(start as u32, b, &spans).unwrap();
            for i in 0..b {
                let (left, right) = vis[i];
                let (off, cnt) = image_spans::raw_window(n_raw_before, i as u32, left, right);
                let causal_end = n_raw_before + i as u32 + 1;
                let legacy = (causal_end.saturating_sub(SWA_WINDOW), causal_end.min(SWA_WINDOW));
                if (left, right) == (0, 0) {
                    assert_eq!(
                        (off, cnt),
                        legacy,
                        "{}: text row {} diverged from the causal window",
                        c.name,
                        start + i
                    );
                    // (0,0) rows are exactly the non-image tokens, except the
                    // single-token edge the generator never produces.
                    assert!(
                        !is_image_token(c.ids[start + i]) || c.rows[start + i][1] == 0,
                        "{}: row {} flagged text but is an image token",
                        c.name,
                        start + i
                    );
                } else {
                    assert!(
                        is_image_token(c.ids[start + i]),
                        "{}: row {} widened but is a text token",
                        c.name,
                        start + i
                    );
                    assert!(cnt >= legacy.1, "{}: row {} narrowed", c.name, start + i);
                }
            }
            start = end;
        }
    }
}

/// Router-side image flag: exactly the synthetic ids, and `image_runs`
/// covers exactly those rows.
#[test]
fn image_runs_cover_synthetic_ids() {
    let cs = load();
    for c in &cs.cases {
        let flagged: BTreeSet<usize> = c
            .ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id >= image_spans::IMAGE_TOKEN_BASE)
            .map(|(i, _)| i)
            .collect();
        let mut covered = BTreeSet::new();
        for (r0, n) in image_spans::image_runs(&c.ids) {
            assert!(n > 0);
            for r in r0..r0 + n {
                assert!(covered.insert(r), "{}: row {r} covered twice", c.name);
            }
        }
        assert_eq!(covered, flagged, "{}: image_runs != synthetic ids", c.name);
        // Compress pads sit BEFORE IMAGE_START, so the router flag covers
        // strictly more rows than the attention spans do.
        for &[s, len] in &c.spans {
            for p in s..s + len {
                assert!(flagged.contains(&(p as usize)), "{}: span row {p} not flagged", c.name);
            }
        }
    }
}

/// Chunk / lane cuts: the planner never cuts strictly inside a span, and
/// the generator's own "illegal" hints are rejected by `cut_ok`.
#[test]
fn planner_never_cuts_inside_a_span() {
    let cs = load();
    for c in &cs.cases {
        let spans = spans_of(c);
        let t = c.ids.len();
        for &p in &c.legal_cuts {
            assert!(image_spans::cut_ok(p as u64, &spans), "{}: cut {p} should be legal", c.name);
        }
        for &p in &c.illegal_cuts {
            assert!(!image_spans::cut_ok(p as u64, &spans), "{}: cut {p} should be illegal", c.name);
        }
        // Pipelined driver: chunk ends AND lane cuts must both be legal,
        // and both lanes must fit their scratch.
        let (cap_a, cap_b) = (512usize, 512usize);
        let mut start = 0usize;
        while start < t {
            let (end, b_a) =
                image_spans::plan_chunk(0, start, t, 1024, Some((cap_a, cap_b)), &spans)
                    .unwrap_or_else(|e| panic!("{}: plan_chunk at {start}: {e}", c.name));
            let b = end - start;
            assert!(b > 0 && b <= 1024, "{}: chunk {b} rows", c.name);
            assert!(image_spans::cut_ok(end as u64, &spans), "{}: chunk end {end} inside a span", c.name);
            if b >= 2 {
                assert!(b_a >= 1 && b_a < b, "{}: lane split {b_a}/{b}", c.name);
                assert!(b_a <= cap_a && b - b_a <= cap_b, "{}: lane caps", c.name);
                assert!(
                    image_spans::cut_ok((start + b_a) as u64, &spans),
                    "{}: lane cut {} inside a span",
                    c.name,
                    start + b_a
                );
                // Every span touching this chunk must lie wholly in ONE lane.
                for &(s, len) in &spans {
                    let (s, e) = (s as usize, (s + len - 1) as usize);
                    if e < start || s >= end {
                        continue;
                    }
                    let lane_of = |p: usize| if p < start + b_a { 0 } else { 1 };
                    assert_eq!(
                        lane_of(s),
                        lane_of(e),
                        "{}: span ({s},{e}) straddles the lane cut {}",
                        c.name,
                        start + b_a
                    );
                }
                // Which is exactly what rows_visibility enforces.
                image_spans::rows_visibility(start as u32, b_a, &spans).unwrap();
                image_spans::rows_visibility((start + b_a) as u32, b - b_a, &spans).unwrap();
            }
            start = end;
        }
    }
}

/// A span longer than the chunk / lane capacity must be a clean error,
/// never a panic and never a silently-truncated window.
///
/// Note the planner errors on the chunk that must CONTAIN the span, not on
/// the first call: it legitimately emits the short text chunk that ends at
/// the span start first. So drive the whole loop, as the driver does.
fn drive(t: usize, cap: usize, lanes: Option<(usize, usize)>, spans: &[ImageSpan]) -> Result<(), String> {
    let mut start = 0usize;
    while start < t {
        let (end, b_a) = image_spans::plan_chunk(0, start, t, cap, lanes, spans)
            .map_err(|e| e.to_string())?;
        assert!(end > start, "plan_chunk made no progress at {start}");
        let b = end - start;
        // Whatever it returned must be usable by the layer code.
        if lanes.is_some() && b >= 2 {
            image_spans::rows_visibility(start as u32, b_a, spans).map_err(|e| e.to_string())?;
            image_spans::rows_visibility((start + b_a) as u32, b - b_a, spans)
                .map_err(|e| e.to_string())?;
        } else {
            image_spans::rows_visibility(start as u32, b, spans).map_err(|e| e.to_string())?;
        }
        start = end;
    }
    Ok(())
}

#[test]
fn oversized_span_errors_cleanly() {
    // Production shape: B_MAX = 1024, lanes at B_MAX/2 = 512 rows each.
    // A 600-token span cannot fit in one 512-row lane.
    let big = [(10u32, 600u32)];
    let e = drive(2000, 1024, Some((512, 512)), &big).unwrap_err();
    assert!(e.contains("lane") || e.contains("chunk"), "unhelpful error: {e}");
    // The same span fits when the whole chunk is one lane (single-lane driver).
    drive(2000, 1024, None, &big).unwrap();
    // ...but not when it is longer than the chunk capacity itself.
    let e = drive(4000, 512, None, &big).unwrap_err();
    assert!(e.contains("chunk"), "unhelpful error: {e}");
    // Real Vision-Exp block sizes (366 / 354 / 198 tokens) always fit both.
    for len in [366u32, 354, 198] {
        drive(4000, 1024, Some((512, 512)), &[(700, len)]).unwrap();
        drive(4000, 1024, None, &[(700, len)]).unwrap();
    }
    // Two big blocks back to back, straddling the natural chunk end.
    drive(4000, 1024, Some((512, 512)), &[(900, 366), (1300, 354)]).unwrap();

    // Straddling a batch is an error, not a panic.
    assert!(image_spans::rows_visibility(0, 100, &[(90, 20)]).is_err());
    // Malformed spans.
    assert!(image_spans::validate_spans(&[(5, 1)], 0, 100).is_err());
    assert!(image_spans::validate_spans(&[(5, 10), (12, 10)], 0, 100).is_err());
    assert!(image_spans::validate_spans(&[(95, 10)], 0, 100).is_err());
    assert!(image_spans::validate_spans(&[(5, 10), (20, 10)], 0, 100).is_ok());
}
