//! Regression: the expert ALLOCATOR and the remote-EXCLUSION builder must be
//! used as matched pairs.
//!
//! The pager has two residency structures, each with its own exclusion builder:
//!
//! | allocator                        | exclusion builder          | slot rule        |
//! |----------------------------------|----------------------------|------------------|
//! | `ensure` (sparse decode LRU)     | `mark_remote_after_ensure` | slot anywhere    |
//! | `ensure_layer_union`/`_dense`    | `set_remote_exclusion`     | slot == expert id|
//!
//! Crossing them is SILENT at the pager level and only shows up as a routing
//! error: `set_remote_exclusion` rebuilds the whole remap from the layer's
//! nominal window, so under the sparse LRU every expert whose slot falls
//! outside that window gets stamped 0 ("another device owns it"). Box 2 does
//! not own those either, so the pick is computed by NOBODY and its
//! contribution is silently missing from `ffn_combine`.
//!
//! Observed before the fix, on the speculative verify with both boxes active:
//!   L18 expert 251: computed by 0 devices [] (need exactly 1). remap[251]=0
//!
//! These tests model the two remap SHAPES the builders produce and assert that
//! `verify_routing_exactly_once` — the invariant check that caught the bug —
//! rejects the crossed pairing and accepts the matched one.

use v4flash_kernels::config::N_EXPERT;
use v4flash_kernels::het::forward_layer::verify_routing_exactly_once;

const LAYER: i32 = 18;

/// Experts this layer routes to, and the pool slots a sparse LRU handed them.
/// Slots are scattered across the pool — NOT equal to the expert id, and not
/// inside layer 18's nominal dense window.
fn sparse_assignment() -> Vec<(u32, usize)> {
    vec![(251, 0), (12, 1730), (200, 44), (7, 2199), (130, 903), (255, 17)]
}

/// What `mark_remote_after_ensure` leaves behind: `ensure` already wrote the
/// negative self-map `-(slot)-1` for every requested id, and the builder
/// overwrites ONLY the entries box 2 owns with 0.
fn remap_sparse_matched(remote: &[u32]) -> Vec<i32> {
    // `ensure` resets everything else to `-(e)-1` ("ours, at slot e").
    let mut remap: Vec<i32> = (0..N_EXPERT as i32).map(|e| -e - 1).collect();
    for &(e, slot) in &sparse_assignment() {
        remap[e as usize] = -(slot as i32) - 1;
    }
    for &e in remote {
        remap[e as usize] = 0;
    }
    remap
}

/// What `set_remote_exclusion` would produce over the SAME sparse assignment:
/// it keeps an entry only when the expert's slot lies inside this layer's
/// window `[base, base+width)`, and writes 0 otherwise. This is the bug.
fn remap_sparse_crossed(remote: &[u32], base: usize, width: usize) -> Vec<i32> {
    let assign = sparse_assignment();
    (0..N_EXPERT as usize)
        .map(|e| {
            if remote.contains(&(e as u32)) {
                return 0;
            }
            match assign.iter().find(|(id, _)| *id == e as u32) {
                Some(&(_, sl)) if sl >= base && sl < base + width => -((sl - base) as i32) - 1,
                _ => 0,
            }
        })
        .collect()
}

fn picks() -> Vec<i32> {
    sparse_assignment().iter().map(|&(e, _)| e as i32).collect()
}

#[test]
fn crossed_pairing_drops_picks_to_nobody() {
    // Layer 18's dense window: slots [18*384, 19*384). None of the sparse LRU
    // slots above fall inside it, which is the whole point of a sparse pool.
    let (base, width) = (LAYER as usize * 384, 384);
    let remote: Vec<u32> = vec![]; // box 2 owns none of these picks
    let remap = remap_sparse_crossed(&remote, base, width);
    let owns: Vec<bool> = vec![false; N_EXPERT as usize];

    // Expert 251 got LRU slot 0, outside the window, so it is stamped 0.
    assert_eq!(remap[251], 0, "crossed builder should stamp the out-of-window slot");

    let err = verify_routing_exactly_once(LAYER, &picks(), &remap, Some(&owns))
        .expect_err("crossing the allocator and exclusion builder must be caught");
    let msg = err.to_string();
    assert!(
        msg.contains("computed by 0 devices"),
        "expected a dropped-pick error, got: {msg}"
    );
}

#[test]
fn matched_pairing_claims_every_pick_exactly_once() {
    let remote: Vec<u32> = vec![]; // every pick is box 1's
    let remap = remap_sparse_matched(&remote);
    let owns: Vec<bool> = vec![false; N_EXPERT as usize];

    // The LRU slot assignment survives: local experts keep their negative map.
    assert!(remap[251] < 0, "matched builder must not disturb the LRU slot");
    assert_eq!(remap[251], -1, "expert 251 is at slot 0 => -(0)-1");

    verify_routing_exactly_once(LAYER, &picks(), &remap, Some(&owns))
        .expect("matched pairing must claim every pick exactly once");
}

#[test]
fn matched_pairing_hands_the_catchall_picks_to_box_two() {
    // The catch-all case: box 1 declined 251 and 7 for lack of residency, so
    // `owns_eff` marks them remote. Exactly one device must still claim each.
    let remote: Vec<u32> = vec![251, 7];
    let remap = remap_sparse_matched(&remote);
    let mut owns: Vec<bool> = vec![false; N_EXPERT as usize];
    for &e in &remote {
        owns[e as usize] = true;
    }

    assert_eq!(remap[251], 0, "a box-2 pick must read as not-ours locally");
    assert!(remap[200] < 0, "a box-1 pick must keep its slot");

    verify_routing_exactly_once(LAYER, &picks(), &remap, Some(&owns))
        .expect("split picks must still be claimed exactly once");
}

#[test]
fn double_claim_is_also_caught() {
    // The other direction of the same invariant: an expert left resident
    // locally AND advertised by box 2 is DOUBLE-COUNTED at ffn_combine.
    let remap = remap_sparse_matched(&[]);
    let mut owns: Vec<bool> = vec![false; N_EXPERT as usize];
    owns[200] = true; // box 2 claims it too

    let err = verify_routing_exactly_once(LAYER, &picks(), &remap, Some(&owns))
        .expect_err("a pick claimed by both devices must be caught");
    assert!(err.to_string().contains("computed by 2 devices"), "{err}");
}
