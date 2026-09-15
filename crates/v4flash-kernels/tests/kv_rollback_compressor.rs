//! Regression: `rollback_kv` must restore the COMPRESSOR store, not just the
//! raw SWA window.
//!
//! The bug (fixed 2026-09-15): `KvMark` carried only per-layer
//! `(n_raw, raw_off)`. A speculative batch fires ~b/ratio compressor boundaries,
//! and those rows stayed in the compressed KV permanently after a rollback, for
//! decode to attend to. The DSpark accept path's partial rollback had the same
//! hole, so REJECTED drafts polluted compressed KV cumulatively. It showed up as
//! a verify-vs-decode fidelity cliff at the batch width where a second boundary
//! can fire (ratio 2, so B>=3): argmax agreement 0.875 at B<=2 vs 0.12-0.39
//! above it.
//!
//! This test pokes the compressor state directly rather than running a forward,
//! so it needs a GPU but no checkpoint:
//!
//!   cargo test -p v4flash-kernels --features v41 --test kv_rollback_compressor \
//!     -- --ignored --nocapture

#![cfg(feature = "v41")]

use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::het::state::HetModelState;

fn devices() -> Option<(Device, Device)> {
    let all = Device::all().ok()?;
    let ig = all.iter().find(|d| {
        d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1151")).unwrap_or(false)
    })?;
    let dg = all
        .iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx12")).unwrap_or(false))
        .unwrap_or(ig);
    Some((*dg, *ig))
}

#[test]
#[ignore = "needs a GPU"]
fn rollback_restores_compressor_counters_and_accumulators() {
    install_panic_handler().ok();
    let Some((dgpu, igpu)) = devices() else {
        eprintln!("no suitable GPU; skipping");
        return;
    };
    let mut state = HetModelState::alloc(dgpu, igpu, 4096).expect("alloc state");

    // Find a layer that OWNS a compressor (only the KV-source layers do).
    let li = state
        .layers
        .iter()
        .position(|l| l.compressor.is_some())
        .expect("no layer owns a compressor");

    let mark = state.mark_kv();

    // Snapshot what the mark should restore.
    let (n_comp0, n_index0, kv0, sc0) = {
        let cs = state.layers[li].compressor.as_ref().unwrap();
        let mut kv = vec![0f32; cs.state_kv.len()];
        let mut sc = vec![0f32; cs.state_score.len()];
        cs.state_kv.copy_to_host(&mut kv).unwrap();
        cs.state_score.copy_to_host(&mut sc).unwrap();
        (cs.n_comp, cs.n_index_comp, kv, sc)
    };

    // Simulate a speculative batch: advance the counters AND perturb the running
    // segment accumulators, exactly what firing compressor boundaries does.
    {
        let cs = state.layers[li].compressor.as_mut().unwrap();
        cs.n_comp += 3;
        cs.n_index_comp += 3;
        let poisoned_kv = vec![1.5f32; cs.state_kv.len()];
        let poisoned_sc = vec![-2.5f32; cs.state_score.len()];
        cs.state_kv.copy_from_host(&poisoned_kv).unwrap();
        cs.state_score.copy_from_host(&poisoned_sc).unwrap();
    }
    // And advance the raw window, as an ingest would.
    state.layers[li].n_raw += 6;

    state.rollback_kv(&mark).expect("rollback");

    let cs = state.layers[li].compressor.as_ref().unwrap();
    assert_eq!(cs.n_comp, n_comp0, "n_comp not restored (the original bug)");
    assert_eq!(cs.n_index_comp, n_index0, "n_index_comp not restored");

    let mut kv = vec![0f32; cs.state_kv.len()];
    let mut sc = vec![0f32; cs.state_score.len()];
    cs.state_kv.copy_to_host(&mut kv).unwrap();
    cs.state_score.copy_to_host(&mut sc).unwrap();
    assert_eq!(kv, kv0, "state_kv accumulator not restored");
    assert_eq!(sc, sc0, "state_score accumulator not restored");
    assert_eq!(state.layers[li].n_raw, mark.per_layer[li].0, "n_raw not restored");
}

/// The accept path's PARTIAL rollback keeps `keep` rows. `n_comp` is positional,
/// so boundaries fired inside the ACCEPTED prefix must survive while the rest is
/// discarded — and it must never rewind below the mark or past what the batch
/// actually wrote.
#[test]
#[ignore = "needs a GPU"]
fn partial_rollback_keeps_accepted_boundaries_only() {
    install_panic_handler().ok();
    let Some((dgpu, igpu)) = devices() else {
        eprintln!("no suitable GPU; skipping");
        return;
    };
    let mut state = HetModelState::alloc(dgpu, igpu, 4096).expect("alloc state");
    let li = state.layers.iter().position(|l| l.compressor.is_some()).expect("compressor layer");

    let mark = state.mark_kv();
    let base_n_comp = state.layers[li].compressor.as_ref().unwrap().n_comp;
    let base_n_raw = state.layers[li].n_raw;

    // A 6-row verify of which only ONE row is accepted, row 0 at `pos`. Keeping
    // just one row is what makes this discriminating: the batch fires 3
    // boundaries, and a rollback that ignores the compressor leaves all 3.
    let pos = 64u32;
    let keep = 1u32;
    let partial = mark.advanced_by(keep, pos);

    {
        let cs = state.layers[li].compressor.as_mut().unwrap();
        cs.n_comp += 3; // boundaries fired across the whole batch
    }
    state.layers[li].n_raw += 6;
    state.rollback_kv(&partial).expect("partial rollback");

    let cs = state.layers[li].compressor.as_ref().unwrap();
    assert_eq!(
        state.layers[li].n_raw,
        base_n_raw + keep,
        "raw window must keep exactly the accepted rows"
    );
    assert!(cs.n_comp >= base_n_comp, "partial rollback rewound BELOW the mark");
    assert!(
        cs.n_comp <= base_n_comp + keep as u32,
        "partial rollback kept more compressed rows than the accepted prefix earns"
    );
}
