//! KvArena::compact_stores: after releasing a middle stream, a request larger
//! than any free run but smaller than the total free space is refused, then
//! fits after compaction, and the surviving streams' comp rows / index keys are
//! byte-identical at their new bases. Small arena: runs alongside a live server.
//! `cargo test --release -p v4flash-kernels --features v41 --test kv_arena_compact -- --nocapture`
#![cfg(feature = "v41")]

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_kernels::het::kv_arena::KvArena;
use v4flash_kernels::index_kv_e2m1::E2M1_KEY_ROW_BYTES;

fn fill(arena: &mut KvArena, slot: u32, seed: u16) -> eyre::Result<Vec<(Vec<u16>, Vec<u8>)>> {
    let mut out = Vec::new();
    for si in 0..arena.stores.len() {
        let l = arena.stores[si].layer;
        let width = arena.stores[si].width as usize;
        let r = arena.stream(slot).unwrap().comp[si].clone();
        let n = r.n_comp as usize;
        let rows: Vec<u16> = (0..n * width).map(|i| seed.wrapping_mul(31).wrapping_add((i % 65521) as u16)).collect();
        let keys: Vec<u8> = (0..n * E2M1_KEY_ROW_BYTES).map(|i| (seed as usize * 7 + i) as u8).collect();
        let cs = arena.state.layers[l].compressor.as_mut().unwrap();
        cs.comp_kv.f16_mut().unwrap().slice_view_mut(r.base as usize * width, n * width).copy_from_host(&rows)?;
        cs.index_k.as_mut().unwrap().slice_view_mut(r.base as usize * E2M1_KEY_ROW_BYTES, n * E2M1_KEY_ROW_BYTES).copy_from_host(&keys)?;
        out.push((rows, keys));
    }
    Ok(out)
}

fn check(arena: &KvArena, slot: u32, want: &[(Vec<u16>, Vec<u8>)]) -> eyre::Result<()> {
    for si in 0..arena.stores.len() {
        let l = arena.stores[si].layer;
        let width = arena.stores[si].width as usize;
        let r = &arena.stream(slot).unwrap().comp[si];
        let n = r.n_comp as usize;
        let cs = arena.state.layers[l].compressor.as_ref().unwrap();
        let mut rows = vec![0u16; n * width];
        cs.comp_kv.f16().unwrap().slice_view(r.base as usize * width, n * width).copy_to_host(&mut rows)?;
        let mut keys = vec![0u8; n * E2M1_KEY_ROW_BYTES];
        cs.index_k.as_ref().unwrap().slice_view(r.base as usize * E2M1_KEY_ROW_BYTES, n * E2M1_KEY_ROW_BYTES).copy_to_host(&mut keys)?;
        assert_eq!(rows, want[si].0, "slot {slot} store {si} comp rows differ after compaction");
        assert_eq!(keys, want[si].1, "slot {slot} store {si} index keys differ after compaction");
    }
    Ok(())
}

#[test]
fn compaction_makes_free_space_contiguous_and_preserves_rows() -> eyre::Result<()> {
    color_eyre::install().ok();
    let dgpu = Device::new(std::env::var("DGPU").ok().and_then(|s| s.parse().ok()).unwrap_or(1));
    dgpu.set_current()?;
    let stream = Stream::new(dgpu.id)?;
    let mut bounce_f16 = DeviceBuffer::<u16>::new(dgpu.id, 300 * 512)?; // deliberately small: forces chunking
    let mut bounce_u8 = DeviceBuffer::<u8>::new(dgpu.id, 300 * E2M1_KEY_ROW_BYTES)?;
    // 4096 comp rows per store; three streams of ctx 2048 (ratio-1 store: 2048
    // rows each) -> the third does not fit at all; two fit.
    let mut arena = KvArena::alloc(dgpu, 4, 4096)?;
    let a = arena.admit(1500, 0)?;
    let b = arena.admit(1500, 0)?;
    let c = arena.admit(1000, 0)?;
    // Populate counters: advance each stream some steps.
    for _ in 0..900 { arena.advance(a)?; }
    for _ in 0..1300 { arena.advance(b)?; }
    for _ in 0..700 { arena.advance(c)?; }
    let da = fill(&mut arena, a, 11)?;
    let dc = fill(&mut arena, c, 29)?;
    let _ = fill(&mut arena, b, 17)?;
    arena.release(b)?;
    // ratio-1 store: free = 1500 (middle) + 96 (tail) = 1596; largest run 1500.
    let want = 1550;
    assert!(arena.admit(want, 0).is_err(), "should not fit before compaction");
    assert!(arena.fits_after_compaction(want));
    let old_a = arena.stream(a).unwrap().comp.clone();
    arena.compact_stores(&stream, &mut bounce_f16, &mut bounce_u8)?;
    let new_a = arena.stream(a).unwrap().comp.clone();
    let new_c = arena.stream(c).unwrap().comp.clone();
    for si in 0..arena.stores.len() {
        assert_eq!(new_a[si].base, old_a[si].base, "a was first; must not move");
        assert_eq!(new_c[si].base, old_a[si].base + old_a[si].cap, "c must slide down to right after a");
        assert_eq!(arena.stores[si].free.free_rows(), arena.stores[si].free.largest_run(), "free space must be one run");
    }
    check(&arena, a, &da)?;
    check(&arena, c, &dc)?;
    let d = arena.admit(want, 0)?;
    assert_eq!(arena.stream(d).unwrap().comp[arena.stores.len() - 1].base, new_c[arena.stores.len() - 1].base + new_c[arena.stores.len() - 1].cap);
    // Compaction with nothing to move is a no-op.
    arena.compact_stores(&stream, &mut bounce_f16, &mut bounce_u8)?;
    check(&arena, a, &da)?;
    check(&arena, c, &dc)?;
    println!("kv_arena_compact: OK (stores {})", arena.stores.len());
    Ok(())
}
