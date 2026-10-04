//! KvArena::compact_stores: after releasing a middle stream, a request larger
//! than any free run but smaller than the total free space is refused, then
//! fits after compaction, and the surviving streams' comp rows / index keys are
//! byte-identical at their new bases. Small arena: runs alongside a live server.
//! `cargo test --release -p v4flash-kernels --features v41 --test kv_arena_compact -- --nocapture`

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

fn bases(arena: &KvArena, slot: u32) -> Vec<(u32, u32)> {
    arena.stream(slot).unwrap().comp.iter().map(|r| (r.base, r.cap)).collect()
}

/// `KvArena::grow` (2026-09-27): in place into the run after the region, by
/// relocation to a run that holds the grown size, and by compacting the store
/// AROUND the region (lower regions pack down, higher ones pack up, both with
/// overlapping moves through a small bounce). Every stream's comp rows and
/// index keys must survive each of them byte for byte; a growth that cannot
/// fit changes nothing.
#[test]
fn grow_in_place_relocated_and_compacted_preserves_rows() -> eyre::Result<()> {
    use v4flash_kernels::het::kv_arena::GrowHow;
    color_eyre::install().ok();
    let dgpu = Device::new(std::env::var("DGPU").ok().and_then(|s| s.parse().ok()).unwrap_or(1));
    dgpu.set_current()?;
    let stream = Stream::new(dgpu.id)?;
    let mut bounce_f16 = DeviceBuffer::<u16>::new(dgpu.id, 300 * 512)?; // small: forces chunked, overlapping moves
    let mut bounce_u8 = DeviceBuffer::<u8>::new(dgpu.id, 300 * E2M1_KEY_ROW_BYTES)?;
    // 4096 rows per store; stores are ratio 2, 2, 2, 1 (V4.1 KV sources).
    let mut arena = KvArena::alloc(dgpu, 4, 4096)?;
    let a = arena.reserve(1000)?;
    let b = arena.reserve(1000)?;
    let c = arena.reserve(1000)?;
    assert_eq!(arena.reserved_positions(a), 1000);
    for _ in 0..900 { arena.advance(a)?; }
    for _ in 0..950 { arena.advance(b)?; }
    for _ in 0..100 { arena.advance(c)?; }
    let da = fill(&mut arena, a, 11)?;
    let _ = fill(&mut arena, c, 29)?;
    arena.release(c)?;

    // 1. In place: c's rows freed the run right after b in every store.
    let before = bases(&arena, b);
    assert_eq!(arena.grow(b, 1500, &stream, &mut bounce_f16, &mut bounce_u8)?, Some(GrowHow::InPlace));
    for (si, (&(b0, _), &(b1, c1))) in before.iter().zip(&bases(&arena, b)).enumerate() {
        assert_eq!(b1, b0, "store {si}: in-place growth moved the region");
        assert_eq!(c1, 1500u32.div_ceil(arena.stores[si].ratio));
    }
    assert_eq!(arena.reserved_positions(b), 1500);
    for _ in 0..400 { arena.advance(b)?; } // 1350 positions: past the old cap
    assert!(arena.can_step(b));
    let db = fill(&mut arena, b, 17)?;
    check(&arena, a, &da)?;

    // 2. Relocated: b sits right after a, and a run elsewhere holds a's grown size.
    let before = bases(&arena, a);
    assert_eq!(arena.grow(a, 1200, &stream, &mut bounce_f16, &mut bounce_u8)?, Some(GrowHow::Relocated));
    assert!(bases(&arena, a).iter().zip(&before).all(|(n, o)| n.0 != o.0), "every store must have moved a");
    check(&arena, a, &da)?;
    check(&arena, b, &db)?;

    // 3. Too big for the free rows: refused, nothing changes.
    let (ba, bb) = (bases(&arena, a), bases(&arena, b));
    let free: Vec<u32> = arena.stores.iter().map(|s| s.free.free_rows()).collect();
    assert_eq!(arena.grow(b, 4000, &stream, &mut bounce_f16, &mut bounce_u8)?, None);
    assert_eq!((bases(&arena, a), bases(&arena, b)), (ba, bb));
    assert_eq!(arena.stores.iter().map(|s| s.free.free_rows()).collect::<Vec<_>>(), free);

    // 4. Compacted: in the ratio-1 store the free rows are split [0,1000) +
    // [3700,4096) around a at 2500, so b (at 1000, 1350 rows written) must
    // move DOWN over itself and a UP over itself.
    assert_eq!(arena.grow(b, 2700, &stream, &mut bounce_f16, &mut bounce_u8)?, Some(GrowHow::Compacted));
    let last = arena.stores.len() - 1;
    assert_eq!(arena.stores[last].ratio, 1);
    let (nb, na) = (bases(&arena, b)[last], bases(&arena, a)[last]);
    assert_eq!(nb, (0, 2700), "b packs down to row 0 and grows");
    assert_eq!(na.0 + na.1, 4096, "a packs up against the end");
    assert_eq!(arena.stores[last].free.free_rows(), arena.stores[last].free.largest_run(), "free space is one run");
    check(&arena, a, &da)?;
    check(&arena, b, &db)?;
    assert_eq!(arena.reserved_positions(b), 2700);
    println!("grow_in_place_relocated_and_compacted_preserves_rows: OK");
    Ok(())
}

/// `reserve` + `fill_reserved` (2026-09-27): the slot is carved before the
/// prefill and filled after it; a reservation that cannot hold the prompt, or
/// one already filled, is refused and left for the caller to release.
#[test]
fn reserve_then_fill() -> eyre::Result<()> {
    use v4flash_kernels::het::state::HetModelState;
    color_eyre::install().ok();
    let dgpu = Device::new(std::env::var("DGPU").ok().and_then(|s| s.parse().ok()).unwrap_or(1));
    dgpu.set_current()?;
    let stream = Stream::new(dgpu.id)?;
    let mut arena = KvArena::alloc(dgpu, 4, 4096)?;
    let mut src = HetModelState::alloc(dgpu, dgpu, 4096)?;
    let pos = 400u32;
    for si in 0..arena.stores.len() {
        let l = arena.stores[si].layer;
        let ratio = arena.stores[si].ratio;
        let cs = src.layers[l].compressor.as_mut().unwrap();
        cs.n_comp = pos / ratio;
        cs.n_index_comp = cs.n_comp;
    }
    let small = arena.reserve(300)?;
    assert!(arena.fill_reserved(small, &src, pos, &stream).is_err(), "300 positions cannot hold a 400-token prompt");
    assert_eq!(arena.live(), 1, "a refused fill leaves the reservation to the caller");
    arena.release(small)?;
    let slot = arena.reserve(1000)?;
    arena.fill_reserved(slot, &src, pos, &stream)?;
    stream.synchronize()?;
    let s = arena.stream(slot).unwrap();
    assert_eq!(s.pos, pos);
    for (si, r) in s.comp.iter().enumerate() {
        assert_eq!(r.n_comp, pos / arena.stores[si].ratio);
    }
    assert!(arena.fill_reserved(slot, &src, pos, &stream).is_err(), "a filled slot is not a fresh reservation");
    println!("reserve_then_fill: OK");
    Ok(())
}

/// Regression (2026-09-23 review): a failed `admit_from_state` used to leave
/// its freshly admitted slot allocated with no Stream owning it, and a parked
/// request retried every tick. Also the stricter key check: a source whose
/// index keys do not cover every compressed row is refused (the indexer scores
/// `n_comp` rows, so a gap would be scored as another stream's keys).
#[test]
fn failed_admit_from_state_releases_its_slot() -> eyre::Result<()> {
    use v4flash_kernels::het::state::HetModelState;
    color_eyre::install().ok();
    let dgpu = Device::new(std::env::var("DGPU").ok().and_then(|s| s.parse().ok()).unwrap_or(1));
    dgpu.set_current()?;
    let stream = Stream::new(dgpu.id)?;
    let mut arena = KvArena::alloc(dgpu, 4, 4096)?;
    let free_before: Vec<u32> = arena.stores.iter().map(|s| s.free.free_rows()).collect();
    let mut src = HetModelState::alloc(dgpu, dgpu, 4096)?;
    let pos = 400u32;
    for si in 0..arena.stores.len() {
        let l = arena.stores[si].layer;
        let ratio = arena.stores[si].ratio;
        let cs = src.layers[l].compressor.as_mut().unwrap();
        cs.n_comp = pos / ratio;
        cs.n_index_comp = cs.n_comp;
    }
    // Keys short on the last store: must be refused, and leave nothing behind.
    {
        let l = arena.stores[arena.stores.len() - 1].layer;
        let cs = src.layers[l].compressor.as_mut().unwrap();
        cs.n_index_comp = cs.n_comp / 2;
    }
    let err = arena.admit_from_state(&src, 1000, pos, &stream).expect_err("keys short of n_comp must be refused");
    assert!(format!("{err:#}").contains("comp rows"), "unexpected error: {err:#}");
    assert_eq!(arena.live(), 0, "the slot admitted before the failure must be released");
    let free_after: Vec<u32> = arena.stores.iter().map(|s| s.free.free_rows()).collect();
    assert_eq!(free_after, free_before, "every store's rows must be given back");
    // Positive control: a consistent source admits, and releases cleanly.
    {
        let l = arena.stores[arena.stores.len() - 1].layer;
        let cs = src.layers[l].compressor.as_mut().unwrap();
        cs.n_index_comp = cs.n_comp;
    }
    stream.synchronize()?;
    let slot = arena.admit_from_state(&src, 1000, pos, &stream)?;
    stream.synchronize()?;
    assert_eq!(arena.live(), 1);
    arena.release(slot)?;
    let free_end: Vec<u32> = arena.stores.iter().map(|s| s.free.free_rows()).collect();
    assert_eq!(free_end, free_before);
    println!("failed_admit_from_state_releases_its_slot: OK");
    Ok(())
}
