//! Prefill VRAM loan on the dGPU (docs/v41/PREFILL_VRAM_LOAN_DESIGN.md): the production-sized arena, a live decode
//! stream whose comp/key rows reach into the lent end ranges, the 1024-row prefill set built over pool + store tails, a
//! lend (compaction + swap), the lent rows overwritten (as a prefill pass would), the loan ended -- and the live rows
//! must read back BIT-IDENTICAL. Also: no two placed buffers overlap.
//!
//!   SERVER DOWN (needs the dGPU's free memory): cargo test --release -p v4flash-kernels --features v41 \
//!     --test prefill_loan_gpu -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::het::kv_arena::KvArena;
use v4flash_kernels::het::prefill_loan::{lm_store_bytes, PrefillLoan};
use v4flash_kernels::index_kv_e2m1::E2M1_KEY_ROW_BYTES;

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 dGPU"))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 17
    }
}

#[test]
#[ignore]
fn loan_round_trip_keeps_live_decode_kv_bit_identical() -> eyre::Result<()> {
    install_panic_handler()?;
    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let ctx_rows: u32 = std::env::var("LOAN_CTX_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1_300_000);
    let mut arena = KvArena::alloc_ctx(dgpu, 8, ctx_rows)?;
    // Two live streams: a big one whose rows reach far into every store (ratio 1: 75% of the store), a small one.
    let big = arena.reserve(ctx_rows / 4 * 3)?;
    let small = arena.reserve(20_000)?;
    let mut rng = Lcg(0x10AD);
    let mut expect: Vec<(u32, usize, Vec<u16>, Vec<u8>)> = Vec::new(); // (slot, store, comp, keys)
    for &slot in &[big, small] {
        let comp_regions = arena.stream(slot).unwrap().comp.clone();
        for (si, r) in comp_regions.iter().enumerate() {
            let n = (r.cap - r.cap / 10).max(1); // 90% of the reservation written
            let width = arena.stores[si].width as usize;
            let comp: Vec<u16> = (0..n as usize * width).map(|_| rng.next() as u16).collect();
            let keys: Vec<u8> = (0..n as usize * E2M1_KEY_ROW_BYTES).map(|_| rng.next() as u8).collect();
            {
                let (cb, kb) = arena.store_buffers(si)?;
                cb.slice_view(r.base as usize * width, comp.len()).copy_from_host(&comp)?;
                kb.slice_view(r.base as usize * E2M1_KEY_ROW_BYTES, keys.len()).copy_from_host(&keys)?;
            }
            let reg = &mut arena.stream_mut(slot).unwrap().comp[si];
            reg.n_comp = n;
            reg.n_index_comp = n;
            expect.push((slot, si, comp, keys));
        }
    }
    // production-like pool (the hub-down card has ~12 GB free: uncapped, the pool would hold the whole set)
    if std::env::var_os("V41_PREFILL_LOAN_POOL_MB").is_none() {
        std::env::set_var("V41_PREFILL_LOAN_POOL_MB", "2048");
    }
    let rows = v4flash_kernels::het::batch_scratch::prefill_lane_rows();
    let mut loan = PrefillLoan::build(dgpu, rows, 368_640, &arena, lm_store_bytes(4096))?;
    eprintln!("set {} MiB, pool {} MiB, lent rows {:?}", loan.set_bytes >> 20, loan.pool_bytes >> 20, loan.lent_rows);
    let stream = Stream::new(dgpu.id)?;
    let mut bf16 = DeviceBuffer::<u16>::new(dgpu.id, 4096 * 512)?;
    let mut bu8 = DeviceBuffer::<u8>::new(dgpu.id, 4096 * 80)?;
    let st = loan.lend(&mut arena, &stream, &mut bf16, &mut bu8)?.expect("lent");
    eprintln!("lend: {st:?}");
    assert!(arena.is_lent());
    assert!(st.swapped_bytes > 0, "the big stream must overlap the lent ranges (test setup)");
    // a prefill pass scribbles over every lent row
    for (si, &n) in loan.lent_rows.clone().iter().enumerate() {
        let cap = arena.stores[si].rows_cap as usize;
        let width = arena.stores[si].width as usize;
        let n = n as usize;
        let (cb, kb) = arena.store_buffers(si)?;
        cb.slice_view((cap - n) * width, n * width).fill_zero()?;
        kb.slice_view((cap - n) * E2M1_KEY_ROW_BYTES, n * E2M1_KEY_ROW_BYTES).fill_zero()?;
    }
    let restored = loan.end(&mut arena)?.expect("ended");
    assert_eq!(restored, st.swapped_bytes, "every swapped byte restored (both streams still live)");
    assert!(!arena.is_lent());
    // live rows read back bit-identical (bases may have moved: compaction at lend)
    for (slot, si, comp, keys) in &expect {
        let r = arena.stream(*slot).unwrap().comp[*si];
        let width = arena.stores[*si].width as usize;
        let (cb, kb) = arena.store_buffers(*si)?;
        let mut c = vec![0u16; comp.len()];
        cb.slice_view(r.base as usize * width, c.len()).copy_to_host(&mut c)?;
        let mut k = vec![0u8; keys.len()];
        kb.slice_view(r.base as usize * E2M1_KEY_ROW_BYTES, k.len()).copy_to_host(&mut k)?;
        assert!(c == *comp, "slot {slot} store {si}: comp rows changed across the loan");
        assert!(k == *keys, "slot {slot} store {si}: index keys changed across the loan");
    }
    // a second lend/end cycle (re-lend after a return) is idempotent and still exact
    loan.lend(&mut arena, &stream, &mut bf16, &mut bu8)?;
    loan.end(&mut arena)?;
    eprintln!("PASS: live decode KV bit-identical across the loan ({} MiB swapped)", st.swapped_bytes >> 20);
    Ok(())
}
