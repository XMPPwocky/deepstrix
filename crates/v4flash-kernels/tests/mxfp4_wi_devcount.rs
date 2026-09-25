//! `V41_MOE_WI_DEVCOUNT`: the MXFP4 kwide gate/up and kwide2 down kernels
//! launched with an UPPER-BOUND grid plus the builder's device-side count must
//! be bit-identical to the exact-grid launch. The work-item buffer past the
//! count is padded with a DECOY item (a real expert whose one member is an
//! otherwise untouched (row, slot)), so a missing guard writes where nothing
//! should. A device count of 0 must compute nothing.
//!
//!   cargo test -p v4flash-kernels --release --features v41 --test mxfp4_wi_devcount -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT_USED, N_FF_EXP};
use v4flash_kernels::mxfp4::Mxfp4Matvec;
use v4flash_kernels::mxfp4_pair::Mxfp4PairMatvec;
use v4flash_kernels::mxfp4_tables::SUPER_MXFP4_BYTES;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// MXFP4 v2 super-blocks with sane E8M0 scales (2^-10..2^5).
fn mxfp4_weights(rng: &mut Lcg, n_experts: usize, n_rows: usize, nb: usize) -> Vec<u8> {
    let mut w = vec![0u8; n_experts * n_rows * nb * SUPER_MXFP4_BYTES];
    for sb in w.chunks_exact_mut(SUPER_MXFP4_BYTES) {
        for b8 in 0..8 {
            sb[128 + b8] = 118 + (rng.next() & 0x0f) as u8;
            for j in 0..16 {
                sb[b8 * 16 + j] = rng.next() as u8;
            }
        }
    }
    w
}

/// Q8_K blocks with a positive scale and random int8s (bsums consistent).
fn q8k_blocks(rng: &mut Lcg, n_blocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; n_blocks * BLOCK_Q8_K_BYTES];
    for blk in out.chunks_exact_mut(BLOCK_Q8_K_BYTES) {
        let d = 0.001f32 + (rng.next() % 1000) as f32 * 1e-5;
        blk[..4].copy_from_slice(&d.to_le_bytes());
        let mut bsums = [0i16; 16];
        for k in 0..256 {
            let q = (rng.next() % 255) as i32 - 127;
            blk[4 + k] = q as i8 as u8;
            bsums[k / 16] += q as i16;
        }
        for (j, s) in bsums.iter().enumerate() {
            blk[260 + 2 * j..262 + 2 * j].copy_from_slice(&s.to_le_bytes());
        }
    }
    out
}

fn upload<T: Copy>(id: i32, h: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
#[ignore]
fn mxfp4_moe_device_count_is_bit_exact() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let s = Stream::new(igpu.id)?;
    let pair = Mxfp4PairMatvec::for_arch(&arch)?;
    let down = Mxfp4Matvec::for_arch(&arch)?;
    let mut rng = Lcg(0x5749_4443); // "WIDC"

    let n_used = N_EXPERT_USED;
    let (b, chunk) = (5usize, 32u32);
    let n_slots = 8usize; // materialised experts
    // Real groups: expert -> members; members are distinct (row, slot) pairs.
    let groups: [(usize, usize); 3] = [(5, 3), (2, 2), (7, 1)];
    let decoy_expert = 1usize;
    let decoy_member = (4usize, n_used - 1); // (row, slot) no real member uses
    let max_per_expert = b.max(3);
    let mut gc = vec![0i32; n_slots];
    let mut em = vec![0i32; n_slots * max_per_expert];
    let mut wi: Vec<i32> = Vec::new();
    let mut np = 0usize;
    for &(e, n) in &groups {
        gc[e] = n as i32;
        wi.push((e as i32) << 16);
        for m in 0..n {
            let (row, slot) = (np % 4, np / 4); // rows 0..3 only: row 4 is the decoy's
            em[e * max_per_expert + m] = ((row as i32) << 16) | slot as i32;
            np += 1;
        }
    }
    let n_real = wi.len();
    gc[decoy_expert] = 1;
    em[decoy_expert * max_per_expert] = ((decoy_member.0 as i32) << 16) | decoy_member.1 as i32;
    let bound = b * n_used; // the route's upper bound
    let mut wi_pad = wi.clone();
    wi_pad.resize(bound, (decoy_expert as i32) << 16);

    let gc_d = upload(igpu.id, &gc)?;
    let em_d = upload(igpu.id, &em)?;
    let wi_exact_d = upload(igpu.id, &wi)?;
    let wi_pad_d = upload(igpu.id, &wi_pad)?;
    let cnt_d = upload(igpu.id, &[n_real as i32])?;
    let zero_d = upload(igpu.id, &[0i32])?;

    // ---- gate/up (pair kwide): mid [B, n_used, N_FF_EXP]
    let (nr, nb) = (N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize);
    let bpe = nr * nb * SUPER_MXFP4_BYTES;
    let gate_d = upload(igpu.id, &mxfp4_weights(&mut rng, n_slots, nr, nb))?;
    let up_d = upload(igpu.id, &mxfp4_weights(&mut rng, n_slots, nr, nb))?;
    let xq_d = upload(igpu.id, &q8k_blocks(&mut rng, b * nb))?;
    let ew: Vec<f32> = (0..b * n_used).map(|i| 0.05 + 0.01 * (i % 17) as f32).collect();
    let ew_d = upload(igpu.id, &ew)?;
    let mut mid_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, b * n_used * nr)?;
    let run_pair = |mid_d: &mut DeviceBuffer<f32>, wi: &DeviceBuffer<i32>, n: u32, c: Option<&DeviceBuffer<i32>>| -> eyre::Result<Vec<f32>> {
        mid_d.fill_zero()?;
        pair.launch_fused_swiglu_kwide_ex(
            &s, mid_d, &gate_d, &up_d, &xq_d, &ew_d, &gc_d, &em_d, wi, n, bpe as u32, bpe as u32,
            n_used as u32, max_per_expert as u32, chunk, 10.0, nr as u32, nb as u32, c,
        )?;
        s.synchronize()?;
        let mut h = vec![0f32; b * n_used * nr];
        mid_d.copy_to_host(&mut h)?;
        Ok(h)
    };
    let exact = run_pair(&mut mid_d, &wi_exact_d, n_real as u32, None)?;
    let dev = run_pair(&mut mid_d, &wi_pad_d, bound as u32, Some(&cnt_d))?;
    let none = run_pair(&mut mid_d, &wi_pad_d, bound as u32, Some(&zero_d))?;
    assert!(exact.iter().any(|&v| v != 0.0), "gate/up computed nothing");
    assert_eq!(bits(&dev), bits(&exact), "gate/up: device count differs from exact grid");
    let decoy_off = (decoy_member.0 * n_used + decoy_member.1) * nr;
    assert!(dev[decoy_off..decoy_off + nr].iter().all(|&v| v == 0.0), "gate/up: a work item past the count ran");
    assert!(none.iter().all(|&v| v == 0.0), "gate/up: count 0 still computed");
    // Sanity: WITHOUT the device count the decoy padding does run (the test can fail).
    let unguarded = run_pair(&mut mid_d, &wi_pad_d, bound as u32, None)?;
    assert!(unguarded[decoy_off..decoy_off + nr].iter().any(|&v| v != 0.0), "decoy never ran: the test cannot detect a missing guard");
    eprintln!("gate/up: bit-exact over {} values, {n_real} of {bound} work items live", exact.len());

    // ---- down (kwide2): partials [B * n_used, N_EMBD]
    let (nr, nb) = (N_EMBD as usize, BLOCKS_Q8K_DOWN_IN as usize);
    let dbpe = nr * nb * SUPER_MXFP4_BYTES;
    let w_d = upload(igpu.id, &mxfp4_weights(&mut rng, n_slots, nr, nb))?;
    let slot_stride = nb * BLOCK_Q8_K_BYTES;
    let midq_d = upload(igpu.id, &q8k_blocks(&mut rng, b * n_used * nb))?;
    let mut part_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, b * n_used * nr)?;
    let run_down = |part_d: &mut DeviceBuffer<f32>, wi: &DeviceBuffer<i32>, n: u32, c: Option<&DeviceBuffer<i32>>| -> eyre::Result<Vec<f32>> {
        part_d.fill_zero()?;
        down.launch_by_expert_kwide2_ex(
            &s, part_d, &w_d, &midq_d, &gc_d, &em_d, wi, n, dbpe as u32, slot_stride as u32,
            n_used as u32, max_per_expert as u32, chunk, nr as u32, nb as u32, c,
        )?;
        s.synchronize()?;
        let mut h = vec![0f32; b * n_used * nr];
        part_d.copy_to_host(&mut h)?;
        Ok(h)
    };
    let exact = run_down(&mut part_d, &wi_exact_d, n_real as u32, None)?;
    let dev = run_down(&mut part_d, &wi_pad_d, bound as u32, Some(&cnt_d))?;
    let none = run_down(&mut part_d, &wi_pad_d, bound as u32, Some(&zero_d))?;
    assert!(exact.iter().any(|&v| v != 0.0), "down computed nothing");
    assert_eq!(bits(&dev), bits(&exact), "down: device count differs from exact grid");
    let decoy_off = (decoy_member.0 * n_used + decoy_member.1) * nr;
    assert!(dev[decoy_off..decoy_off + nr].iter().all(|&v| v == 0.0), "down: a work item past the count ran");
    assert!(none.iter().all(|&v| v == 0.0), "down: count 0 still computed");
    let unguarded = run_down(&mut part_d, &wi_pad_d, bound as u32, None)?;
    assert!(unguarded[decoy_off..decoy_off + nr].iter().any(|&v| v != 0.0), "decoy never ran: the test cannot detect a missing guard");
    eprintln!("down: bit-exact over {} values, {n_real} of {bound} work items live", exact.len());

    // A grid bound past the work-item buffer is refused, not launched.
    assert!(pair
        .launch_fused_swiglu_kwide_ex(
            &s, &mut mid_d, &gate_d, &up_d, &xq_d, &ew_d, &gc_d, &em_d, &wi_exact_d, bound as u32,
            bpe as u32, bpe as u32, n_used as u32, max_per_expert as u32, chunk, 10.0, N_FF_EXP,
            BLOCKS_Q8K_GATE_IN, Some(&cnt_d),
        )
        .is_err());
    Ok(())
}
