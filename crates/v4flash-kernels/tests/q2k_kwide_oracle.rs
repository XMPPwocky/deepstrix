//! Oracle: `q2_k_matvec_par_by_expert_kwide` vs production
//! `q2_k_matvec_par_by_expert` on RANDOM weights/activations.
//!
//! Hardening (per the tile8 d-cache lesson):
//!   - xq d varies per (token, super-block)
//!   - weight d AND dmin vary per (row, super-block)
//!   - random 4-bit group scales/mins (sc bytes fully random)
//!   - one FULL chunk (32 members) + one PARTIAL chunk (19) — exercises both
//!     dot-loop paths and the kwide unroll tail
//!
//! The folded-scale integer sum is exactly associative (3·15 < 256, no
//! cross-byte carries), so kwide should match by_expert to f32 round-off;
//! tolerance 5e-3 rel as for the other MoE oracles.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test q2k_kwide_oracle -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, N_EMBD, N_EXPERT, N_EXPERT_USED};
use v4flash_kernels::q2_k::{Q2KAccumulateMatvec, BLOCK_Q2_K_BYTES};
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
    fn new(seed: u64) -> Self { Lcg(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 { (self.next() & 0xff) as u8 }
}

/// Small f16 scales: 1/64, 1/32, 1/16, 1/8.
const F16_SCALES: [u16; 4] = [0x2400, 0x2800, 0x2c00, 0x3000];

#[test]
#[ignore]
fn q2k_kwide_matches_by_expert() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let q2k = Q2KAccumulateMatvec::for_arch(&arch)?;

    let b: u32 = 48;
    let n_used = N_EXPERT_USED as u32;
    let n_rows = N_EMBD;
    let n_blocks_in = BLOCKS_Q8K_DOWN_IN;
    let dbpe = (n_rows as usize) * (n_blocks_in as usize) * BLOCK_Q2_K_BYTES;
    let xq_slot_stride = (n_blocks_in as u32) * (BLOCK_Q8_K_BYTES as u32);
    let max_per_expert = b;
    let chunk_size: u32 = 32;
    // Two experts: full chunk + partial chunk.
    let experts: [(usize, i32); 2] = [(1, 32), (7, 19)];

    let mut rng = Lcg::new(0xfeedface);

    // Random weights for the two experts only.
    let mut down_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * dbpe)?;
    let mut w_host = vec![0u8; (N_EXPERT as usize) * dbpe];
    for (e, _) in experts {
        for r in 0..(n_rows as usize) {
            for bi in 0..(n_blocks_in as usize) {
                let o = e * dbpe + (r * (n_blocks_in as usize) + bi) * BLOCK_Q2_K_BYTES;
                for k in 0..80 {
                    w_host[o + k] = rng.next_byte(); // sc[16] + q2[64], fully random
                }
                let d = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
                let dm = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
                w_host[o + 80..o + 82].copy_from_slice(&d);
                w_host[o + 82..o + 84].copy_from_slice(&dm);
            }
        }
    }
    down_w.copy_from_host(&w_host)?;

    // Random xq with CORRECT bsums (the min-term path consumes them).
    let xq_total = (b as usize) * (n_used as usize) * (xq_slot_stride as usize);
    let mut xq: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq_total)?;
    let mut xq_host = vec![0u8; xq_total];
    for t in 0..(b as usize) * (n_used as usize) {
        for bi in 0..(n_blocks_in as usize) {
            let o = t * (xq_slot_stride as usize) + bi * BLOCK_Q8_K_BYTES;
            let d_val = 0.02f32 + ((rng.next() & 0xff) as f32) * 0.0008f32;
            xq_host[o..o + 4].copy_from_slice(&d_val.to_le_bytes());
            let mut bsums = [0i16; 16];
            for k in 0..256 {
                let v = rng.next_byte() as i8;
                xq_host[o + 4 + k] = v as u8;
                bsums[k / 16] += v as i16;
            }
            for (j, s) in bsums.iter().enumerate() {
                let by = s.to_le_bytes();
                xq_host[o + 260 + 2 * j..o + 260 + 2 * j + 2].copy_from_slice(&by);
            }
        }
    }
    xq.copy_from_host(&xq_host)?;

    // Routing arrays.
    let mut gc_h = vec![0i32; N_EXPERT as usize];
    let mut em_h = vec![0i32; (N_EXPERT as usize) * (max_per_expert as usize)];
    let mut wi_h: Vec<i32> = Vec::new();
    let mut touched: Vec<(usize, usize)> = Vec::new();
    let mut next_pair = 0usize;
    for (e, group_n) in experts {
        gc_h[e] = group_n;
        for i in 0..(group_n as usize) {
            let b_idx = next_pair % (b as usize);
            let slot = (next_pair / (b as usize)) % (n_used as usize);
            em_h[e * (max_per_expert as usize) + i] = ((b_idx as i32) << 16) | (slot as i32);
            touched.push((b_idx, slot));
            next_pair += 1;
        }
        wi_h.push(((e as i32) << 16) | 0i32);
    }
    let n_work_items = wi_h.len() as u32;
    let mut group_count_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, N_EXPERT as usize)?;
    let mut expert_members_d: DeviceBuffer<i32> =
        DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * (max_per_expert as usize))?;
    let mut work_items_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_work_items as usize)?;
    group_count_d.copy_from_host(&gc_h)?;
    expert_members_d.copy_from_host(&em_h)?;
    work_items_d.copy_from_host(&wi_h)?;

    let part_elems = (b as usize) * (n_used as usize) * (n_rows as usize);
    let mut part_ref: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, part_elems)?;
    let mut part_kw: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, part_elems)?;
    part_ref.fill_zero()?;
    part_kw.fill_zero()?;

    q2k.launch_by_expert(
        &stream, &mut part_ref, &down_w, &xq,
        &group_count_d, &expert_members_d, &work_items_d,
        dbpe as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
        n_rows, n_blocks_in, n_work_items,
    )?;
    q2k.launch_by_expert_kwide(
        &stream, &mut part_kw, &down_w, &xq,
        &group_count_d, &expert_members_d, &work_items_d,
        dbpe as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
        n_rows, n_blocks_in, n_work_items,
    )?;
    stream.synchronize()?;

    let mut ref_host = vec![0f32; part_elems];
    let mut kw_host = vec![0f32; part_elems];
    part_ref.copy_to_host(&mut ref_host)?;
    part_kw.copy_to_host(&mut kw_host)?;

    let mut max_abs_diff = 0f32;
    let mut max_abs_ref = 0f32;
    let mut sum_sq = 0f64;
    let mut n = 0usize;
    for &(b_idx, slot) in &touched {
        for row in 0..(n_rows as usize) {
            let off = (b_idx * (n_used as usize) + slot) * (n_rows as usize) + row;
            let r = ref_host[off];
            let t = kw_host[off];
            let d = (r - t).abs();
            if d > max_abs_diff { max_abs_diff = d; }
            if r.abs() > max_abs_ref { max_abs_ref = r.abs(); }
            sum_sq += (d as f64) * (d as f64);
            n += 1;
        }
    }
    let rmse = (sum_sq / n as f64).sqrt();
    if max_abs_ref < 1e-6 {
        return Err(eyre!("reference output near-zero — degenerate test"));
    }
    let rel = max_abs_diff / max_abs_ref;
    eprintln!(
        "q2k kwide vs by_expert: n={n} max|ref|={max_abs_ref:.4} \
         max_abs_diff={max_abs_diff:.6} rmse={rmse:.6} rel={rel:.6}"
    );
    assert!(rel < 5e-3, "q2k kwide diverges: rel={rel}");
    Ok(())
}
