//! Oracle: `iq2_xxs_pair_matvec_fused_swiglu_tile8_row32` vs the production
//! `iq2_xxs_pair_matvec_fused_swiglu_chunked_staged`.
//!
//! Synthetic small inputs (random iq2 + random q8_K) — both kernels see the
//! exact same buffers, run, and the per-element max-abs diff is reported.
//! Tolerance: 5e-3 (different f32 reduction order, different chunk_size, so
//! we allow ~half a ULP per dp4a × the per-row depth).
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test iq2_tile8_oracle -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{
    BLOCKS_Q8K_GATE_IN, N_EXPERT, N_EXPERT_USED, N_FF_EXP, SWIGLU_CLAMP_EXP,
};
use v4flash_kernels::iq2_xxs::Iq2XxsPairMatvec;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

const BLOCK_IQ2_XXS_BYTES: usize = 66;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

/// Cheap LCG over a seed so we don't pull in `rand`.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 { (self.next() & 0xff) as u8 }
}

#[test]
#[ignore]
fn iq2_tile8_matches_staged() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2 = Iq2XxsPairMatvec::for_arch(&arch)?;

    let b: u32 = 16;
    let cs_n_used = N_EXPERT_USED as u32;
    // Use small chunk_size so both kernels' work_items layout is similar.
    let chunk_size_staged: u32 = 8;  // staged supports up to 32 but tile8 caps at 8
    let chunk_size_tile8: u32 = 8;

    // Just one expert, full of work — easy to inspect.
    let n_work_items: u32 = 1;
    let n_distinct = 1usize;
    let max_per_expert: u32 = b;

    let gate_bpe = (N_FF_EXP as usize) * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_IQ2_XXS_BYTES;
    let up_bpe   = gate_bpe;
    let total_gate = gate_bpe * (N_EXPERT as usize);
    let total_up   = up_bpe   * (N_EXPERT as usize);

    let mut gate_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_gate)?;
    let mut up_w:   DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_up)?;

    // Random iq2_xxs blocks. Per 66-byte block: d (uint16) is a half-float
    // scale — we set to ~0x2c00 (0.0625) so the f32 sums stay bounded.
    // qs (64 bytes) are 8-bit grid indices interleaved with sign indices.
    // For oracle purposes any well-formed pattern works; we just need
    // non-zero outputs.
    let mut rng = Lcg::new(0xcafef00d);
    let mut gate_host = vec![0u8; total_gate];
    let mut up_host   = vec![0u8; total_up];
    // Only initialize the one expert we use.
    let e = 0usize;
    for r in 0..(N_FF_EXP as usize) {
        let off = e * gate_bpe + r * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_IQ2_XXS_BYTES;
        for bi in 0..(BLOCKS_Q8K_GATE_IN as usize) {
            let b_off = off + bi * BLOCK_IQ2_XXS_BYTES;
            // d = 0.0625 in f16 = 0x2c00
            gate_host[b_off + 0] = 0x00; gate_host[b_off + 1] = 0x2c;
            up_host  [b_off + 0] = 0x00; up_host  [b_off + 1] = 0x2c;
            for k in 0..64 {
                gate_host[b_off + 2 + k] = rng.next_byte();
                up_host  [b_off + 2 + k] = rng.next_byte();
            }
        }
    }
    gate_w.copy_from_host(&gate_host)?;
    up_w  .copy_from_host(&up_host)?;

    // xq: B tokens × q8_K blocks. Each 292-byte block: d (f32) + 256 int8 qs + 32 int16 bsums.
    let xq_per_tok = (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES;
    let total_xq = xq_per_tok * (b as usize);
    let mut xq: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_xq)?;
    let mut xq_host = vec![0u8; total_xq];
    for t in 0..(b as usize) {
        for bi in 0..(BLOCKS_Q8K_GATE_IN as usize) {
            let o = t * xq_per_tok + bi * BLOCK_Q8_K_BYTES;
            // d = 0.0625 (f32 little-endian)
            let d_bytes = 0.0625f32.to_le_bytes();
            xq_host[o..o+4].copy_from_slice(&d_bytes);
            for k in 0..256 {
                let v = rng.next_byte() as i8;
                xq_host[o + 4 + k] = v as u8;
            }
            // bsums (last 64 bytes = 32 int16) — production code computes
            // these correctly via Q8KQuantize; for matvec they're unused, so
            // leaving zeros is fine.
        }
    }
    xq.copy_from_host(&xq_host)?;

    let mut expert_w: DeviceBuffer<f32> =
        DeviceBuffer::new(igpu.id, (b as usize) * (cs_n_used as usize))?;
    let ew_host = vec![1.0f32; (b as usize) * (cs_n_used as usize)];
    expert_w.copy_from_host(&ew_host)?;

    let mut group_count: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, N_EXPERT as usize)?;
    let mut expert_members: DeviceBuffer<i32> =
        DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * (max_per_expert as usize))?;
    let mut work_items_st: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_work_items as usize)?;
    let mut work_items_t8: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_work_items as usize)?;

    let mut gc_h = vec![0i32; N_EXPERT as usize];
    let mut em_h = vec![0i32; (N_EXPERT as usize) * (max_per_expert as usize)];
    let group_n: i32 = chunk_size_tile8 as i32;  // both variants use this many members
    gc_h[e] = group_n;
    for i in 0..(group_n as usize) {
        let b_idx = i % (b as usize);
        let slot = i % (cs_n_used as usize);
        em_h[e * (max_per_expert as usize) + i] = ((b_idx as i32) << 16) | (slot as i32);
    }
    group_count.copy_from_host(&gc_h)?;
    expert_members.copy_from_host(&em_h)?;
    let wi_h = vec![((e as i32) << 16) | 0i32; n_work_items as usize];
    work_items_st.copy_from_host(&wi_h)?;
    work_items_t8.copy_from_host(&wi_h)?;

    let out_elems = (b as usize) * (cs_n_used as usize) * (N_FF_EXP as usize);
    let mut mid_st: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, out_elems)?;
    let mut mid_t8: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, out_elems)?;
    mid_st.fill_zero()?;
    mid_t8.fill_zero()?;

    iq2.launch_fused_swiglu_chunked_staged(
        &stream, &mut mid_st, &gate_w, &up_w, &xq, &expert_w,
        &group_count, &expert_members, &work_items_st,
        gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
        chunk_size_staged, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
        n_work_items,
    )?;
    iq2.launch_fused_swiglu_tile8_row32(
        &stream, &mut mid_t8, &gate_w, &up_w, &xq, &expert_w,
        &group_count, &expert_members, &work_items_t8,
        gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
        chunk_size_tile8, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
        n_work_items,
    )?;
    stream.synchronize()?;

    let mut st_host = vec![0f32; out_elems];
    let mut t8_host = vec![0f32; out_elems];
    mid_st.copy_to_host(&mut st_host)?;
    mid_t8.copy_to_host(&mut t8_host)?;

    // Stats on only the (b, slot) entries we actually wrote (others stay 0).
    let mut touched = 0usize;
    let mut max_abs_diff = 0f32;
    let mut max_abs_st = 0f32;
    let mut max_abs_t8 = 0f32;
    let mut sum_sq_diff = 0f64;
    for i in 0..(group_n as usize) {
        let b_idx = i % (b as usize);
        let slot = i % (cs_n_used as usize);
        for row in 0..(N_FF_EXP as usize) {
            let off = ((b_idx * (cs_n_used as usize)) + slot) * (N_FF_EXP as usize) + row;
            let s = st_host[off];
            let t = t8_host[off];
            let d = (s - t).abs();
            if d > max_abs_diff { max_abs_diff = d; }
            if s.abs() > max_abs_st { max_abs_st = s.abs(); }
            if t.abs() > max_abs_t8 { max_abs_t8 = t.abs(); }
            sum_sq_diff += (d as f64) * (d as f64);
            touched += 1;
        }
    }
    let rmse = (sum_sq_diff / (touched as f64)).sqrt() as f32;

    eprintln!(
        "tile8 vs staged: touched={touched} max|st|={max_abs_st:.4} max|t8|={max_abs_t8:.4} \
         max_abs_diff={max_abs_diff:.6} rmse={rmse:.6}"
    );
    if max_abs_st < 1e-6 && max_abs_t8 < 1e-6 {
        return Err(eyre!("both kernels produced near-zero output — test is degenerate"));
    }
    let rel = max_abs_diff / max_abs_st.max(1e-6);
    eprintln!("relative max_abs_diff = {rel:.6}");
    assert!(rel < 5e-3, "tile8 diverges from staged: rel={rel}");
    Ok(())
}
