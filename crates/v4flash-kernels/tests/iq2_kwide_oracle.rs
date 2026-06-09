//! Oracle: `iq2_xxs_pair_matvec_fused_swiglu_kwide` (and `_chunked_staged_v2`)
//! vs the production `iq2_xxs_pair_matvec_fused_swiglu_chunked_staged`.
//!
//! Hardening beyond the tile8 oracle template:
//!   - xq d varies per (token, super-block)  (the tile8 d-cache bug class)
//!   - WEIGHT d varies per (row, matrix, super-block) (kwide re-associates the
//!     wd/ls scale chain — a wd mixup between the pair's two super-blocks
//!     would pass a uniform-d test)
//!   - two experts, two work items
//!   - one FULL chunk (32 members) and one PARTIAL chunk (19 members, not a
//!     multiple of the unroll) to exercise both dot-loop paths
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test iq2_kwide_oracle -- --ignored --nocapture

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

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 { (self.next() & 0xff) as u8 }
}

/// Small f16 scales so f32 sums stay bounded: 1/64, 1/32, 1/16, 1/8.
const F16_SCALES: [u16; 4] = [0x2400, 0x2800, 0x2c00, 0x3000];

struct OracleSetup {
    gate_w: DeviceBuffer<u8>,
    up_w: DeviceBuffer<u8>,
    xq: DeviceBuffer<u8>,
    expert_w: DeviceBuffer<f32>,
    group_count: DeviceBuffer<i32>,
    expert_members: DeviceBuffer<i32>,
    work_items: DeviceBuffer<i32>,
    gate_bpe: u32,
    up_bpe: u32,
    cs_n_used: u32,
    max_per_expert: u32,
    n_work_items: u32,
    out_elems: usize,
    // (b_idx, slot) pairs that receive output
    touched: Vec<(usize, usize)>,
}

fn build_setup(igpu: &Device) -> eyre::Result<OracleSetup> {
    let b: u32 = 48;
    let cs_n_used = N_EXPERT_USED as u32;
    let max_per_expert: u32 = b;
    // Two experts: e0 full chunk (32 members), e1 partial (19 members).
    let experts: [(usize, i32); 2] = [(0, 32), (3, 19)];
    let n_work_items: u32 = 2;

    let gate_bpe = (N_FF_EXP as usize) * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_IQ2_XXS_BYTES;
    let up_bpe = gate_bpe;
    let total_gate = gate_bpe * (N_EXPERT as usize);
    let total_up = up_bpe * (N_EXPERT as usize);

    let mut gate_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_gate)?;
    let mut up_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_up)?;

    let mut rng = Lcg::new(0xdecafbad);
    let mut gate_host = vec![0u8; total_gate];
    let mut up_host = vec![0u8; total_up];
    for (e, _) in experts {
        for r in 0..(N_FF_EXP as usize) {
            let off = e * gate_bpe + r * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_IQ2_XXS_BYTES;
            for bi in 0..(BLOCKS_Q8K_GATE_IN as usize) {
                let b_off = off + bi * BLOCK_IQ2_XXS_BYTES;
                // Weight d varies per (row, matrix, super-block).
                let dg = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
                let du = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
                gate_host[b_off..b_off + 2].copy_from_slice(&dg);
                up_host[b_off..b_off + 2].copy_from_slice(&du);
                for k in 0..64 {
                    gate_host[b_off + 2 + k] = rng.next_byte();
                    up_host[b_off + 2 + k] = rng.next_byte();
                }
            }
        }
    }
    gate_w.copy_from_host(&gate_host)?;
    up_w.copy_from_host(&up_host)?;

    let xq_per_tok = (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES;
    let total_xq = xq_per_tok * (b as usize);
    let mut xq: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_xq)?;
    let mut xq_host = vec![0u8; total_xq];
    for t in 0..(b as usize) {
        for bi in 0..(BLOCKS_Q8K_GATE_IN as usize) {
            let o = t * xq_per_tok + bi * BLOCK_Q8_K_BYTES;
            // d varies PER (token, super-block).
            let d_val = 0.03f32 + ((rng.next() & 0xff) as f32) * 0.001f32;
            xq_host[o..o + 4].copy_from_slice(&d_val.to_le_bytes());
            for k in 0..256 {
                xq_host[o + 4 + k] = rng.next_byte();
            }
        }
    }
    xq.copy_from_host(&xq_host)?;

    let mut expert_w: DeviceBuffer<f32> =
        DeviceBuffer::new(igpu.id, (b as usize) * (cs_n_used as usize))?;
    let mut ew_host = vec![0.0f32; (b as usize) * (cs_n_used as usize)];
    for v in ew_host.iter_mut() {
        *v = 0.5f32 + ((rng.next() & 0xff) as f32) * 0.004f32;
    }
    expert_w.copy_from_host(&ew_host)?;

    let mut group_count: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, N_EXPERT as usize)?;
    let mut expert_members: DeviceBuffer<i32> =
        DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * (max_per_expert as usize))?;
    let mut work_items: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_work_items as usize)?;

    let mut gc_h = vec![0i32; N_EXPERT as usize];
    let mut em_h = vec![0i32; (N_EXPERT as usize) * (max_per_expert as usize)];
    let mut wi_h = Vec::with_capacity(n_work_items as usize);
    let mut touched = Vec::new();
    let mut next_pair = 0usize; // distinct (b_idx, slot) per member across experts
    for (e, group_n) in experts {
        gc_h[e] = group_n;
        for i in 0..(group_n as usize) {
            let b_idx = next_pair % (b as usize);
            let slot = (next_pair / (b as usize)) % (cs_n_used as usize);
            em_h[e * (max_per_expert as usize) + i] = ((b_idx as i32) << 16) | (slot as i32);
            touched.push((b_idx, slot));
            next_pair += 1;
        }
        wi_h.push(((e as i32) << 16) | 0i32);
    }
    group_count.copy_from_host(&gc_h)?;
    expert_members.copy_from_host(&em_h)?;
    work_items.copy_from_host(&wi_h)?;

    let out_elems = (b as usize) * (cs_n_used as usize) * (N_FF_EXP as usize);
    Ok(OracleSetup {
        gate_w, up_w, xq, expert_w, group_count, expert_members, work_items,
        gate_bpe: gate_bpe as u32, up_bpe: up_bpe as u32, cs_n_used,
        max_per_expert, n_work_items, out_elems, touched,
    })
}

fn compare(
    name: &str,
    ref_host: &[f32],
    test_host: &[f32],
    s: &OracleSetup,
) -> eyre::Result<()> {
    let mut max_abs_diff = 0f32;
    let mut max_abs_ref = 0f32;
    let mut sum_sq_diff = 0f64;
    let mut n = 0usize;
    for &(b_idx, slot) in &s.touched {
        for row in 0..(N_FF_EXP as usize) {
            let off = (b_idx * (s.cs_n_used as usize) + slot) * (N_FF_EXP as usize) + row;
            let r = ref_host[off];
            let t = test_host[off];
            let d = (r - t).abs();
            if d > max_abs_diff { max_abs_diff = d; }
            if r.abs() > max_abs_ref { max_abs_ref = r.abs(); }
            sum_sq_diff += (d as f64) * (d as f64);
            n += 1;
        }
    }
    let rmse = (sum_sq_diff / (n as f64)).sqrt();
    if max_abs_ref < 1e-6 {
        return Err(eyre!("{name}: reference output near-zero — degenerate test"));
    }
    let rel = max_abs_diff / max_abs_ref.max(1e-6);
    eprintln!(
        "{name}: n={n} max|ref|={max_abs_ref:.4} max_abs_diff={max_abs_diff:.6} \
         rmse={rmse:.6} rel={rel:.6}"
    );
    if rel >= 5e-3 {
        return Err(eyre!("{name} diverges from staged: rel={rel}"));
    }
    Ok(())
}

#[test]
#[ignore]
fn iq2_kwide_and_v2_match_staged() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2 = Iq2XxsPairMatvec::for_arch(&arch)?;

    let mut s = build_setup(&igpu)?;
    let chunk_size: u32 = 32;

    let mut mid_ref: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, s.out_elems)?;
    let mut mid_kw: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, s.out_elems)?;
    let mut mid_v2: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, s.out_elems)?;
    mid_ref.fill_zero()?;
    mid_kw.fill_zero()?;
    mid_v2.fill_zero()?;

    macro_rules! launch {
        ($method:ident, $mid:expr) => {
            iq2.$method(
                &stream, $mid, &s.gate_w, &s.up_w, &s.xq, &s.expert_w,
                &s.group_count, &s.expert_members, &s.work_items,
                s.gate_bpe, s.up_bpe, s.cs_n_used, s.max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
                s.n_work_items,
            )?
        };
    }
    launch!(launch_fused_swiglu_chunked_staged, &mut mid_ref);
    launch!(launch_fused_swiglu_kwide, &mut mid_kw);
    launch!(launch_fused_swiglu_chunked_staged_v2, &mut mid_v2);
    stream.synchronize()?;

    let mut ref_host = vec![0f32; s.out_elems];
    let mut kw_host = vec![0f32; s.out_elems];
    let mut v2_host = vec![0f32; s.out_elems];
    mid_ref.copy_to_host(&mut ref_host)?;
    mid_kw.copy_to_host(&mut kw_host)?;
    mid_v2.copy_to_host(&mut v2_host)?;

    compare("kwide", &ref_host, &kw_host, &s)?;
    compare("staged_v2", &ref_host, &v2_host, &s)?;
    Ok(())
}
