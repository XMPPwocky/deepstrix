//! The 2026-09-26 A_moe_smallb / B_moe_prefill sweep kernels vs the kernels they
//! replace, on synthetic MXFP4 experts (no model load; < 450 MB of iGPU memory):
//!
//!   1. `mxfp4_matvec_par_by_expert_smallb` (A, dn2_member_outer) vs
//!      `mxfp4_matvec_par_by_expert_kwide2`: BIT-EXACT partials (whole buffer,
//!      sentinel-filled, so the written set must match too) at b = 1..8 in three
//!      selection regimes (every row on the same 6 experts, hub-like distinct
//!      picks, few shared experts with chunk 4 = split groups), exact grid (null
//!      count) and bound grid + device count; plus `dispatch::moe_down_mxfp4` at
//!      rows = b (`V41_MOE_DOWN_DN2`) and at rows = 16 (kwide2).
//!   2. `mxfp4_pair_matvec_fused_swiglu_wmma` / `mxfp4_matvec_par_by_expert_wmma`
//!      (B, int8 WMMA) vs kwide / kwide2: NOT bit-exact by design (f32
//!      re-association of identical int32 dots). Reports rel_rmse / max_abs /
//!      differing lanes and asserts rel_rmse < 1e-5, no non-finite values and the
//!      identical written set; plus `dispatch::moe_gate_up_chunked_rows` /
//!      `moe_down_mxfp4` at rows = B (WMMA above 256 / 128 rows under the knobs).
//!
//! gfx1151 only (the MXFP4 MoE runs on the iGPUs; the WMMA arm is RDNA3-only).
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test
//! mxfp4_moe_sweep -- --ignored --test-threads=1 --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::GgufType;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT_USED, N_FF_EXP, SWIGLU_CLAMP_EXP};
use v4flash_kernels::het::dispatch;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mxfp4_tables::SUPER_MXFP4_BYTES;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

fn pick_igpu() -> eyre::Result<Option<Device>> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(Some(d));
        }
    }
    Ok(None)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// MXFP4 v2 super-blocks, E8M0 scales 2^(s-128) with s in [sc_lo, sc_lo + span).
fn mxfp4_weights(rng: &mut Lcg, n_experts: usize, n_rows: usize, nb: usize, sc_lo: u8, span: u32) -> Vec<u8> {
    let mut w = vec![0u8; n_experts * n_rows * nb * SUPER_MXFP4_BYTES];
    for sb in w.chunks_exact_mut(SUPER_MXFP4_BYTES) {
        for b8 in 0..8 {
            sb[128 + b8] = sc_lo + (rng.next() % span) as u8;
            for j in 0..16 {
                sb[b8 * 16 + j] = rng.next() as u8;
            }
        }
    }
    w
}

/// Q8_K blocks with a positive scale, random int8s and CONSISTENT bsums (the
/// WMMA arm reads them).
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
    let mut d = DeviceBuffer::new(id, h.len().max(1))?;
    if !h.is_empty() {
        d.copy_from_host(h)?;
    }
    Ok(d)
}

fn download(d: &DeviceBuffer<f32>, n: usize) -> eyre::Result<Vec<f32>> {
    let mut v = vec![0f32; d.len()];
    d.copy_to_host(&mut v)?;
    v.truncate(n);
    Ok(v)
}

const SENTINEL: f32 = f32::from_bits(0x7FC0_BEEF);

/// Groups built on the host exactly like the production builder's output:
/// group_count[e], expert_members[e * max + m] = (row << 16) | slot, work items
/// (e << 16) | member_start per `chunk` members.
struct Groups {
    gc: Vec<i32>,
    em: Vec<i32>,
    wi: Vec<i32>,
    max_per_expert: usize,
}

/// Every row picks `n_used` DISTINCT virtual experts out of `n_virt`; picks >=
/// `n_phys` are not materialised (production: another box / a miss) and do not
/// join a group.
fn build_groups(rng: &mut Lcg, b: usize, n_used: usize, n_virt: u32, n_phys: usize, chunk: usize) -> Groups {
    let max_per_expert = b;
    let mut gc = vec![0i32; n_phys];
    let mut em = vec![0i32; n_phys * max_per_expert];
    for row in 0..b {
        let mut picks: Vec<u32> = Vec::new();
        while picks.len() < n_used {
            let e = rng.below(n_virt);
            if !picks.contains(&e) {
                picks.push(e);
            }
        }
        for (slot, &e) in picks.iter().enumerate() {
            let e = e as usize;
            if e < n_phys {
                em[e * max_per_expert + gc[e] as usize] = ((row as i32) << 16) | slot as i32;
                gc[e] += 1;
            }
        }
    }
    let mut wi = Vec::new();
    for (e, &n) in gc.iter().enumerate() {
        let mut s = 0;
        while s < n as usize {
            wi.push(((e as i32) << 16) | s as i32);
            s += chunk;
        }
    }
    Groups { gc, em, wi, max_per_expert }
}

struct Stats {
    rel_rmse: f64,
    max_abs: f32,
    n_diff: usize,
    non_finite: usize,
    written_mismatch: usize,
}

fn stats(want: &[f32], got: &[f32]) -> Stats {
    let (mut se, mut sw) = (0f64, 0f64);
    let (mut max_abs, mut n_diff, mut non_finite, mut written_mismatch) = (0f32, 0usize, 0usize, 0usize);
    for (&a, &b) in want.iter().zip(got) {
        let (sa, sb) = (a.to_bits() == SENTINEL.to_bits(), b.to_bits() == SENTINEL.to_bits());
        if sa != sb {
            written_mismatch += 1;
            continue;
        }
        if sa {
            continue;
        }
        if !b.is_finite() || !a.is_finite() {
            non_finite += 1;
            continue;
        }
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
        }
        let d = (a - b) as f64;
        se += d * d;
        sw += (a as f64) * (a as f64);
        max_abs = max_abs.max((a - b).abs());
    }
    Stats { rel_rmse: if sw > 0.0 { (se / sw).sqrt() } else { 0.0 }, max_abs, n_diff, non_finite, written_mismatch }
}

// ------------------------------------------------------------- 1. A: smallb

#[test]
#[ignore]
fn smallb_down_matches_kwide2() -> eyre::Result<()> {
    install_panic_handler()?;
    let Some(dev) = pick_igpu()? else {
        eprintln!("SKIP: no gfx1151 device");
        return Ok(());
    };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_A001);
    let nu = N_EXPERT_USED;
    let (nr, nb) = (N_EMBD as usize, BLOCKS_Q8K_DOWN_IN as usize);
    let dbpe = nr * nb * SUPER_MXFP4_BYTES;
    const N_PHYS: usize = 12;
    let w = upload(id, &mxfp4_weights(&mut rng, N_PHYS, nr, nb, 118, 8))?;
    let slot_stride = nb * BLOCK_Q8_K_BYTES;
    let mut cases = 0usize;
    // (n_virt, chunk): every row on the SAME 6 experts (6 groups of b); hub-like
    // distinct picks over 384 (most groups 1 member, most picks unmaterialised
    // -> use 12 physical of 24 virtual); few shared experts with chunk 4 (splits).
    for &(n_virt, chunk) in &[(6u32, 32usize), (24, 32), (8, 4)] {
        for b in 1..=8usize {
            let g = build_groups(&mut rng, b, nu, n_virt, N_PHYS, chunk);
            let midq = upload(id, &q8k_blocks(&mut rng, b * nu * nb))?;
            let gc = upload(id, &g.gc)?;
            let em = upload(id, &g.em)?;
            let n_real = g.wi.len();
            // Bound grid + device count: pad with a stale-looking item (a real expert).
            let bound = (b * nu).max(n_real);
            let mut wi_pad = g.wi.clone();
            wi_pad.resize(bound, 0);
            let wi_exact = upload(id, &g.wi)?;
            let wi_bound = upload(id, &wi_pad)?;
            let cnt = upload(id, &[n_real as i32])?;
            let n_out = b * nu * nr;
            let mut outs: Vec<(String, Vec<f32>)> = Vec::new();
            for (name, bounded) in [("exact", false), ("bound+count", true)] {
                let (wi_d, n_wi, c) = if bounded { (&wi_bound, bound as u32, Some(&cnt)) } else { (&wi_exact, n_real as u32, None) };
                let mut p_old = upload(id, &vec![SENTINEL; n_out])?;
                e.mxfp4.launch_by_expert_kwide2_ex(&s, &mut p_old, &w, &midq, &gc, &em, wi_d, n_wi, dbpe as u32, slot_stride as u32,
                    nu as u32, g.max_per_expert as u32, chunk as u32, nr as u32, nb as u32, c)?;
                let mut p_new = upload(id, &vec![SENTINEL; n_out])?;
                e.mxfp4.launch_by_expert_smallb_ex(&s, &mut p_new, &w, &midq, &gc, &em, wi_d, n_wi, dbpe as u32, slot_stride as u32,
                    nu as u32, g.max_per_expert as u32, chunk as u32, nr as u32, nb as u32, c)?;
                let mut p_disp = upload(id, &vec![SENTINEL; n_out])?;
                dispatch::moe_down_mxfp4(&e, &s, &mut p_disp, &w, &midq, &gc, &em, wi_d, n_wi, dbpe as u32, slot_stride as u32,
                    nu as u32, g.max_per_expert as u32, chunk as u32, nr as u32, nb as u32, c, b as u32)?;
                let mut p_16 = upload(id, &vec![SENTINEL; n_out])?;
                dispatch::moe_down_mxfp4(&e, &s, &mut p_16, &w, &midq, &gc, &em, wi_d, n_wi, dbpe as u32, slot_stride as u32,
                    nu as u32, g.max_per_expert as u32, chunk as u32, nr as u32, nb as u32, c, 16)?;
                s.synchronize()?;
                let old = download(&p_old, n_out)?;
                let written = old.iter().filter(|v| v.to_bits() != SENTINEL.to_bits()).count();
                assert!(n_real == 0 || written > 0, "kwide2 wrote nothing");
                for (tag, buf) in [("smallb", &p_new), ("dispatch(rows=b)", &p_disp), ("dispatch(rows=16)", &p_16)] {
                    let got = download(buf, n_out)?;
                    let d = old.iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    assert_eq!(d, 0, "{tag} differs from kwide2 ({name}): b={b} n_virt={n_virt} chunk={chunk}");
                }
                outs.push((name.to_string(), old));
            }
            assert_eq!(outs[0].1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                       outs[1].1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "exact vs bound grid differ (kwide2)");
            eprintln!("smallb b={b} n_virt={n_virt} chunk={chunk}: {n_real} work items, bit-exact (exact + bound grids, dispatch)");
            cases += 1;
        }
    }
    // The compiled n_blocks_in is enforced, not silently skipped.
    let g = build_groups(&mut rng, 2, nu, 6, N_PHYS, 32);
    let (gc, em, wi) = (upload(id, &g.gc)?, upload(id, &g.em)?, upload(id, &g.wi)?);
    let midq = upload(id, &q8k_blocks(&mut rng, 2 * nu * nb))?;
    let mut p = upload(id, &vec![SENTINEL; 2 * nu * nr])?;
    assert!(e.mxfp4.launch_by_expert_smallb_ex(&s, &mut p, &w, &midq, &gc, &em, &wi, g.wi.len() as u32, dbpe as u32,
        slot_stride as u32, nu as u32, 2, 32, nr as u32, 8, None).is_err(), "n_blocks_in != 9 must be refused");
    eprintln!("PASS: {cases} smallb cases bit-exact vs kwide2");
    Ok(())
}

// --------------------------------------------------------------- 2. B: WMMA

#[test]
#[ignore]
fn wmma_arm_matches_kwide_within_reassociation() -> eyre::Result<()> {
    install_panic_handler()?;
    let Some(dev) = pick_igpu()? else {
        eprintln!("SKIP: no gfx1151 device");
        return Ok(());
    };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    if !e.mxfp4pair.has_wmma() || !e.mxfp4.has_wmma() {
        return Err(eyre!("gfx1151 engine without the WMMA module"));
    }
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_B001);
    let nu = N_EXPERT_USED;
    const N_PHYS: usize = 8;
    const CHUNK: usize = 32;
    let (gnr, gnb) = (N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize);
    let (dnr, dnb) = (N_EMBD as usize, BLOCKS_Q8K_DOWN_IN as usize);
    let gbpe = gnr * gnb * SUPER_MXFP4_BYTES;
    let dbpe = dnr * dnb * SUPER_MXFP4_BYTES;
    // Realistic scale range (2^-10..2^-3): few gate/up outputs hit the +-10 clamp.
    let gate = upload(id, &mxfp4_weights(&mut rng, N_PHYS, gnr, gnb, 118, 8))?;
    let up = upload(id, &mxfp4_weights(&mut rng, N_PHYS, gnr, gnb, 118, 8))?;
    let down = upload(id, &mxfp4_weights(&mut rng, N_PHYS, dnr, dnb, 118, 8))?;
    let mut worst_gu = 0f64;
    let mut worst_dn = 0f64;
    let (mut worst_gu16, mut worst_dn16) = (0f64, 0f64);
    // (B, n_virt): n_virt = 48 -> ~B/8 members per expert (chunked at B >= 256);
    // n_virt = 8 -> 0.75 B members per expert (many work items per expert).
    for &(b, n_virt) in &[(37usize, 48u32), (128, 48), (256, 48), (512, 48), (300, 8)] {
        let g = build_groups(&mut rng, b, nu, n_virt, N_PHYS, CHUNK);
        let gc = upload(id, &g.gc)?;
        let em = upload(id, &g.em)?;
        let n_real = g.wi.len();
        let bound = dispatch::moe_wi_upper_bound(b * nu, N_PHYS as u32, CHUNK as u32, 1 << 20) as usize;
        let bound = bound.max(n_real);
        let mut wi_pad = g.wi.clone();
        wi_pad.resize(bound, 0);
        let wi = upload(id, &wi_pad)?;
        let cnt = upload(id, &[n_real as i32])?;
        let xq = upload(id, &q8k_blocks(&mut rng, b * gnb))?;
        let ew_h: Vec<f32> = (0..b * nu).map(|i| 0.05 + 0.01 * (i % 17) as f32).collect();
        let ew = upload(id, &ew_h)?;

        // ---- gate+up: kwide vs WMMA vs dispatch(rows = b)
        let n_mid = b * nu * gnr;
        let mut m_old = upload(id, &vec![SENTINEL; n_mid])?;
        e.mxfp4pair.launch_fused_swiglu_kwide_ex(&s, &mut m_old, &gate, &up, &xq, &ew, &gc, &em, &wi, bound as u32,
            gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP, gnr as u32, gnb as u32, Some(&cnt))?;
        let mut m_new = upload(id, &vec![SENTINEL; n_mid])?;
        e.mxfp4pair.launch_fused_swiglu_wmma_ex(&s, &mut m_new, &gate, &up, &xq, &ew, &gc, &em, &wi, bound as u32,
            gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP, gnr as u32, gnb as u32, Some(&cnt))?;
        let mut m_disp = upload(id, &vec![SENTINEL; n_mid])?;
        let handled = dispatch::moe_gate_up_chunked_rows(&e, GgufType::MXFP4, &s, &mut m_disp, &gate, &up, &xq, &ew, &gc, &em, &wi,
            bound as u32, gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP,
            gnr as u32, gnb as u32, Some(&cnt), b as u32)?;
        assert!(handled);
        s.synchronize()?;
        let (old, new, disp) = (download(&m_old, n_mid)?, download(&m_new, n_mid)?, download(&m_disp, n_mid)?);
        let st = stats(&old, &new);
        eprintln!("gate+up B={b} n_virt={n_virt} ({n_real} work items): wmma vs kwide rel_rmse={:.3e} max_abs={:.3e} differing={} of {} written, non_finite={}, written-set mismatch={}",
            st.rel_rmse, st.max_abs, st.n_diff, old.iter().filter(|v| v.to_bits() != SENTINEL.to_bits()).count(), st.non_finite, st.written_mismatch);
        assert_eq!(st.written_mismatch, 0, "WMMA gate+up wrote a different set");
        assert_eq!(st.non_finite, 0);
        assert!(st.rel_rmse < 1e-5, "WMMA gate+up rel_rmse {:.3e} >= 1e-5 at B={b}", st.rel_rmse);
        let want_disp = if dispatch::moe_wmma_gateup_for(b as u32) { &new } else { &old };
        assert_eq!(want_disp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), disp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "dispatch gate+up at rows={b} took the wrong arm");
        worst_gu = worst_gu.max(st.rel_rmse);

        // ---- f16-WMMA arm (V41_MOE_WMMA_F16): vs kwide, and dispatch takes it with the knob on
        if e.mxfp4pair.has_wmma_f16() {
            let mut m_f16 = upload(id, &vec![SENTINEL; n_mid])?;
            e.mxfp4pair.launch_fused_swiglu_wmma_arm(true, &s, &mut m_f16, &gate, &up, &xq, &ew, &gc, &em, &wi, bound as u32,
                gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP, gnr as u32, gnb as u32, Some(&cnt))?;
            let mut m_d16 = upload(id, &vec![SENTINEL; n_mid])?;
            assert!(v4flash_kernels::knobs::MOE_WMMA_F16.set("1"));
            let handled = dispatch::moe_gate_up_chunked_rows(&e, GgufType::MXFP4, &s, &mut m_d16, &gate, &up, &xq, &ew, &gc, &em, &wi,
                bound as u32, gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP,
                gnr as u32, gnb as u32, Some(&cnt), b as u32);
            assert!(v4flash_kernels::knobs::MOE_WMMA_F16.set("0"));
            assert!(handled?);
            s.synchronize()?;
            let (f16, d16) = (download(&m_f16, n_mid)?, download(&m_d16, n_mid)?);
            let st = stats(&old, &f16);
            eprintln!("gate+up B={b} n_virt={n_virt}: f16 vs kwide rel_rmse={:.3e} max_abs={:.3e} non_finite={} written-set mismatch={}",
                st.rel_rmse, st.max_abs, st.non_finite, st.written_mismatch);
            assert_eq!(st.written_mismatch, 0, "f16 gate+up wrote a different set");
            assert_eq!(st.non_finite, 0);
            assert!(st.rel_rmse < 1e-5, "f16 gate+up rel_rmse {:.3e} >= 1e-5 at B={b}", st.rel_rmse);
            let want = if dispatch::moe_wmma_gateup_for(b as u32) { &f16 } else { &old };
            assert_eq!(want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), d16.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "dispatch gate+up (V41_MOE_WMMA_F16=1) at rows={b} took the wrong arm");
            worst_gu16 = worst_gu16.max(st.rel_rmse);
        }

        // ---- down: kwide2 vs WMMA vs dispatch(rows = b), on random midq
        let midq = upload(id, &q8k_blocks(&mut rng, b * nu * dnb))?;
        let slot_stride = dnb * BLOCK_Q8_K_BYTES;
        let n_part = b * nu * dnr;
        let mut p_old = upload(id, &vec![SENTINEL; n_part])?;
        e.mxfp4.launch_by_expert_kwide2_ex(&s, &mut p_old, &down, &midq, &gc, &em, &wi, bound as u32, dbpe as u32, slot_stride as u32,
            nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, Some(&cnt))?;
        let mut p_new = upload(id, &vec![SENTINEL; n_part])?;
        e.mxfp4.launch_by_expert_wmma_ex(&s, &mut p_new, &down, &midq, &gc, &em, &wi, bound as u32, dbpe as u32, slot_stride as u32,
            nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, Some(&cnt))?;
        let mut p_disp = upload(id, &vec![SENTINEL; n_part])?;
        dispatch::moe_down_mxfp4(&e, &s, &mut p_disp, &down, &midq, &gc, &em, &wi, bound as u32, dbpe as u32, slot_stride as u32,
            nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, Some(&cnt), b as u32)?;
        s.synchronize()?;
        let (old, new, disp) = (download(&p_old, n_part)?, download(&p_new, n_part)?, download(&p_disp, n_part)?);
        let st = stats(&old, &new);
        eprintln!("down    B={b} n_virt={n_virt}: wmma vs kwide2 rel_rmse={:.3e} max_abs={:.3e} differing={} of {} written, non_finite={}, written-set mismatch={}",
            st.rel_rmse, st.max_abs, st.n_diff, old.iter().filter(|v| v.to_bits() != SENTINEL.to_bits()).count(), st.non_finite, st.written_mismatch);
        assert_eq!(st.written_mismatch, 0, "WMMA down wrote a different set");
        assert_eq!(st.non_finite, 0);
        assert!(st.rel_rmse < 1e-5, "WMMA down rel_rmse {:.3e} >= 1e-5 at B={b}", st.rel_rmse);
        let want_disp = if dispatch::moe_wmma_down_for(b as u32) { &new } else { &old };
        assert_eq!(want_disp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), disp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "dispatch down at rows={b} took the wrong arm");
        worst_dn = worst_dn.max(st.rel_rmse);

        if e.mxfp4.has_wmma_f16() {
            let mut p_f16 = upload(id, &vec![SENTINEL; n_part])?;
            e.mxfp4.launch_by_expert_wmma_arm(true, &s, &mut p_f16, &down, &midq, &gc, &em, &wi, bound as u32, dbpe as u32,
                slot_stride as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, Some(&cnt))?;
            let mut p_d16 = upload(id, &vec![SENTINEL; n_part])?;
            assert!(v4flash_kernels::knobs::MOE_WMMA_F16.set("1"));
            let r = dispatch::moe_down_mxfp4(&e, &s, &mut p_d16, &down, &midq, &gc, &em, &wi, bound as u32, dbpe as u32,
                slot_stride as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, Some(&cnt), b as u32);
            assert!(v4flash_kernels::knobs::MOE_WMMA_F16.set("0"));
            r?;
            s.synchronize()?;
            let (f16, d16) = (download(&p_f16, n_part)?, download(&p_d16, n_part)?);
            let st = stats(&old, &f16);
            eprintln!("down    B={b} n_virt={n_virt}: f16 vs kwide2 rel_rmse={:.3e} max_abs={:.3e} non_finite={} written-set mismatch={}",
                st.rel_rmse, st.max_abs, st.non_finite, st.written_mismatch);
            assert_eq!(st.written_mismatch, 0, "f16 down wrote a different set");
            assert_eq!(st.non_finite, 0);
            assert!(st.rel_rmse < 1e-5, "f16 down rel_rmse {:.3e} >= 1e-5 at B={b}", st.rel_rmse);
            let want = if dispatch::moe_wmma_down_for(b as u32) { &f16 } else { &old };
            assert_eq!(want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), d16.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "dispatch down (V41_MOE_WMMA_F16=1) at rows={b} took the wrong arm");
            worst_dn16 = worst_dn16.max(st.rel_rmse);
        }
    }
    eprintln!("PASS: WMMA within f32 re-association: worst rel_rmse gate+up {worst_gu:.3e}, down {worst_dn:.3e}; \
f16 arm gate+up {worst_gu16:.3e}, down {worst_dn16:.3e}");
    Ok(())
}

/// The f16 arm folds the E8M0 scale into its perm LUT, exact for e8 in [114, 137] only (`wm_lut` clamps outside).
/// Every scale in that range, uniformly, vs kwide / kwide2 (which scale in f32 and have no range limit).
#[test]
#[ignore]
fn wmma_f16_arm_exact_over_its_whole_scale_range() -> eyre::Result<()> {
    install_panic_handler()?;
    let Some(dev) = pick_igpu()? else {
        eprintln!("SKIP: no gfx1151 device");
        return Ok(());
    };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    if !e.mxfp4pair.has_wmma_f16() || !e.mxfp4.has_wmma_f16() {
        return Err(eyre!("gfx1151 engine without the f16 WMMA module"));
    }
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_F16F);
    let nu = N_EXPERT_USED;
    const N_PHYS: usize = 8;
    const CHUNK: usize = 32;
    let (gnr, gnb) = (N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize);
    let (dnr, dnb) = (N_EMBD as usize, BLOCKS_Q8K_DOWN_IN as usize);
    let gbpe = gnr * gnb * SUPER_MXFP4_BYTES;
    let dbpe = dnr * dnb * SUPER_MXFP4_BYTES;
    // (lo, span, label, asserted): the model's own range 115..128 (e8_scan.py over all 47,232 expert scale tensors,
    // 2026-10-10) uniformly mixed inside every super-block is the worst REAL case; the 24-octave mix is printed only
    // (the f16 arm accumulates a super-block's 8 blocks in one f32 chain, so it loses more there than the int8 arm).
    for &(lo, span, label, asserted) in &[
        (114u8, 1u32, "e8 = 114 (all)", true),
        (137, 1, "e8 = 137 (all)", true),
        (115, 14, "e8 in [115,128] mixed (the model's range)", true),
        (114, 24, "e8 in [114,137] mixed (report only)", false),
    ] {
    let gate = upload(id, &mxfp4_weights(&mut rng, N_PHYS, gnr, gnb, lo, span))?;
    let up = upload(id, &mxfp4_weights(&mut rng, N_PHYS, gnr, gnb, lo, span))?;
    let down = upload(id, &mxfp4_weights(&mut rng, N_PHYS, dnr, dnb, lo, span))?;
    let b = 512usize;
    let g = build_groups(&mut rng, b, nu, 48, N_PHYS, CHUNK);
    let (gc, em) = (upload(id, &g.gc)?, upload(id, &g.em)?);
    let n_wi = g.wi.len();
    let wi = upload(id, &g.wi)?;
    let xq = upload(id, &q8k_blocks(&mut rng, b * gnb))?;
    let ew = upload(id, &(0..b * nu).map(|i| 0.05 + 0.01 * (i % 17) as f32).collect::<Vec<f32>>())?;
    let n_mid = b * nu * gnr;
    let (mut m_ref, mut m_f16, mut m_i8) = (upload(id, &vec![SENTINEL; n_mid])?, upload(id, &vec![SENTINEL; n_mid])?, upload(id, &vec![SENTINEL; n_mid])?);
    e.mxfp4pair.launch_fused_swiglu_kwide_ex(&s, &mut m_ref, &gate, &up, &xq, &ew, &gc, &em, &wi, n_wi as u32,
        gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP, gnr as u32, gnb as u32, None)?;
    e.mxfp4pair.launch_fused_swiglu_wmma_arm(true, &s, &mut m_f16, &gate, &up, &xq, &ew, &gc, &em, &wi, n_wi as u32,
        gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP, gnr as u32, gnb as u32, None)?;
    e.mxfp4pair.launch_fused_swiglu_wmma_arm(false, &s, &mut m_i8, &gate, &up, &xq, &ew, &gc, &em, &wi, n_wi as u32,
        gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP, gnr as u32, gnb as u32, None)?;
    let midq = upload(id, &q8k_blocks(&mut rng, b * nu * dnb))?;
    let slot_stride = dnb * BLOCK_Q8_K_BYTES;
    let n_part = b * nu * dnr;
    let (mut p_ref, mut p_f16, mut p_i8) = (upload(id, &vec![SENTINEL; n_part])?, upload(id, &vec![SENTINEL; n_part])?, upload(id, &vec![SENTINEL; n_part])?);
    e.mxfp4.launch_by_expert_kwide2_ex(&s, &mut p_ref, &down, &midq, &gc, &em, &wi, n_wi as u32, dbpe as u32, slot_stride as u32,
        nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, None)?;
    e.mxfp4.launch_by_expert_wmma_arm(true, &s, &mut p_f16, &down, &midq, &gc, &em, &wi, n_wi as u32, dbpe as u32, slot_stride as u32,
        nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, None)?;
    e.mxfp4.launch_by_expert_wmma_arm(false, &s, &mut p_i8, &down, &midq, &gc, &em, &wi, n_wi as u32, dbpe as u32, slot_stride as u32,
        nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, None)?;
    s.synchronize()?;
    for (name, r, f, i) in [("gate+up", download(&m_ref, n_mid)?, download(&m_f16, n_mid)?, download(&m_i8, n_mid)?),
                            ("down", download(&p_ref, n_part)?, download(&p_f16, n_part)?, download(&p_i8, n_part)?)] {
        let (st, si) = (stats(&r, &f), stats(&r, &i));
        eprintln!("{name} {label}: f16 vs kwide rel_rmse={:.3e} max_abs={:.3e} | int8 arm vs kwide rel_rmse={:.3e} max_abs={:.3e} | non_finite={} mismatch={}",
            st.rel_rmse, st.max_abs, si.rel_rmse, si.max_abs, st.non_finite, st.written_mismatch);
        assert_eq!(st.written_mismatch, 0);
        assert_eq!(st.non_finite, 0);
        // The LUT is exact for every e8 in range, so the f16 arm must never be worse than the int8 arm that ships. Absolute
        // rel_rmse is NOT the bar here: at e8 = 137 the gate outputs run far past the SwiGLU clamp and BOTH arms sit near
        // 1e-5 (measured 10-10: f16 1.2e-5, int8 1.9e-5); the 24-octave mixed case also loses small blocks to f32
        // accumulation inside a super-block. Real weights span e8 115..128 (e8_scan.py, all 47K scale tensors).
        assert!(!asserted || st.rel_rmse <= 1.25 * si.rel_rmse.max(2e-7),
            "{name} {label}: f16 arm rel_rmse {:.3e} worse than the int8 arm's {:.3e}", st.rel_rmse, si.rel_rmse);
    }
    }
    eprintln!("PASS: f16 arm no worse than the int8 arm at uniform e8 114 / 137 and over the model's range [115, 128]");
    Ok(())
}

/// REAL weights: 8 experts from each of 10 layers of the V4.1-Flash checkpoint (engine form, `read_expert_into`), Q8_K
/// activations, B = 512 / 3072 rows: the f16 arm vs the int8 arm, both vs kwide / kwide2. Gate for `V41_MOE_WMMA_F16` at
/// kernel level: the int8 arm passed the end-to-end golden gate (09-26 sweep), so "f16 no worse than int8 on real weights"
/// is the claim. Reads ~1.5 GB from the checkpoint; runs beside a live hub (iGPU only, ~0.5 GB device).
///   V41_MODEL=<hf snapshot dir> (default ~/.cache/deepstrix/models/dsv4.1f)
#[test]
#[ignore]
fn wmma_f16_arm_on_real_weights() -> eyre::Result<()> {
    use v4flash_core::{V41HfWeights, WeightSrc};
    install_panic_handler()?;
    let Some(dev) = pick_igpu()? else {
        eprintln!("SKIP: no gfx1151 device");
        return Ok(());
    };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    if !e.mxfp4pair.has_wmma_f16() || !e.mxfp4.has_wmma_f16() {
        return Err(eyre!("gfx1151 engine without the f16 WMMA module"));
    }
    let dir = std::env::var("V41_MODEL")
        .unwrap_or_else(|_| format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap()));
    let hf = V41HfWeights::open(&dir, None)?;
    let src = WeightSrc::from(&hf);
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_4EA1);
    let nu = N_EXPERT_USED;
    const N_PHYS: usize = 8;
    const CHUNK: usize = 32;
    let (gnr, gnb) = (N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize);
    let (dnr, dnb) = (N_EMBD as usize, BLOCKS_Q8K_DOWN_IN as usize);
    let gbpe = gnr * gnb * SUPER_MXFP4_BYTES;
    let dbpe = dnr * dnb * SUPER_MXFP4_BYTES;
    let load = |layer: u32, which: &str, bpe: usize, experts: &[usize]| -> eyre::Result<(Vec<u8>, u8, u8)> {
        let name = format!("blk.{layer}.ffn_{which}_exps.weight");
        let t = src.tensor(&name).ok_or_else(|| eyre!("{name}"))?;
        let mut host = vec![0u8; experts.len() * bpe];
        for (k, &x) in experts.iter().enumerate() {
            src.read_expert_into(t, x, &mut host[k * bpe..(k + 1) * bpe])?;
        }
        let (mut lo, mut hi) = (255u8, 0u8);
        for sb in host.chunks_exact(SUPER_MXFP4_BYTES) {
            for &e8 in &sb[128..136] {
                lo = lo.min(e8);
                hi = hi.max(e8);
            }
        }
        Ok((host, lo, hi))
    };
    let (mut worst16, mut worst8) = ([0f64; 2], [0f64; 2]);
    for &layer in &[0u32, 2, 5, 9, 13, 17, 19, 24, 31, 39] {
        let experts: Vec<usize> = (0..N_PHYS).map(|k| (k * 48 + layer as usize * 7) % 384).collect();
        let (gh, glo, ghi) = load(layer, "gate", gbpe, &experts)?;
        let (uh, ulo, uhi) = load(layer, "up", gbpe, &experts)?;
        let (dh, dlo, dhi) = load(layer, "down", dbpe, &experts)?;
        let (gate, up, down) = (upload(id, &gh)?, upload(id, &uh)?, upload(id, &dh)?);
        for &(b, n_virt) in &[(512usize, 48u32), (3072, 48)] {
            let g = build_groups(&mut rng, b, nu, n_virt, N_PHYS, CHUNK);
            let (gc, em) = (upload(id, &g.gc)?, upload(id, &g.em)?);
            let n_wi = g.wi.len();
            let wi = upload(id, &g.wi)?;
            let xq = upload(id, &q8k_blocks(&mut rng, b * gnb))?;
            let ew = upload(id, &(0..b * nu).map(|i| 0.05 + 0.01 * (i % 17) as f32).collect::<Vec<f32>>())?;
            let n_mid = b * nu * gnr;
            let mut m = [upload(id, &vec![SENTINEL; n_mid])?, upload(id, &vec![SENTINEL; n_mid])?, upload(id, &vec![SENTINEL; n_mid])?];
            e.mxfp4pair.launch_fused_swiglu_kwide_ex(&s, &mut m[0], &gate, &up, &xq, &ew, &gc, &em, &wi, n_wi as u32,
                gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP, gnr as u32, gnb as u32, None)?;
            for (k, f16) in [(1usize, false), (2, true)] {
                e.mxfp4pair.launch_fused_swiglu_wmma_arm(f16, &s, &mut m[k], &gate, &up, &xq, &ew, &gc, &em, &wi, n_wi as u32,
                    gbpe as u32, gbpe as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, SWIGLU_CLAMP_EXP, gnr as u32, gnb as u32, None)?;
            }
            let midq = upload(id, &q8k_blocks(&mut rng, b * nu * dnb))?;
            let slot_stride = dnb * BLOCK_Q8_K_BYTES;
            let n_part = b * nu * dnr;
            let mut p = [upload(id, &vec![SENTINEL; n_part])?, upload(id, &vec![SENTINEL; n_part])?, upload(id, &vec![SENTINEL; n_part])?];
            e.mxfp4.launch_by_expert_kwide2_ex(&s, &mut p[0], &down, &midq, &gc, &em, &wi, n_wi as u32, dbpe as u32, slot_stride as u32,
                nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, None)?;
            for (k, f16) in [(1usize, false), (2, true)] {
                e.mxfp4.launch_by_expert_wmma_arm(f16, &s, &mut p[k], &down, &midq, &gc, &em, &wi, n_wi as u32, dbpe as u32,
                    slot_stride as u32, nu as u32, g.max_per_expert as u32, CHUNK as u32, dnr as u32, dnb as u32, None)?;
            }
            s.synchronize()?;
            let mr: Vec<Vec<f32>> = m.iter().map(|x| download(x, n_mid)).collect::<eyre::Result<_>>()?;
            let pr: Vec<Vec<f32>> = p.iter().map(|x| download(x, n_part)).collect::<eyre::Result<_>>()?;
            for (r, name, out) in [(0usize, "gate+up", &mr), (1, "down", &pr)] {
                let (s8, s16) = (stats(&out[0], &out[1]), stats(&out[0], &out[2]));
                eprintln!("L{layer:2} B={b:4} {name:7} e8 g[{glo},{ghi}] u[{ulo},{uhi}] d[{dlo},{dhi}]: f16 rel_rmse {:.3e} max_abs {:.3e} | int8 {:.3e} max_abs {:.3e}",
                    s16.rel_rmse, s16.max_abs, s8.rel_rmse, s8.max_abs);
                assert_eq!(s16.written_mismatch, 0);
                assert_eq!(s16.non_finite, 0);
                assert!(s16.rel_rmse <= 1.25 * s8.rel_rmse.max(2e-7),
                    "L{layer} B={b} {name}: f16 rel_rmse {:.3e} worse than int8 {:.3e}", s16.rel_rmse, s8.rel_rmse);
                worst16[r] = worst16[r].max(s16.rel_rmse);
                worst8[r] = worst8[r].max(s8.rel_rmse);
            }
        }
    }
    eprintln!("PASS real weights (10 layers x 8 experts, B 512/3072): worst rel_rmse vs kwide -- f16 gate+up {:.3e} down {:.3e}; \
int8 gate+up {:.3e} down {:.3e}", worst16[0], worst16[1], worst8[0], worst8[1]);
    Ok(())
}

