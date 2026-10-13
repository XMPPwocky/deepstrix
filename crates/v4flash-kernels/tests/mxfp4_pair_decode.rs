//! The decode-regime MXFP4 gate+up kernel (`mxfp4_pair_matvec_fused_swiglu_dec_*`,
//! kernels/mxfp4_pair_decode.hip, `V41_MOE_PAIR_DEC`) vs the kernel production
//! runs at rows < 256 (`mxfp4_pair_matvec_fused_swiglu_kwide`):
//!
//!   1. `dec_matches_kwide_bit_exact`: BIT-IDENTICAL `mid` (whole buffer,
//!      sentinel-filled, so the written set must match too) at b = 1..8 over
//!      selection regimes (hub-like distinct picks, every row on the same experts,
//!      split work items with a small chunk), two scale ranges, several persistent
//!      grid / work-group geometries, the test-only 2-members-per-pass entry (the
//!      multi-pass path), the exact count and the production bound + device count,
//!      and `dispatch::moe_gate_up_chunked_rows` with the knob on.
//!   2. `bench_dec_decode_regime`: the production decode regime -- b in {1,2,4,8},
//!      D in {4,8,16} distinct experts, the device-count bound grid (6b work-item
//!      slots), COLD weights (each call's experts are fresh: the slots rotate over a
//!      ~0.8 GB pool, every call reads 50-200 MB against a 32 MB MALL). In ONE
//!      process, interleaved per round: kwide (twice: the A/A spread), the new
//!      kernel, its load-only sub-stage, and the roofline probe (the same bytes
//!      read by an ideal streaming kernel); plus the contiguous streaming ceiling.
//!      Reports medians and p90 per variant.
//!
//! gfx1151 only. Production shares this iGPU: run the bench beside it and read
//! the A/A spread before any difference.
//!
//!   cargo test --release --features v41 -p v4flash-kernels --test mxfp4_pair_decode \
//!     -- --ignored --test-threads=1 --nocapture
//!
//! Bench env: DEC_REPS (default 60 rounds), DEC_POOL (experts in the pool, default
//! 64 = 0.8 GB), DEC_WAVES (comma list of waves per work-group to time, default 16).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, launch_kernel, Device, DeviceBuffer, Event, LaunchConfig, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EXPERT_USED, N_FF_EXP, SWIGLU_CLAMP_EXP};
use v4flash_kernels::device_ceilings::DeviceCeilings;
use v4flash_kernels::het::dispatch;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mxfp4_pair::MXFP4_DEC_WAVES_DEFAULT;
use v4flash_kernels::mxfp4_tables::SUPER_MXFP4_BYTES;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;
use v4flash_core::gguf::GgufType;

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
fn mxfp4_weights(rng: &mut Lcg, n_bytes: usize, sc_lo: u8, span: u32) -> Vec<u8> {
    let mut w = vec![0u8; n_bytes];
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

/// Q8_K blocks with a positive scale, random int8s and consistent bsums.
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

fn download(d: &DeviceBuffer<f32>) -> eyre::Result<Vec<f32>> {
    let mut v = vec![0f32; d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

const SENTINEL: f32 = f32::from_bits(0x7FC0_BEEF);

/// The production builder's output, built on the host: group_count[e],
/// expert_members[e * max + m] = (row << 16) | slot, work items (e << 16) |
/// member_start per `chunk` members, in expert-id order.
struct Groups {
    gc: Vec<i32>,
    em: Vec<i32>,
    wi: Vec<i32>,
    max_per_expert: usize,
    distinct: usize,
    members: usize,
}

/// `picks[row]` = (expert, slot) pairs; experts index the pool (< n_pool).
fn groups_from_picks(picks: &[Vec<(usize, usize)>], n_pool: usize, chunk: usize) -> Groups {
    let b = picks.len();
    let max_per_expert = b.max(1);
    let mut gc = vec![0i32; n_pool];
    let mut em = vec![0i32; n_pool * max_per_expert];
    let mut members = 0;
    for (row, p) in picks.iter().enumerate() {
        for &(e, slot) in p {
            em[e * max_per_expert + gc[e] as usize] = ((row as i32) << 16) | slot as i32;
            gc[e] += 1;
            members += 1;
        }
    }
    let mut wi = Vec::new();
    let mut distinct = 0;
    for (e, &n) in gc.iter().enumerate() {
        if n > 0 {
            distinct += 1;
        }
        let mut s = 0;
        while s < n as usize {
            wi.push(((e as i32) << 16) | s as i32);
            s += chunk;
        }
    }
    Groups { gc, em, wi, max_per_expert, distinct, members }
}

/// Every row picks `n_used` DISTINCT virtual experts out of `n_virt`; picks >=
/// `n_phys` are not ours (another box / the dGPU) and join no group.
fn random_picks(rng: &mut Lcg, b: usize, n_used: usize, n_virt: u32, n_phys: usize) -> Vec<Vec<(usize, usize)>> {
    (0..b)
        .map(|_| {
            let mut p: Vec<u32> = Vec::new();
            while p.len() < n_used {
                let e = rng.below(n_virt);
                if !p.contains(&e) {
                    p.push(e);
                }
            }
            p.iter().enumerate().filter(|(_, &e)| (e as usize) < n_phys).map(|(s, &e)| (e as usize, s)).collect()
        })
        .collect()
}

/// The bench regime: `b` rows, exactly `d` distinct experts (the given pool
/// slots), each row picking `p = clamp(ceil(d / b), 3, 6)` (<= d) of them in
/// distinct slots -- box 1's ~half of a row's 6 picks, more when d needs it.
fn regime_picks(rng: &mut Lcg, b: usize, experts: &[usize]) -> Vec<Vec<(usize, usize)>> {
    let d = experts.len();
    let p = d.div_ceil(b).clamp(3, 6).min(d);
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); b];
    for (j, &e) in experts.iter().enumerate() {
        rows[j % b].push(e); // every expert at least once
    }
    for r in rows.iter_mut() {
        while r.len() < p {
            let e = experts[rng.below(d as u32) as usize];
            if !r.contains(&e) {
                r.push(e);
            }
        }
    }
    rows.into_iter().map(|r| r.into_iter().enumerate().map(|(s, e)| (e, s)).collect()).collect()
}

// ------------------------------------------------------------ 1. exactness

#[test]
#[ignore]
fn dec_matches_kwide_bit_exact() -> eyre::Result<()> {
    install_panic_handler()?;
    let Some(dev) = pick_igpu()? else {
        eprintln!("SKIP: no gfx1151 device");
        return Ok(());
    };
    // Before the first dispatch call (the knob is read once).
    // SAFETY: single-threaded test start, nothing else reads the environment yet.
    unsafe { std::env::set_var("V41_MOE_PAIR_DEC", "1") };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    if !e.mxfp4pair.has_dec() {
        return Err(eyre!("gfx1151 engine without the decode module"));
    }
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_DEC1);
    let nu = N_EXPERT_USED;
    let (nr, nb) = (N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize);
    let bpe = nr * nb * SUPER_MXFP4_BYTES;
    const N_PHYS: usize = 12;
    // Realistic scales (2^-10..2^-3: few outputs hit the clamp) and a wide range
    // (2^-127..2^-96 + large: the e8m0 subnormal path, clamped outputs).
    let gate = upload(id, &mxfp4_weights(&mut rng, N_PHYS * bpe, 118, 8))?;
    let up = upload(id, &mxfp4_weights(&mut rng, N_PHYS * bpe, 118, 8))?;
    let gate_w = upload(id, &mxfp4_weights(&mut rng, N_PHYS * bpe, 0, 140))?;
    let mut cases = 0usize;
    let mut checked = 0usize;
    // The dynamic order's ticket: zeroed once; every launch must leave it zeroed.
    let ticket = upload(id, &[0u32, 0u32])?;
    // (b list, n_virt, chunk): hub-like distinct picks over 24 virtual (12 ours);
    // every row on the same 6 experts (groups of b); few shared experts with a
    // small chunk (split work items, member_start > 0).
    let regimes: &[(&[usize], u32, usize)] = &[
        (&[1, 2, 3, 4, 5, 6, 7, 8], 24, 32),
        (&[1, 2, 4, 7, 8], 6, 32),
        (&[3, 6, 8], 8, 4),
        (&[8], 8, 3),
    ];
    for (ri, &(bs, n_virt, chunk)) in regimes.iter().enumerate() {
        for &b in bs {
            let picks = random_picks(&mut rng, b, nu, n_virt, N_PHYS);
            let g = groups_from_picks(&picks, N_PHYS, chunk);
            let xq = upload(id, &q8k_blocks(&mut rng, b * nb))?;
            let ew_h: Vec<f32> = (0..b * nu).map(|_| 0.5 + (rng.next() % 1000) as f32 * 1e-3).collect();
            let ew = upload(id, &ew_h)?;
            let gc = upload(id, &g.gc)?;
            let em = upload(id, &g.em)?;
            let n_real = g.wi.len();
            let bound = (b * nu).max(n_real);
            let mut wi_pad = g.wi.clone();
            // Stale entries past the device count look like real items.
            wi_pad.resize(bound, g.wi.first().copied().unwrap_or(0));
            let wi_exact = upload(id, &g.wi)?;
            let wi_bound = upload(id, &wi_pad)?;
            let cnt = upload(id, &[n_real as i32])?;
            let n_out = b * nu * nr;
            for (wname, gw) in [("scales 2^-10..-3", &gate), ("scales wide", &gate_w)] {
                for (gname, bounded) in [("exact", false), ("bound+count", true)] {
                    let (wi_d, n_wi, c) =
                        if bounded { (&wi_bound, bound as u32, Some(&cnt)) } else { (&wi_exact, n_real as u32, None) };
                    let mut want = upload(id, &vec![SENTINEL; n_out])?;
                    e.mxfp4pair.launch_fused_swiglu_kwide_ex(
                        &s, &mut want, gw, &up, &xq, &ew, &gc, &em, wi_d, n_wi, bpe as u32, bpe as u32,
                        nu as u32, g.max_per_expert as u32, chunk as u32, SWIGLU_CLAMP_EXP, nr as u32, nb as u32, c,
                    )?;
                    s.synchronize()?;
                    let want = download(&want)?;
                    let written = want.iter().filter(|v| v.to_bits() != SENTINEL.to_bits()).count();
                    assert_eq!(written, g.members * nr, "kwide wrote {written}, expected {} (b={b})", g.members * nr);
                    let mut runs: Vec<(String, DeviceBuffer<f32>)> = Vec::new();
                    let geoms: &[(&str, u32, u32, bool)] = &[
                        ("mxfp4_pair_matvec_fused_swiglu_dec", 40, MXFP4_DEC_WAVES_DEFAULT, true),
                        ("mxfp4_pair_matvec_fused_swiglu_dec", 40, MXFP4_DEC_WAVES_DEFAULT, false),
                        ("mxfp4_pair_matvec_fused_swiglu_dec", 1, 16, true),
                        ("mxfp4_pair_matvec_fused_swiglu_dec", 7, 3, false),
                        ("mxfp4_pair_matvec_fused_swiglu_dec", 7, 3, true),
                        ("mxfp4_pair_matvec_fused_swiglu_dec", 80, 8, true),
                        ("mxfp4_pair_matvec_fused_swiglu_dec", 300, 1, false),
                        ("mxfp4_pair_matvec_fused_swiglu_dec", 2000, 16, true),
                        ("mxfp4_pair_dec_2row", 40, 16, true),
                        ("mxfp4_pair_dec_2row", 7, 3, false),
                        ("mxfp4_pair_dec_2row_m2_test", 40, 16, true),
                        ("mxfp4_pair_dec_m2_test", 40, 16, true),
                        ("mxfp4_pair_dec_m2_test", 13, 5, false),
                    ];
                    for &(sym, wgs, waves, dynamic) in geoms {
                        let mut got = upload(id, &vec![SENTINEL; n_out])?;
                        e.mxfp4pair.launch_fused_swiglu_dec_raw(
                            sym, wgs, waves, if dynamic { Some(&ticket) } else { None }, None, &s, &mut got, gw, &up,
                            &xq, &ew, &gc, &em, wi_d, n_wi, bpe as u32, bpe as u32, nu as u32, g.max_per_expert as u32,
                            chunk as u32, SWIGLU_CLAMP_EXP, nr as u32, nb as u32, c, b as u32,
                        )?;
                        runs.push((format!("{sym} wgs={wgs} waves={waves} dynamic={dynamic}"), got));
                    }
                    let mut disp = upload(id, &vec![SENTINEL; n_out])?;
                    let handled = dispatch::moe_gate_up_chunked_rows(
                        &e, GgufType::MXFP4, &s, &mut disp, gw, &up, &xq, &ew, &gc, &em, wi_d, n_wi,
                        bpe as u32, bpe as u32, nu as u32, g.max_per_expert as u32, chunk as u32, SWIGLU_CLAMP_EXP,
                        nr as u32, nb as u32, c, b as u32,
                    )?;
                    assert!(handled);
                    runs.push(("dispatch(V41_MOE_PAIR_DEC=1)".into(), disp));
                    s.synchronize()?;
                    let mut tk = [7u32; 2];
                    ticket.copy_to_host(&mut tk)?;
                    assert_eq!(tk, [0, 0], "the ticket did not reset after the launches");
                    // The engine's own ticket for this work-item buffer (the dispatch launch).
                    if let Some(own) = e.mxfp4pair.dec_ticket(wi_d) {
                        own.copy_to_host(&mut tk)?;
                        assert_eq!(tk, [0, 0], "the engine's ticket did not reset after the dispatch launch");
                    }
                    for (tag, buf) in &runs {
                        let got = download(buf)?;
                        let d = want.iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                        if d != 0 {
                            let (i, (a, bb)) = want.iter().zip(&got).enumerate().find(|(_, (a, b))| a.to_bits() != b.to_bits()).unwrap();
                            return Err(eyre!(
                                "{tag} differs from kwide in {d} of {n_out} ({wname}, {gname}, regime {ri}, b={b}, chunk={chunk}): first at {i}: {a:e} vs {bb:e}"
                            ));
                        }
                        checked += 1;
                    }
                }
            }
            eprintln!(
                "b={b:2} n_virt={n_virt:2} chunk={chunk:2}: {} distinct, {} members, {n_real} work items: bit-exact ({} launches)",
                g.distinct, g.members, 2 * 2 * 14
            );
            cases += 1;
        }
    }
    // Refusals: the compiled width, odd rows, more token rows than staged.
    let g = groups_from_picks(&random_picks(&mut rng, 2, nu, 24, N_PHYS), N_PHYS, 32);
    let (gc, em, wi) = (upload(id, &g.gc)?, upload(id, &g.em)?, upload(id, &g.wi)?);
    let xq = upload(id, &q8k_blocks(&mut rng, 2 * nb))?;
    let ew = upload(id, &vec![1.0f32; 2 * nu])?;
    let mut m = upload(id, &vec![SENTINEL; 2 * nu * nr])?;
    assert!(e.mxfp4pair.launch_fused_swiglu_dec_ex(&s, &mut m, &gate, &up, &xq, &ew, &gc, &em, &wi, g.wi.len() as u32,
        bpe as u32, bpe as u32, nu as u32, 2, 32, SWIGLU_CLAMP_EXP, nr as u32, 16, None, 2).is_err(), "n_blocks != 20 must be refused");
    assert!(e.mxfp4pair.launch_fused_swiglu_dec_ex(&s, &mut m, &gate, &up, &xq, &ew, &gc, &em, &wi, g.wi.len() as u32,
        bpe as u32, bpe as u32, nu as u32, 2, 32, SWIGLU_CLAMP_EXP, nr as u32 - 1, nb as u32, None, 2).is_err(), "odd n_rows must be refused");
    assert!(e.mxfp4pair.launch_fused_swiglu_dec_ex(&s, &mut m, &gate, &up, &xq, &ew, &gc, &em, &wi, g.wi.len() as u32,
        bpe as u32, bpe as u32, nu as u32, 2, 32, SWIGLU_CLAMP_EXP, nr as u32, nb as u32, None, 9).is_err(), "rows > 8 must be refused");
    eprintln!("PASS: {cases} selections, {checked} launches bit-identical to kwide");
    Ok(())
}

// ------------------------------------------------------------------ 2. bench

fn pct(v: &mut [f32], q: f64) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((v.len() - 1) as f64 * q).round() as usize;
    v[i]
}

/// Bench variants, interleaved in a fresh random order every round.
const VARIANTS: &[&str] = &[
    "kwide",           // production today (rows < 256)
    "kwide(A/A)",      // the same again: the noise floor
    "dec",             // the new kernel as dispatch launches it (dynamic unit order, the default)
    "dec-static",      // static unit order (V41_MOE_PAIR_DEC_DYN=0)
    "dec-2row",        // reference mapping (bit-identical, two rows per wave)
    "dec-loadonly",    // sub-stage: the kernel's loads, no arithmetic
    "dec-compute",     // sub-stage: the kernel's arithmetic, no loads after the prologue
    "probe",           // roofline: the same bytes, ideal streaming read
];

#[test]
#[ignore]
fn bench_dec_decode_regime() -> eyre::Result<()> {
    install_panic_handler()?;
    let Some(dev) = pick_igpu()? else {
        eprintln!("SKIP: no gfx1151 device");
        return Ok(());
    };
    dev.set_current()?;
    let props = dev.properties()?;
    let arch = props.gcn_arch_name.clone();
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let ceil = DeviceCeilings::for_arch(&arch)?;
    let dec = e.mxfp4pair.dec_module().ok_or_else(|| eyre!("no decode module"))?;
    let s = Stream::new(id)?;
    let reps: usize = std::env::var("DEC_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    let n_pool: usize = std::env::var("DEC_POOL").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let waves: u32 = std::env::var("DEC_WAVES").ok().and_then(|v| v.parse().ok()).unwrap_or(MXFP4_DEC_WAVES_DEFAULT);
    // The roofline probe's work-groups per matrix (DEC_PROBE_NX, default 64).
    let probe_nx: u32 = std::env::var("DEC_PROBE_NX").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let nu = N_EXPERT_USED;
    let (nr, nb) = (N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize);
    let bpe = nr * nb * SUPER_MXFP4_BYTES;
    let wgs = e.mxfp4pair.dec_wgs(&s)?;
    let ticket = upload(id, &[0u32, 0u32])?;
    let mut rng = Lcg(0xBE7C_DEC0);

    // The pool: gate and up, n_pool experts each, filled in ~65 MB pieces.
    let mut gate = DeviceBuffer::<u8>::new(id, n_pool * bpe)?;
    let mut up = DeviceBuffer::<u8>::new(id, n_pool * bpe)?;
    {
        let piece = mxfp4_weights(&mut rng, 470 * SUPER_MXFP4_BYTES * 1024, 118, 8);
        for buf in [&mut gate, &mut up] {
            let mut off = 0usize;
            while off < n_pool * bpe {
                let n = piece.len().min(n_pool * bpe - off);
                buf.slice_view_mut(off, n).copy_from_host(&piece[..n])?;
                off += n;
            }
        }
    }
    let mut sink = DeviceBuffer::<u32>::new(id, 16)?;
    eprintln!(
        "pool: {n_pool} experts x (gate + up) = {:.2} GB; bpe {bpe} B; {} CUs; reps {reps}; dec grid {wgs} WGs x {waves} waves",
        2.0 * (n_pool * bpe) as f64 / 1e9,
        props.multi_processor_count * 2
    );

    // Contiguous streaming ceiling: ceil_read_v4 over a cold 200 MB window of the
    // pool (rotating), best of a few grid sizes.
    {
        let win = 200usize << 20;
        let mut best = (0f64, 0u32);
        for &grid in &[320u32, 640, 1280, 2560, 5120] {
            let mut t = Vec::new();
            for r in 0..12 {
                let off = ((r * 7 + grid as usize) * win) % (n_pool * bpe - win);
                let view = gate.slice_view(off & !255, win);
                let (a, b) = (Event::new()?, Event::new()?);
                a.record(&s)?;
                ceil.read_v4(&s, &view, &mut sink, grid, 256)?;
                b.record(&s)?;
                s.synchronize()?;
                t.push(Event::elapsed_ms(&a, &b)?);
            }
            let med = pct(&mut t, 0.5);
            let gbps = win as f64 / (med as f64 * 1e-3) / 1e9;
            if gbps > best.0 {
                best = (gbps, grid);
            }
        }
        eprintln!("contiguous streaming read (ceil_read_v4, 200 MB cold): {:.1} GB/s (grid {} x 256)", best.0, best.1);
    }

    let configs: &[(usize, usize)] = &[(1, 4), (2, 4), (2, 8), (4, 4), (4, 8), (4, 16), (8, 4), (8, 8), (8, 16)];
    let mut cursor = 0usize;
    let mut next_experts = |d: usize, rng: &mut Lcg| -> Vec<usize> {
        let mut v: Vec<usize> = (0..d).map(|i| (cursor + i) % n_pool).collect();
        cursor = (cursor + d) % n_pool;
        for i in (1..v.len()).rev() {
            let j = rng.below(i as u32 + 1) as usize;
            v.swap(i, j);
        }
        v
    };
    eprintln!("\np50 / p90 us (GB/s at p50); every call on fresh (cold) experts; the work-item grid is the");
    eprintln!("production device-count bound (6b slots). 'members' = (row, pick) pairs over the D experts.");
    let mut summary = Vec::new();
    for &(b, d) in configs {
        let xq = upload(id, &q8k_blocks(&mut rng, b * nb))?;
        let ew = upload(id, &vec![1.0f32; b * nu])?;
        let mut mid = DeviceBuffer::<f32>::new(id, b * nu * nr)?;
        let bound = b * nu; // dispatch::moe_wi_upper_bound at decode = M
        let mut wi_d = DeviceBuffer::<i32>::new(id, bound)?;
        let mut gc_d = DeviceBuffer::<i32>::new(id, n_pool)?;
        let mut em_d = DeviceBuffer::<i32>::new(id, n_pool * b)?;
        let mut cnt_d = DeviceBuffer::<i32>::new(id, 1)?;
        let mut t: Vec<Vec<f32>> = vec![Vec::new(); VARIANTS.len()];
        let mut order: Vec<usize> = (0..VARIANTS.len()).collect();
        let mut members = 0;
        for _rep in 0..reps {
            for i in (1..order.len()).rev() {
                let j = rng.below(i as u32 + 1) as usize;
                order.swap(i, j);
            }
            for &v in &order {
                // Fresh experts for every call (cold), production-shaped groups.
                let experts = next_experts(d, &mut rng);
                let g = groups_from_picks(&regime_picks(&mut rng, b, &experts), n_pool, 32);
                assert_eq!(g.distinct, d);
                members = g.members;
                let mut wi = g.wi.clone();
                wi.resize(bound, g.wi[0]);
                wi_d.copy_from_host(&wi)?;
                gc_d.copy_from_host(&g.gc)?;
                em_d.copy_from_host(&g.em)?;
                cnt_d.copy_from_host(&[g.wi.len() as i32])?;
                let mpe = g.max_per_expert as u32;
                let dec_launch = |sym: &str, tk: Option<&DeviceBuffer<u32>>, snk: Option<&mut DeviceBuffer<u32>>,
                                  mid: &mut DeviceBuffer<f32>| {
                    e.mxfp4pair.launch_fused_swiglu_dec_raw(
                        sym, wgs, waves, tk, snk, &s, mid, &gate, &up, &xq, &ew, &gc_d, &em_d, &wi_d,
                        bound as u32, bpe as u32, bpe as u32, nu as u32, mpe, 32, SWIGLU_CLAMP_EXP, nr as u32,
                        nb as u32, Some(&cnt_d), b as u32,
                    )
                };
                let (a, z) = (Event::new()?, Event::new()?);
                a.record(&s)?;
                match VARIANTS[v] {
                    "kwide" | "kwide(A/A)" => e.mxfp4pair.launch_fused_swiglu_kwide_ex(
                        &s, &mut mid, &gate, &up, &xq, &ew, &gc_d, &em_d, &wi_d, bound as u32, bpe as u32, bpe as u32,
                        nu as u32, mpe, 32, SWIGLU_CLAMP_EXP, nr as u32, nb as u32, Some(&cnt_d),
                    )?,
                    "dec" => dec_launch("mxfp4_pair_matvec_fused_swiglu_dec", Some(&ticket), None, &mut mid)?,
                    "dec-static" => dec_launch("mxfp4_pair_matvec_fused_swiglu_dec", None, None, &mut mid)?,
                    "dec-2row" => dec_launch("mxfp4_pair_dec_2row", Some(&ticket), None, &mut mid)?,
                    "dec-loadonly" => dec_launch("mxfp4_pair_dec_loadonly", Some(&ticket), Some(&mut sink), &mut mid)?,
                    "dec-compute" => dec_launch("mxfp4_pair_dec_noload", Some(&ticket), None, &mut mid)?,
                    "probe" => {
                        // Ideal reader of the same bytes: exact grid (no dead slots).
                        let f = dec.get_function("mxfp4_pair_dec_probe_read")?;
                        let cfg = LaunchConfig { grid: (probe_nx, g.wi.len() as u32, 2), block: (256, 1, 1), shared_mem_bytes: 0 };
                        launch_kernel!(f, cfg, &s, [gate.raw(), up.raw(), wi_d.raw(), bpe as u32, sink.raw()])?;
                    }
                    other => unreachable!("{other}"),
                }
                z.record(&s)?;
                s.synchronize()?;
                t[v].push(Event::elapsed_ms(&a, &z)? * 1e3);
            }
        }
        let mb = (2 * d * bpe) as f64 / 1e6;
        eprintln!("\nb={b} D={d} members={members} bytes={mb:.1} MB");
        let mut p50 = vec![0f32; VARIANTS.len()];
        for (v, name) in VARIANTS.iter().enumerate() {
            let m = pct(&mut t[v], 0.5);
            let p9 = pct(&mut t[v], 0.9);
            p50[v] = m;
            eprintln!("  {name:<13} {m:7.0} / {p9:7.0} us   {:6.1} GB/s", mb * 1e6 / (m as f64 * 1e-6) / 1e9);
        }
        let probe = p50[VARIANTS.len() - 1];
        eprintln!(
            "  -> dec {:.1}% of the probe (kwide {:.1}%), dec / kwide = {:.3}, A/A spread {:.1}%",
            100.0 * probe / p50[2],
            100.0 * probe / p50[0],
            p50[2] / p50[0],
            100.0 * (p50[1] - p50[0]).abs() / p50[0]
        );
        summary.push((b, d, members, mb, p50));
    }
    eprintln!("\nSUMMARY (p50 us)            kwide     A/A     dec  static  dec-2row loadonly compute   probe  dec%probe kwide%probe");
    for (b, d, m, mb, p) in &summary {
        eprintln!(
            "  b={b} D={d:2} m={m:2} {mb:6.1} MB {:7.0} {:7.0} {:7.0} {:7.0} {:7.0} {:7.0} {:7.0} {:7.0}   {:5.1}%    {:5.1}%",
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 100.0 * p[7] / p[2], 100.0 * p[7] / p[0]
        );
    }
    Ok(())
}
