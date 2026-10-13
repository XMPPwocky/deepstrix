//! The decode MXFP4 down `mxfp4_matvec_par_by_expert_dn3_*` (`V41_MOE_DOWN_DN3`) vs the
//! kernels it would replace at rows <= 8: `mxfp4_matvec_par_by_expert_smallb` (dn2, the
//! production decode arm) and `mxfp4_matvec_par_by_expert_kwide2`.
//!
//!   1. `dn3_bit_identical_to_dn2`: every compiled dn3 geometry == dn2 == kwide2 BIT FOR
//!      BIT over the whole sentinel-filled partials buffer (so the written set matches
//!      too), at b = 1..8 in the sweep's three selection regimes, the DECODE regime
//!      (D = 2..16 distinct experts of 1-3 members at scattered pool slots), a split-chunk
//!      regime, all-zero expert rows against negative Q8_K scales (sign of zero), exact and
//!      bound grid + device count; `dispatch::moe_down_mxfp4` with `V41_MOE_DOWN_DN3=1`;
//!      the refusals. Also a CPU replica of dn2's per-lane order + reduction tree (f32
//!      fma) that must match bit for bit, and the plain CPU oracle
//!      (`cpu_dot_mxfp4_q8_k`, a different summation order) for max rel error.
//!   2. `bench_dn3_decode`: the decode regime, COLD weights (expert slots rotate over a
//!      ~2.9 GB pool, so no call re-reads anything the 32 MB MALL could hold), b in
//!      {1,2,4,8} x distinct experts in {4,8,16} (where b allows), interleaved arms in
//!      one process (dn2, dn2 again = A/A, every dn3 geometry, the slot-streaming read
//!      probe = the achievable ceiling for exactly the call's bytes), medians + p90.
//!
//! gfx1151 only. Runs beside a live hub (iGPU, < 3.1 GB device memory):
//!   cargo test --release --features v41 -p v4flash-kernels --test mxfp4_down_dn3 \
//!     -- --ignored --test-threads=1 --nocapture
//! Bench env: DN3_REPS (default 60), DN3_POOL_SLOTS (default 470 = 2.95 GB).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, N_EMBD, N_EXPERT_USED};
use v4flash_kernels::device_ceilings::DeviceCeilings;
use v4flash_kernels::het::dispatch;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mxfp4::{Dn3Geom, DN3_DEFAULT, DN3_GEOMS};
use v4flash_kernels::mxfp4_tables::{cpu_dot_mxfp4_q8_k, SUPER_MXFP4_BYTES};
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

/// Q8_K blocks with random int8s and bsums; the scale's SIGN is random when `signed_d`
/// (ggml's Q8_K keeps the sign of the max-magnitude value, so d < 0 is real).
fn q8k_blocks(rng: &mut Lcg, n_blocks: usize, signed_d: bool) -> Vec<u8> {
    let mut out = vec![0u8; n_blocks * BLOCK_Q8_K_BYTES];
    for blk in out.chunks_exact_mut(BLOCK_Q8_K_BYTES) {
        let mut d = 0.001f32 + (rng.next() % 1000) as f32 * 1e-5;
        if signed_d && rng.next() & 1 == 1 {
            d = -d;
        }
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

/// The production builder's output: group_count[e], expert_members[e * max + m] =
/// (row << 16) | slot, work items (e << 16) | member_start per `chunk` members.
struct Groups {
    gc: Vec<i32>,
    em: Vec<i32>,
    wi: Vec<i32>,
    max_per_expert: usize,
    members: usize,
}

fn work_items(gc: &[i32], chunk: usize) -> Vec<i32> {
    let mut wi = Vec::new();
    for (e, &n) in gc.iter().enumerate() {
        let mut s = 0;
        while s < n as usize {
            wi.push(((e as i32) << 16) | s as i32);
            s += chunk;
        }
    }
    wi
}

/// tests/mxfp4_moe_sweep.rs's builder: every row picks `n_used` DISTINCT virtual experts
/// out of `n_virt`; picks >= `n_phys` are not materialised (another box / a miss).
fn build_groups(rng: &mut Lcg, b: usize, n_used: usize, n_virt: u32, n_phys: usize, chunk: usize) -> Groups {
    let max_per_expert = b;
    let mut gc = vec![0i32; n_phys];
    let mut em = vec![0i32; n_phys * max_per_expert];
    let mut members = 0;
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
                members += 1;
            }
        }
    }
    let wi = work_items(&gc, chunk);
    Groups { gc, em, wi, max_per_expert, members }
}

/// Members box 1 computes for `b` rows x 6 picks at ~50% of the pick mass over `d`
/// distinct experts: clamp(3b, d, min(d * min(3, b), 6b)).
fn decode_members(b: usize, d: usize) -> usize {
    (3 * b).max(d).min((d * b.min(3)).min(N_EXPERT_USED * b))
}

/// The DECODE regime: `d` distinct experts at pool slots `slots` (group arrays sized for
/// `n_pool` slots), `decode_members(b, d)` members spread round-robin over the rows (an
/// expert at most once per row, at most 6 picks per row), members in row order.
fn decode_groups(b: usize, slots: &[u32], n_pool: usize, chunk: usize) -> Groups {
    let d = slots.len();
    let m_total = decode_members(b, d);
    let max_per_expert = b;
    let mut gc = vec![0i32; n_pool];
    let mut em = vec![0i32; n_pool * max_per_expert];
    let mut row_fill = vec![0usize; b];
    let mut row_has = vec![vec![false; d]; b];
    let mut cursor = 0usize;
    let mut placed = 0usize;
    for j in 0..d {
        let m_j = m_total / d + usize::from(j < m_total % d);
        let e = slots[j] as usize;
        for _ in 0..m_j {
            let mut tries = 0;
            loop {
                let r = cursor % b;
                cursor += 1;
                tries += 1;
                assert!(tries <= 2 * b, "decode_groups: cannot place b={b} d={d}");
                if !row_has[r][j] && row_fill[r] < N_EXPERT_USED {
                    row_has[r][j] = true;
                    em[e * max_per_expert + gc[e] as usize] = ((r as i32) << 16) | row_fill[r] as i32;
                    row_fill[r] += 1;
                    gc[e] += 1;
                    placed += 1;
                    break;
                }
            }
        }
    }
    let wi = work_items(&gc, chunk);
    Groups { gc, em, wi, max_per_expert, members: placed }
}

/// CPU replica of dn2's arithmetic for one (member, row): per lane (bl, sub) f = fma(sc1,
/// s1, fma(sc0, s0, 0)), pacc = fma(yd, f, 0) [+ fma(yd8, f8, pacc) on lanes 0-3], then
/// the shfl_down tree 16, 8, 4, 2, 1.
fn cpu_dn2_replica(w_row: &[u8], y: &[u8]) -> f32 {
    use v4flash_kernels::mxfp4_tables::{e8m0_half_to_f32, KVALUES_MXFP4};
    let lane_f = |sb: usize, sub: usize| -> f32 {
        let ws = &w_row[sb * SUPER_MXFP4_BYTES..(sb + 1) * SUPER_MXFP4_BYTES];
        let q8 = &y[sb * BLOCK_Q8_K_BYTES + 4..sb * BLOCK_Q8_K_BYTES + 260];
        let mut f = 0.0f32;
        for t in 0..2 {
            let blk = 2 * sub + t;
            let qs = &ws[blk * 16..blk * 16 + 16];
            let mut s: i32 = 0;
            for j in 0..16 {
                s += KVALUES_MXFP4[(qs[j] & 0x0F) as usize] as i32 * (q8[blk * 32 + j] as i8) as i32;
                s += KVALUES_MXFP4[(qs[j] >> 4) as usize] as i32 * (q8[blk * 32 + 16 + j] as i8) as i32;
            }
            f = e8m0_half_to_f32(ws[128 + blk]).mul_add(s as f32, f);
        }
        f
    };
    let yd = |sb: usize| f32::from_le_bytes(y[sb * BLOCK_Q8_K_BYTES..sb * BLOCK_Q8_K_BYTES + 4].try_into().unwrap());
    let mut v = [0f32; 32];
    for (l, vl) in v.iter_mut().enumerate() {
        let (bl, sub) = (l >> 2, l & 3);
        let mut pacc = yd(bl).mul_add(lane_f(bl, sub), 0.0);
        if bl == 0 {
            pacc = yd(8).mul_add(lane_f(8, sub), pacc);
        }
        *vl = pacc;
    }
    for o in [16usize, 8, 4, 2, 1] {
        for l in 0..o {
            v[l] += v[l + o];
        }
    }
    v[0]
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// --------------------------------------------------------------------- 1. exactness

#[test]
#[ignore]
fn dn3_bit_identical_to_dn2() -> eyre::Result<()> {
    install_panic_handler()?;
    // The dispatch knob is read once (LazyLock): set before any dispatch call.
    std::env::set_var("V41_MOE_DOWN_DN3", "1");
    let Some(dev) = pick_igpu()? else {
        eprintln!("SKIP: no gfx1151 device");
        return Ok(());
    };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_D103);
    let nu = N_EXPERT_USED;
    let (nr, nb) = (N_EMBD as usize, BLOCKS_Q8K_DOWN_IN as usize);
    let dbpe = nr * nb * SUPER_MXFP4_BYTES;
    const N_PHYS: usize = 20;
    let mut w_host = mxfp4_weights(&mut rng, N_PHYS, nr, nb, 118, 8);
    // Expert 19 has all-zero nibbles (every product 0): with negative Q8_K scales the
    // partials are signed zeros, which a different fma/mul shape would flip.
    for sb in w_host[19 * dbpe..20 * dbpe].chunks_exact_mut(SUPER_MXFP4_BYTES) {
        sb[..128].fill(0);
    }
    let w = upload(id, &w_host)?;
    let slot_stride = nb * BLOCK_Q8_K_BYTES;

    // (label, groups, chunk)
    let mut cases: Vec<(String, Groups, usize)> = Vec::new();
    for &(n_virt, chunk) in &[(6u32, 32usize), (24, 32), (8, 4)] {
        for b in 1..=8usize {
            cases.push((format!("sweep n_virt={n_virt} chunk={chunk} b={b}"), build_groups(&mut rng, b, nu, n_virt, N_PHYS, chunk), chunk));
        }
    }
    for b in 1..=8usize {
        for d in [2usize, 3, 4, 6, 8, 12, 16] {
            if d > nu * b {
                continue;
            }
            // Scattered slots incl. the zero expert 19 at a random position.
            let mut slots: Vec<u32> = Vec::new();
            while slots.len() < d {
                let x = rng.below(N_PHYS as u32);
                if !slots.contains(&x) {
                    slots.push(x);
                }
            }
            if b % 3 == 0 && !slots.contains(&19) {
                slots[0] = 19;
            }
            for chunk in [32usize, 2] {
                cases.push((format!("decode b={b} D={d} chunk={chunk}"), decode_groups(b, &slots, N_PHYS, chunk), chunk));
            }
        }
    }

    let (mut n_cases, mut n_launch_cmp) = (0usize, 0usize);
    let (mut worst_rel_oracle, mut replica_checked) = (0f64, 0usize);
    for (label, g, chunk) in &cases {
        let b = g.max_per_expert;
        let midq_h = q8k_blocks(&mut rng, b * nu * nb, true);
        let midq = upload(id, &midq_h)?;
        let gc = upload(id, &g.gc)?;
        let em = upload(id, &g.em)?;
        let n_real = g.wi.len();
        let bound = dispatch::moe_wi_upper_bound(b * nu, N_PHYS as u32, *chunk as u32, 1 << 20).max(n_real as u32) as usize;
        let mut wi_pad = g.wi.clone();
        wi_pad.resize(bound, 0); // stale-looking items past the count (a real expert)
        let wi_exact = upload(id, &g.wi)?;
        let wi_bound = upload(id, &wi_pad)?;
        let cnt = upload(id, &[n_real as i32])?;
        let n_out = b * nu * nr;
        let mut reference: Option<Vec<u32>> = None;
        for bounded in [false, true] {
            let (wi_d, n_wi, c) = if bounded { (&wi_bound, bound as u32, Some(&cnt)) } else { (&wi_exact, n_real as u32, None) };
            let mut outs: Vec<(String, DeviceBuffer<f32>)> = Vec::new();
            let mut p = upload(id, &vec![SENTINEL; n_out])?;
            e.mxfp4.launch_by_expert_kwide2_ex(&s, &mut p, &w, &midq, &gc, &em, wi_d, n_wi, dbpe as u32, slot_stride as u32,
                nu as u32, g.max_per_expert as u32, *chunk as u32, nr as u32, nb as u32, c)?;
            outs.push(("kwide2".into(), p));
            let mut p = upload(id, &vec![SENTINEL; n_out])?;
            e.mxfp4.launch_by_expert_smallb_ex(&s, &mut p, &w, &midq, &gc, &em, wi_d, n_wi, dbpe as u32, slot_stride as u32,
                nu as u32, g.max_per_expert as u32, *chunk as u32, nr as u32, nb as u32, c)?;
            outs.push(("dn2".into(), p));
            for geom in DN3_GEOMS {
                let mut p = upload(id, &vec![SENTINEL; n_out])?;
                e.mxfp4.launch_by_expert_dn3_geom(*geom, &s, &mut p, &w, &midq, &gc, &em, wi_d, n_wi, dbpe as u32, slot_stride as u32,
                    nu as u32, g.max_per_expert as u32, *chunk as u32, nr as u32, nb as u32, c)?;
                outs.push((geom.label(), p));
            }
            let mut p = upload(id, &vec![SENTINEL; n_out])?;
            dispatch::moe_down_mxfp4(&e, &s, &mut p, &w, &midq, &gc, &em, wi_d, n_wi, dbpe as u32, slot_stride as u32,
                nu as u32, g.max_per_expert as u32, *chunk as u32, nr as u32, nb as u32, c, b as u32)?;
            outs.push(("dispatch(rows=b, DN3=1)".into(), p));
            s.synchronize()?;
            let base = bits(&download(&outs[1].1, n_out)?);
            let written = base.iter().filter(|&&v| v != SENTINEL.to_bits()).count();
            assert_eq!(written, g.members * nr, "{label}: dn2 wrote {written} of {} partials", g.members * nr);
            for (name, buf) in &outs {
                let got = bits(&download(buf, n_out)?);
                let nd = base.iter().zip(&got).filter(|(a, b)| a != b).count();
                assert_eq!(nd, 0, "{label} ({}): {name} differs from dn2 in {nd} of {n_out} partials",
                    if bounded { "bound+count" } else { "exact" });
                n_launch_cmp += 1;
            }
            if let Some(r) = &reference {
                assert_eq!(r, &base, "{label}: exact vs bound grid differ");
            }
            // CPU: dn2 replica (bit for bit) + plain oracle (rel error), sampled rows.
            if !bounded {
                let part = download(&outs[1].1, n_out)?;
                for (ee, &n) in g.gc.iter().enumerate() {
                    for m in 0..n as usize {
                        let packed = g.em[ee * g.max_per_expert + m];
                        let (row_b, slot) = ((packed >> 16) as usize, (packed & 0xffff) as usize);
                        let y = &midq_h[(row_b * nu + slot) * slot_stride..(row_b * nu + slot + 1) * slot_stride];
                        for k in 0..6 {
                            let r = (k * 997 + m * 131 + ee * 17) % nr;
                            let w_row = &w_host[ee * dbpe + r * nb * SUPER_MXFP4_BYTES..ee * dbpe + (r + 1) * nb * SUPER_MXFP4_BYTES];
                            let gpu = part[(row_b * nu + slot) * nr + r];
                            let rep = cpu_dn2_replica(w_row, y);
                            assert_eq!(gpu.to_bits(), rep.to_bits(), "{label}: dn2 != CPU replica at e={ee} m={m} row={r}: {gpu:e} vs {rep:e}");
                            replica_checked += 1;
                            let ora = cpu_dot_mxfp4_q8_k(nb, w_row, y);
                            if ora != 0.0 {
                                worst_rel_oracle = worst_rel_oracle.max(((gpu - ora).abs() / ora.abs()) as f64);
                            }
                        }
                    }
                }
            }
            reference = Some(base);
        }
        n_cases += 1;
    }
    // Refusals: compiled n_blocks_in, n_rows granularity.
    let g = build_groups(&mut rng, 2, nu, 6, N_PHYS, 32);
    let (gc, em, wi) = (upload(id, &g.gc)?, upload(id, &g.em)?, upload(id, &g.wi)?);
    let midq = upload(id, &q8k_blocks(&mut rng, 2 * nu * nb, false))?;
    let mut p = upload(id, &vec![SENTINEL; 2 * nu * nr])?;
    assert!(e.mxfp4.launch_by_expert_dn3_ex(&s, &mut p, &w, &midq, &gc, &em, &wi, g.wi.len() as u32, dbpe as u32,
        slot_stride as u32, nu as u32, 2, 32, nr as u32, 8, None).is_err(), "n_blocks_in != 9 must be refused");
    let rows_wg = DN3_DEFAULT.rows_per_wave * DN3_DEFAULT.waves;
    assert!(e.mxfp4.launch_by_expert_dn3_ex(&s, &mut p, &w, &midq, &gc, &em, &wi, g.wi.len() as u32, dbpe as u32,
        slot_stride as u32, nu as u32, 2, 32, rows_wg + rows_wg / 2, nb as u32, None).is_err(), "n_rows % rows-per-WG != 0 must be refused");
    eprintln!("PASS: {n_cases} cases x (exact, bound+count): {n_launch_cmp} launches bit-identical to dn2 \
(kwide2, {} dn3 geometries, dispatch with V41_MOE_DOWN_DN3=1); dn2 == CPU replica of its order on {replica_checked} outputs; \
max rel err vs the plain CPU oracle (different order) {worst_rel_oracle:.3e}", DN3_GEOMS.len());
    Ok(())
}

// ------------------------------------------------------------------------- 2. bench

#[derive(Clone)]
enum Arm {
    Dn2,
    Dn2Aa,
    Dn3(Dn3Geom),
    Probe(u32),
    ProbeMx(u32),
}

impl Arm {
    fn name(&self) -> String {
        match self {
            Arm::Dn2 => "dn2 (current)".into(),
            Arm::Dn2Aa => "dn2 A/A".into(),
            Arm::Dn3(g) => g.label(),
            Arm::Probe(c) => format!("read probe {}K", c / 1024),
            Arm::ProbeMx(c) => format!("read probe mx-lanes {}B", c),
        }
    }
}

fn pct(v: &mut [f32], p: f64) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

#[test]
#[ignore]
fn bench_dn3_decode() -> eyre::Result<()> {
    install_panic_handler()?;
    let Some(dev) = pick_igpu()? else {
        eprintln!("SKIP: no gfx1151 device");
        return Ok(());
    };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let ceil = DeviceCeilings::for_arch(&arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_BE4C);
    let nu = N_EXPERT_USED;
    let (nr, nb) = (N_EMBD as usize, BLOCKS_Q8K_DOWN_IN as usize);
    let dbpe = nr * nb * SUPER_MXFP4_BYTES;
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let reps = env("DN3_REPS", 60);
    let n_pool = env("DN3_POOL_SLOTS", 470);
    let bound_grid = env("DN3_BOUND", 1) != 0;
    eprintln!("pool {n_pool} slots x {dbpe} B = {:.2} GB; reps {reps}; grid {}", (n_pool * dbpe) as f64 / 1e9,
        if bound_grid { "bound + device count (production)" } else { "exact" });
    let mut pool = DeviceBuffer::<u8>::new(id, n_pool * dbpe)?;
    {
        let pats: Vec<Vec<u8>> = (0..4).map(|_| mxfp4_weights(&mut rng, 1, nr, nb, 118, 8)).collect();
        for k in 0..n_pool {
            pool.slice_view_mut(k * dbpe, dbpe).copy_from_host(&pats[k % pats.len()])?;
        }
    }
    let mut sink = DeviceBuffer::<u32>::new(id, 16)?;

    // ---- steady-state streaming ceiling over the cold pool (1 GB per launch)
    {
        let gb1 = (1usize << 30).min(n_pool * dbpe);
        let mut t = Vec::new();
        for k in 0..12 {
            let off = ((k * 7) % 2) * gb1.min(n_pool * dbpe - gb1);
            let view = pool.slice_view(off, gb1);
            let (a, b) = (Event::new()?, Event::new()?);
            a.record(&s)?;
            ceil.read_v4(&s, &view, &mut sink, 40 * 16, 256)?;
            b.record(&s)?;
            s.synchronize()?;
            if k >= 2 {
                t.push(gb1 as f32 / (Event::elapsed_ms(&a, &b)? * 1e-3) / 1e9);
            }
        }
        let med = pct(&mut t.clone(), 0.5);
        let best = pct(&mut t, 1.0);
        eprintln!("CEILING ceil_read_v4 1 GiB cold, 640 WG x 256: median {med:.1} GB/s, best {best:.1} GB/s");
    }

    // ---- shapes
    // DN3_SHAPES="b:d,b:d,..." overrides the default sweep.
    let shapes: Vec<(usize, usize)> = match std::env::var("DN3_SHAPES") {
        Ok(v) => v.split(',').filter_map(|x| {
            let (b, d) = x.split_once(':')?;
            Some((b.trim().parse().ok()?, d.trim().parse().ok()?))
        }).collect(),
        Err(_) => vec![(1usize, 2usize), (1, 4), (2, 4), (2, 8), (4, 4), (4, 6), (4, 8), (4, 10), (4, 12), (4, 16), (8, 4), (8, 8), (8, 16)],
    }
    .into_iter().filter(|&(b, d)| d <= nu * b).collect();
    let mut arms = vec![Arm::Dn2, Arm::Dn2Aa];
    let only = std::env::var("DN3_ONLY").ok();
    for g in DN3_GEOMS {
        let a = Arm::Dn3(*g);
        if only.as_deref().is_none_or(|o| o.split(',').any(|x| a.name().contains(x))) {
            arms.push(a);
        }
    }
    // DN3_PROBES="19584,65536,..." (chunk bytes) overrides the default 64 KB probe.
    for c in std::env::var("DN3_PROBES").unwrap_or_else(|_| "65536".into()).split(',') {
        if let Ok(c) = c.trim().parse::<u32>() {
            arms.push(Arm::Probe(c));
        }
    }
    // DN3_PROBES_MX="34816,..." (multiples of 1088 dividing the expert): the same read in
    // the kernels' per-lane MXFP4 pattern.
    for c in std::env::var("DN3_PROBES_MX").unwrap_or_default().split(',') {
        if let Ok(c) = c.trim().parse::<u32>() {
            arms.push(Arm::ProbeMx(c));
        }
    }

    struct ShapeBufs {
        b: usize,
        d: usize,
        members: usize,
        rot: Vec<(DeviceBuffer<i32>, DeviceBuffer<i32>, DeviceBuffer<i32>, DeviceBuffer<i32>, DeviceBuffer<i32>, usize, u32)>,
        midq: DeviceBuffer<u8>,
        part: DeviceBuffer<f32>,
    }
    const N_ROT: usize = 48;
    let mut slot_ctr = 0usize;
    let mut sb: Vec<ShapeBufs> = Vec::new();
    for &(b, d) in &shapes {
        let mut rot = Vec::new();
        let mut members = 0;
        for _ in 0..N_ROT {
            let slots: Vec<u32> = (0..d).map(|k| (((slot_ctr + k) * 97) % n_pool) as u32).collect();
            slot_ctr += d;
            let g = decode_groups(b, &slots, n_pool, 32);
            members = g.members;
            let n_real = g.wi.len();
            let bound = if bound_grid { (b * nu).max(n_real) } else { n_real };
            let mut wi_pad = g.wi.clone();
            wi_pad.resize(bound, 0);
            let slots_i: Vec<i32> = slots.iter().map(|&x| x as i32).collect();
            rot.push((upload(id, &g.gc)?, upload(id, &g.em)?, upload(id, &wi_pad)?, upload(id, &[n_real as i32])?, upload(id, &slots_i)?, bound, g.max_per_expert as u32));
        }
        let midq = upload(id, &q8k_blocks(&mut rng, b * nu * nb, true))?;
        let part = DeviceBuffer::<f32>::new(id, b * nu * nr)?;
        sb.push(ShapeBufs { b, d, members, rot, midq, part });
    }

    // ---- launch floor: an empty-ish launch (16 B probe) through the same event pair
    let floor_us = {
        let tiny = upload(id, &[0i32])?;
        let mut t = Vec::new();
        for _ in 0..200 {
            let (a, b) = (Event::new()?, Event::new()?);
            a.record(&s)?;
            ceil.read_slots(&s, &pool, &tiny, 1, 16, 16, &mut sink)?;
            b.record(&s)?;
            s.synchronize()?;
            t.push(Event::elapsed_ms(&a, &b)? * 1000.0);
        }
        pct(&mut t, 0.5)
    };
    eprintln!("launch floor (1-WG kernel, event pair): median {floor_us:.1} us");

    // ---- interleaved A/B: per rep, per shape, arms in a rotated order
    let mut times: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); arms.len()]; sb.len()];
    let mut rot_i = 0usize;
    let slot_stride = (nb * BLOCK_Q8_K_BYTES) as u32;
    for rep in 0..reps + 3 {
        for (si, x) in sb.iter_mut().enumerate() {
            for k in 0..arms.len() {
                let ai = (k + rep + si) % arms.len();
                let (gc, em, wi, cnt, slots, bound, mpe) = &x.rot[rot_i % N_ROT];
                rot_i += 1;
                let (a, bev) = (Event::new()?, Event::new()?);
                let c = if bound_grid { Some(cnt) } else { None };
                a.record(&s)?;
                match &arms[ai] {
                    Arm::Dn2 | Arm::Dn2Aa => e.mxfp4.launch_by_expert_smallb_ex(&s, &mut x.part, &pool, &x.midq, gc, em, wi, *bound as u32,
                        dbpe as u32, slot_stride, nu as u32, *mpe, 32, nr as u32, nb as u32, c)?,
                    Arm::Dn3(g) => e.mxfp4.launch_by_expert_dn3_geom(*g, &s, &mut x.part, &pool, &x.midq, gc, em, wi, *bound as u32,
                        dbpe as u32, slot_stride, nu as u32, *mpe, 32, nr as u32, nb as u32, c)?,
                    Arm::Probe(chunk) => ceil.read_slots(&s, &pool, slots, x.d as u32, dbpe, *chunk, &mut sink)?,
                    Arm::ProbeMx(chunk) => ceil.read_slots_mx(&s, &pool, slots, x.d as u32, dbpe, *chunk, &mut sink)?,
                }
                bev.record(&s)?;
                s.synchronize()?;
                if rep >= 3 {
                    times[si][ai].push(Event::elapsed_ms(&a, &bev)? * 1000.0);
                }
            }
        }
    }

    eprintln!("\nDECODE down, cold weights, {} reps/arm/shape; us = HIP-event kernel time; GB/s = weight bytes / median", reps);
    eprintln!("bytes/call = D x {dbpe} B weights (+ members x {} B activations, + members x {} B partials written)", nb * BLOCK_Q8_K_BYTES, nr * 4);
    for (si, x) in sb.iter().enumerate() {
        let bytes = (x.d * dbpe) as f64;
        eprintln!("\nb={} D={} members={} weights {:.2} MB, launches 1 (grid.y {})", x.b, x.d, x.members, bytes / 1e6, x.rot[0].5);
        for (ai, arm) in arms.iter().enumerate() {
            let mut v = times[si][ai].clone();
            let (p50, p90, p10, p25) = (pct(&mut v, 0.5), pct(&mut v, 0.9), pct(&mut v, 0.1), pct(&mut v, 0.25));
            eprintln!("  {:<28} p10 {:7.1}  p25 {:7.1}  p50 {:7.1}  p90 {:7.1} us   {:6.1} GB/s (p50)  {:6.1} GB/s (p10)",
                arm.name(), p10, p25, p50, p90, bytes / (p50 as f64 * 1e-6) / 1e9, bytes / (p10 as f64 * 1e-6) / 1e9);
        }
    }
    Ok(())
}
