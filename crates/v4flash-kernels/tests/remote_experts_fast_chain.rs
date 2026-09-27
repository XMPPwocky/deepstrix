//! Box 2's short batched chain (`V41_B2_FAST_CHAIN`, `knobs::fast_chain`) vs
//! the old chain, on the gfx1151 iGPU: every output BIT-IDENTICAL.
//!
//! `fast_chain_matches_old_chain` drives the REAL `MoeExecutor::run_path` on a
//! small PAGED pool of one layer (24 slots, picks over 40 experts: misses,
//! hits-first two passes), toggling the knob in process:
//!   1. every request run by BOTH chains back to back on one executor (the
//!      first to run takes the misses, so each chain gets its two-pass share),
//!      f32 and f16 replies (and a partner-style partial f16 read) compared;
//!   2. the same requests by the fast chain alone (its no-memset steady state)
//!      against phase 1's outputs, asserting no partials memset after warm-up;
//!   3. errors injected mid-request (after pass A of a two-pass, i.e. partials
//!      written and never reduced; and after the reduce, with the GPU work
//!      still in flight), each followed by a normal request that must match.
//! Shapes: b = 1..16, NO_PICK padding (sentinels on the device), rows with no
//! pick, duplicate experts across rows, one distinct expert for the whole
//! batch, up to 14 distinct, f32 replies.
//!
//! `fast_kernels_match_old_kernels` checks the two new kernels directly on
//! synthetic inputs the executor never produces (mode 1, over-cap residents,
//! group ids past the bound, member overflow): the fused builder against
//! `moe_group_builder_hetsplit` + `moe_work_items_builder` as sets, the zeroing
//! reduce against `q2_k_reduce_partials_hetsplit` + `f32_to_f16_cast` bit for
//! bit, and exactly which partial slots it zeroes.
//!
//! Run (box 1, one GPU test process at a time; ~0.5 GB of GTT, ~1-2 s of iGPU
//! time, a few hundred MB of expert reads):
//!   cargo test --release -p v4flash-kernels --features v41 --test remote_experts_fast_chain -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, PinnedBuffer};
use v4flash_kernels::b2_fast_chain::B2FastChain;
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT_USED};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::remote_experts::{
    knobs, Assignment, ExecTiming, ExpertShard, MoeExecutor, CHUNK_SIZE, NO_PICK, REMAP_LEN, SENTINEL_EXPERT,
    XQ_BYTES_PER_TOKEN,
};

const LAYER: u32 = 20;
const SEEDED: u32 = 24;
const RANGE: u32 = 40;
const WINDOW: u32 = 14;
const ROWS: usize = 16;
const NU: usize = N_EXPERT_USED;
const NE: usize = N_EMBD as usize;

fn model_dir() -> String {
    std::env::var("V41_MODEL")
        .unwrap_or_else(|_| format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap()))
}

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// A valid Q8_K row (per 256-block `d` f32, 256 int8 `qs`, 16 int16 `bsums`).
fn q8k_row(rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(XQ_BYTES_PER_TOKEN);
    for _ in 0..BLOCKS_Q8K_GATE_IN {
        let d = 0.002f32 + (rng.below(1000) as f32) * 1e-5;
        out.extend_from_slice(&d.to_le_bytes());
        let qs: Vec<i8> = (0..256).map(|_| (rng.below(255) as i32 - 127) as i8).collect();
        out.extend(qs.iter().map(|&q| q as u8));
        for g in 0..16 {
            let s: i16 = qs[g * 16..(g + 1) * 16].iter().map(|&q| q as i16).sum();
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    assert_eq!(out.len(), XQ_BYTES_PER_TOKEN);
    out
}

#[derive(Clone)]
struct Spec {
    b: usize,
    xq: Vec<u8>,
    sel: Vec<i32>,
    ew: Vec<f32>,
    f32_reply: bool,
}

/// Request `i`: b = 1 + i % 16; picks from a window of `WINDOW` experts
/// somewhere in `0..RANGE` (so a request always fits the pool, and the pool
/// churns across requests).
fn gen_spec(rng: &mut Rng, i: usize) -> Spec {
    let b = 1 + i % ROWS;
    let w0 = rng.below(u64::from(RANGE - WINDOW + 1)) as u32;
    let mut sel = vec![NO_PICK; b * NU];
    let mut ew = vec![0f32; b * NU];
    let shape = rng.below(6);
    let one = (w0 + rng.below(u64::from(WINDOW)) as u32) as i32;
    for t in 0..b {
        let k = match shape {
            0 => usize::from(t == 0 || rng.below(4) != 0), // ONE expert for the whole batch (some rows empty)
            1 => NU,                                       // dense rows
            2 => rng.below(3) as usize,                    // sparse, often empty rows
            _ => rng.below(NU as u64 + 1) as usize,
        };
        let mut row: Vec<i32> = Vec::new();
        while row.len() < k {
            let e = if shape == 0 { one } else { (w0 + rng.below(u64::from(WINDOW)) as u32) as i32 };
            if !row.contains(&e) {
                row.push(e);
            }
        }
        // Random distinct slots (NO_PICK padding anywhere, like the wire).
        let mut slots: Vec<usize> = (0..NU).collect();
        for j in (1..NU).rev() {
            slots.swap(j, rng.below(j as u64 + 1) as usize);
        }
        for (e, &s) in row.iter().zip(&slots) {
            sel[t * NU + s] = *e;
            ew[t * NU + s] = 0.05 + rng.below(950) as f32 / 1000.0;
        }
    }
    let xq: Vec<u8> = (0..b).flat_map(|_| q8k_row(rng)).collect();
    Spec { b, xq, sel, ew, f32_reply: rng.below(7) == 0 }
}

struct Out {
    f32: Vec<u32>,
    f16: Vec<u16>,
}

type Hook<'a> = &'a mut dyn FnMut(&mut ExpertShard, bool) -> eyre::Result<()>;

fn run_spec(exec: &mut MoeExecutor, shard: &mut ExpertShard, s: &Spec, fast: bool, hook: Hook<'_>) -> eyre::Result<(Out, ExecTiming)> {
    knobs::set_fast_chain(fast);
    exec.set_reply_f16(!s.f32_reply);
    let t = exec.run_path(shard, LAYER, s.b, &s.xq, &s.sel, &s.ew, true, hook)?;
    assert_eq!(t.fast_chain, fast, "b={} took the wrong chain", s.b);
    let mut f = vec![0f32; s.b * NE];
    exec.read_f32_at(0, s.b, &mut f)?;
    let mut h = vec![0u16; s.b * NE];
    exec.read_f16_at(0, s.b, &mut h)?;
    if s.b >= 2 {
        // A merged partner's reply: rows [off, b) of the same pass.
        let mut h2 = vec![0u16; (s.b - 1) * NE];
        exec.read_f16_at(1, s.b - 1, &mut h2)?;
        assert!(h2 == h[NE..], "partial f16 read differs from the full read (fast={fast})");
    }
    Ok((Out { f32: f.iter().map(|x| x.to_bits()).collect(), f16: h }, t))
}

fn same(a: &Out, b: &Out, what: &str) {
    let df = a.f32.iter().zip(&b.f32).filter(|(x, y)| x != y).count();
    let dh = a.f16.iter().zip(&b.f16).filter(|(x, y)| x != y).count();
    assert!(
        a.f32.len() == b.f32.len() && df == 0 && dh == 0,
        "{what}: {df} f32 and {dh} f16 elements differ (of {})",
        a.f32.len()
    );
}

#[test]
#[ignore]
fn fast_chain_matches_old_chain() -> eyre::Result<()> {
    install_panic_handler()?;
    for (k, v) in [("V41_B2_HITS_FIRST", "1"), ("V41_B2_PARK", "1"), ("V41_B2_PREFILL_STAGE", "0")] {
        std::env::set_var(k, v);
    }
    std::env::remove_var("V41_B2_FAST_CHAIN");
    std::env::remove_var("V41_B2_DECODE_DOWN");
    let igpu = pick_igpu()?;
    let hf = V41HfWeights::open(&model_dir(), None)?;
    let asg = Assignment::parse(&format!("L{LAYER}:0-{}", SEEDED - 1))?;
    let mut shard = ExpertShard::load(hf, igpu, &asg, 4, 5, ROWS as u32, 1)?;
    shard.enable_paging()?;
    let mut exec = MoeExecutor::new(igpu, ROWS, 1)?;
    let mut rng = Rng(0xfa57_c4a1_b2b2_0001);
    let specs: Vec<Spec> = (0..3 * ROWS).map(|i| gen_spec(&mut rng, i)).collect();
    let mut ok = |_: &mut ExpertShard, _: bool| -> eyre::Result<()> { Ok(()) };

    // 1. Both chains on every request, alternating which runs first.
    let mut refs: Vec<Out> = Vec::new();
    let (mut two_old, mut two_fast, mut f32_replies) = (0, 0, 0);
    for (i, s) in specs.iter().enumerate() {
        let fast_first = i % 2 == 0;
        let (a, ta) = run_spec(&mut exec, &mut shard, s, fast_first, &mut ok)?;
        let (b, tb) = run_spec(&mut exec, &mut shard, s, !fast_first, &mut ok)?;
        same(&a, &b, &format!("request {i} (b={}, fast first={fast_first}, two-pass {}/{})", s.b, ta.two_pass, tb.two_pass));
        // Not vacuous: every row with a pick has a non-zero result, every row without is zero.
        for t in 0..s.b {
            let has = s.sel[t * NU..(t + 1) * NU].iter().any(|&e| e != NO_PICK);
            let nz = a.f32[t * NE..(t + 1) * NE].iter().filter(|&&x| x != 0).count();
            assert_eq!(has, nz > NE / 2, "request {i} row {t}: picks {has}, {nz} non-zero outputs");
        }
        for (t, fast) in [(&ta, fast_first), (&tb, !fast_first)] {
            if t.two_pass {
                if fast { two_fast += 1 } else { two_old += 1 }
            }
        }
        f32_replies += usize::from(s.f32_reply);
        refs.push(a);
    }
    eprintln!(
        "phase 1: {} requests x 2 chains bit-identical (f32 + f16); two-pass runs old {two_old} fast {two_fast}; f32 replies {f32_replies}",
        specs.len()
    );
    assert!(two_old >= 3 && two_fast >= 3, "two-pass not exercised: old {two_old} fast {two_fast}");

    // 2. Fast chain only: its steady state (no memset once every row count has been seen).
    let m0 = exec.fast_chain_memsets();
    let mut two = 0;
    for (i, s) in specs.iter().enumerate() {
        let (o, t) = run_spec(&mut exec, &mut shard, s, true, &mut ok)?;
        same(&o, &refs[i], &format!("fast-only request {i} (b={}, two-pass {})", s.b, t.two_pass));
        two += usize::from(t.two_pass);
    }
    let m_steady = exec.fast_chain_memsets() - m0;
    eprintln!("phase 2: {} fast-only requests bit-identical; two-pass {two}; partials memsets {m_steady}", specs.len());
    // Phase 1 ended with a fast run of 16 rows after an old-chain run of 16
    // (whose dirty rows it re-zeroed): every row is clean, so no memset at all.
    assert_eq!(m_steady, 0, "fast chain memset partials {m_steady} times in steady state");

    // 3. Errors mid-request, then a normal request.
    let mut n_err = 0;
    for (k, &j) in [5usize, 15, 22, 31, 40, 47].iter().enumerate() {
        let s = &specs[j];
        let fast_err = k % 3 != 2; // mostly the fast chain; one in three the old one
        let after_reduce = k % 2 == 1;
        // (a) after pass A of a two-pass (partials written, never reduced): the
        // request plus one expert that cannot be resident; (b) after the reduce,
        // GPU work still in flight: the request as is.
        run_spec(&mut exec, &mut shard, s, true, &mut ok)?; // its own picks resident now
        let mut bad = s.clone();
        if !after_reduce {
            let e_miss = (RANGE..384).find(|&e| !shard.is_resident_pool(LAYER, e)).unwrap() as i32;
            let slot = bad.sel.iter().position(|&e| e == NO_PICK).unwrap_or(0);
            bad.sel[slot] = e_miss;
            bad.ew[slot] = 0.5;
            if !bad.sel.iter().any(|&e| e >= 0 && e != e_miss) {
                continue; // no hit left for pass A
            }
        }
        let m_before = exec.fast_chain_memsets();
        let mut fired = false;
        let mut fail = |_: &mut ExpertShard, park: bool| -> eyre::Result<()> {
            if park != after_reduce && !fired {
                fired = true;
                return Err(eyre!("injected"));
            }
            Ok(())
        };
        knobs::set_fast_chain(fast_err);
        let r = exec.run_path(&mut shard, LAYER, bad.b, &bad.xq, &bad.sel, &bad.ew, true, &mut fail);
        let msg = format!("{:?}", r.as_ref().err());
        assert!(r.is_err() && msg.contains("injected"), "injection {k} did not fail as planned: {msg}");
        n_err += 1;
        let (o, t) = run_spec(&mut exec, &mut shard, s, true, &mut ok)?;
        same(&o, &refs[j], &format!("request {j} after an injected error ({}, fast={fast_err})", if after_reduce { "after reduce" } else { "after pass A" }));
        let m = exec.fast_chain_memsets() - m_before;
        eprintln!(
            "error {k}: {} chain, {} -> next request b={} bit-identical (two-pass {}, partials memsets {m})",
            if fast_err { "fast" } else { "old" },
            if after_reduce { "after the reduce" } else { "after pass A" },
            s.b,
            t.two_pass
        );
        if !after_reduce || !fast_err {
            assert!(m >= 1, "partials left dirty by the failed request were not re-zeroed");
        }
    }
    assert!(n_err >= 4, "only {n_err} injections ran");
    knobs::set_fast_chain(true);
    Ok(())
}

/// The fused builder and the zeroing reduce against the kernels they replace,
/// on synthetic inputs (no model).
#[test]
#[ignore]
fn fast_kernels_match_old_kernels() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let e = DeviceEngine::for_arch(igpu, &arch)?;
    let fk = B2FastChain::for_arch(&arch)?;
    let s = &e.compute;
    let id = igpu.id;
    let mut rng = Rng(0xb2b2_fa57_0000_0002);
    let n_rows = 256u32; // reduce row width (small: the reduce is per element)
    let mut cases = 0;
    for it in 0..96 {
        let b = 1 + it % 16;
        let mode = (it / 16 % 2) as u32;
        let cap = if it % 3 == 0 { 2 } else { NU as u32 };
        let gbound: u32 = 500;
        let maxpe = if it % 11 == 0 { 2 } else { b as u32 }; // overflow now and then
        // remap: resident on "this side" as -(slot)-1 (some slots past the bound),
        // dGPU-dense as >= 0, sentinel = 0.
        let remap: Vec<i32> = (0..REMAP_LEN)
            .map(|i| {
                if i as i32 == SENTINEL_EXPERT {
                    return 0;
                }
                match rng.below(4) {
                    0 => rng.below(u64::from(gbound)) as i32,
                    _ => -(rng.below(u64::from(gbound) + 20) as i32) - 1,
                }
            })
            .collect();
        let pool: Vec<i32> = (0..(4 + rng.below(20) as i32)).map(|_| rng.below(384) as i32).collect();
        let mut sel = vec![SENTINEL_EXPERT; b * NU];
        for t in 0..b {
            for sl in 0..NU {
                match rng.below(5) {
                    0 => sel[t * NU + sl] = NO_PICK,
                    1 => {}
                    _ => {
                        let v = pool[rng.below(pool.len() as u64) as usize];
                        // distinct within a row unless this case tests overflow
                        if maxpe == 2 || !sel[t * NU..t * NU + sl].contains(&v) {
                            sel[t * NU + sl] = v;
                        }
                    }
                }
            }
        }
        let total = b * NU;
        let mut d_sel = DeviceBuffer::<i32>::new(id, total)?;
        d_sel.copy_from_host(&sel)?;
        let mut d_remap = DeviceBuffer::<i32>::new(id, REMAP_LEN)?;
        d_remap.copy_from_host(&remap)?;
        let max_items = gbound + (ROWS * NU) as u32;
        // Old: memset + hetsplit builder + memset + work items.
        let mut gc_o = DeviceBuffer::<i32>::new(id, gbound as usize)?;
        let mut em_o = DeviceBuffer::<i32>::new(id, (gbound * maxpe) as usize)?;
        let mut wi_o = DeviceBuffer::<i32>::new(id, max_items as usize)?;
        let mut nwi_o = DeviceBuffer::<i32>::new(id, 1)?;
        gc_o.fill_zero_async(s)?;
        e.moe_group_builder.launch_hetsplit(s, &mut gc_o, &mut em_o, &d_sel, &d_remap, mode, cap, b as u32, NU as u32, gbound, maxpe)?;
        nwi_o.fill_zero_async(s)?;
        e.moe_group_builder.launch_work_items(s, &mut wi_o, &mut nwi_o, &gc_o, gbound, CHUNK_SIZE, max_items)?;
        // New: garbage everywhere first (it must not rely on any zeroing).
        let mut gc_n = DeviceBuffer::<i32>::new(id, gbound as usize)?;
        let mut em_n = DeviceBuffer::<i32>::new(id, (gbound * maxpe) as usize)?;
        let mut wi_n = DeviceBuffer::<i32>::new(id, max_items as usize)?;
        let mut nwi_n = DeviceBuffer::<i32>::new(id, 1)?;
        gc_n.copy_from_host(&vec![0x0777_7777; gbound as usize])?;
        em_n.copy_from_host(&vec![-5; (gbound * maxpe) as usize])?;
        wi_n.copy_from_host(&vec![-7; max_items as usize])?;
        nwi_n.copy_from_host(&[12345])?;
        fk.launch_builder(s, &mut gc_n, &mut em_n, &mut wi_n, &mut nwi_n, &d_sel, &d_remap, mode, cap, b as u32, NU as u32, gbound, maxpe, CHUNK_SIZE, max_items)?;
        s.synchronize()?;
        let rd = |d: &DeviceBuffer<i32>| -> eyre::Result<Vec<i32>> {
            let mut v = vec![0i32; d.len()];
            d.copy_to_host(&mut v)?;
            Ok(v)
        };
        let (gco, emo, wio, nwo) = (rd(&gc_o)?, rd(&em_o)?, rd(&wi_o)?, rd(&nwi_o)?[0]);
        let (gcn, emn, win, nwn) = (rd(&gc_n)?, rd(&em_n)?, rd(&wi_n)?, rd(&nwi_n)?[0]);
        assert_eq!(nwo, nwn, "case {it}: work-item count");
        let mut a: Vec<i32> = wio[..nwo as usize].to_vec();
        let mut bb: Vec<i32> = win[..nwn as usize].to_vec();
        a.sort_unstable();
        bb.sort_unstable();
        assert_eq!(a, bb, "case {it}: work items (as a set)");
        for g in 0..gbound as usize {
            if gco[g] == 0 {
                // Untouched: the new builder must not have written it, and no work item names it.
                assert_eq!(gcn[g], 0x0777_7777, "case {it}: untouched group {g} written");
                continue;
            }
            assert_eq!(gco[g], gcn[g], "case {it}: group {g} count");
            let c = (gco[g] as u32).min(maxpe) as usize;
            let mut mo: Vec<i32> = emo[g * maxpe as usize..g * maxpe as usize + c].to_vec();
            let mut mn: Vec<i32> = emn[g * maxpe as usize..g * maxpe as usize + c].to_vec();
            mn.windows(2).for_each(|w| assert!(w[0] < w[1], "case {it}: members not in (b, slot) order"));
            if gco[g] as u32 <= maxpe {
                mo.sort_unstable();
                mn.sort_unstable();
                assert_eq!(mo, mn, "case {it}: group {g} members");
            }
        }

        // Reduce: old hetsplit + cast vs the zeroing reduce (f32 + host-mapped f16).
        let np = total * n_rows as usize;
        let parts: Vec<f32> = (0..np)
            .map(|_| match rng.below(10) {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(rng.below(0x0080_0000) as u32), // denormal
                3 => (rng.below(2000) as f32 - 1000.0) * 70.0,       // f16 overflow range
                _ => (rng.below(200_000) as f32 - 100_000.0) * 1e-5,
            })
            .collect();
        let mut p_o = DeviceBuffer::<f32>::new(id, np)?;
        p_o.copy_from_host(&parts)?;
        let mut p_n = DeviceBuffer::<f32>::new(id, np)?;
        p_n.copy_from_host(&parts)?;
        let nout = b * n_rows as usize;
        let mut out_o = DeviceBuffer::<f32>::new(id, nout)?;
        let mut o16 = DeviceBuffer::<u16>::new(id, nout)?;
        e.q2k.launch_reduce_partials_hetsplit(s, &mut out_o, &p_o, &d_sel, &d_remap, mode, cap, NU as u32, n_rows, b as u32)?;
        e.q8k.launch_cast_f16(s, &mut o16, &out_o, nout as u32)?;
        let mut out_n = DeviceBuffer::<f32>::new(id, nout)?;
        let mut pin16 = PinnedBuffer::<u16>::new(nout)?;
        fk.launch_reduce_zero(s, &mut out_n, Some(&mut pin16), &mut p_n, &d_sel, &d_remap, mode, cap, NU as u32, n_rows, b as u32, 384)?;
        s.synchronize()?;
        let mut fo = vec![0f32; nout];
        out_o.copy_to_host(&mut fo)?;
        let mut fnn = vec![0f32; nout];
        out_n.copy_to_host(&mut fnn)?;
        let mut ho = vec![0u16; nout];
        o16.copy_to_host(&mut ho)?;
        assert!(fo.iter().zip(&fnn).all(|(x, y)| x.to_bits() == y.to_bits()), "case {it}: reduce f32 differs");
        assert!(ho == pin16.as_slice(), "case {it}: reduce f16 differs from f32_to_f16_cast");
        // Zeroed exactly: consumed slots and real-id slots; everything else untouched.
        let mut pn = vec![0f32; np];
        p_n.copy_to_host(&mut pn)?;
        for t in 0..b {
            let mut rank = 0u32;
            for sl in 0..NU {
                let ev = sel[t * NU + sl];
                let resident = ev >= 0 && remap[ev as usize] >= 0;
                let dgpu = resident && rank < cap;
                if resident {
                    rank += 1;
                }
                let take = if mode == 1 { dgpu } else { !dgpu };
                let zero = take || (0..384).contains(&ev);
                let base = (t * NU + sl) * n_rows as usize;
                for r in 0..n_rows as usize {
                    let want = if zero { 0f32.to_bits() } else { parts[base + r].to_bits() };
                    assert_eq!(pn[base + r].to_bits(), want, "case {it}: partial ({t},{sl}) row {r} zeroing");
                }
            }
        }
        cases += 1;
    }
    eprintln!("fast kernels: {cases} synthetic cases match (builder as sets incl. mode 1 / over-cap / bound drops / overflow; reduce f32 + f16 bit-exact; zeroing exact)");
    Ok(())
}
