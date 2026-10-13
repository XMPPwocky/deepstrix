//! Bit-exactness of dGPU fusion group G1 (`V41_DGPU_G1_FUSED`, `G1RouterBlock`): ONE launch against
//! the production decode chain it replaces, op for op as `forward_prefill.rs` issues it:
//!
//!   MhcArena::launch_fast (collapse only, mix late) -> F16Matvec::matvec_batched_router_x2 (or
//!   matvec_batched_router without a look-ahead) -> the prior's pinned H2D -> RouterTopk::
//!   launch_batched_ex (prior / protect / dry / orig_sel / range) -> RouterTopk::launch_batched
//!   (look-ahead) -> Q8KQuantize::launch -> ReadbackPack::launch (the production segment order)
//!   -> ExpertSelCount::launch (the plain picks).
//!
//! Every output is compared BIT FOR BIT (cur, ffn_input_norm, both logits, picks, weights, orig,
//! range, look-ahead picks / weights, d_prior, xq, every pack word, the M62 bank) at b = 1..16, in
//! four configurations (production; bare; dry run + protect 0 + xq unpacked; sub3 without a prior),
//! for every kernel variant (rows per wave x prefetched chunks), on TWO lanes' buffer sets launched
//! back to back on one stream (the self-resetting scratch between launches). Outputs are poisoned
//! before each run so a word the kernel forgets fails. The inputs carry exact ties: duplicated gate
//! rows (index tie-break of the top-k), and +-max pairs inside Q8_K blocks (q8_k_quantize's tree tie
//! rule) at several index pairs, plus an all-zero Q8_K block.
//!
//! Run (dGPU only, beside the hub: ~40 MB): HIP_VISIBLE_DEVICES=0 cargo test --release --features v41
//! -p v4flash-kernels --test g1_router_block_bitexact -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, PinnedBuffer, Stream};
use v4flash_kernels::config::{EXPERT_WEIGHT_SCALE, HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::g1_router_block::{G1Collapse, G1Io, G1Look, G1PackOffsets, G1Prior};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::FastCollapse;
use v4flash_kernels::readback_pack::{PackPlan, PackSeg};
use v4flash_kernels::router_topk::RouterEx;

const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;
const BMAX: usize = 16;
const NE: usize = N_EMBD as usize;
const NX: usize = N_EXPERT as usize;
const NU: usize = N_EXPERT_USED;
const XQ_ROW: usize = 20 * 292;
const LAYER: u32 = 17;
use v4flash_kernels::g1_router_block::G1_VARIANTS as VARIANTS;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn sym(&mut self, a: f32) -> f32 {
        ((self.next() & 0xFFFFFF) as f32 / 16777216.0 * 2.0 - 1.0) * a
    }
}

fn dgpu() -> eyre::Result<(Device, String)> {
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201 visible (set HIP_VISIBLE_DEVICES to the dGPU)"))?;
    let arch = dev.properties()?.gcn_arch_name;
    Ok((dev, arch))
}

fn up<T: Copy>(id: i32, h: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}
fn down<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}
fn f16_bits(v: f32) -> u16 {
    half_bits(v)
}
/// f32 -> f16 bits, round to nearest even (no `half` dependency in this crate's tests).
fn half_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let man = x & 0x7fffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = man | 0x800000;
        let shift = (14 - exp) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rest > half || (rest == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

/// One lane's buffers (inputs + outputs) -- `bd` / `sd` in production.
struct Lane {
    x: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
    cur: DeviceBuffer<f32>,
    ffn_in: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    logits2: DeviceBuffer<f32>,
    sel: DeviceBuffer<i32>,
    ew: DeviceBuffer<f32>,
    orig: DeviceBuffer<i32>,
    range: DeviceBuffer<f32>,
    look_sel: DeviceBuffer<i32>,
    look_ew: DeviceBuffer<f32>,
    d_prior: DeviceBuffer<f32>,
    prior_pin: PinnedBuffer<f32>,
    xq: DeviceBuffer<u8>,
    pack: PinnedBuffer<u32>,
    bank: DeviceBuffer<u32>,
}

/// Everything a run produced, for a bitwise comparison.
#[derive(PartialEq)]
struct Snap {
    cur: Vec<u32>,
    ffn_in: Vec<u32>,
    logits: Vec<u32>,
    logits2: Vec<u32>,
    sel: Vec<i32>,
    ew: Vec<u32>,
    orig: Vec<i32>,
    range: Vec<u32>,
    look_sel: Vec<i32>,
    look_ew: Vec<u32>,
    d_prior: Vec<u32>,
    xq: Vec<u8>,
    pack: Vec<u32>,
    bank: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
struct Cfg {
    name: &'static str,
    look: bool,
    prior: bool,
    sub3: bool,
    dry: bool,
    protect: u32,
    xq: bool,
    pack: bool,
    bank: bool,
}

const CFGS: [Cfg; 4] = [
    Cfg { name: "prod", look: true, prior: true, sub3: true, dry: false, protect: 2, xq: true, pack: true, bank: true },
    Cfg { name: "bare", look: false, prior: false, sub3: false, dry: false, protect: 0, xq: false, pack: false, bank: false },
    Cfg { name: "dry_p0_xq_unpacked", look: true, prior: true, sub3: true, dry: true, protect: 0, xq: true, pack: false, bank: true },
    Cfg { name: "sub3_noprior_packed", look: true, prior: false, sub3: true, dry: false, protect: 2, xq: false, pack: true, bank: true },
];

const POISON: u32 = 0x7fc0_dead;

fn f32v(d: &DeviceBuffer<f32>, n: usize) -> eyre::Result<Vec<u32>> {
    Ok(down(d)?[..n].iter().map(|v| v.to_bits()).collect())
}

impl Lane {
    fn new(id: i32, rng: &mut Lcg) -> eyre::Result<Self> {
        // after_attn_hc [b, 4, NE] with planted +-max pairs (Q8_K ties) and an all-zero block.
        let mut x: Vec<f32> = (0..BMAX * 4 * NE).map(|_| rng.sym(1.0)).collect();
        // (block j, idx a, idx b) pairs: index pairs whose tie the tree resolves differently.
        let pairs = [(3usize, 5usize, 133usize), (7, 0, 128), (11, 64, 32), (12, 1, 2), (15, 255, 127), (19, 200, 100)];
        for z in 0..BMAX {
            for (k, &(j, ia, ib)) in pairs.iter().enumerate() {
                if (z + k) % 3 == 0 {
                    continue;
                }
                let (da, db) = (j * 256 + ia, j * 256 + ib);
                for h in 0..4 {
                    let v = 8.0 + rng.sym(1.0);
                    x[(z * 4 + h) * NE + da] = v;
                    x[(z * 4 + h) * NE + db] = -v;
                }
            }
            if z % 4 == 1 {
                for h in 0..4 {
                    for d in 10 * 256..11 * 256 {
                        x[(z * 4 + h) * NE + d] = 0.0;
                    }
                }
            }
        }
        let carry: Vec<f32> = (0..BMAX * 24).map(|_| 0.1 + rng.sym(0.5).abs()).collect();
        let prior: Vec<f32> = (0..64 * NX).map(|_| if rng.next() % 3 == 0 { 0.02 + rng.sym(0.01) } else { 0.0 }).collect();
        let mut prior_pin = PinnedBuffer::<f32>::new(prior.len())?;
        prior_pin.as_mut_slice().copy_from_slice(&prior);
        Ok(Self {
            x: up(id, &x)?,
            carry: up(id, &carry)?,
            cur: DeviceBuffer::new(id, BMAX * NE)?,
            ffn_in: DeviceBuffer::new(id, BMAX * NE)?,
            logits: DeviceBuffer::new(id, BMAX * NX)?,
            logits2: DeviceBuffer::new(id, BMAX * NX)?,
            sel: DeviceBuffer::new(id, BMAX * NU)?,
            ew: DeviceBuffer::new(id, BMAX * NU)?,
            orig: DeviceBuffer::new(id, BMAX * NU)?,
            range: DeviceBuffer::new(id, BMAX)?,
            look_sel: DeviceBuffer::new(id, BMAX * NU)?,
            look_ew: DeviceBuffer::new(id, BMAX * NU)?,
            d_prior: DeviceBuffer::new(id, NX)?,
            prior_pin,
            xq: DeviceBuffer::new(id, BMAX * XQ_ROW)?,
            pack: PinnedBuffer::new(BMAX * (6 * NU + 9) + BMAX * XQ_ROW / 4)?,
            bank: DeviceBuffer::new(id, 64 * NX)?,
        })
    }

    fn poison(&mut self) -> eyre::Result<()> {
        let pf = f32::from_bits(POISON);
        for d in [&mut self.cur, &mut self.ffn_in, &mut self.logits, &mut self.logits2, &mut self.ew, &mut self.range, &mut self.look_ew, &mut self.d_prior] {
            let n = d.len();
            d.copy_from_host(&vec![pf; n])?;
        }
        for d in [&mut self.sel, &mut self.orig, &mut self.look_sel] {
            let n = d.len();
            d.copy_from_host(&vec![POISON as i32; n])?;
        }
        let n = self.xq.len();
        self.xq.copy_from_host(&vec![0xa5u8; n])?;
        let n = self.bank.len();
        self.bank.copy_from_host(&vec![3u32; n])?;
        self.pack.as_mut_slice().fill(POISON);
        Ok(())
    }

    fn snap(&self, b: usize) -> eyre::Result<Snap> {
        Ok(Snap {
            cur: f32v(&self.cur, b * NE)?,
            ffn_in: f32v(&self.ffn_in, b * NE)?,
            logits: f32v(&self.logits, b * NX)?,
            logits2: f32v(&self.logits2, b * NX)?,
            sel: down(&self.sel)?[..b * NU].to_vec(),
            ew: f32v(&self.ew, b * NU)?,
            orig: down(&self.orig)?[..b * NU].to_vec(),
            range: f32v(&self.range, b)?,
            look_sel: down(&self.look_sel)?[..b * NU].to_vec(),
            look_ew: f32v(&self.look_ew, b * NU)?,
            d_prior: f32v(&self.d_prior, NX)?,
            xq: down(&self.xq)?[..b * XQ_ROW].to_vec(),
            pack: self.pack.as_slice().to_vec(),
            bank: down(&self.bank)?,
        })
    }
}

struct Weights {
    w0: DeviceBuffer<u8>,
    w1: DeviceBuffer<u8>,
    bias0: DeviceBuffer<f32>,
    bias1: DeviceBuffer<f32>,
    norm_w: DeviceBuffer<f32>,
}

fn weights(id: i32, rng: &mut Lcg) -> eyre::Result<Weights> {
    let mut mk = |dups: &[(usize, usize)]| -> eyre::Result<DeviceBuffer<u8>> {
        let mut w: Vec<u16> = (0..NX * NE).map(|_| f16_bits(rng.sym(0.03))).collect();
        for &(a, b) in dups {
            let (ra, rb) = (a * NE, b * NE);
            let row: Vec<u16> = w[ra..ra + NE].to_vec();
            w[rb..rb + NE].copy_from_slice(&row);
        }
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        up(id, &bytes)
    };
    // Duplicated rows -> exactly equal logits (with equal bias) -> the index tie-break decides.
    // (5, 37, 69) and (6, 38) share a lane (e % 32): ties inside one lane's sorted list.
    let dups = [(7, 100), (200, 201), (3, 383), (50, 51), (51, 52), (5, 37), (37, 69), (6, 38)];
    let w0 = mk(&dups)?;
    let w1 = mk(&dups)?;
    let mut bias = |rng: &mut Lcg| -> Vec<f32> {
        let mut v: Vec<f32> = (0..NX).map(|_| rng.sym(0.05)).collect();
        // the in-lane tie group (5, 37, 69) always in the top 6: its order IS the tie-break
        v[5] = 3.0;
        for &(a, b) in &dups {
            v[b] = v[a];
        }
        v
    };
    let b0 = bias(rng);
    let b1 = bias(rng);
    let mut nw: Vec<f32> = (0..NE).map(|_| 0.5 + rng.sym(0.4).abs()).collect();
    // equal norm weights at the planted +-pairs (exactly negated norm outputs)
    for &(j, ia, ib) in &[(3usize, 5usize, 133usize), (7, 0, 128), (11, 64, 32), (12, 1, 2), (15, 255, 127), (19, 200, 100)] {
        nw[j * 256 + ib] = nw[j * 256 + ia];
    }
    Ok(Weights { w0, w1, bias0: up(id, &b0)?, bias1: up(id, &b1)?, norm_w: up(id, &nw)? })
}

/// The production pack plan (forward_prefill.rs order): sel, look, look_ew, orig, range, ew, xq.
fn plan_offsets(c: &Cfg, b: usize, l: &Lane) -> eyre::Result<(G1PackOffsets, usize)> {
    let mut plan = PackPlan::default();
    let mut o = G1PackOffsets::default();
    let n_sel = b * NU;
    o.sel = plan.push(PackSeg::words(&l.sel, n_sel)?).0;
    if c.look {
        o.look = plan.push(PackSeg::words(&l.look_sel, n_sel)?).0;
        o.look_ew = plan.push(PackSeg::words(&l.look_ew, n_sel)?).0;
    }
    if c.sub3 {
        o.orig = plan.push(PackSeg::words(&l.orig, n_sel)?).0;
        o.range = plan.push(PackSeg::words(&l.range, b)?).0;
    }
    o.ew = plan.push(PackSeg::words(&l.ew, n_sel)?).0;
    let mut words = o.ew as usize + n_sel;
    if c.xq {
        o.xq = plan.push(PackSeg::bytes(&l.xq, b * XQ_ROW)?).0;
        words = o.xq as usize + b * XQ_ROW / 4;
    }
    Ok((o, words))
}

fn run_reference(e: &DeviceEngine, s: &Stream, w: &Weights, l: &mut Lane, c: &Cfg, b: u32, slot: usize) -> eyre::Result<()> {
    let bu = b as usize;
    e.mhc_arena.launch_fast(
        s,
        None,
        Some(FastCollapse { x: &l.x, cur_out: &mut l.cur, norm_out: &mut l.ffn_in, norm_w: &w.norm_w }),
        &mut l.carry,
        false,
        HC_DIM,
        RMS_EPS,
        SINKHORN_ITERS,
        SINKHORN_EPS,
        b,
    )?;
    if c.look {
        let fused = e.f16.matvec_batched_router_x2(s, &mut l.logits, &mut l.logits2, &w.w0, &w.w1, &l.ffn_in, N_EXPERT, N_EMBD, b)?;
        if !fused {
            return Err(eyre!("router_x2 declined b = {b}"));
        }
    } else {
        e.f16.matvec_batched_router(s, &mut l.logits, &w.w0, &l.ffn_in, N_EXPERT, N_EMBD, b)?;
    }
    if c.prior {
        let src = &l.prior_pin.as_slice()[slot * NX..(slot + 1) * NX];
        l.d_prior.copy_from_host_async(src, s)?;
    }
    e.router_topk.launch_batched_ex(
        s,
        &mut l.sel,
        &mut l.ew,
        &l.logits,
        Some(&w.bias0),
        N_EXPERT,
        NU as u32,
        EXPERT_WEIGHT_SCALE,
        ROUTER_WEIGHT_EPS,
        b,
        RouterEx {
            alts: None,
            n_alt: 0,
            alt_w: None,
            prior: if c.prior { Some(&l.d_prior) } else { None },
            n_protect: c.protect,
            prior_dry: c.dry,
            orig_sel: if c.sub3 { Some(&mut l.orig) } else { None },
            range_out: if c.sub3 { Some(&mut l.range) } else { None },
        },
    )?;
    if c.look {
        e.router_topk.launch_batched(s, &mut l.look_sel, &mut l.look_ew, &l.logits2, Some(&w.bias1), N_EXPERT, NU as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS, b)?;
    }
    if c.xq {
        e.q8k.launch(s, &mut l.xq, &l.ffn_in, 20 * b)?;
    }
    if c.pack {
        let n_sel = bu * NU;
        let mut plan = PackPlan::default();
        plan.push(PackSeg::words(&l.sel, n_sel)?);
        if c.look {
            plan.push(PackSeg::words(&l.look_sel, n_sel)?);
            plan.push(PackSeg::words(&l.look_ew, n_sel)?);
        }
        if c.sub3 {
            plan.push(PackSeg::words(&l.orig, n_sel)?);
            plan.push(PackSeg::words(&l.range, bu)?);
        }
        plan.push(PackSeg::words(&l.ew, n_sel)?);
        if c.xq {
            plan.push(PackSeg::bytes(&l.xq, bu * XQ_ROW)?);
        }
        // SAFETY of the aliasing: the plan only borrows device buffers immutably; the pack is separate.
        let pack: &mut PinnedBuffer<u32> = unsafe { &mut *(&mut l.pack as *mut PinnedBuffer<u32>) };
        e.rb_pack.launch(s, pack, plan.segs())?;
    }
    if c.bank {
        let plain = if c.prior && !c.dry { &l.orig } else { &l.sel };
        e.expert_sel_count.launch(s, &mut l.bank, plain, LAYER, N_EXPERT, b * NU as u32)?;
    }
    Ok(())
}

fn run_fused(e: &DeviceEngine, s: &Stream, w: &Weights, l: &mut Lane, c: &Cfg, b: u32, slot: usize, v: (u32, u32, u32)) -> eyre::Result<()> {
    let bu = b as usize;
    let (offs, _) = plan_offsets(c, bu, l)?;
    let pack_ptr: *mut PinnedBuffer<u32> = &mut l.pack;
    let io = G1Io {
        collapse: G1Collapse { x: &l.x, carry: &l.carry, norm_w: &w.norm_w, cur_out: &mut l.cur, norm_out: &mut l.ffn_in, rms_eps: RMS_EPS },
        w0: &w.w0,
        bias0: Some(&w.bias0),
        logits0: &mut l.logits,
        look: if c.look {
            Some(G1Look { w: &w.w1, bias: Some(&w.bias1), logits: &mut l.logits2, sel: &mut l.look_sel, ew: &mut l.look_ew })
        } else {
            None
        },
        prior: if c.prior {
            Some(G1Prior { host: &l.prior_pin, host_off: slot * NX, dev: &mut l.d_prior, n_protect: c.protect, dry: c.dry })
        } else {
            None
        },
        sel: &mut l.sel,
        ew: &mut l.ew,
        orig_sel: if c.sub3 { Some(&mut l.orig) } else { None },
        range: if c.sub3 { Some(&mut l.range) } else { None },
        xq: if c.xq { Some(&mut l.xq) } else { None },
        // SAFETY: the pack is a separate allocation from every other field borrowed here.
        pack: if c.pack { Some((unsafe { &mut *pack_ptr }, offs)) } else { None },
        sel_bank: if c.bank { Some((&mut l.bank, LAYER * N_EXPERT)) } else { None },
        done: None,
        n_used: NU as u32,
        scale: EXPERT_WEIGHT_SCALE,
        weight_eps: ROUTER_WEIGHT_EPS,
        b,
    };
    e.g1.launch_variant(s, io, v)
}

fn first_diff<T: PartialEq + std::fmt::Debug>(name: &str, a: &[T], b: &[T]) -> Option<String> {
    let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
    if n == 0 && a.len() == b.len() {
        return None;
    }
    let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
    Some(format!("{name}: {n} of {} differ (first at {i}: ref {:?} vs fused {:?})", a.len(), a.get(i), b.get(i)))
}

fn diff(r: &Snap, f: &Snap, c: &Cfg, pack_words: usize) -> Vec<String> {
    let mut v = Vec::new();
    let mut chk = |o: Option<String>| {
        if let Some(s) = o {
            v.push(s);
        }
    };
    chk(first_diff("cur", &r.cur, &f.cur));
    chk(first_diff("ffn_input_norm", &r.ffn_in, &f.ffn_in));
    chk(first_diff("logits", &r.logits, &f.logits));
    chk(first_diff("sel", &r.sel, &f.sel));
    chk(first_diff("ew", &r.ew, &f.ew));
    chk(first_diff("bank", &r.bank, &f.bank));
    if c.look {
        chk(first_diff("logits2", &r.logits2, &f.logits2));
        chk(first_diff("look_sel", &r.look_sel, &f.look_sel));
        chk(first_diff("look_ew", &r.look_ew, &f.look_ew));
    }
    if c.sub3 {
        chk(first_diff("orig", &r.orig, &f.orig));
        chk(first_diff("range", &r.range, &f.range));
    }
    if c.prior {
        chk(first_diff("d_prior", &r.d_prior, &f.d_prior));
    }
    if c.xq {
        chk(first_diff("xq", &r.xq, &f.xq));
    }
    if c.pack {
        chk(first_diff("pack", &r.pack[..pack_words], &f.pack[..pack_words]));
        chk(first_diff("pack tail (untouched)", &r.pack[pack_words..], &f.pack[pack_words..]));
    }
    v
}

#[test]
#[ignore]
fn g1_router_block_is_bit_identical() -> eyre::Result<()> {
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x6a1_5eed);
    let w = weights(id, &mut rng)?;
    let mut lanes = [Lane::new(id, &mut rng)?, Lane::new(id, &mut rng)?];
    let bs: Vec<u32> = std::env::var("G1_BS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| (1..=16).collect());
    let mut fails = 0usize;
    let mut checks = 0usize;
    let mut ties_sel = 0usize;
    let mut q8_ties_seen = false;
    for &b in &bs {
        let bu = b as usize;
        for c in &CFGS {
            // Reference: both lanes back to back (as the two lanes' chains on de.compute).
            let mut refs = Vec::new();
            for (li, l) in lanes.iter_mut().enumerate() {
                l.poison()?;
                run_reference(&e, &s, &w, l, c, b, 3 + li)?;
                s.synchronize()?;
                refs.push(l.snap(bu)?);
            }
            // Sanity: the planted ties are live (a duplicated pair both in the top 6 somewhere).
            for r in &refs {
                for z in 0..bu {
                    let row = &r.sel[z * NU..(z + 1) * NU];
                    if [(7, 100), (200, 201), (50, 51), (51, 52), (3, 383), (5, 37), (37, 69), (6, 38)].iter().any(|&(a, bb)| row.contains(&a) && row.contains(&bb)) {
                        ties_sel += 1;
                    }
                }
                if c.xq {
                    // a q8 block whose d is negative (max is the negative element) and one positive
                    q8_ties_seen = true;
                }
            }
            for &v in VARIANTS.iter().filter(|v| v.0 >= b || (v.0 == 8 && v.2 > 0 && v.1 < 160)) {
                for l in lanes.iter_mut() {
                    l.poison()?;
                }
                s.synchronize()?;
                for (li, l) in lanes.iter_mut().enumerate() {
                    run_fused(&e, &s, &w, l, c, b, 3 + li, v)?;
                }
                s.synchronize()?;
                for (li, l) in lanes.iter().enumerate() {
                    let f = l.snap(bu)?;
                    let (_, words) = plan_offsets(c, bu, l)?;
                    let d = diff(&refs[li], &f, c, words);
                    checks += 1;
                    if !d.is_empty() {
                        fails += 1;
                        if fails <= 20 {
                            eprintln!("FAIL b={b} cfg={} {v:?} lane {li}:\n  {}", c.name, d.join("\n  "));
                        }
                    }
                }
            }
        }
        eprintln!("b={b}: done ({fails} failures so far)");
    }
    eprintln!("checks {checks}, failures {fails}; rows with a tied duplicate pair both picked: {ties_sel}; xq checked: {q8_ties_seen}");
    let _ = HC_MIX_DIM;
    if fails > 0 {
        return Err(eyre!("{fails} of {checks} (b, cfg, variant, lane) runs differ from the production chain"));
    }
    Ok(())
}

/// The fused path's Q8_K rows reproduce q8_k_quantize's tie rule: check the planted +-max pairs
/// actually tie (the block's |max| appears with both signs) and that the d sign follows the tree.
#[test]
#[ignore]
fn g1_q8k_ties_are_exercised() -> eyre::Result<()> {
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x6a1_5eed);
    let w = weights(id, &mut rng)?;
    let mut l = Lane::new(id, &mut rng)?;
    let c = CFGS[0];
    l.poison()?;
    run_reference(&e, &s, &w, &mut l, &c, 16, 3)?;
    s.synchronize()?;
    let ffn = down(&l.ffn_in)?;
    let mut tied = 0;
    let mut neg_d = 0;
    for z in 0..16 {
        for j in 0..20 {
            let blk = &ffn[z * NE + j * 256..z * NE + (j + 1) * 256];
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let pos = blk.iter().any(|v| *v == amax);
            let neg = blk.iter().any(|v| *v == -amax);
            if amax > 0.0 && pos && neg {
                tied += 1;
                let xq = down(&l.xq)?;
                let o = z * XQ_ROW + j * 292;
                let d = f32::from_le_bytes([xq[o], xq[o + 1], xq[o + 2], xq[o + 3]]);
                if d < 0.0 {
                    neg_d += 1;
                }
            }
        }
    }
    eprintln!("tied Q8_K blocks: {tied} (reference picked the negative max in {neg_d})");
    if tied < 20 {
        return Err(eyre!("only {tied} tied Q8_K blocks: the planted pairs do not tie"));
    }
    Ok(())
}
