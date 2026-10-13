//! Bit-exactness of the decode attention FRONT in two launches (`V41_DGPU_QKV_FRONT`, fusion
//! group G2, src/qkv_front.rs) against the production chain it replaces at decode rows 1..=8
//! (forward_prefill.rs q_chain + kv_chain + window append, dp4a arm, V41_DEC_FUSE):
//!
//!   q8.quantize_input_batched(attn_input_norm, 5120) -> q8.matvec_batched(q_a 1280 x 5120)
//!   -> rms_w.launch_weighted_quant_q8_1280 -> q8.matvec_batched(q_b 32768 x 1280)
//!   -> rope.launch_forward_batched_copy(q_normed := rope(q))
//!   -> q8.matvec_batched(kv 512 x 5120) -> rope.launch_kv_rms_rope_fp8 -> kv_append.launch_batched_rows
//!
//! vs `QkvFront::launch_a` + `launch_b`. Compared bit for bit: qr, qr_normed, qr_xq, qr_xscale,
//! kv_raw, q_normed (every row), and the whole f16 window cache (appended rows + untouched
//! sentinel rows). Synthetic Q8_0 weights (~55 MB, beside a live hub), plain and YaRN rope,
//! adversarial activations (exact zeros, an all-zero 32-block, spikes, tiny values), arena
//! slots (`slot_per`) and contiguous appends (`slot_base`), TWO lanes' buffer sets with their own
//! tickets interleaved on one stream, a ticket shared by consecutive launches, and every launch-A
//! grid the bench sweeps. Rows 9..=16 take the f16x q_a arm in production (SMALL_B_DENSE_MAX = 8)
//! and the new path falls back there: `launch_a` / `launch_b` must refuse them.
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test qkv_front_bitexact --
//! --ignored --test-threads=1 --nocapture` (HIP_VISIBLE_DEVICES=0 = the dGPU on box 1).
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::q8_0::Q8_0Matvec;
use v4flash_kernels::qkv_front::{QkvFront, MAX_B, MAX_GRID_ONE};
use v4flash_kernels::{KvCacheAppend, RmsNorm, RopeParams, RopeTail};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 device"))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn sym(&mut self, a: f32) -> f32 {
        ((self.next() & 0xFFFFFF) as f32 / 16777216.0 * 2.0 - 1.0) * a
    }
    fn val(&mut self, a: f32) -> f32 {
        match self.next() % 331 {
            0 => 0.0,
            1 => 2000.0 * a,
            2 => 1e-7 * a,
            _ => self.sym(a),
        }
    }
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}
fn download<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}
fn f16_bits(v: f32) -> u16 {
    half_bits(v)
}
/// f32 -> f16 bits (round to nearest even; finite inputs only, enough for scales).
fn half_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    let mut h = ((e as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// Q8_0 weight in the repacked row layout the GEMVs read: per row [f16 scale x blocks | i8 x 32 x blocks].
pub fn q8_weight(rng: &mut Lcg, rows: u32, k: u32) -> Vec<u8> {
    let blocks = (k / 32) as usize;
    let mut out = Vec::with_capacity(rows as usize * blocks * 34);
    for _ in 0..rows {
        for _ in 0..blocks {
            let s = 0.002 + (rng.next() % 1000) as f32 * 1e-5;
            let s = if rng.next() % 2 == 0 { s } else { -s };
            out.extend_from_slice(&f16_bits(s).to_le_bytes());
        }
        for _ in 0..blocks * 32 {
            out.push((rng.next() & 0xff) as u8);
        }
    }
    out
}

fn rope_params() -> [(&'static str, RopeParams); 2] {
    [
        ("plain", RopeParams { freq_base: 10000.0, freq_scale: 1.0, ext_factor: 0.0, attn_factor: 1.0,
                               beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 0 }),
        ("yarn", RopeParams { freq_base: 160000.0, freq_scale: 0.0625, ext_factor: 1.0, attn_factor: 1.0,
                              beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65536 }),
    ]
}

const CACHE_ROWS: usize = 96;

/// One lane's scratch (the buffers of `BatchScratch` the chain touches).
struct Lane {
    x: DeviceBuffer<f32>,
    xq: DeviceBuffer<i8>,
    xs: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    qr_normed: DeviceBuffer<f32>,
    qr_xq: DeviceBuffer<i8>,
    qr_xs: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    kv_raw: DeviceBuffer<f32>,
    kv_normed: DeviceBuffer<f32>,
    cache: DeviceBuffer<u16>,
    pos: DeviceBuffer<i32>,
    slots: DeviceBuffer<i32>,
    ticket: DeviceBuffer<u32>,
    epoch: u32,
}

impl Lane {
    fn new(id: i32, b: usize) -> eyre::Result<Self> {
        let bm = 8usize.max(b);
        Ok(Self {
            x: upload(id, &vec![0f32; bm * 5120])?,
            xq: upload(id, &vec![0x11i8; bm * 5120])?,
            xs: upload(id, &vec![f32::NAN; bm * 160])?,
            qr: upload(id, &vec![f32::NAN; bm * 1280])?,
            qr_normed: upload(id, &vec![f32::NAN; bm * 1280])?,
            qr_xq: upload(id, &vec![0x22i8; bm * 1280])?,
            qr_xs: upload(id, &vec![f32::NAN; bm * 40])?,
            q: upload(id, &vec![f32::NAN; bm * 32768])?,
            q_normed: upload(id, &vec![f32::NAN; bm * 32768])?,
            kv_raw: upload(id, &vec![f32::NAN; bm * 512])?,
            kv_normed: upload(id, &vec![f32::NAN; bm * 512])?,
            cache: upload(id, &vec![0x7c01u16; CACHE_ROWS * 512])?,
            pos: upload(id, &vec![0i32; bm])?,
            slots: upload(id, &vec![0i32; bm])?,
            ticket: upload(id, &[0u32; 4])?,
            epoch: 0,
        })
    }
    /// Fresh inputs for `b` rows; poisons the outputs.
    fn fill(&mut self, rng: &mut Lcg, b: usize, case: u32) -> eyre::Result<()> {
        let mut x: Vec<f32> = (0..b * 5120).map(|_| rng.val(1.5)).collect();
        // an all-zero 32-block (d = 0 path) and one all-tiny block
        let zb = (case as usize * 37) % 160;
        for e in 0..32 {
            x[zb * 32 + e] = 0.0;
        }
        let tb = (case as usize * 53 + 7) % 160;
        if b > 1 {
            for e in 0..32 {
                x[5120 + tb * 32 + e] = 1e-30 * (e as f32 - 16.0);
            }
        }
        self.x.copy_from_host(&{
            let mut v = x;
            v.resize(self.x.len(), 0.0);
            v
        })?;
        let pos: Vec<i32> = (0..self.pos.len()).map(|i| (1 + (70_001u64 * (i as u64 + 1 + case as u64)) % 300_000) as i32).collect();
        self.pos.copy_from_host(&pos)?;
        // distinct slots, scattered
        let mut slots: Vec<i32> = Vec::new();
        while slots.len() < self.slots.len() {
            let s = (rng.next() as usize % CACHE_ROWS) as i32;
            if !slots.contains(&s) {
                slots.push(s);
            }
        }
        self.slots.copy_from_host(&slots)?;
        self.qr.copy_from_host(&vec![f32::NAN; self.qr.len()])?;
        self.qr_normed.copy_from_host(&vec![f32::NAN; self.qr_normed.len()])?;
        self.qr_xq.copy_from_host(&vec![0x22i8; self.qr_xq.len()])?;
        self.qr_xs.copy_from_host(&vec![f32::NAN; self.qr_xs.len()])?;
        self.q_normed.copy_from_host(&vec![f32::NAN; self.q_normed.len()])?;
        self.kv_raw.copy_from_host(&vec![f32::NAN; self.kv_raw.len()])?;
        self.cache.copy_from_host(&vec![0x7c01u16; self.cache.len()])?;
        Ok(())
    }
}

struct W {
    qa: DeviceBuffer<u8>,
    kv: DeviceBuffer<u8>,
    qb: DeviceBuffer<u8>,
    qa_norm: DeviceBuffer<f32>,
    kv_norm: DeviceBuffer<f32>,
}

struct Mods {
    q8: Q8_0Matvec,
    rms: RmsNorm,
    rope: RopeTail,
    app: KvCacheAppend,
    qf: QkvFront,
}

const EPS: f32 = 1e-6;

/// The production chain (forward_prefill.rs, decode rows, dp4a arm, V41_DEC_FUSE + SKIP_DEAD).
fn chain(m: &Mods, s: &Stream, w: &W, l: &mut Lane, p: &RopeParams, b: u32, arena: bool, base: u32) -> eyre::Result<()> {
    m.q8.quantize_input_batched(s, &mut l.xq, &mut l.xs, &l.x, 5120, b)?;
    m.q8.matvec_batched(s, &mut l.qr, &w.qa, &l.xq, &l.xs, 1280, 5120, b)?;
    m.rms.launch_weighted_quant_q8_1280(s, &mut l.qr_normed, &mut l.qr_xq, &mut l.qr_xs, &l.qr, &w.qa_norm, EPS, b)?;
    m.q8.matvec_batched(s, &mut l.q, &w.qb, &l.qr_xq, &l.qr_xs, 32768, 1280, b)?;
    m.rope.launch_forward_batched_copy(s, &mut l.q_normed, &l.q, &l.pos, 64, 512, 64, b, p)?;
    m.q8.matvec_batched(s, &mut l.kv_raw, &w.kv, &l.xq, &l.xs, 512, 5120, b)?;
    m.rope.launch_kv_rms_rope_fp8(s, &mut l.kv_normed, &l.kv_raw, &w.kv_norm, EPS, &l.pos, 64, b, p)?;
    m.app.launch_batched_rows(s, &mut l.cache, &l.kv_normed, base, 512, b, if arena { Some(&l.slots) } else { None })
}

/// `QkvFront::launch_one` (`V41_DGPU_QKV_FRONT_ONE`).
fn fused_one(m: &Mods, s: &Stream, w: &W, l: &mut Lane, p: &RopeParams, b: u32, arena: bool, base: u32, grid: u32) -> eyre::Result<()> {
    l.epoch = l.epoch.wrapping_add(1).max(1);
    m.qf.launch_one(s, &mut l.qr, &mut l.kv_raw, &w.qa, &w.kv, &l.x, &mut l.qr_normed, &mut l.qr_xq, &mut l.qr_xs,
        &w.qa_norm, EPS, &mut l.ticket, l.epoch, &mut l.q_normed, &w.qb, &l.pos, p, &mut l.cache, &w.kv_norm,
        if arena { Some(&l.slots) } else { None }, base, b, grid)
}

fn fused(m: &Mods, s: &Stream, w: &W, l: &mut Lane, p: &RopeParams, b: u32, arena: bool, base: u32, grid: u32) -> eyre::Result<()> {
    m.qf.launch_a(s, &mut l.qr, &mut l.kv_raw, &w.qa, &w.kv, &l.x, &mut l.qr_normed, &mut l.qr_xq, &mut l.qr_xs,
        &w.qa_norm, EPS, &mut l.ticket, b, grid)?;
    m.qf.launch_b(s, &mut l.q_normed, &w.qb, &l.qr_xq, &l.qr_xs, &l.pos, p, &mut l.cache, &l.kv_raw, &w.kv_norm, EPS,
        if arena { Some(&l.slots) } else { None }, base, b)
}

struct Snap {
    qr: Vec<f32>,
    qr_normed: Vec<f32>,
    qr_xq: Vec<i8>,
    qr_xs: Vec<f32>,
    kv_raw: Vec<f32>,
    q_normed: Vec<f32>,
    cache: Vec<u16>,
}
fn snap(l: &Lane, b: usize) -> eyre::Result<Snap> {
    Ok(Snap {
        qr: download(&l.qr)?[..b * 1280].to_vec(),
        qr_normed: download(&l.qr_normed)?[..b * 1280].to_vec(),
        qr_xq: download(&l.qr_xq)?[..b * 1280].to_vec(),
        qr_xs: download(&l.qr_xs)?[..b * 40].to_vec(),
        kv_raw: download(&l.kv_raw)?[..b * 512].to_vec(),
        q_normed: download(&l.q_normed)?[..b * 32768].to_vec(),
        cache: download(&l.cache)?,
    })
}
fn fdiff(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}
fn compare(r: &Snap, f: &Snap, what: &str) -> eyre::Result<()> {
    let d = [
        ("qr", fdiff(&r.qr, &f.qr)),
        ("qr_normed", fdiff(&r.qr_normed, &f.qr_normed)),
        ("qr_xq", r.qr_xq.iter().zip(&f.qr_xq).filter(|(a, b)| a != b).count()),
        ("qr_xscale", fdiff(&r.qr_xs, &f.qr_xs)),
        ("kv_raw", fdiff(&r.kv_raw, &f.kv_raw)),
        ("q_normed", fdiff(&r.q_normed, &f.q_normed)),
        ("kv_cache", r.cache.iter().zip(&f.cache).filter(|(a, b)| a != b).count()),
    ];
    let bad: Vec<_> = d.iter().filter(|(_, n)| *n != 0).collect();
    if !bad.is_empty() {
        if let Some(i) = (0..r.cache.len()).find(|&i| r.cache[i] != f.cache[i]) {
            eprintln!("  kv_cache[{i}] (slot {} dim {}): chain {:#06x} fused {:#06x}", i / 512, i % 512, r.cache[i], f.cache[i]);
            for row in 0..r.kv_raw.len() / 512 {
                let d = i % 512;
                let blk = d / 32 * 32;
                let amax = (blk..blk + 32).map(|e| r.kv_raw[row * 512 + e].abs()).fold(0f32, f32::max);
                eprintln!("    row {row}: kv_raw[{d}] = {:e} ({:#010x}) block amax {:e}", r.kv_raw[row * 512 + d], r.kv_raw[row * 512 + d].to_bits(), amax);
            }
        }
        // first differing q_normed element, for diagnosis
        if let Some(i) = (0..r.q_normed.len()).find(|&i| r.q_normed[i].to_bits() != f.q_normed[i].to_bits()) {
            eprintln!("  q_normed[{i}] (row {} dim {}): chain {:e} fused {:e}", i / 32768, i % 32768, r.q_normed[i], f.q_normed[i]);
        }
        return Err(eyre!("{what}: differs {bad:?}"));
    }
    // the outputs must be real (a NaN poison surviving in both would hide a skipped write)
    let nan = r.q_normed.iter().filter(|v| v.is_nan()).count() + r.qr_normed.iter().filter(|v| v.is_nan()).count();
    if nan != 0 {
        return Err(eyre!("{what}: {nan} NaN outputs in the chain (poison not overwritten)"));
    }
    Ok(())
}

#[test]
#[ignore]
fn qkv_front_matches_chain() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let m = Mods {
        q8: Q8_0Matvec::for_arch(&arch)?,
        rms: RmsNorm::for_arch(&arch)?,
        rope: RopeTail::for_arch(&arch)?,
        app: KvCacheAppend::for_arch(&arch)?,
        qf: QkvFront::for_arch(&arch)?,
    };
    let mut rng = Lcg(0x6A2F_00D5);
    let w = W {
        qa: upload(id, &q8_weight(&mut rng, 1280, 5120))?,
        kv: upload(id, &q8_weight(&mut rng, 512, 5120))?,
        qb: upload(id, &q8_weight(&mut rng, 32768, 1280))?,
        qa_norm: upload(id, &(0..1280).map(|_| 0.2 + rng.sym(1.0).abs()).collect::<Vec<f32>>())?,
        kv_norm: upload(id, &(0..512).map(|_| 0.2 + rng.sym(1.0).abs()).collect::<Vec<f32>>())?,
    };
    // reference and fused copies of two lanes
    let mut ref_l = [Lane::new(id, 8)?, Lane::new(id, 8)?];
    let mut new_l = [Lane::new(id, 8)?, Lane::new(id, 8)?];
    let mut one_l = [Lane::new(id, 8)?, Lane::new(id, 8)?];
    let grids = [64u32, 32, 56, 96, 112, 224, 1, 7];
    let mut cases = 0;
    for b in 1..=MAX_B {
        let bu = b as usize;
        for (pi, (pname, p)) in rope_params().iter().enumerate() {
            for arena in [true, false] {
                let grid = grids[(b as usize + pi * 3 + arena as usize) % grids.len()];
                let case = b * 10 + pi as u32 * 2 + arena as u32;
                let base = if arena { 0 } else { (case * 7) % (CACHE_ROWS as u32 - b) };
                for lane in 0..2 {
                    let seed = rng.next() as u64 | ((rng.next() as u64) << 32);
                    let mut r1 = Lcg(seed);
                    let mut r2 = Lcg(seed);
                    ref_l[lane].fill(&mut r1, bu, case + lane as u32)?;
                    new_l[lane].fill(&mut r2, bu, case + lane as u32)?;
                    one_l[lane].fill(&mut Lcg(seed), bu, case + lane as u32)?;
                }
                let g1 = (grid % MAX_GRID_ONE).max(1);
                for lane in 0..2 {
                    fused_one(&m, &s, &w, &mut one_l[lane], p, b, arena, base, g1)?;
                }
                // lanes interleaved on one stream, as the two-lane decode issues them
                for lane in 0..2 {
                    chain(&m, &s, &w, &mut ref_l[lane], p, b, arena, base)?;
                }
                for lane in 0..2 {
                    let (l, rest) = new_l.split_at_mut(1);
                    let ln = if lane == 0 { &mut l[0] } else { &mut rest[0] };
                    m.qf.launch_a(&s, &mut ln.qr, &mut ln.kv_raw, &w.qa, &w.kv, &ln.x, &mut ln.qr_normed,
                        &mut ln.qr_xq, &mut ln.qr_xs, &w.qa_norm, EPS, &mut ln.ticket, b, grid)?;
                }
                for lane in 0..2 {
                    let ln = &mut new_l[lane];
                    m.qf.launch_b(&s, &mut ln.q_normed, &w.qb, &ln.qr_xq, &ln.qr_xs, &ln.pos, p, &mut ln.cache,
                        &ln.kv_raw, &w.kv_norm, EPS, if arena { Some(&ln.slots) } else { None }, base, b)?;
                }
                s.synchronize()?;
                for lane in 0..2 {
                    let what = format!("b={b} {pname} arena={arena} grid={grid} lane={lane}");
                    compare(&snap(&ref_l[lane], bu)?, &snap(&new_l[lane], bu)?, &what)?;
                    compare(&snap(&ref_l[lane], bu)?, &snap(&one_l[lane], bu)?, &format!("{what} ONE g={g1}"))?;
                    let t = download(&new_l[lane].ticket)?;
                    if t[0] != 0 {
                        return Err(eyre!("{what}: ticket left at {}", t[0]));
                    }
                    let t = download(&one_l[lane].ticket)?;
                    if t[0] != 0 || t[2] != 0 || t[3] != 0 || t[1] != one_l[lane].epoch {
                        return Err(eyre!("{what} ONE: ctl left at {t:?} (epoch {})", one_l[lane].epoch));
                    }
                    cases += 1;
                }
                // the same lane again with the same ticket (back-to-back launches reuse it)
                let seed = rng.next() as u64;
                ref_l[0].fill(&mut Lcg(seed), bu, case + 100)?;
                new_l[0].fill(&mut Lcg(seed), bu, case + 100)?;
                chain(&m, &s, &w, &mut ref_l[0], p, b, arena, base)?;
                fused(&m, &s, &w, &mut new_l[0], p, b, arena, base, grid)?;
                fused(&m, &s, &w, &mut new_l[0], p, b, arena, base, grid)?; // idempotent re-run
                one_l[0].fill(&mut Lcg(seed), bu, case + 100)?;
                fused_one(&m, &s, &w, &mut one_l[0], p, b, arena, base, g1)?;
                fused_one(&m, &s, &w, &mut one_l[0], p, b, arena, base, g1)?;
                s.synchronize()?;
                compare(&snap(&ref_l[0], bu)?, &snap(&new_l[0], bu)?, &format!("b={b} {pname} arena={arena} rerun"))?;
                compare(&snap(&ref_l[0], bu)?, &snap(&one_l[0], bu)?, &format!("b={b} {pname} arena={arena} ONE rerun"))?;
                cases += 1;
            }
        }
        eprintln!("b={b}: identical");
    }
    // every grid at b = 4 and b = 8
    for b in [4u32, 8] {
        for &grid in &grids {
            let seed = rng.next() as u64;
            ref_l[1].fill(&mut Lcg(seed), b as usize, 900 + grid)?;
            new_l[1].fill(&mut Lcg(seed), b as usize, 900 + grid)?;
            let p = &rope_params()[1].1;
            chain(&m, &s, &w, &mut ref_l[1], p, b, true, 0)?;
            fused(&m, &s, &w, &mut new_l[1], p, b, true, 0, grid)?;
            one_l[1].fill(&mut Lcg(seed), b as usize, 900 + grid)?;
            fused_one(&m, &s, &w, &mut one_l[1], p, b, true, 0, grid.min(MAX_GRID_ONE))?;
            s.synchronize()?;
            compare(&snap(&ref_l[1], b as usize)?, &snap(&new_l[1], b as usize)?, &format!("b={b} grid={grid}"))?;
            compare(&snap(&ref_l[1], b as usize)?, &snap(&one_l[1], b as usize)?, &format!("b={b} ONE grid={}", grid.min(MAX_GRID_ONE)))?;
            cases += 1;
        }
    }
    // rows the new path does not serve
    for b in [0u32, 9, 12, 16] {
        let l = &mut new_l[0];
        let ra = m.qf.launch_a(&s, &mut l.qr, &mut l.kv_raw, &w.qa, &w.kv, &l.x, &mut l.qr_normed, &mut l.qr_xq,
            &mut l.qr_xs, &w.qa_norm, EPS, &mut l.ticket, b, 64);
        assert!(ra.is_err(), "launch_a must refuse b={b}");
    }
    eprintln!("qkv_front: {cases} cases bit-identical, two-launch AND one-launch (b = 1..={MAX_B}, plain + YaRN, arena + contiguous, 2 lanes)");
    Ok(())
}
