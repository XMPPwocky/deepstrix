//! G3 (dGPU fusion group "attention block"): `AttnDecG3::launch` (kernels/attn_dec_g3.hip, ONE launch)
//! against the production chain it replaces, launched through the production wrappers in the
//! production order (het/forward_prefill.rs, decode lanes of <= 16 rows):
//!
//!   AttnMetaFill::launch (n_raw_per, n_raw_offset_per, final n_comp_per)
//!   -> IndexerGather::launch_batched_rows (top-512 rows -> attn_active_comp_kv)
//!   -> AttentionDec::launch_fused_f16s_rows (score + softmax + W.V)
//!   -> RopeTail::launch_inverse_quant_q8 (inverse rope + Q8_0 of heads)
//!
//! Synthetic data (no model load, ~80 MB of device memory: runs beside the live hub).
//!
//!   attn_dec_g3_is_bit_identical_to_the_chain: heads / xq / xscale BIT FOR BIT, b = 1..16, both
//!     lanes' buffer sets, plain + YaRN rope, five key layouts (steady 128 + 512 selected, ragged
//!     rows incl. a 1-key row and partial tiles, dense store with per-row bases, dense shared store,
//!     no compressed keys).
//!   bench_attn_g3_chain_wall: the per-lane-layer chain WALL both ways (queued behind a spin, so it
//!     is device time incl. every inter-kernel gap), interleaved in random order per round, MALL
//!     flushed (cold) and warm, plus an A/A copy of the chain; medians / p10 / p90 / paired diffs.
//!   bench_attn_g3_floors: the floors in this process -- MALL-cold read bandwidth, the dependent
//!     launch cost, one WGP's f16 WMMA rate and the dependent-WMMA latency.
//!
//! `G3_VARIANTS=a.hsaco,b.hsaco` adds code objects with the same `attn_dec_g3` symbol as extra arms
//! (kernel iteration without a cargo rebuild); each is checked bit for bit before it is timed.
//!
//!   HIP_VISIBLE_DEVICES=<dGPU> HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
//!   DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 CARGO_TARGET_DIR=target-fuse nix develop -c cargo test \
//!     --release --features v41 -p v4flash-kernels --test attn_dec_g3_bitexact -- --ignored --nocapture --test-threads=1
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::attention_dec::AttentionDec;
use v4flash_kernels::attn_dec_g3::{AttnDecG3, G3Comp, G3Rows};
use v4flash_kernels::attn_meta::{AttnMetaFill, MetaDst};
use v4flash_kernels::device_ceilings::DeviceCeilings;
use v4flash_kernels::wmma_probe::WmmaProbe;
use v4flash_kernels::{IndexerGather, RopeParams, RopeTail};

const BMAX: usize = 16;
const NH: u32 = 64;
const HD: u32 = 512;
const QF: usize = (NH * HD) as usize; // 32768
const TOPK: u32 = 512;
const KV_ROWS: usize = 4096;
const STORE_ROWS: usize = 16384;
const MB: usize = 1 << 20;
const FLUSH_BYTES: usize = 80 * MB;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn sym(&mut self, a: f32) -> f32 {
        ((self.next() & 0xFFFFFF) as f32 / 16777216.0 * 2.0 - 1.0) * a
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n.max(1)
    }
}

fn f16_bits(v: f32) -> u16 {
    // round-to-nearest-even f32 -> f16 for the finite, normal-range values used here
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = v.abs();
    if a < 6.1e-5 {
        return sign;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}
fn zeros<T: Copy + Default>(id: i32, n: usize) -> eyre::Result<DeviceBuffer<T>> {
    upload(id, &vec![T::default(); n])
}
fn download<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

fn dgpu() -> eyre::Result<(Device, String)> {
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201 visible (set HIP_VISIBLE_DEVICES to the dGPU)"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    // beside the live hub: production holds most of the 16 GB
    let (free, total) = Device::mem_info_current()?;
    eprintln!("dGPU {arch}: {} MB free of {} MB", free >> 20, total >> 20);
    if free < 300 * MB {
        return Err(eyre!("only {} MB free on the dGPU: not starting beside production", free >> 20));
    }
    Ok((dev, arch))
}

fn rope_params() -> [(&'static str, RopeParams); 2] {
    [
        ("plain", RopeParams { freq_base: 10000.0, freq_scale: 1.0, ext_factor: 0.0, attn_factor: 1.0,
                               beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 0 }),
        // the V4.1 compressed-attention layers' YaRN parameters
        ("yarn", RopeParams { freq_base: 160000.0, freq_scale: 0.0625, ext_factor: 1.0, attn_factor: 1.0,
                              beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65536 }),
    ]
}

/// The production launchers this group touches.
struct Kernels {
    meta: AttnMetaFill,
    gather: IndexerGather,
    dec: AttentionDec,
    rope: RopeTail,
    g3: AttnDecG3,
    variants: Vec<(String, AttnDecG3)>,
}

impl Kernels {
    fn new(arch: &str) -> eyre::Result<Self> {
        let mut variants = Vec::new();
        if let Ok(list) = std::env::var("G3_VARIANTS") {
            let vdir = std::env::var("G3_VDIR").unwrap_or_else(|_| ".".into());
            for p in list.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                // a bare name is <G3_VDIR>/v_<name>.hsaco
                let p = if p.contains('/') { p.to_string() } else { format!("{vdir}/v_{p}.hsaco") };
                let p = p.as_str();
                let img = std::fs::read(p).map_err(|e| eyre!("G3_VARIANTS {p}: {e}"))?;
                let name = std::path::Path::new(p).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                variants.push((name, AttnDecG3::from_image(&img)?));
            }
        }
        Ok(Self {
            meta: AttnMetaFill::for_arch(arch)?,
            gather: IndexerGather::for_arch(arch)?,
            dec: AttentionDec::for_arch(arch)?,
            rope: RopeTail::for_arch(arch)?,
            g3: AttnDecG3::for_arch(arch)?,
            variants,
        })
    }
    /// (name, launcher) of every new arm: the built-in kernel first.
    fn arms(&self) -> Vec<(&str, &AttnDecG3)> {
        let mut v = vec![("g3", &self.g3)];
        v.extend(self.variants.iter().map(|(n, k)| (n.as_str(), k)));
        v
    }
}

/// One lane's buffer set (the production `sd` scratch + the layer's caches).
struct Lane {
    q: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    kv: DeviceBuffer<u16>,
    store: DeviceBuffer<u16>,
    sel: DeviceBuffer<i32>,
    base: DeviceBuffer<i32>,
    pos: DeviceBuffer<i32>,
    active: DeviceBuffer<u16>,
    n_raw_per: DeviceBuffer<i32>,
    n_raw_off_per: DeviceBuffer<i32>,
    n_comp_per: DeviceBuffer<i32>,
    heads: DeviceBuffer<f32>,
    xq: DeviceBuffer<i8>,
    xs: DeviceBuffer<f32>,
    heads2: DeviceBuffer<f32>,
    xq2: DeviceBuffer<i8>,
    xs2: DeviceBuffer<f32>,
    zero_row: DeviceBuffer<u16>,
}

impl Lane {
    fn new(id: i32, rng: &mut Lcg, q_amp: f32) -> eyre::Result<Self> {
        let q: Vec<f32> = (0..BMAX * QF).map(|i| {
            // a few spiky heads so some softmax rows are peaked, most are flat
            let spike = if (i / HD as usize) % 23 == 0 { 6.0 } else { 1.0 };
            rng.sym(q_amp) * spike
        }).collect();
        let sinks: Vec<f32> = (0..NH).map(|_| rng.sym(3.0)).collect();
        let kv: Vec<u16> = (0..KV_ROWS * HD as usize).map(|_| f16_bits(rng.sym(1.0))).collect();
        let store: Vec<u16> = (0..STORE_ROWS * HD as usize).map(|_| f16_bits(rng.sym(1.0))).collect();
        Ok(Self {
            q: upload(id, &q)?,
            sinks: upload(id, &sinks)?,
            kv: upload(id, &kv)?,
            store: upload(id, &store)?,
            sel: zeros(id, BMAX * TOPK as usize)?,
            base: zeros(id, BMAX)?,
            pos: zeros(id, BMAX)?,
            active: zeros(id, BMAX * TOPK as usize * HD as usize)?,
            n_raw_per: zeros(id, BMAX)?,
            n_raw_off_per: zeros(id, BMAX)?,
            n_comp_per: zeros(id, BMAX)?,
            heads: zeros(id, BMAX * QF)?,
            xq: zeros(id, BMAX * QF)?,
            xs: zeros(id, BMAX * QF / 32)?,
            heads2: zeros(id, BMAX * QF)?,
            xq2: zeros(id, BMAX * QF)?,
            xs2: zeros(id, BMAX * QF / 32)?,
            zero_row: zeros(id, HD as usize)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// indexer fired / S2 reuse: the selection through the gather (the decode common case)
    Selected,
    /// indexer did not fire, arena: per-row bases into the shared store
    DenseBase,
    /// indexer did not fire, no arena: every row reads the store from row 0
    DenseShared,
    /// ratio-0 layer: no compressed keys
    NoComp,
}

/// One case: per-row metadata + the selection / bases / positions written into the lane.
struct Case {
    mode: Mode,
    n_raw: Vec<i32>,
    raw_off: Vec<i32>,
    n_comp: Vec<i32>,
}

fn make_case(lane: &mut Lane, rng: &mut Lcg, b: usize, mode: Mode, ragged: bool) -> eyre::Result<Case> {
    let mut n_raw = Vec::new();
    let mut raw_off = Vec::new();
    let mut n_comp = Vec::new();
    let mut base = Vec::new();
    let mut pos = Vec::new();
    // rows come in streams of up to 4 consecutive positions (DSpark verify / two-stream lanes)
    let mut stream_base = 0i32;
    let mut stream_pos = 0i32;
    for i in 0..b {
        if i % 4 == 0 {
            stream_base = rng.below((STORE_ROWS - 2 * TOPK as usize) as u32) as i32;
            stream_pos = 448 + rng.below(200_000) as i32;
        }
        let (nr, nc) = if ragged {
            let nr = [1, 17, 100, 128, 128, 64][rng.below(6) as usize];
            let nc = [0, 1, 15, 33, 300, 512, 512, 511][rng.below(8) as usize];
            (nr, nc)
        } else {
            (128, 512)
        };
        let nc = match mode {
            Mode::NoComp => 0,
            _ => nc,
        };
        n_raw.push(nr);
        raw_off.push(rng.below((KV_ROWS - 200) as u32) as i32);
        n_comp.push(nc);
        base.push(stream_base);
        pos.push(stream_pos + (i % 4) as i32);
    }
    // the selection: distinct rows of the stream's store region, sentinels from n_comp on (top-K)
    let mut sel = vec![-1i32; BMAX * TOPK as usize];
    for i in 0..b {
        let span = (STORE_ROWS as i32 - base[i]).min(4 * TOPK as i32);
        let mut picked = std::collections::HashSet::new();
        for j in 0..n_comp[i] as usize {
            loop {
                let r = rng.below(span as u32) as i32;
                if picked.insert(r) {
                    sel[i * TOPK as usize + j] = r;
                    break;
                }
            }
        }
    }
    lane.sel.copy_from_host(&sel)?;
    let mut bpad = base.clone();
    bpad.resize(BMAX, 0);
    lane.base.copy_from_host(&bpad)?;
    let mut ppad = pos.clone();
    ppad.resize(BMAX, 0);
    lane.pos.copy_from_host(&ppad)?;
    Ok(Case { mode, n_raw, raw_off, n_comp })
}

/// The production chain (forward_prefill.rs: meta fill -> gather -> fused attention -> rope_inv_quant).
fn run_chain(k: &Kernels, s: &Stream, l: &mut Lane, c: &Case, params: &RopeParams) -> eyre::Result<()> {
    let b = c.n_raw.len() as u32;
    k.meta.launch(s, [
        Some(MetaDst::new(&mut l.n_raw_per, &c.n_raw)?),
        Some(MetaDst::new(&mut l.n_raw_off_per, &c.raw_off)?),
        Some(MetaDst::new(&mut l.n_comp_per, &c.n_comp)?),
        None,
    ])?;
    if c.mode == Mode::Selected {
        k.gather.launch_batched_rows(s, &mut l.active, &l.store, &l.sel, TOPK, HD, b, Some(&l.base))?;
    }
    let n_total_max = c.n_raw.iter().zip(&c.n_comp).map(|(&r, &n)| (r + n) as u32).max().unwrap_or(0);
    let nrp = l.n_raw_per.slice_view(0, b as usize);
    let nrop = l.n_raw_off_per.slice_view(0, b as usize);
    let ncp = l.n_comp_per.slice_view(0, b as usize);
    let (comp, stride, base): (Option<&DeviceBuffer<u16>>, u32, Option<&DeviceBuffer<i32>>) = match c.mode {
        Mode::Selected => (Some(&l.active), TOPK, None),
        Mode::DenseBase => (Some(&l.store), 0, Some(&l.base)),
        Mode::DenseShared => (Some(&l.store), 0, None),
        Mode::NoComp => (None, 0, None),
    };
    k.dec.launch_fused_f16s_rows(s, &mut l.heads, &l.sinks, &l.q, &l.kv, comp, &nrp, &nrop, &ncp, NH, HD,
                                 n_total_max, b, stride, base)?;
    k.rope.launch_inverse_quant_q8(s, &mut l.heads, &mut l.xq, &mut l.xs, &l.pos, NH, HD, 64, b, params)
}

fn run_g3(g3: &AttnDecG3, s: &Stream, l: &mut Lane, c: &Case, params: &RopeParams, with_heads: bool) -> eyre::Result<()> {
    let comp = match c.mode {
        Mode::Selected => G3Comp::Selected { store: &l.store, sel: &l.sel, sel_stride: TOPK, base: Some(&l.base) },
        Mode::DenseBase => G3Comp::Dense { store: &l.store, base: Some(&l.base), batch_stride: 0 },
        Mode::DenseShared => G3Comp::Dense { store: &l.store, base: None, batch_stride: 0 },
        Mode::NoComp => G3Comp::None,
    };
    let heads = if with_heads { Some(&mut l.heads2) } else { None };
    g3.launch(s, heads, &mut l.xq2, &mut l.xs2, &l.zero_row, &l.sinks, &l.q, &l.kv, comp,
              G3Rows { n_raw: &c.n_raw, raw_off: &c.raw_off, n_comp: &c.n_comp }, &l.pos, NH, HD, 64, params)
}

fn bits_diff_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut n = 0;
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            worst = worst.max((x - y).abs());
        }
    }
    (n, worst)
}

#[test]
#[ignore]
fn attn_dec_g3_is_bit_identical_to_the_chain() -> eyre::Result<()> {
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let id = dev.id;
    let k = Kernels::new(&arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5eed_6a3);
    let mut lanes = [Lane::new(id, &mut rng, 0.5)?, Lane::new(id, &mut rng, 2.0)?];
    let mut total = 0usize;
    let mut bad = 0usize;
    for (pname, params) in rope_params() {
        for mode in [Mode::Selected, Mode::DenseBase, Mode::DenseShared, Mode::NoComp] {
            for ragged in [false, true] {
                for b in 1..=BMAX {
                    let li = (b + ragged as usize) % 2;
                    let lane = &mut lanes[li];
                    let c = make_case(lane, &mut rng, b, mode, ragged)?;
                    // poison the outputs so a missed write shows
                    lane.heads2.copy_from_host(&vec![f32::NAN; BMAX * QF])?;
                    lane.xq2.copy_from_host(&vec![0x55i8; BMAX * QF])?;
                    lane.xs2.copy_from_host(&vec![f32::NAN; BMAX * QF / 32])?;
                    run_chain(&k, &s, lane, &c, &params)?;
                    s.synchronize()?;
                    let n = b * QF;
                    let h_ref = download(&lane.heads)?;
                    let q_ref = download(&lane.xq)?;
                    let s_ref = download(&lane.xs)?;
                    for (aname, g3) in k.arms() {
                        lane.heads2.copy_from_host(&vec![f32::NAN; BMAX * QF])?;
                        lane.xq2.copy_from_host(&vec![0x55i8; BMAX * QF])?;
                        lane.xs2.copy_from_host(&vec![f32::NAN; BMAX * QF / 32])?;
                        run_g3(g3, &s, lane, &c, &params, true)?;
                        s.synchronize()?;
                        let h = download(&lane.heads2)?;
                        let qn = download(&lane.xq2)?;
                        let sn = download(&lane.xs2)?;
                        let (dh, wh) = bits_diff_f32(&h_ref[..n], &h[..n]);
                        let dq = q_ref[..n].iter().zip(&qn[..n]).filter(|(a, b)| a != b).count();
                        let (ds, _) = bits_diff_f32(&s_ref[..n / 32], &sn[..n / 32]);
                        total += 1;
                        if dh + dq + ds != 0 {
                            bad += 1;
                            if bad <= 3 {
                                // where: per 32-dim block of the head (0..15), and a few samples
                                let mut blk = [0usize; 16];
                                let mut shown = 0;
                                for i in 0..n {
                                    if h_ref[i].to_bits() != h[i].to_bits() {
                                        blk[(i % 512) / 32] += 1;
                                        if shown < 6 {
                                            shown += 1;
                                            eprintln!("    row {} head {} dim {}: chain {:e} g3 {:e}", i / QF, (i % QF) / 512, i % 512, h_ref[i], h[i]);
                                        }
                                    }
                                }
                                eprintln!("    mismatches per 32-dim block: {blk:?}");
                            }
                            eprintln!("MISMATCH {aname} rope={pname} mode={mode:?} ragged={ragged} b={b} lane={li}: \
                                       heads {dh} (max |d| {wh:e}) xq {dq} xscale {ds}  n_raw {:?} n_comp {:?}",
                                      c.n_raw, c.n_comp);
                        }
                    }
                }
            }
        }
    }
    eprintln!("attn_dec_g3 vs chain: {total} comparisons, {bad} mismatching");
    if bad != 0 {
        return Err(eyre!("{bad} of {total} comparisons not bit-identical"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Chain-wall bench
// ---------------------------------------------------------------------------------------------
fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        return f64::NAN;
    }
    s[((s.len() - 1) as f64 * p).round() as usize]
}

struct Rig {
    s: Stream,
    ea: Event,
    eb: Event,
    probe: WmmaProbe,
    ceil: DeviceCeilings,
    flush: DeviceBuffer<u8>,
    sink: DeviceBuffer<u32>,
    spin_out: DeviceBuffer<f32>,
    spin_a: DeviceBuffer<f32>,
    spin_b: DeviceBuffer<f32>,
    spin_iters: u32,
}

impl Rig {
    fn new(id: i32, arch: &str) -> eyre::Result<Self> {
        let mut r = Self {
            s: Stream::new(id)?,
            ea: Event::new()?,
            eb: Event::new()?,
            probe: WmmaProbe::for_arch(arch)?,
            ceil: DeviceCeilings::for_arch(arch)?,
            flush: zeros(id, FLUSH_BYTES)?,
            sink: zeros(id, 1 << 16)?,
            spin_out: zeros(id, 32)?,
            spin_a: upload(id, &[1.0f32])?,
            spin_b: upload(id, &[0.999f32])?,
            spin_iters: 2000,
        };
        let mut t = Vec::new();
        for _ in 0..12 {
            t.push(r.time(false, &mut |_| Ok(()))?);
        }
        let t2000 = pct(&t, 0.5);
        // the spin only has to cover the host enqueue of one sample (~10 us): aim for ~40 us
        let mut ts = Vec::new();
        for _ in 0..12 {
            let st = unsafe { &*(&r.s as *const Stream) };
            r.ea.record(st)?;
            r.probe.launch_fma_f32(st, &mut r.spin_out, &r.spin_a, &r.spin_b, 2000, 1, 32)?;
            r.eb.record(st)?;
            st.synchronize()?;
            ts.push(Event::elapsed_ms(&r.ea, &r.eb)? as f64 * 1e3);
        }
        let per = pct(&ts, 0.5).max(1.0);
        r.spin_iters = ((2000.0 * 40.0 / per) as u32).clamp(200, 400_000);
        eprintln!("rig: empty event pair {t2000:.1} us; spin 2000 iters = {per:.1} us -> {} iters", r.spin_iters);
        Ok(r)
    }
    /// Device time of `f` queued behind a spin: [flush?] spin evA f evB.
    fn time(&mut self, cold: bool, f: &mut dyn FnMut(&Stream) -> eyre::Result<()>) -> eyre::Result<f64> {
        let st = unsafe { &*(&self.s as *const Stream) };
        if cold {
            self.ceil.read_v4(st, &self.flush, &mut self.sink, 64 * 8, 256)?;
        }
        self.probe.launch_fma_f32(st, &mut self.spin_out, &self.spin_a, &self.spin_b, self.spin_iters, 1, 32)?;
        self.ea.record(st)?;
        f(st)?;
        self.eb.record(st)?;
        st.synchronize()?;
        Ok(Event::elapsed_ms(&self.ea, &self.eb)? as f64 * 1e3)
    }
}

#[test]
#[ignore]
fn bench_attn_g3_chain_wall() -> eyre::Result<()> {
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let id = dev.id;
    let k = Kernels::new(&arch)?;
    let mut rig = Rig::new(id, &arch)?;
    let mut rng = Lcg(0xbe_4c4);
    let mut lane = Lane::new(id, &mut rng, 0.5)?;
    let params = rope_params()[1].1;
    let rounds: usize = std::env::var("G3_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
    let bs: Vec<usize> = std::env::var("G3_B").ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16]);
    let arms = k.arms();
    // arm ids: 0 = chain, 1 = chain#AA, 2 = attention kernel alone (prod), 3.. = new arms
    let names: Vec<String> = ["chain (meta+gather+attn+ropeq)", "chain#AA", "attn_dec_fused alone"]
        .iter().map(|s| s.to_string())
        .chain(arms.iter().map(|(n, _)| format!("{n} (1 launch)")))
        .collect();
    for &b in &bs {
        let c = make_case(&mut lane, &mut rng, b, Mode::Selected, false)?;
        // the chain once so `active` / n_*_per hold this case for the attention-alone arm
        run_chain(&k, &rig.s, &mut lane, &c, &params)?;
        rig.s.synchronize()?;
        for cold in [true, false] {
            let mut t: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
            for _ in 0..rounds {
                let mut order: Vec<usize> = (0..names.len()).collect();
                for i in (1..order.len()).rev() {
                    let j = rng.below(i as u32 + 1) as usize;
                    order.swap(i, j);
                }
                for &a in &order {
                    let l = unsafe { &mut *(&mut lane as *mut Lane) };
                    let us = match a {
                        0 | 1 => rig.time(cold, &mut |s| run_chain(&k, s, l, &c, &params))?,
                        2 => rig.time(cold, &mut |s| {
                            let n_total_max = 640;
                            let nrp = l.n_raw_per.slice_view(0, b);
                            let nrop = l.n_raw_off_per.slice_view(0, b);
                            let ncp = l.n_comp_per.slice_view(0, b);
                            k.dec.launch_fused_f16s_rows(s, &mut l.heads, &l.sinks, &l.q, &l.kv, Some(&l.active), &nrp,
                                                         &nrop, &ncp, NH, HD, n_total_max, b as u32, TOPK, None)
                        })?,
                        _ => {
                            let g3 = arms[a - 3].1;
                            rig.time(cold, &mut |s| run_g3(g3, s, l, &c, &params, false))?
                        }
                    };
                    t[a].push(us);
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            eprintln!("\n== b = {b}, {} ({rounds} rounds, us, device time incl. gaps) ==", if cold { "COLD (MALL flushed)" } else { "warm" });
            for (i, n) in names.iter().enumerate() {
                eprintln!("  {n:<34} p10 {:7.1}  p50 {:7.1}  p90 {:7.1}", pct(&t[i], 0.1), pct(&t[i], 0.5), pct(&t[i], 0.9));
            }
            let paired = |x: usize, y: usize| -> (f64, f64, f64) {
                let d: Vec<f64> = t[x].iter().zip(&t[y]).map(|(a, b)| a - b).collect();
                (pct(&d, 0.25), pct(&d, 0.5), pct(&d, 0.75))
            };
            let (lo, md, hi) = paired(1, 0);
            eprintln!("  A/A  chain#AA - chain            p50 {md:+7.1} (p25 {lo:+.1}, p75 {hi:+.1})");
            for a in 3..names.len() {
                let (lo, md, hi) = paired(a, 0);
                eprintln!("  {:<32} p50 {md:+7.1} (p25 {lo:+.1}, p75 {hi:+.1})", format!("{} - chain", names[a]));
            }
        }
    }
    Ok(())
}

/// The floors this group is measured against, in THIS process beside the hub (dGPU, production
/// kernarg mode): streaming read bandwidth (MALL-cold 80 MB), the dependent-launch cost (N empty
/// `attn_meta_fill` launches), the f16 WMMA rate of ONE WGP (one 512-thread WG, 8 independent
/// accumulators per warp) and the dependent-WMMA latency (one warp, one accumulator chain).
#[test]
#[ignore]
fn bench_attn_g3_floors() -> eyre::Result<()> {
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    let id = dev.id;
    let k = Kernels::new(&arch)?;
    let mut rig = Rig::new(id, &arch)?;
    let rounds = 40;
    // (1) read bandwidth: the whole 80 MB flush buffer (> 64 MB MALL) read twice back to back; the
    // second read is the one timed (its first half was evicted by its own second half).
    {
        let mut t = Vec::new();
        for _ in 0..rounds {
            let ceil = unsafe { &*(&rig.ceil as *const DeviceCeilings) };
            let buf = unsafe { &*(&rig.flush as *const DeviceBuffer<u8>) };
            let sink = unsafe { &mut *(&mut rig.sink as *mut DeviceBuffer<u32>) };
            t.push(rig.time(true, &mut |s| ceil.read_v4(s, buf, sink, 64 * 8, 256))?);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let us = pct(&t, 0.5);
        eprintln!("read 80 MB MALL-cold: p50 {us:.1} us = {:.0} GB/s (p10 {:.0} GB/s)", FLUSH_BYTES as f64 / us / 1e3,
                  FLUSH_BYTES as f64 / pct(&t, 0.1) / 1e3);
    }
    // (2) dependent launches: N x attn_meta_fill (1 WG, 32 threads, 4 B written) on one stream
    {
        let mut dst: DeviceBuffer<i32> = zeros(id, 16)?;
        let vals = [1i32];
        for n in [1usize, 17] {
            let mut t = Vec::new();
            for _ in 0..rounds {
                let d = unsafe { &mut *(&mut dst as *mut DeviceBuffer<i32>) };
                t.push(rig.time(false, &mut |s| {
                    for _ in 0..n {
                        k.meta.launch(s, [Some(MetaDst::new(d, &vals)?), None, None, None])?;
                    }
                    Ok(())
                })?);
            }
            eprintln!("{n:>2} dependent empty launches: p10 {:.1} p50 {:.1} us", pct(&t, 0.1), pct(&t, 0.5));
        }
    }
    // (3) WMMA: rate of one WGP and dependent latency of one warp
    {
        let a = upload(id, &vec![f16_bits(0.001); 16])?;
        let bb = upload(id, &vec![f16_bits(0.001); 16])?;
        let mut out: DeviceBuffer<f32> = zeros(id, 64 * 16 * 8)?;
        let iters = 2000u32;
        let mut tp = Vec::new();
        let mut tl = Vec::new();
        for _ in 0..rounds {
            let o = unsafe { &mut *(&mut out as *mut DeviceBuffer<f32>) };
            let probe = unsafe { &*(&rig.probe as *const WmmaProbe) };
            tp.push(rig.time(false, &mut |s| probe.launch_parallel(s, o, &a, &bb, iters, 1, 512))?);
            let o = unsafe { &mut *(&mut out as *mut DeviceBuffer<f32>) };
            tl.push(rig.time(false, &mut |s| probe.launch(s, o, &a, &bb, iters, 1, 32))?);
        }
        let n_wmma_wgp = 16.0 * 8.0 * iters as f64;
        let tpu = pct(&tp, 0.5);
        eprintln!("WMMA f16 16x16x16, ONE WGP (16 warps x 8 chains): {:.1} us for {} WMMAs = {:.2} ns per WMMA per SIMD \
                   ({:.1} TFLOP/s per WGP)", tpu, n_wmma_wgp, tpu * 1e3 / (n_wmma_wgp / 4.0), n_wmma_wgp * 8192.0 / tpu / 1e6);
        let tlu = pct(&tl, 0.5);
        eprintln!("WMMA dependent chain, one warp: {:.1} us / {iters} = {:.2} ns per dependent WMMA", tlu, tlu * 1e3 / iters as f64);
    }
    Ok(())
}
