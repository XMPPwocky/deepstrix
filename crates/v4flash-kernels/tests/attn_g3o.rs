//! G3-optimal (`AttnDecG3o`, kernels/attn_dec_g3o.hip) against the production chain it replaces
//! (meta fill -> gather -> fused attention -> rope_inv_quant, through the production wrappers in the
//! production order) and against round 1's one-launch `attn_dec_g3`.
//!
//! Synthetic data (no model load; ~100 MB of device memory: runs beside the live hub).
//!
//!   g3o_bit_identical_to_the_chain: heads / xq / xscale BIT FOR BIT, b = 1..16, two lanes that
//!     SHARE the score / sync scratch (as production's `sd`) launched back to back without a sync,
//!     three rope sets (plain, YaRN, YaRN with ext 0.5 / attn 1.3), every key layout (selected with
//!     and without per-row bases, dense with bases, dense shared, no compressed keys), steady, ragged
//!     and edge rows (1 key, 41 tiles, partial tiles on both sides), every arm (two-launch, ticketed,
//!     G3O_VARIANTS code objects); the ticketed scratch must be zero again after every launch and its
//!     error word clear; a negative control must differ.
//!   bench_g3o_chain_wall: the per-lane-layer wall (queued behind a spin: device time incl. every
//!     gap, plus an ~8 us event pair) of chain / chain#AA / g3 / g3o arms, per layer kind (reuse
//!     128 + 512 selected, index-source = reuse + the remaining meta launch, ratio-0 128 raw, short
//!     dense 128 + 200), cold (MALL flushed) and warm, random order per round; medians / p10 / p90 and
//!     paired differences.
//!   bench_g3o_floors: read bandwidth MALL-cold (DRAM) and L2-resident, the dependent-launch cost.
//!   g3o_trace: in-kernel phase timestamps per work-group (G3O_TRACE_B, default 4).
//!
//! `G3O_VARIANTS=a,b` (names under G3O_VDIR as v_<name>.hsaco, or paths) adds code objects.
//!
//!   HIP_VISIBLE_DEVICES=<dGPU> HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
//!   DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 cargo test --release --features v41 -p v4flash-kernels \
//!     --test attn_g3o -- --ignored --nocapture --test-threads=1
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::attention_dec::AttentionDec;
use v4flash_kernels::attn_dec_g3::{AttnDecG3, G3Comp, G3Rows};
use v4flash_kernels::attn_dec_g3o::{AttnDecG3o, G3oMode, G3O_DEFAULT_CFG, G3O_SYNC_ERR, G3O_SYNC_WORDS, G3O_TS_SLOTS};
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
    let (free, total) = Device::mem_info_current()?;
    eprintln!("dGPU {arch}: {} MB free of {} MB", free >> 20, total >> 20);
    if free < 300 * MB {
        return Err(eyre!("only {} MB free on the dGPU: not starting beside production", free >> 20));
    }
    Ok((dev, arch))
}

fn rope_params() -> [(&'static str, RopeParams); 3] {
    [
        ("plain", RopeParams { freq_base: 10000.0, freq_scale: 1.0, ext_factor: 0.0, attn_factor: 1.0,
                               beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 0 }),
        ("yarn", RopeParams { freq_base: 160000.0, freq_scale: 0.0625, ext_factor: 1.0, attn_factor: 1.0,
                              beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65536 }),
        ("yarn2", RopeParams { freq_base: 160000.0, freq_scale: 0.0625, ext_factor: 0.5, attn_factor: 1.3,
                               beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65536 }),
    ]
}

/// One new arm: a code object and the way it runs.
struct Arm {
    name: String,
    k: usize,
    mode: G3oMode,
    /// the production policy: `auto_mode` per case, round 1's kernel when it says None
    auto: bool,
}

struct Kernels {
    meta: AttnMetaFill,
    gather: IndexerGather,
    dec: AttentionDec,
    rope: RopeTail,
    g3: AttnDecG3,
    /// [0] = the built-in code object, then G3O_VARIANTS
    g3o: Vec<(String, AttnDecG3o)>,
    arms: Vec<Arm>,
}

impl Kernels {
    fn new(arch: &str, id: i32, s: &Stream) -> eyre::Result<Self> {
        let builtin = AttnDecG3o::for_arch(arch)?;
        let q = builtin.query(id, s)?;
        if q != G3O_DEFAULT_CFG {
            return Err(eyre!("built-in attn_dec_g3o reports {q:?}, the launcher assumes {G3O_DEFAULT_CFG:?}"));
        }
        let mut g3o = vec![("g3o".to_string(), builtin)];
        if let Ok(list) = std::env::var("G3O_VARIANTS") {
            let vdir = std::env::var("G3O_VDIR").unwrap_or_else(|_| "scratch/g3o".into());
            for p in list.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                let p = if p.contains('/') { p.to_string() } else { format!("{vdir}/v_{p}.hsaco") };
                let img = std::fs::read(&p).map_err(|e| eyre!("G3O_VARIANTS {p}: {e}"))?;
                let name = std::path::Path::new(&p).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let k = AttnDecG3o::from_image(&img, id, s)?;
                eprintln!("variant {name}: {:?}", k.cfg());
                g3o.push((name, k));
            }
        }
        let only: Option<Vec<String>> = std::env::var("G3O_MODES").ok().map(|v| v.split(',').map(|x| x.trim().to_string()).collect());
        let mut arms = Vec::new();
        for (i, (n, k)) in g3o.iter().enumerate() {
            for (suffix, mode) in [("2L", G3oMode::TwoLaunch), ("T", G3oMode::Ticketed),
                                   ("W1", G3oMode::Whole { parts: 1, warps: 16 }), ("W2", G3oMode::Whole { parts: 2, warps: 16 }),
                                   ("W4", G3oMode::Whole { parts: 4, warps: 16 }), ("X1", G3oMode::Whole { parts: 1, warps: 32 }),
                                   ("X2", G3oMode::Whole { parts: 2, warps: 32 }), ("Sonly", G3oMode::SOnly),
                                   ("Vonly", G3oMode::VOnly)] {
                // the timing-only arms run only when asked for by name
                if matches!(mode, G3oMode::SOnly | G3oMode::VOnly) && only.is_none() {
                    continue;
                }
                if !k.supports(mode) || only.as_ref().is_some_and(|o| !o.iter().any(|x| x == suffix)) {
                    continue;
                }
                arms.push(Arm { name: format!("{n}:{suffix}"), k: i, mode, auto: false });
            }
        }
        if i_want_auto(&only) {
            arms.push(Arm { name: "auto".into(), k: 0, mode: G3oMode::Whole { parts: 1, warps: 16 }, auto: true });
        }
        Ok(Self {
            meta: AttnMetaFill::for_arch(arch)?,
            gather: IndexerGather::for_arch(arch)?,
            dec: AttentionDec::for_arch(arch)?,
            rope: RopeTail::for_arch(arch)?,
            // G3O_G3_HSACO: another attn_dec_g3 code object for the round-1 arm (e.g. a G3_DBG_STOP
            // timing build)
            g3: match std::env::var("G3O_G3_HSACO") {
                Ok(p) => AttnDecG3::from_image(&std::fs::read(&p).map_err(|e| eyre!("G3O_G3_HSACO {p}: {e}"))?)?,
                Err(_) => AttnDecG3::for_arch(arch)?,
            },
            g3o,
            arms,
        })
    }
}

fn i_want_auto(only: &Option<Vec<String>>) -> bool {
    only.as_ref().map_or(true, |o| o.iter().any(|x| x == "auto"))
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
    n_idx_per: DeviceBuffer<i32>,
    heads: DeviceBuffer<f32>,
    xq: DeviceBuffer<i8>,
    xs: DeviceBuffer<f32>,
    heads2: DeviceBuffer<f32>,
    xq2: DeviceBuffer<i8>,
    xs2: DeviceBuffer<f32>,
    zero_row: DeviceBuffer<u16>,
}

impl Lane {
    fn new(id: i32, rng: &mut Lcg, q_amp: f32, kv_amp: f32) -> eyre::Result<Self> {
        let q: Vec<f32> = (0..BMAX * QF).map(|i| {
            let spike = if (i / HD as usize) % 23 == 0 { 6.0 } else { 1.0 };
            rng.sym(q_amp) * spike
        }).collect();
        let sinks: Vec<f32> = (0..NH).map(|_| rng.sym(3.0)).collect();
        // f16 KV with the edge values a transposition must carry exactly: subnormals (every 97th) and
        // signed zeros (every 89th); everything else uniform in +-kv_amp
        let mut kv_val = |i: usize, rng: &mut Lcg| -> u16 {
            if i % 97 == 13 {
                ((rng.below(1023) + 1) as u16) | if rng.below(2) == 1 { 0x8000 } else { 0 }
            } else if i % 89 == 7 {
                if rng.below(2) == 1 { 0x8000 } else { 0 }
            } else {
                f16_bits(rng.sym(kv_amp))
            }
        };
        let kv: Vec<u16> = (0..KV_ROWS * HD as usize).map(|i| kv_val(i, rng)).collect();
        let store: Vec<u16> = (0..STORE_ROWS * HD as usize).map(|i| kv_val(i, rng)).collect();
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
            n_idx_per: zeros(id, BMAX)?,
            heads: zeros(id, BMAX * QF)?,
            xq: zeros(id, BMAX * QF)?,
            xs: zeros(id, BMAX * QF / 32)?,
            heads2: zeros(id, BMAX * QF)?,
            xq2: zeros(id, BMAX * QF)?,
            xs2: zeros(id, BMAX * QF / 32)?,
            zero_row: zeros(id, HD as usize)?,
        })
    }
    fn poison(&mut self) -> eyre::Result<()> {
        self.heads2.copy_from_host(&vec![f32::NAN; BMAX * QF])?;
        self.xq2.copy_from_host(&vec![0x55i8; BMAX * QF])?;
        self.xs2.copy_from_host(&vec![f32::NAN; BMAX * QF / 32])?;
        Ok(())
    }
}

/// The score / sync scratch the lanes share (production: `sd`, one per device).
struct Shared {
    sc: DeviceBuffer<u16>,
    sync: DeviceBuffer<u32>,
    ts: DeviceBuffer<u64>,
}
impl Shared {
    fn new(id: i32) -> eyre::Result<Self> {
        Ok(Self {
            sc: zeros(id, AttnDecG3o::scratch_halves(BMAX))?,
            sync: zeros(id, G3O_SYNC_WORDS)?,
            ts: zeros(id, G3O_TS_SLOTS * 16)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// indexer fired / S2 reuse: the selection with per-row store bases (arena)
    Selected,
    /// the selection without bases (non-arena gather)
    SelectedNoBase,
    /// indexer did not fire, arena: per-row bases into the shared store
    DenseBase,
    /// indexer did not fire, no arena: every row reads the store from row 0
    DenseShared,
    /// ratio-0 layer: no compressed keys
    NoComp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Rows {
    /// fixed (n_raw, n_comp) on every row
    Steady(i32, i32),
    Ragged,
    /// tile-boundary / wide-window rows
    Edges,
}

const EDGES: [(i32, i32); 13] = [
    (1, 0), (1, 512), (129, 511), (127, 511), (128, 512), (16, 1), (17, 15), (600, 40), (625, 15),
    (640, 0), (512, 128), (33, 0), (113, 497),
];

struct Case {
    mode: Mode,
    n_raw: Vec<i32>,
    raw_off: Vec<i32>,
    n_comp: Vec<i32>,
}

fn make_case(lane: &mut Lane, rng: &mut Lcg, b: usize, mode: Mode, rows: Rows) -> eyre::Result<Case> {
    let mut n_raw = Vec::new();
    let mut raw_off = Vec::new();
    let mut n_comp = Vec::new();
    let mut base = Vec::new();
    let mut pos = Vec::new();
    let mut stream_base = 0i32;
    let mut stream_pos = 0i32;
    let e0 = rng.below(EDGES.len() as u32) as usize;
    for i in 0..b {
        if i % 4 == 0 {
            stream_base = rng.below((STORE_ROWS - 2 * TOPK as usize) as u32) as i32;
            stream_pos = [0, 448, 1_000_000, 70_000][rng.below(4) as usize] + rng.below(20_000) as i32;
        }
        let (nr, nc) = match rows {
            Rows::Steady(r, c) => (r, c),
            Rows::Ragged => (
                [1, 17, 100, 128, 128, 64][rng.below(6) as usize],
                [0, 1, 15, 33, 300, 512, 512, 511][rng.below(8) as usize],
            ),
            Rows::Edges => EDGES[(e0 + i) % EDGES.len()],
        };
        let nc = match mode {
            Mode::NoComp => 0,
            _ => nc,
        };
        let nr = if mode == Mode::NoComp && nr == 0 { 1 } else { nr };
        n_raw.push(nr);
        raw_off.push(rng.below((KV_ROWS - 700) as u32) as i32);
        n_comp.push(nc);
        base.push(stream_base);
        pos.push(stream_pos + (i % 4) as i32);
    }
    let mut sel = vec![-1i32; BMAX * TOPK as usize];
    for i in 0..b {
        let b0 = if mode == Mode::SelectedNoBase { 0 } else { base[i] };
        let span = (STORE_ROWS as i32 - b0).min(4 * TOPK as i32);
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
/// `index_source`: the meta fill also writes n_index_comp_per_b (4 destinations).
fn run_chain(k: &Kernels, s: &Stream, l: &mut Lane, c: &Case, params: &RopeParams, index_source: bool) -> eyre::Result<()> {
    let b = c.n_raw.len() as u32;
    k.meta.launch(s, [
        Some(MetaDst::new(&mut l.n_raw_per, &c.n_raw)?),
        Some(MetaDst::new(&mut l.n_raw_off_per, &c.raw_off)?),
        Some(MetaDst::new(&mut l.n_comp_per, &c.n_comp)?),
        if index_source { Some(MetaDst::new(&mut l.n_idx_per, &c.n_comp)?) } else { None },
    ])?;
    let sel_base = match c.mode {
        Mode::Selected => Some(&l.base),
        _ => None,
    };
    if matches!(c.mode, Mode::Selected | Mode::SelectedNoBase) {
        k.gather.launch_batched_rows(s, &mut l.active, &l.store, &l.sel, TOPK, HD, b, sel_base)?;
    }
    let n_total_max = c.n_raw.iter().zip(&c.n_comp).map(|(&r, &n)| (r + n) as u32).max().unwrap_or(0);
    let nrp = l.n_raw_per.slice_view(0, b as usize);
    let nrop = l.n_raw_off_per.slice_view(0, b as usize);
    let ncp = l.n_comp_per.slice_view(0, b as usize);
    let (comp, stride, base): (Option<&DeviceBuffer<u16>>, u32, Option<&DeviceBuffer<i32>>) = match c.mode {
        Mode::Selected | Mode::SelectedNoBase => (Some(&l.active), TOPK, None),
        Mode::DenseBase => (Some(&l.store), 0, Some(&l.base)),
        Mode::DenseShared => (Some(&l.store), 0, None),
        Mode::NoComp => (None, 0, None),
    };
    k.dec.launch_fused_f16s_rows(s, &mut l.heads, &l.sinks, &l.q, &l.kv, comp, &nrp, &nrop, &ncp, NH, HD,
                                 n_total_max, b, stride, base)?;
    k.rope.launch_inverse_quant_q8(s, &mut l.heads, &mut l.xq, &mut l.xs, &l.pos, NH, HD, 64, b, params)
}

fn comp_of<'a>(l: &'a Lane, c: &Case) -> G3Comp<'a> {
    match c.mode {
        Mode::Selected => G3Comp::Selected { store: &l.store, sel: &l.sel, sel_stride: TOPK, base: Some(&l.base) },
        Mode::SelectedNoBase => G3Comp::Selected { store: &l.store, sel: &l.sel, sel_stride: TOPK, base: None },
        Mode::DenseBase => G3Comp::Dense { store: &l.store, base: Some(&l.base), batch_stride: 0 },
        Mode::DenseShared => G3Comp::Dense { store: &l.store, base: None, batch_stride: 0 },
        Mode::NoComp => G3Comp::None,
    }
}

/// Round 1's one-launch G3 (+ the index-source layers' remaining 1-destination meta fill).
fn run_g3(k: &Kernels, s: &Stream, l: &mut Lane, c: &Case, params: &RopeParams, index_source: bool) -> eyre::Result<()> {
    if index_source {
        k.meta.launch(s, [None, None, None, Some(MetaDst::new(&mut l.n_idx_per, &c.n_comp)?)])?;
    }
    let comp = comp_of(l, c);
    let l2 = unsafe { &mut *(l as *const Lane as *mut Lane) };
    k.g3.launch(s, None, &mut l2.xq2, &mut l2.xs2, &l.zero_row, &l.sinks, &l.q, &l.kv, comp,
                G3Rows { n_raw: &c.n_raw, raw_off: &c.raw_off, n_comp: &c.n_comp }, &l.pos, NH, HD, 64, params)
}

#[allow(clippy::too_many_arguments)]
fn run_g3o(k: &Kernels, arm: &Arm, s: &Stream, l: &mut Lane, sh: &mut Shared, c: &Case, params: &RopeParams,
           with_heads: bool, index_source: bool, trace: bool) -> eyre::Result<()> {
    let mode = if arm.auto {
        match v4flash_kernels::attn_dec_g3o::auto_mode(&c.n_raw, &c.n_comp) {
            Some(m) => m,
            None => {
                // round 1's kernel (+ its index-source meta fill); heads_out only through G3's own path
                if with_heads {
                    let l2 = unsafe { &mut *(l as *const Lane as *mut Lane) };
                    let comp = comp_of(l, c);
                    if index_source {
                        k.meta.launch(s, [None, None, None, Some(MetaDst::new(&mut l2.n_idx_per, &c.n_comp)?)])?;
                    }
                    return k.g3.launch(s, Some(&mut l2.heads2), &mut l2.xq2, &mut l2.xs2, &l.zero_row, &l.sinks, &l.q,
                                       &l.kv, comp, G3Rows { n_raw: &c.n_raw, raw_off: &c.raw_off, n_comp: &c.n_comp },
                                       &l.pos, NH, HD, 64, params);
                }
                return run_g3(k, s, l, c, params, index_source);
            }
        }
    } else {
        arm.mode
    };
    if index_source {
        k.meta.launch(s, [None, None, None, Some(MetaDst::new(&mut l.n_idx_per, &c.n_comp)?)])?;
    }
    let comp = comp_of(l, c);
    let l2 = unsafe { &mut *(l as *const Lane as *mut Lane) };
    let heads = if with_heads { Some(&mut l2.heads2) } else { None };
    let ts = if trace { Some(&mut sh.ts) } else { None };
    let scratch = if matches!(mode, G3oMode::Whole { .. }) { None } else { Some((&mut sh.sc, &mut sh.sync)) };
    k.g3o[arm.k].1.launch(s, mode, heads, &mut l2.xq2, &mut l2.xs2, &l.zero_row, &l.sinks, &l.q, &l.kv, comp,
                          G3Rows { n_raw: &c.n_raw, raw_off: &c.raw_off, n_comp: &c.n_comp }, &l.pos, NH, HD, 64,
                          params, scratch, ts)
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

struct Ref {
    h: Vec<f32>,
    q: Vec<i8>,
    s: Vec<f32>,
}

fn compare(l: &Lane, r: &Ref, b: usize) -> eyre::Result<(usize, usize, usize, f32)> {
    let n = b * QF;
    let h = download(&l.heads2)?;
    let qn = download(&l.xq2)?;
    let sn = download(&l.xs2)?;
    let (dh, wh) = bits_diff_f32(&r.h[..n], &h[..n]);
    let dq = r.q[..n].iter().zip(&qn[..n]).filter(|(a, b)| a != b).count();
    let (ds, _) = bits_diff_f32(&r.s[..n / 32], &sn[..n / 32]);
    Ok((dh, dq, ds, wh))
}

#[test]
#[ignore]
fn g3o_bit_identical_to_the_chain() -> eyre::Result<()> {
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let k = Kernels::new(&arch, id, &s)?;
    let mut rng = Lcg(0x5eed_6a3);
    let mut lanes = [Lane::new(id, &mut rng, 0.5, 1.0)?, Lane::new(id, &mut rng, 2.0, 1.0)?];
    let mut sh = Shared::new(id)?;
    let zero_sync = vec![0u32; G3O_SYNC_WORDS];
    let mut total = 0usize;
    let mut bad = 0usize;
    let only_b: Option<Vec<usize>> = std::env::var("G3O_B").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    for (pname, params) in rope_params() {
        for mode in [Mode::Selected, Mode::SelectedNoBase, Mode::DenseBase, Mode::DenseShared, Mode::NoComp] {
            for rows in [Rows::Steady(128, 512), Rows::Ragged, Rows::Edges] {
                for b in 1..=BMAX {
                    if only_b.as_ref().is_some_and(|o| !o.contains(&b)) {
                        continue;
                    }
                    // both lanes get a case (lane 1 at 17 - b rows: the other lane of the step)
                    let bb = [b, (BMAX + 1 - b).max(1)];
                    let mut cases = Vec::new();
                    let mut refs = Vec::new();
                    for li in 0..2 {
                        let c = make_case(&mut lanes[li], &mut rng, bb[li], mode, rows)?;
                        run_chain(&k, &s, &mut lanes[li], &c, &params, false)?;
                        s.synchronize()?;
                        refs.push(Ref { h: download(&lanes[li].heads)?, q: download(&lanes[li].xq)?, s: download(&lanes[li].xs)? });
                        cases.push(c);
                    }
                    for arm in &k.arms {
                        for l in lanes.iter_mut() {
                            l.poison()?;
                        }
                        // lane 0, lane 1, lane 0 again: back to back on one stream, shared scratch
                        // lane 1 runs without heads_out, as production does (only xq / xscale compared)
                        for &li in &[0usize, 1, 0] {
                            let l = unsafe { &mut *(&mut lanes[li] as *mut Lane) };
                            run_g3o(&k, arm, &s, l, &mut sh, &cases[li], &params, li == 0, false, false)?;
                        }
                        s.synchronize()?;
                        for li in 0..2 {
                            let (dh, dq, ds, wh) = compare(&lanes[li], &refs[li], bb[li])?;
                            let dh = if li == 0 { dh } else { 0 };
                            total += 1;
                            if dh + dq + ds != 0 {
                                bad += 1;
                                eprintln!("MISMATCH {} rope={pname} mode={mode:?} rows={rows:?} b={} lane={li}: heads {dh} \
                                           (max |d| {wh:e}) xq {dq} xscale {ds}  n_raw {:?} n_comp {:?}",
                                          arm.name, bb[li], cases[li].n_raw, cases[li].n_comp);
                            }
                        }
                        if arm.mode == G3oMode::Ticketed {
                            let sy = download(&sh.sync)?;
                            if sy != zero_sync {
                                bad += 1;
                                let nz: Vec<(usize, u32)> = sy.iter().copied().enumerate().filter(|&(_, v)| v != 0).take(8).collect();
                                eprintln!("SCRATCH NOT RESET {} (err word {}): {nz:?}", arm.name, sy[G3O_SYNC_ERR]);
                                sh.sync.copy_from_host(&zero_sync)?;
                            }
                        }
                    }
                }
            }
        }
        eprintln!("rope {pname}: {total} comparisons so far, {bad} bad");
    }
    // negative control: a 1.001 scale on one lane's q must change the outputs of every arm
    {
        let c = make_case(&mut lanes[0], &mut rng, 4, Mode::Selected, Rows::Steady(128, 512))?;
        let params = rope_params()[1].1;
        run_chain(&k, &s, &mut lanes[0], &c, &params, false)?;
        s.synchronize()?;
        let r = Ref { h: download(&lanes[0].heads)?, q: download(&lanes[0].xq)?, s: download(&lanes[0].xs)? };
        let q0 = download(&lanes[0].q)?;
        let q1: Vec<f32> = q0.iter().enumerate().map(|(i, &v)| if i < QF { v * 1.001 } else { v }).collect();
        lanes[0].q.copy_from_host(&q1)?;
        for arm in &k.arms {
            lanes[0].poison()?;
            let l = unsafe { &mut *(&mut lanes[0] as *mut Lane) };
            run_g3o(&k, arm, &s, l, &mut sh, &c, &params, true, false, false)?;
            s.synchronize()?;
            let (dh, dq, ds, _) = compare(&lanes[0], &r, 4)?;
            eprintln!("negative control {}: heads {dh} xq {dq} xscale {ds} differ (must be > 0)", arm.name);
            if dh == 0 {
                bad += 1;
            }
        }
        lanes[0].q.copy_from_host(&q0)?;
    }
    eprintln!("attn_dec_g3o vs chain: {total} comparisons, {bad} bad (arms: {})",
              k.arms.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", "));
    if bad != 0 {
        return Err(eyre!("{bad} of {total} comparisons failed"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Benches
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
    pair_us: f64,
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
            pair_us: 0.0,
        };
        let mut t = Vec::new();
        for _ in 0..12 {
            t.push(r.time(false, &mut |_| Ok(()))?);
        }
        let t2000 = pct(&t, 0.5);
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
        let mut t = Vec::new();
        for _ in 0..24 {
            t.push(r.time(false, &mut |_| Ok(()))?);
        }
        r.pair_us = pct(&t, 0.5);
        eprintln!("rig: empty event pair {t2000:.1} us (behind the spin: {:.1}); spin 2000 iters = {per:.1} us -> {} iters",
                  r.pair_us, r.spin_iters);
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

/// (name, mode, rows, index_source)
fn kinds() -> Vec<(&'static str, Mode, Rows, bool)> {
    let all = vec![
        ("reuse", Mode::Selected, Rows::Steady(128, 512), false),
        ("index", Mode::Selected, Rows::Steady(128, 512), true),
        ("ratio0", Mode::NoComp, Rows::Steady(128, 0), false),
        ("short", Mode::DenseBase, Rows::Steady(128, 200), false),
    ];
    match std::env::var("G3O_KINDS") {
        Ok(v) => {
            let want: Vec<&str> = v.split(',').map(str::trim).collect();
            all.into_iter().filter(|k| want.contains(&k.0)).collect()
        }
        Err(_) => all,
    }
}

#[test]
#[ignore]
fn bench_g3o_chain_wall() -> eyre::Result<()> {
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let id = dev.id;
    let mut rig = Rig::new(id, &arch)?;
    let k = Kernels::new(&arch, id, &rig.s)?;
    let mut rng = Lcg(0xbe_4c4);
    let mut lane = Lane::new(id, &mut rng, 0.5, 1.0)?;
    let mut sh = Shared::new(id)?;
    let params = rope_params()[1].1;
    let rounds: usize = std::env::var("G3O_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
    let bs: Vec<usize> = std::env::var("G3O_B").ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16]);
    let colds: Vec<bool> = match std::env::var("G3O_COLD").as_deref() {
        Ok("1") => vec![true],
        Ok("0") => vec![false],
        _ => vec![true, false],
    };
    let no_chain = std::env::var("G3O_NO_CHAIN").as_deref() == Ok("1");
    // arm ids: 0 = chain, 1 = chain#AA, 2 = g3 (round 1), 3.. = g3o arms
    let mut names: Vec<String> = vec!["chain (meta+gather+attn+ropeq)".into(), "chain#AA".into(), "g3 (round 1)".into()];
    names.extend(k.arms.iter().map(|a| a.name.clone()));
    for (kname, mode, rows, index_source) in kinds() {
        for &b in &bs {
            let c = make_case(&mut lane, &mut rng, b, mode, rows)?;
            for &cold in &colds {
                let mut t: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
                for _ in 0..rounds {
                    let mut order: Vec<usize> = (0..names.len()).filter(|&a| !(no_chain && a < 2)).collect();
                    for i in (1..order.len()).rev() {
                        let j = rng.below(i as u32 + 1) as usize;
                        order.swap(i, j);
                    }
                    for &a in &order {
                        let l = unsafe { &mut *(&mut lane as *mut Lane) };
                        let shp = unsafe { &mut *(&mut sh as *mut Shared) };
                        let us = match a {
                            0 | 1 => rig.time(cold, &mut |s| run_chain(&k, s, l, &c, &params, index_source))?,
                            2 => rig.time(cold, &mut |s| run_g3(&k, s, l, &c, &params, index_source))?,
                            _ => {
                                let arm = &k.arms[a - 3];
                                rig.time(cold, &mut |s| run_g3o(&k, arm, s, l, shp, &c, &params, false, index_source, false))?
                            }
                        };
                        t[a].push(us);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                eprintln!("\n== {kname} b = {b} {} ({rounds} rounds, us, device time incl. gaps + the {:.1} us event pair) ==",
                          if cold { "COLD" } else { "warm" }, rig.pair_us);
                for (i, n) in names.iter().enumerate() {
                    if t[i].is_empty() {
                        continue;
                    }
                    eprintln!("  {n:<32} p10 {:7.1}  p50 {:7.1}  p90 {:7.1}", pct(&t[i], 0.1), pct(&t[i], 0.5), pct(&t[i], 0.9));
                }
                let paired = |x: usize, y: usize| -> (f64, f64, f64) {
                    let d: Vec<f64> = t[x].iter().zip(&t[y]).map(|(a, b)| a - b).collect();
                    (pct(&d, 0.25), pct(&d, 0.5), pct(&d, 0.75))
                };
                if !no_chain {
                    let (lo, md, hi) = paired(1, 0);
                    eprintln!("  A/A  chain#AA - chain           p50 {md:+7.1} (p25 {lo:+.1}, p75 {hi:+.1})");
                    for a in 2..names.len() {
                        let (lo, md, hi) = paired(a, 0);
                        eprintln!("  {:<31} p50 {md:+7.1} (p25 {lo:+.1}, p75 {hi:+.1})", format!("{} - chain", names[a]));
                    }
                }
                for a in 3..names.len() {
                    let (lo, md, hi) = paired(a, 2);
                    eprintln!("  {:<31} p50 {md:+7.1} (p25 {lo:+.1}, p75 {hi:+.1})", format!("{} - g3", names[a]));
                }
            }
        }
    }
    Ok(())
}

/// Floors in THIS process beside the hub: MALL-cold read bandwidth, L2-resident read bandwidth
/// (slope between 2 and 6 MB), the dependent-launch cost.
#[test]
#[ignore]
fn bench_g3o_floors() -> eyre::Result<()> {
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    let id = dev.id;
    let k = Kernels::new(&arch, id, &Stream::new(id)?)?;
    let mut rig = Rig::new(id, &arch)?;
    let rounds = 60;
    {
        let mut t = Vec::new();
        for _ in 0..rounds {
            let ceil = unsafe { &*(&rig.ceil as *const DeviceCeilings) };
            let buf = unsafe { &*(&rig.flush as *const DeviceBuffer<u8>) };
            let sink = unsafe { &mut *(&mut rig.sink as *mut DeviceBuffer<u32>) };
            t.push(rig.time(true, &mut |s| ceil.read_v4(s, buf, sink, 64 * 8, 256))?);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let us = pct(&t, 0.5) - rig.pair_us;
        eprintln!("read 80 MB MALL-cold: p50 {us:.1} us = {:.0} GB/s (p10 {:.0} GB/s)", FLUSH_BYTES as f64 / us / 1e3,
                  FLUSH_BYTES as f64 / (pct(&t, 0.1) - rig.pair_us) / 1e3);
    }
    {
        let mut res = Vec::new();
        for mb in [2usize, 6] {
            let buf: DeviceBuffer<u8> = zeros(id, mb * MB)?;
            let mut t = Vec::new();
            for _ in 0..rounds {
                let ceil = unsafe { &*(&rig.ceil as *const DeviceCeilings) };
                let sink = unsafe { &mut *(&mut rig.sink as *mut DeviceBuffer<u32>) };
                t.push(rig.time(false, &mut |s| {
                    ceil.read_v4(s, &buf, sink, 64 * 8, 256)?;
                    ceil.read_v4(s, &buf, sink, 64 * 8, 256)
                })?);
            }
            res.push(pct(&t, 0.5));
            eprintln!("read {mb} MB twice (L2-warm second pass): p50 {:.1} us", pct(&t, 0.5));
        }
        // slope of (cold-ish first + warm second) is not clean; report the warm-pass delta per MB
        eprintln!("  ~ L2 read: {:.0} GB/s from the 2 -> 6 MB slope (two passes)", 8.0 * MB as f64 / (res[1] - res[0]) / 1e3);
    }
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
    Ok(())
}

/// In-kernel phase timestamps (s_memrealtime-class steady counter, 100 MHz) of each arm, cold, one
/// reuse case: per role, the median over reps of [min, median, max] over work-groups of each stamp,
/// relative to the earliest WG start.
#[test]
#[ignore]
fn g3o_trace() -> eyre::Result<()> {
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let id = dev.id;
    let mut rig = Rig::new(id, &arch)?;
    let k = Kernels::new(&arch, id, &rig.s)?;
    let mut rng = Lcg(0x7ace);
    let mut lane = Lane::new(id, &mut rng, 0.5, 1.0)?;
    let mut sh = Shared::new(id)?;
    let params = rope_params()[1].1;
    let b: usize = std::env::var("G3O_TRACE_B").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let reps = 40;
    for (kname, mode, rows, _) in kinds() {
        let c = make_case(&mut lane, &mut rng, b, mode, rows)?;
        for arm in k.arms.iter().filter(|a| !a.auto) {
            let cfg = k.g3o[arm.k].1.cfg();
            let ns = b * (4 / cfg.hgp as usize) * cfg.nch as usize;
            let nv = b * 4 * cfg.dp as usize;
            // per stamp k: Vec over reps of (min, med, max) over S / V WGs
            let mut s_st: Vec<Vec<[f64; 3]>> = vec![Vec::new(); 16];
            let mut v_st: Vec<Vec<[f64; 3]>> = vec![Vec::new(); 16];
            let mut walls = Vec::new();
            for _ in 0..reps {
                sh.ts.copy_from_host(&vec![0u64; G3O_TS_SLOTS * 16])?;
                let l = unsafe { &mut *(&mut lane as *mut Lane) };
                let shp = unsafe { &mut *(&mut sh as *mut Shared) };
                walls.push(rig.time(true, &mut |s| run_g3o(&k, arm, s, l, shp, &c, &params, false, false, true))?);
                let ts = download(&sh.ts)?;
                let (s_slots, v_slots): (Vec<usize>, Vec<usize>) = match arm.mode {
                    G3oMode::TwoLaunch | G3oMode::SOnly | G3oMode::VOnly => ((0..ns).collect(), (1024..1024 + nv).collect()),
                    G3oMode::Ticketed => ((0..ns).collect(), (ns..ns + nv).collect()),
                    G3oMode::Whole { parts, .. } => ((0..4 * b * parts as usize).collect(), Vec::new()),
                };
                let t0 = s_slots.iter().chain(&v_slots).map(|&i| ts[i * 16]).filter(|&v| v != 0).min().unwrap_or(0);
                for (slots, st) in [(&s_slots, &mut s_st), (&v_slots, &mut v_st)] {
                    for kk in 0..16 {
                        let mut v: Vec<f64> = slots.iter().map(|&i| ts[i * 16 + kk]).filter(|&x| x != 0)
                            .map(|x| (x as f64 - t0 as f64) * 0.01).collect();
                        if v.is_empty() {
                            continue;
                        }
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        st[kk].push([v[0], v[v.len() / 2], v[v.len() - 1]]);
                    }
                }
            }
            eprintln!("\n== trace {kname} b={b} {} (cold, {reps} reps; us from the first WG start; * = landed only in G3O_TSW builds; wall p50 {:.1} - pair {:.1}) ==",
                      arm.name, pct(&walls, 0.5), rig.pair_us);
            let labels_s = ["start", "q in LDS (barrier)", "sel landed*", "K landed*", "chain done", "scores stored", "-", "end",
                            "-", "-", "-", "-", "-", "-", "-", "-"];
            let labels_v = ["start", "V issued (landed*)", "scores ready", "softmax done", "W.V done (wave 0)", "table built", "W.V done (all)", "end",
                            "-", "-", "-", "-", "-", "-", "-", "-"];
            for (role, st, labels) in [("S", &s_st, labels_s), ("V", &v_st, labels_v)] {
                for kk in 0..16 {
                    if st[kk].is_empty() {
                        continue;
                    }
                    let med = |j: usize| pct(&st[kk].iter().map(|x| x[j]).collect::<Vec<_>>(), 0.5);
                    eprintln!("  {role} {:<24} min {:6.2}  med {:6.2}  max {:6.2}", labels[kk], med(0), med(1), med(2));
                }
            }
        }
    }
    Ok(())
}

/// Load-pattern probe (scratch/g3o/probe.hip compiled to the code object named by G3O_PROBE): the
/// time one warp-tile of 16 KB costs in the WMMA fragment pattern, the transposed-load pattern and
/// a coalesced pattern, from an L2-resident 2.5 MB buffer (64 WGs x 16 warps, many tiles per warp).
#[test]
#[ignore]
fn g3o_load_probe() -> eyre::Result<()> {
    use v4flash_hip::{launch_kernel, LaunchConfig, Module};
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    let id = dev.id;
    let mut rig = Rig::new(id, &arch)?;
    let path = std::env::var("G3O_PROBE").map_err(|_| eyre!("set G3O_PROBE to the probe code object"))?;
    let module = Module::load_data(&std::fs::read(&path)?)?;
    let n_tiles = 160u32;
    let buf: DeviceBuffer<u16> = zeros(id, n_tiles as usize * 16 * 512)?;
    let mut sink: DeviceBuffer<u32> = zeros(id, 1024)?;
    for wgs in [32u32, 64, 128] {
        for name in ["probe_frag", "probe_tr", "probe_coal"] {
            let f = module.get_function(name)?;
            let mut res = Vec::new();
            for iters in [4u32, 20] {
                let mut t = Vec::new();
                for _ in 0..40 {
                    let sk = unsafe { &mut *(&mut sink as *mut DeviceBuffer<u32>) };
                    t.push(rig.time(false, &mut |s| {
                        let cfg = LaunchConfig { grid: (wgs, 1, 1), block: (512, 1, 1), shared_mem_bytes: 0 };
                        launch_kernel!(f, cfg, s, [buf.raw(), n_tiles, iters, sk.raw()])
                    })?);
                }
                res.push(pct(&t, 0.5));
            }
            let bytes = wgs as f64 * 16.0 * 16384.0 * 16.0;
            eprintln!("{name:<11} {wgs:>3} WGs: 4 iters {:.1} us, 20 iters {:.1} us -> {:.0} GB/s ({:.2} us per warp-tile)",
                      res[0], res[1], bytes / (res[1] - res[0]) / 1e3, (res[1] - res[0]) / 16.0);
        }
    }
    Ok(())
}

/// Dependent-load latency beside the hub (G3O_PROBE code object, probe_chase_c): one lane hops
/// through a random cyclic permutation -- L2-resident (256 KB, warm) and cold (32 MB after the
/// 80 MB flush).
#[test]
#[ignore]
fn g3o_latency_probe() -> eyre::Result<()> {
    use v4flash_hip::{launch_kernel, LaunchConfig, Module};
    install_panic_handler();
    let (dev, arch) = dgpu()?;
    let id = dev.id;
    let mut rig = Rig::new(id, &arch)?;
    let path = std::env::var("G3O_PROBE").map_err(|_| eyre!("set G3O_PROBE to the probe code object"))?;
    let module = Module::load_data(&std::fs::read(&path)?)?;
    let f = module.get_function("probe_chase_c")?;
    let mut rng = Lcg(0xc4a5e);
    let mut out: DeviceBuffer<u32> = zeros(id, 64)?;
    for (label, bytes, cold) in [("L2-warm 256 KB", 256usize << 10, false), ("cold 32 MB", 32usize << 20, true)] {
        // a single cycle over n slots spaced 128 B apart (one line per hop)
        let n = bytes / 128;
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.below(i as u32 + 1) as usize;
            order.swap(i, j);
        }
        let mut next = vec![0u32; bytes / 4];
        for k in 0..n {
            next[order[k] * 32] = (order[(k + 1) % n] * 32) as u32;
        }
        let buf = upload(id, &next)?;
        let mut res = Vec::new();
        for steps in [16u32, 80] {
            let mut t = Vec::new();
            for _ in 0..30 {
                let o = unsafe { &mut *(&mut out as *mut DeviceBuffer<u32>) };
                if !cold {
                    // warm the region first
                    let cfg = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
                    launch_kernel!(f, cfg, &rig.s, [buf.raw(), n as u32, o.raw()])?;
                }
                t.push(rig.time(cold, &mut |s| {
                    let cfg = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
                    launch_kernel!(f, cfg, s, [buf.raw(), steps, o.raw()])
                })?);
            }
            res.push(pct(&t, 0.5));
        }
        eprintln!("{label}: 16 hops {:.1} us, 80 hops {:.1} us -> {:.0} ns per dependent load", res[0], res[1],
                  (res[1] - res[0]) * 1e3 / 64.0);
    }
    Ok(())
}
