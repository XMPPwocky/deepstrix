//! dGPU fusion group G1 (kernels/g1_router_block.hip, knob `V41_DGPU_G1_FUSED`, default off): the
//! decode lane-layer's router block -- pre-ffn collapse, the router matvecs of layer l and its k1
//! look-ahead, the cache prior's upload, both top-ks, box 2's Q8_K rows, the readback pack and the
//! M62 pick counts -- as ONE launch, bit-identical to the nine queue operations it replaces
//! (tests/g1_router_block_bitexact.rs). See the kernel's header for the roles and the proof of
//! forward progress (ticketed workgroups).
//!
//! The kernel keeps a small self-resetting scratch (`sync`, zero between launches): launches that
//! share one [`G1RouterBlock`] must be STREAM-ORDERED (production: `de.compute` only).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, PinnedBuffer, Stream};

const G1_GFX1201: &[u8] = include_bytes!(env!("KERNEL_G1_ROUTER_BLOCK_GFX1201"));
const G1_GFX1151: &[u8] = include_bytes!(env!("KERNEL_G1_ROUTER_BLOCK_GFX1151"));

/// Rows per launch (the kernel's `G1_MAXB`).
pub const G1_MAX_B: u32 = 16;
/// Router rows per matrix the kernel is built for (V4.1 `N_EXPERT`).
pub const G1_N_EXPERT: u32 = 384;
/// Picks per row the kernel supports.
pub const G1_MAX_USED: u32 = 8;
const G1_RWG: u32 = 48; // router WGs per matrix and batch slice (384 rows / 8 waves)
const S_WORDS: usize = 32 * (5 + G1_MAX_B as usize); // one 128-byte line per word (the kernel's S_*)
const XQ_ROW_BYTES: usize = 20 * 292;
/// A pack segment the launch does not write.
pub const PK_NONE: u32 = u32::MAX;

/// The kernel's `G1Args`, field for field (pointers first, then 32-bit scalars).
#[repr(C)]
#[derive(Clone, Copy)]
struct G1ArgsRaw {
    cx: u64,
    carry: u64,
    norm_w: u64,
    cur_out: u64,
    norm_out: u64,
    w0: u64,
    w1: u64,
    logits0: u64,
    logits1: u64,
    bias0: u64,
    bias1: u64,
    prior_host: u64,
    prior_dev: u64,
    sel: u64,
    ew: u64,
    orig_sel: u64,
    range_out: u64,
    look_sel: u64,
    look_ew: u64,
    xq: u64,
    pack: u64,
    sel_bank: u64,
    sync: u64,
    scr: u64,
    done_word: u64,
    ts: u64,
    b: u32,
    n_used: u32,
    n_protect: u32,
    prior_dry: u32,
    bank_off: u32,
    zb: u32,
    n_mat: u32,
    done_value: u32,
    pk_sel: u32,
    pk_look: u32,
    pk_look_ew: u32,
    pk_orig: u32,
    pk_range: u32,
    pk_ew: u32,
    pk_xq: u32,
    rms_eps: f32,
    scale: f32,
    weight_eps: f32,
    n_pf: u32,
    dbg: u32,
}

/// The pre-ffn collapse's operands (`mhc_arena::FastCollapse` + the carry).
pub struct G1Collapse<'a> {
    /// [b, 4 * n_embd] after_attn_hc.
    pub x: &'a DeviceBuffer<f32>,
    /// [b, 24] the previous sub-block's pre-weights (`hc_pre_carry`).
    pub carry: &'a DeviceBuffer<f32>,
    pub norm_w: &'a DeviceBuffer<f32>,
    pub cur_out: &'a mut DeviceBuffer<f32>,
    /// [b, n_embd] ffn_input_norm.
    pub norm_out: &'a mut DeviceBuffer<f32>,
    pub rms_eps: f32,
}

/// The k1 look-ahead (layer l+1's router on this layer's input).
pub struct G1Look<'a> {
    pub w: &'a DeviceBuffer<u8>,
    pub bias: Option<&'a DeviceBuffer<f32>>,
    pub logits: &'a mut DeviceBuffer<f32>,
    pub sel: &'a mut DeviceBuffer<i32>,
    pub ew: &'a mut DeviceBuffer<f32>,
}

/// The cache prior (`V41_SUB=3`): `host` = this layer's pinned slot, copied by the kernel to `dev`.
pub struct G1Prior<'a> {
    pub host: &'a PinnedBuffer<f32>,
    /// Element offset of the slot in `host`.
    pub host_off: usize,
    pub dev: &'a mut DeviceBuffer<f32>,
    pub n_protect: u32,
    pub dry: bool,
}

/// Pack word offsets (`PK_NONE` = not packed), as `PackPlan::push` laid them out.
#[derive(Clone, Copy, Debug)]
pub struct G1PackOffsets {
    pub sel: u32,
    pub look: u32,
    pub look_ew: u32,
    pub orig: u32,
    pub range: u32,
    pub ew: u32,
    pub xq: u32,
}

impl Default for G1PackOffsets {
    fn default() -> Self {
        Self { sel: PK_NONE, look: PK_NONE, look_ew: PK_NONE, orig: PK_NONE, range: PK_NONE, ew: PK_NONE, xq: PK_NONE }
    }
}

pub struct G1Io<'a> {
    pub collapse: G1Collapse<'a>,
    /// [384, n_embd] f16 router gate of this layer.
    pub w0: &'a DeviceBuffer<u8>,
    pub bias0: Option<&'a DeviceBuffer<f32>>,
    pub logits0: &'a mut DeviceBuffer<f32>,
    pub look: Option<G1Look<'a>>,
    pub prior: Option<G1Prior<'a>>,
    pub sel: &'a mut DeviceBuffer<i32>,
    pub ew: &'a mut DeviceBuffer<f32>,
    /// Written when given (`sub3`).
    pub orig_sel: Option<&'a mut DeviceBuffer<i32>>,
    pub range: Option<&'a mut DeviceBuffer<f32>>,
    /// Box 2's Q8_K rows ([b, 20 x 292] bytes) when given.
    pub xq: Option<&'a mut DeviceBuffer<u8>>,
    /// The pinned readback pack and where each segment lands.
    pub pack: Option<(&'a mut PinnedBuffer<u32>, G1PackOffsets)>,
    /// M62 bank and `layer * n_expert`.
    pub sel_bank: Option<(&'a mut DeviceBuffer<u32>, u32)>,
    /// Pinned completion word and the value to post there after everything else is written.
    pub done: Option<(&'a PinnedBuffer<u32>, usize, u32)>,
    pub n_used: u32,
    pub scale: f32,
    pub weight_eps: f32,
    pub b: u32,
}

pub struct G1RouterBlock {
    module: Module,
    sync: DeviceBuffer<u32>,
    /// The router waves' per-expert probabilities and scores for the selection tails.
    scr: DeviceBuffer<f32>,
    ts: Option<DeviceBuffer<u64>>,
}

/// The kernel variants (zb, ch, phe): ZB = rows per router wave (>= b: one wave per weight row covers
/// every batch row, so each weight byte is read once and the launch is at most 113 WGs, all
/// resident). phe = 0: the register form `g1_router_block_z{zb}_c{ch}` (x read from L2 by every
/// wave, CH weight elements per lane per chunk); phe > 0: the phased-LDS form
/// `g1_router_block_p{zb}_{phe}_{ch}` (x staged in LDS per K-slab of 32 * phe elements, a ring of
/// ch weight elements per lane prefetched before the row wait).
/// ch = 160 (the whole weight row per lane): the full-prefetch phased form (packed d16 loads at entry).
pub const G1_VARIANTS: [(u32, u32, u32); 21] = [
    (1, 20, 0),
    (2, 20, 0),
    (2, 10, 0),
    (4, 10, 0),
    (4, 8, 0),
    (8, 5, 0),
    (8, 4, 0),
    (16, 4, 0),
    (1, 64, 32),
    (2, 64, 32),
    (4, 80, 40),
    (8, 80, 20),
    (8, 40, 20),
    (16, 80, 10),
    (16, 40, 10),
    (1, 160, 32),
    (2, 160, 32),
    (4, 160, 40),
    (8, 160, 20),
    (8, 160, 40),
    (16, 160, 20),
];

/// The symbol of a variant.
pub fn g1_symbol(v: (u32, u32, u32)) -> String {
    if v.2 == 0 {
        format!("g1_router_block_z{}_c{}", v.0, v.1)
    } else {
        format!("g1_router_block_p{}_{}_{}", v.0, v.2, v.1)
    }
}

/// The variant for `b` rows. `V41_DGPU_G1_VARIANT=zb,ch,phe` overrides (bench / tuning).
pub fn g1_variant_for(b: u32) -> (u32, u32, u32) {
    static OVR: std::sync::LazyLock<Option<(u32, u32, u32)>> = std::sync::LazyLock::new(|| {
        let v = std::env::var("V41_DGPU_G1_VARIANT").ok()?;
        let mut it = v.split(',').map(|s| s.trim().parse::<u32>().ok());
        Some((it.next()??, it.next()??, it.next().flatten().unwrap_or(0)))
    });
    if let Some(v) = *OVR {
        if v.0 >= b || (v.0 == 8 && v.2 > 0 && v.1 < 160) {
            return v;
        }
    }
    match b {
        0..=2 => (2, 64, 32),
        3..=4 => (4, 8, 0),
        5..=8 => (8, 40, 20),
        _ => (8, 40, 20),
    }
}

/// Prefetch workgroups per launch (`V41_DGPU_G1_NPF`, default 0 while it is measured).
pub fn g1_n_pf() -> u32 {
    static N: std::sync::LazyLock<u32> =
        std::sync::LazyLock::new(|| std::env::var("V41_DGPU_G1_NPF").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(256));
    *N
}

/// `V41_DGPU_G1_DBG` (bench ablations only; 0 = the real kernel): see the kernel's `G1Args::dbg`.
fn g1_dbg() -> u32 {
    static D: std::sync::LazyLock<u32> =
        std::sync::LazyLock::new(|| std::env::var("V41_DGPU_G1_DBG").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    *D
}

fn sym_ok(v: (u32, u32, u32)) -> bool {
    G1_VARIANTS.contains(&v)
}

fn need<T>(name: &str, buf: &DeviceBuffer<T>, n: usize) -> eyre::Result<u64> {
    if buf.len() < n {
        return Err(eyre!("g1_router_block: {name} has {} elements, needs {n}", buf.len()));
    }
    Ok(buf.raw() as u64)
}

impl G1RouterBlock {
    pub fn for_arch(arch: &str, device_id: i32) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            G1_GFX1201
        } else if arch.starts_with("gfx1151") {
            G1_GFX1151
        } else {
            return Err(eyre!("unsupported arch for g1_router_block: {arch}"));
        };
        let mut sync = DeviceBuffer::<u32>::new(device_id, S_WORDS)?;
        sync.copy_from_host(&[0u32; S_WORDS])?;
        // Kernel iteration aid (benches only): `V41_DGPU_G1_HSACO_<ARCH>=<path>` loads that code object.
        let ovr = std::env::var(format!("V41_DGPU_G1_HSACO_{}", arch.split(':').next().unwrap_or(arch).to_uppercase())).ok().filter(|p| !p.is_empty());
        let module = match ovr {
            Some(p) => Module::load_data(&std::fs::read(&p).map_err(|e| eyre!("g1 hsaco override {p}: {e}"))?)?,
            None => Module::load_data(image)?,
        };
        let ts = if std::env::var_os("V41_DGPU_G1_TS").is_some() { Some(DeviceBuffer::<u64>::new(device_id, 2048 * 16)?) } else { None };
        let scr = DeviceBuffer::<f32>::new(device_id, 2 * 2 * G1_MAX_B as usize * G1_N_EXPERT as usize)?;
        Ok(Self { module, sync, scr, ts })
    }

    /// Debug phase timestamps of the last launch (`V41_DGPU_G1_TS`, 100 MHz steady counter):
    /// `[ticket][8]`, see the kernel's `TS_*` slots. None when the knob is off.
    pub fn ts_dump(&self) -> eyre::Result<Option<Vec<u64>>> {
        match self.ts.as_ref() {
            None => Ok(None),
            Some(t) => {
                let mut v = vec![0u64; t.len()];
                t.copy_to_host(&mut v)?;
                Ok(Some(v))
            }
        }
    }

    /// Workgroups of one launch: `b` collapse WGs, the prior's WG, and 48 router WGs per matrix.
    pub fn grid(b: u32, look: bool, prior: bool) -> u32 {
        b + prior as u32 + g1_n_pf() + (1 + look as u32) * G1_RWG
    }

    /// Launch the whole router block on `stream` (see the module docs: stream-ordered launches only).
    pub fn launch(&self, stream: &Stream, io: G1Io<'_>) -> eyre::Result<()> {
        let v = g1_variant_for(io.b);
        self.launch_variant(stream, io, v)
    }

    pub fn launch_variant(&self, stream: &Stream, io: G1Io<'_>, v: (u32, u32, u32)) -> eyre::Result<()> {
        let zb = v.0;
        let b = io.b;
        if b == 0 || b > G1_MAX_B {
            return Err(eyre!("g1_router_block: b {b} must be in [1, {G1_MAX_B}]"));
        }
        if io.n_used == 0 || io.n_used > G1_MAX_USED {
            return Err(eyre!("g1_router_block: n_used {} must be in [1, {G1_MAX_USED}]", io.n_used));
        }
        // the phased zb = 8 forms (phe > 0, ch < 160) loop over row slices of 8; the others need zb >= b
        if !sym_ok(v) || (zb < b && !(zb == 8 && v.2 > 0 && v.1 < 160)) {
            return Err(eyre!("g1_router_block: no variant {v:?} for b {b}"));
        }
        let ne = crate::config::N_EMBD as usize;
        let nb = b as usize;
        let ns = nb * io.n_used as usize;
        let wbytes = G1_N_EXPERT as usize * ne * 2;
        if io.w0.byte_len() != wbytes {
            return Err(eyre!("g1_router_block: w0 is {} bytes, expected {wbytes}", io.w0.byte_len()));
        }
        let c = &io.collapse;
        let mut a = G1ArgsRaw {
            cx: need("x", c.x, nb * 4 * ne)?,
            carry: need("carry", c.carry, nb * 24)?,
            norm_w: need("norm_w", c.norm_w, ne)?,
            cur_out: need("cur_out", c.cur_out, nb * ne)?,
            norm_out: need("norm_out", c.norm_out, nb * ne)?,
            w0: io.w0.raw() as u64,
            w1: 0,
            logits0: need("logits0", io.logits0, nb * G1_N_EXPERT as usize)?,
            logits1: 0,
            bias0: match io.bias0 {
                Some(x) => need("bias0", x, G1_N_EXPERT as usize)?,
                None => 0,
            },
            bias1: 0,
            prior_host: 0,
            prior_dev: 0,
            sel: need("sel", io.sel, ns)?,
            ew: need("ew", io.ew, ns)?,
            orig_sel: match io.orig_sel.as_ref() {
                Some(x) => need("orig_sel", x, ns)?,
                None => 0,
            },
            range_out: match io.range.as_ref() {
                Some(x) => need("range", x, nb)?,
                None => 0,
            },
            look_sel: 0,
            look_ew: 0,
            xq: match io.xq.as_ref() {
                Some(x) => need("xq", x, nb * XQ_ROW_BYTES)?,
                None => 0,
            },
            pack: 0,
            sel_bank: 0,
            sync: self.sync.raw() as u64,
            scr: self.scr.raw() as u64,
            done_word: 0,
            ts: self.ts.as_ref().map_or(0, |t| t.raw() as u64),
            b,
            n_used: io.n_used,
            n_protect: 0,
            prior_dry: 0,
            bank_off: 0,
            zb,
            n_mat: 1,
            done_value: 0,
            pk_sel: PK_NONE,
            pk_look: PK_NONE,
            pk_look_ew: PK_NONE,
            pk_orig: PK_NONE,
            pk_range: PK_NONE,
            pk_ew: PK_NONE,
            pk_xq: PK_NONE,
            rms_eps: c.rms_eps,
            scale: io.scale,
            weight_eps: io.weight_eps,
            n_pf: g1_n_pf(),
            dbg: g1_dbg(),
        };
        if let Some(l) = io.look.as_ref() {
            if l.w.byte_len() != wbytes {
                return Err(eyre!("g1_router_block: look w is {} bytes, expected {wbytes}", l.w.byte_len()));
            }
            a.w1 = l.w.raw() as u64;
            a.logits1 = need("look logits", l.logits, nb * G1_N_EXPERT as usize)?;
            a.bias1 = match l.bias {
                Some(x) => need("look bias", x, G1_N_EXPERT as usize)?,
                None => 0,
            };
            a.look_sel = need("look sel", l.sel, ns)?;
            a.look_ew = need("look ew", l.ew, ns)?;
            a.n_mat = 2;
        }
        if let Some(p) = io.prior.as_ref() {
            if p.host.len() < p.host_off + G1_N_EXPERT as usize {
                return Err(eyre!("g1_router_block: prior slot {} out of {}", p.host_off, p.host.len()));
            }
            a.prior_host = p.host.device_ptr() as u64 + (p.host_off * 4) as u64;
            a.prior_dev = need("prior dev", p.dev, G1_N_EXPERT as usize)?;
            a.n_protect = p.n_protect;
            a.prior_dry = p.dry as u32;
        }
        if let Some((pk, o)) = io.pack.as_ref() {
            let len = pk.len() as u64;
            let fits = |off: u32, n: usize| off == PK_NONE || (off as u64 + n as u64) <= len;
            let xq_words = nb * XQ_ROW_BYTES / 4;
            if !(fits(o.sel, ns) && fits(o.look, ns) && fits(o.look_ew, ns) && fits(o.orig, ns) && fits(o.range, nb) && fits(o.ew, ns) && fits(o.xq, xq_words)) {
                return Err(eyre!("g1_router_block: pack offsets {o:?} overrun the pack ({len} words)"));
            }
            if (o.look != PK_NONE || o.look_ew != PK_NONE) && io.look.is_none() {
                return Err(eyre!("g1_router_block: look segments packed without a look-ahead"));
            }
            a.pack = pk.device_ptr() as u64;
            (a.pk_sel, a.pk_look, a.pk_look_ew, a.pk_orig, a.pk_range, a.pk_ew, a.pk_xq) = (o.sel, o.look, o.look_ew, o.orig, o.range, o.ew, o.xq);
        }
        if let Some((bank, off)) = io.sel_bank.as_ref() {
            if bank.len() < *off as usize + G1_N_EXPERT as usize {
                return Err(eyre!("g1_router_block: sel bank too small for offset {off}"));
            }
            a.sel_bank = bank.raw() as u64;
            a.bank_off = *off;
        }
        if let Some((w, i, v)) = io.done.as_ref() {
            if *i >= w.len() {
                return Err(eyre!("g1_router_block: done word {i} out of {}", w.len()));
            }
            a.done_word = w.device_ptr() as u64 + (*i * 4) as u64;
            a.done_value = *v;
        }
        let name = g1_symbol(v);
        let function = self.module.get_function(&name)?;
        let cfg = LaunchConfig { grid: (Self::grid(b, io.look.is_some(), io.prior.is_some()), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [a])
    }
}
