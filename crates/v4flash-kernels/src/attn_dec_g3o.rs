//! "G3-optimal" (round 2 of fusion group G3): the decode attention block (`attn_meta_fill` +
//! `indexer_gather` + `attention_dec_fused_vt_qreg_sp_d4` + `rope_inv_quant_q8`) as one launch
//! (kernels/attn_dec_g3o.hip), BIT-IDENTICAL to that chain and to [`crate::attn_dec_g3`]
//! (tests/attn_g3o.rs).
//!
//! Production: [`G3oMode::Whole`] (`V41_ATTN_G3_KERNEL`, [`auto_mode`]) -- one WG per (row, head
//! tile, 1/parts of the dims) with pipelined loads, a faster phase A and epilogue; `parts` 2 / 4
//! replicate the scores + softmax per dim part to use more WGPs at small b. No scratch.
//!
//! Bench-only (they need the score / sync scratch; measured slower than `attn_dec_g3` from b = 3):
//! the split the round-1 verifier proposed -- S work-groups (row, head-tile group, key chunk) write
//! the scores (and V^T fragments) to a scratch, V work-groups (row, head tile, dim part) run the
//! softmax + W.V -- as two launches ([`G3oMode::TwoLaunch`]) or one ticketed launch
//! ([`G3oMode::Ticketed`]: V work-groups wait only on lower tickets; self-resetting counters).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

use crate::attn_dec_g3::{G3Comp, G3Rows, G3_MAX_COMP, G3_MAX_KEYS, G3_MAX_ROWS};
use crate::RopeParams;

const ATTN_DEC_G3O_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ATTN_DEC_G3O_GFX1201"));
const ATTN_DEC_G3O_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ATTN_DEC_G3O_GFX1151"));

/// The score scratch row stride (f16 per head): the kernel's `SC_STRIDE`.
pub const G3O_SC_STRIDE: usize = 640;
/// f16 per row of the V^T fragment scratch (41 key tiles x 32 dim tiles x 256).
pub const G3O_VT_ROW_HALVES: usize = 41 * 32 * 256;
/// The weights scratch row stride (f16 per head): the kernel's `WSTRIDE`.
pub const G3O_W_STRIDE: usize = 664;
/// u32 words of the sync scratch (zeroed once at allocation; the kernel leaves it zeroed).
pub const G3O_SYNC_WORDS: usize = (3 + 2 * G3_MAX_ROWS * 4) * 32 + 32;
/// Index of the sticky error word (a bounded poll timed out) in the sync scratch.
pub const G3O_SYNC_ERR: usize = 2 * 32;
/// Debug timestamp slots: [slot][16] u64 (two launches: S slots < 1024, V slots 1024..).
pub const G3O_TS_SLOTS: usize = 1024 + G3_MAX_ROWS * 4 * 4;

/// How the block runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G3oMode {
    /// `attn_g3o_s` then `attn_g3o_v`: the launch boundary is the score -> softmax dependency.
    TwoLaunch,
    /// `attn_g3o_t`: one launch, ticketed roles (needs the variant's V WGs at 512 threads).
    Ticketed,
    /// `attn_g3o_w{1,2,4}` (16 warps) / `attn_g3o_x{1,2}` (32 warps): one launch, one WG per (head
    /// tile, row, 1/parts of the dims): every score and the softmax computed per WG (replicated over
    /// the dim parts), no inter-WG traffic.
    Whole { parts: u32, warps: u32 },
    /// Timing experiments only: the two-launch mode's S launch alone / V launch alone (the V launch
    /// reads whatever the scratch holds).
    SOnly,
    VOnly,
}

/// The compiled variant's geometry (`attn_g3o_cfg`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G3oCfg {
    /// head tiles per S work-group
    pub hgp: u32,
    /// key tiles per S work-group
    pub ktp: u32,
    /// S chunks per (row, head-tile group)
    pub nch: u32,
    /// dim parts per (row, head tile)
    pub dp: u32,
    /// warps per V work-group
    pub nwv: u32,
    /// key tiles a row may have (V fragments held per warp)
    pub vpf: u32,
    /// warps per S work-group
    pub nws: u32,
}

/// The shipped kernel's geometry (kernels/attn_dec_g3o.hip defaults; tests/attn_g3o.rs checks it
/// against `attn_g3o_cfg`).
pub const G3O_DEFAULT_CFG: G3oCfg = G3oCfg { hgp: 2, ktp: 16, nch: 3, dp: 2, nwv: 16, vpf: 41, nws: 16 };

#[repr(C)]
#[derive(Clone, Copy)]
struct G3Meta {
    n_raw: [i32; G3_MAX_ROWS],
    raw_off: [i32; G3_MAX_ROWS],
    n_comp: [i32; G3_MAX_ROWS],
}

/// The kernel's `G3OArgs` (same field order and types).
#[repr(C)]
#[derive(Clone, Copy)]
struct G3OArgs {
    heads_out: *mut std::ffi::c_void,
    xq: *mut std::ffi::c_void,
    xscale: *mut std::ffi::c_void,
    sinks: *mut std::ffi::c_void,
    q: *mut std::ffi::c_void,
    raw_kv: *mut std::ffi::c_void,
    comp_kv: *mut std::ffi::c_void,
    sel: *mut std::ffi::c_void,
    comp_base_per: *mut std::ffi::c_void,
    pos_per_b: *mut std::ffi::c_void,
    zero_row: *mut std::ffi::c_void,
    sc: *mut std::ffi::c_void,
    vt: *mut std::ffi::c_void,
    sync: *mut std::ffi::c_void,
    ts: *mut std::ffi::c_void,
    sel_stride: u32,
    comp_kv_batch_stride: u32,
    b: u32,
    theta_scale: f32,
    freq_scale: f32,
    ext_factor: f32,
    mscale_eff: f32,
    corr_low: f32,
    corr_high: f32,
    kq_scale: f32,
    meta: G3Meta,
}

/// The `V41_ATTN_G3_KERNEL=auto` policy (tests/attn_g3o.rs bench_g3o_chain_wall, 10-13, paired vs
/// `attn_dec_g3`, cold / warm): `Whole { 4 }` at b = 1 (-3.4 / -3.2 us), `Whole { 2 }` at b = 2..3
/// (-2.3 / -2.2, -1.3 / -1.0), `Whole { 1 }` from b = 4 when every row holds >= 384 keys (the long
/// selected / index-source layers: -0.4..-1.5 us); `None` (keep `attn_dec_g3`) for short rows at b >= 4
/// (ratio-0 / short dense: W1 +0.3..+1.3 us).
pub fn auto_mode(n_raw: &[i32], n_comp: &[i32]) -> Option<G3oMode> {
    let b = n_raw.len();
    match b {
        0 => None,
        1 => Some(G3oMode::Whole { parts: 4, warps: 16 }),
        2 | 3 => Some(G3oMode::Whole { parts: 2, warps: 16 }),
        _ if n_raw.iter().zip(n_comp).all(|(&r, &c)| r + c >= 384) => Some(G3oMode::Whole { parts: 1, warps: 16 }),
        _ => None,
    }
}

pub struct AttnDecG3o {
    module: Module,
    cfg: G3oCfg,
}

impl AttnDecG3o {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ATTN_DEC_G3O_GFX1201
        } else if arch.starts_with("gfx1151") {
            ATTN_DEC_G3O_GFX1151
        } else {
            return Err(eyre!("unsupported arch for attn_dec_g3o: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)?, cfg: G3O_DEFAULT_CFG })
    }

    /// Load another code object with the same symbols (tests: kernel variants); its geometry is
    /// read from `attn_g3o_cfg` (one tiny launch + sync on `stream`).
    pub fn from_image(image: &[u8], id: i32, stream: &Stream) -> eyre::Result<Self> {
        let module = Module::load_data(image)?;
        let cfg = Self::query_cfg(&module, id, stream)?;
        Ok(Self { module, cfg })
    }

    /// The geometry the code object reports (tests compare it with [`G3O_DEFAULT_CFG`]).
    pub fn query(&self, id: i32, stream: &Stream) -> eyre::Result<G3oCfg> {
        Self::query_cfg(&self.module, id, stream)
    }

    fn query_cfg(module: &Module, id: i32, stream: &Stream) -> eyre::Result<G3oCfg> {
        let out: DeviceBuffer<u32> = DeviceBuffer::new(id, 16)?;
        let f = module.get_function("attn_g3o_cfg")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out.raw()])?;
        stream.synchronize()?;
        let mut v = [0u32; 16];
        out.copy_to_host(&mut v)?;
        if v[6] as usize != G3O_SYNC_WORDS || v[7] as usize != G3O_SC_STRIDE || v[9] as usize != G3O_W_STRIDE {
            return Err(eyre!("attn_dec_g3o: code object scratch layout {:?} != the launcher's", &v[..10]));
        }
        Ok(G3oCfg { hgp: v[0], ktp: v[1], nch: v[2], dp: v[3], nwv: v[4], vpf: v[5], nws: v[8] })
    }

    pub fn cfg(&self) -> G3oCfg {
        self.cfg
    }

    /// f16 elements of the scratch for `b` rows: the scores [b][64][640], then the V^T fragments
    /// [b][41 tiles][32 dim tiles][256] (16-B aligned: the score part is a multiple of 16 B).
    pub fn scratch_halves(b: usize) -> usize {
        b * 64 * G3O_SC_STRIDE + b * G3O_VT_ROW_HALVES
    }

    /// Whether this code object can run `mode`.
    pub fn supports(&self, mode: G3oMode) -> bool {
        match mode {
            G3oMode::TwoLaunch | G3oMode::SOnly | G3oMode::VOnly => true,
            G3oMode::Ticketed => self.cfg.nwv == self.cfg.nws,
            G3oMode::Whole { parts, warps } => matches!((parts, warps), (1 | 2 | 4, 16) | (1 | 2, 32)),
        }
    }

    /// One block: `xq` / `xscale` (and `heads_out` when given) receive what the chain's
    /// `rope_inv_quant_q8` leaves in `heads_xq` / `heads_xscale` (/ `heads`). `sc`: >=
    /// [`Self::scratch_halves`]`(b)` f16 of score scratch (dead after the call). `sync`:
    /// [`G3O_SYNC_WORDS`] u32 zeroed once at allocation (ticketed mode; the kernel leaves it zeroed).
    /// The scratch is dead after the call (every value in it is rewritten before it is read).
    /// `ts`: debug timestamps ([`G3O_TS_SLOTS`] x 8 u64) or None. Same aliasing rule as
    /// [`crate::attn_dec_g3::AttnDecG3::launch`]: the outputs must not alias `q`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Stream,
        mode: G3oMode,
        heads_out: Option<&mut DeviceBuffer<f32>>,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        zero_row: &DeviceBuffer<u16>,
        sinks: &DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp: G3Comp<'_>,
        rows: G3Rows<'_>,
        pos_per_b: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        params: &RopeParams,
        scratch: Option<(&mut DeviceBuffer<u16>, &mut DeviceBuffer<u32>)>,
        ts: Option<&mut DeviceBuffer<u64>>,
    ) -> eyre::Result<()> {
        self.launch_epi(crate::attn_dec_g3::G3Epilogue::Q8_0, stream, mode, heads_out, xq, xscale, zero_row, sinks, q,
                        raw_kv, comp, rows, pos_per_b, n_head, head_dim, n_rot, params, scratch, ts)
    }

    /// [`Self::launch`] with the epilogue `epi`. `G3Epilogue::Fp8` (`V41_ATTN_O_FP8`) exists for the
    /// W family only (`attn_g3o_{w,x}*_o8`; production's `V41_ATTN_G3_KERNEL` picks only W modes);
    /// any other mode is an error.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_epi(
        &self,
        epi: crate::attn_dec_g3::G3Epilogue,
        stream: &Stream,
        mode: G3oMode,
        heads_out: Option<&mut DeviceBuffer<f32>>,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        zero_row: &DeviceBuffer<u16>,
        sinks: &DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp: G3Comp<'_>,
        rows: G3Rows<'_>,
        pos_per_b: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        params: &RopeParams,
        scratch: Option<(&mut DeviceBuffer<u16>, &mut DeviceBuffer<u32>)>,
        ts: Option<&mut DeviceBuffer<u64>>,
    ) -> eyre::Result<()> {
        if n_head != 64 || head_dim != 512 || n_rot != 64 {
            return Err(eyre!("attn_dec_g3o: written for 64 x 512 heads, n_rot 64"));
        }
        let o8 = epi == crate::attn_dec_g3::G3Epilogue::Fp8;
        if o8 && !matches!(mode, G3oMode::Whole { .. }) {
            return Err(eyre!("attn_dec_g3o: the FP8 (act_quant) epilogue exists for the W family only, not {mode:?}"));
        }
        let b = rows.n_raw.len();
        if b == 0 {
            return Ok(());
        }
        if !self.supports(mode) {
            return Err(eyre!("attn_dec_g3o: this code object (S {} / V {} warps) has no ticketed entry", self.cfg.nws, self.cfg.nwv));
        }
        if rows.raw_off.len() != b || !crate::attn_dec_g3::AttnDecG3::rows_ok(rows.n_raw, rows.n_comp) {
            return Err(eyre!(
                "attn_dec_g3o: rows outside the kernel's limits (b <= {G3_MAX_ROWS}, n_comp <= {G3_MAX_COMP}, \
                 n_raw + n_comp <= {G3_MAX_KEYS}): n_raw {:?} n_comp {:?}",
                rows.n_raw, rows.n_comp
            ));
        }
        let n = b * 32768;
        if xq.len() < n || xscale.len() < n / 32 || q.len() < n || pos_per_b.len() < b
            || heads_out.as_ref().is_some_and(|h| h.len() < n)
        {
            return Err(eyre!("attn_dec_g3o: buffers too small for b={b}"));
        }
        let split = !matches!(mode, G3oMode::Whole { .. });
        let (sc_ptr, vt_ptr, sync_ptr) = match scratch {
            Some((sc, sync)) => {
                if sc.len() < Self::scratch_halves(b) || sync.len() < G3O_SYNC_WORDS {
                    return Err(eyre!("attn_dec_g3o: score / sync scratch too small for b={b}"));
                }
                (sc.raw(), (sc.raw() as *mut u16).wrapping_add(b * 64 * G3O_SC_STRIDE) as *mut std::ffi::c_void, sync.raw())
            }
            None if split => return Err(eyre!("attn_dec_g3o: mode {mode:?} needs the score / sync scratch")),
            None => (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()),
        };
        if ts.as_ref().is_some_and(|t| t.len() < G3O_TS_SLOTS * 16) {
            return Err(eyre!("attn_dec_g3o: ts too small"));
        }
        if zero_row.len() < head_dim as usize || (zero_row.raw() as usize) % 16 != 0 {
            return Err(eyre!("attn_dec_g3o: zero_row must be >= {head_dim} 16-B aligned f16 zeros"));
        }
        let mut meta = G3Meta { n_raw: [0; G3_MAX_ROWS], raw_off: [0; G3_MAX_ROWS], n_comp: [0; G3_MAX_ROWS] };
        meta.n_raw[..b].copy_from_slice(rows.n_raw);
        meta.raw_off[..b].copy_from_slice(rows.raw_off);
        meta.n_comp[..b].copy_from_slice(rows.n_comp);
        let null = std::ptr::null_mut::<std::ffi::c_void>();
        let (comp_ptr, sel_ptr, sel_stride, base_ptr, batch_stride) = match comp {
            G3Comp::None => {
                meta.n_comp = [0; G3_MAX_ROWS];
                (null, null, 0u32, null, 0u32)
            }
            G3Comp::Selected { store, sel, sel_stride, base } => {
                if sel.len() < b * sel_stride as usize || rows.n_comp.iter().any(|&c| c as u32 > sel_stride) {
                    return Err(eyre!("attn_dec_g3o: selection [{}] too small for b={b} x {sel_stride}", sel.len()));
                }
                (store.raw(), sel.raw(), sel_stride, base.map(|x| x.raw()).unwrap_or(null), 0u32)
            }
            G3Comp::Dense { store, base, batch_stride } => {
                (store.raw(), null, 0u32, base.map(|x| x.raw()).unwrap_or(null), batch_stride)
            }
        };
        let (theta_scale, mscale_eff, corr_low, corr_high) = crate::RopeTail::device_args(params, n_rot);
        let args = G3OArgs {
            heads_out: heads_out.map(|h| h.raw()).unwrap_or(null),
            xq: xq.raw(),
            xscale: xscale.raw(),
            sinks: sinks.raw(),
            q: q.raw(),
            raw_kv: raw_kv.raw(),
            comp_kv: comp_ptr,
            sel: sel_ptr,
            comp_base_per: base_ptr,
            pos_per_b: pos_per_b.raw(),
            zero_row: zero_row.raw(),
            sc: sc_ptr,
            vt: vt_ptr,
            sync: sync_ptr,
            ts: ts.map(|t| t.raw()).unwrap_or(null),
            sel_stride,
            comp_kv_batch_stride: batch_stride,
            b: b as u32,
            theta_scale,
            freq_scale: params.freq_scale,
            ext_factor: params.ext_factor,
            mscale_eff,
            corr_low,
            corr_high,
            kq_scale: 1.0f32 / (head_dim as f32).sqrt(),
            meta,
        };
        let c = self.cfg;
        let b32 = b as u32;
        match mode {
            G3oMode::TwoLaunch | G3oMode::SOnly | G3oMode::VOnly => {
                if mode != G3oMode::VOnly {
                    let fs = self.module.get_function("attn_g3o_s")?;
                    let cs = LaunchConfig { grid: (c.nch, 4 / c.hgp, b32), block: (c.nws * 32, 1, 1), shared_mem_bytes: 0 };
                    launch_kernel!(fs, cs, stream, [args])?;
                }
                if mode != G3oMode::SOnly {
                    let fv = self.module.get_function("attn_g3o_v")?;
                    let cv = LaunchConfig { grid: (c.dp, 4, b32), block: (c.nwv * 32, 1, 1), shared_mem_bytes: 0 };
                    launch_kernel!(fv, cv, stream, [args])?;
                }
                Ok(())
            }
            G3oMode::Ticketed => {
                let ns = b32 * (4 / c.hgp) * c.nch;
                let nv = b32 * 4 * c.dp;
                let ft = self.module.get_function("attn_g3o_t")?;
                let ct = LaunchConfig { grid: (ns + nv, 1, 1), block: (c.nws * 32, 1, 1), shared_mem_bytes: 0 };
                launch_kernel!(ft, ct, stream, [args])
            }
            G3oMode::Whole { parts, warps } => {
                let fw = self.module.get_function(match (parts, warps, o8) {
                    (1, 16, false) => "attn_g3o_w1",
                    (2, 16, false) => "attn_g3o_w2",
                    (4, 16, false) => "attn_g3o_w4",
                    (1, 32, false) => "attn_g3o_x1",
                    (_, _, false) => "attn_g3o_x2",
                    (1, 16, true) => "attn_g3o_w1_o8",
                    (2, 16, true) => "attn_g3o_w2_o8",
                    (4, 16, true) => "attn_g3o_w4_o8",
                    (1, 32, true) => "attn_g3o_x1_o8",
                    (_, _, true) => "attn_g3o_x2_o8",
                })?;
                let cw = LaunchConfig { grid: (4, b32, parts), block: (warps * 32, 1, 1), shared_mem_bytes: 0 };
                launch_kernel!(fw, cw, stream, [args])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_mode_policy() {
        let w = |p| Some(G3oMode::Whole { parts: p, warps: 16 });
        assert_eq!(auto_mode(&[], &[]), None);
        assert_eq!(auto_mode(&[128], &[512]), w(4));
        assert_eq!(auto_mode(&[128], &[0]), w(4));
        assert_eq!(auto_mode(&[128, 128], &[0, 0]), w(2));
        assert_eq!(auto_mode(&[128; 3], &[512; 3]), w(2));
        assert_eq!(auto_mode(&[128; 4], &[512; 4]), w(1));
        assert_eq!(auto_mode(&[128; 16], &[512; 16]), w(1));
        assert_eq!(auto_mode(&[128; 4], &[0; 4]), None);
        assert_eq!(auto_mode(&[128, 128, 128, 100], &[512, 512, 512, 200]), None);
    }

    #[test]
    fn layout_constants_match_the_kernel() {
        // kernels/attn_dec_g3o.hip: S_WORDS = S_VCNT(64) = (3 + 128) * 32, + 32 for the error word line
        assert_eq!(G3O_SYNC_WORDS, (3 + 128) * 32 + 32);
        assert_eq!(G3O_VT_ROW_HALVES, 41 * 32 * 256);
        assert_eq!(AttnDecG3o::scratch_halves(2), 2 * 64 * 640 + 2 * 41 * 32 * 256);
    }
}
