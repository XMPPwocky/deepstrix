//! The decode attention BLOCK as one launch (dGPU fusion group G3, `V41_ATTN_G3`):
//! `attn_meta_fill` + `indexer_gather` + `attention_dec_fused_vt_qreg_sp_d4` +
//! `rope_inv_quant_q8` -> `attn_dec_g3` (kernels/attn_dec_g3.hip).
//!
//! BIT-IDENTICAL to that chain's `heads` / `xq` / `xscale` (tests/attn_dec_g3_bitexact.rs) for every
//! row whose selection has no sentinel inside its `n_comp` (a sentinel row reads a zero key here;
//! the chain reads whatever the gather buffer held: undefined, and never produced by consistent
//! decode state). The per-row metadata travels in the kernel arguments, the selected compressed rows
//! are read in place through the top-K selection (no gathered copy), W.V reads V with
//! `global_load_tr_b128` straight into the WMMA B layout (no LDS ring), and the epilogue applies the
//! inverse rope + Q8_0 quantize per head.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

use crate::RopeParams;

const ATTN_DEC_G3_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ATTN_DEC_G3_GFX1201"));
const ATTN_DEC_G3_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ATTN_DEC_G3_GFX1151"));

/// Rows one launch serves (the kernel's `G3_MAX_ROWS`: the metadata arrays in the arguments).
pub const G3_MAX_ROWS: usize = 16;
/// Most keys (`n_raw + n_comp`) per row (the kernel's LDS weight rows; no device-side guard).
pub const G3_MAX_KEYS: u32 = 640;
/// Most compressed keys per row (one row-table entry per thread of the 512-thread WG).
pub const G3_MAX_COMP: u32 = 512;

#[repr(C)]
#[derive(Clone, Copy)]
struct G3Meta {
    n_raw: [i32; G3_MAX_ROWS],
    raw_off: [i32; G3_MAX_ROWS],
    n_comp: [i32; G3_MAX_ROWS],
}

/// Where a row's compressed keys come from.
pub enum G3Comp<'a> {
    /// No compressed keys (ratio-0 layers).
    None,
    /// The top-K selection: comp key `j` of row `b` is store row `base[b] + sel[b * sel_stride + j]`
    /// (`base` None = 0) -- exactly the rows `IndexerGather::launch_batched_rows` copies.
    Selected { store: &'a DeviceBuffer<u16>, sel: &'a DeviceBuffer<i32>, sel_stride: u32, base: Option<&'a DeviceBuffer<i32>> },
    /// Dense (indexer did not fire): key `j` is store row `base[b] + j`, or `b * batch_stride + j`
    /// when `base` is None (the attention kernels' `comp_base_per` / `comp_kv_batch_stride`).
    Dense { store: &'a DeviceBuffer<u16>, base: Option<&'a DeviceBuffer<i32>>, batch_stride: u32 },
}

/// Per-row metadata: the values `attn_meta_fill` would write (n_comp already in its FINAL form,
/// i.e. min(n, TOP_K) when the selection is used).
pub struct G3Rows<'a> {
    pub n_raw: &'a [i32],
    pub raw_off: &'a [i32],
    pub n_comp: &'a [i32],
}

/// What a G3 epilogue writes into `xq` / `xscale` (both kernels, `attn_dec_g3` and `attn_dec_g3o`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G3Epilogue {
    /// The Q8_0 quantize of `heads` (`rope_inv_quant_q8`'s bytes): the default.
    Q8_0,
    /// `V41_ATTN_O_FP8`: the 1017 reference's act_quant of `heads` -- e4m3 codes + one power-of-two
    /// scale per 32 (`rope_inv_quant_fp8`'s bytes), the FP8 `wo_a` input (`crate::fp8_wo_a`).
    Fp8,
}

pub struct AttnDecG3 {
    module: Module,
}

impl AttnDecG3 {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ATTN_DEC_G3_GFX1201
        } else if arch.starts_with("gfx1151") {
            ATTN_DEC_G3_GFX1151
        } else {
            return Err(eyre!("unsupported arch for attn_dec_g3: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// Load another code object with the same symbol (tests: kernel variants).
    pub fn from_image(image: &[u8]) -> eyre::Result<Self> {
        Ok(Self { module: Module::load_data(image)? })
    }

    /// Whether the fused block serves these rows (else the caller keeps the chain).
    pub fn rows_ok(n_raw: &[i32], n_comp: &[i32]) -> bool {
        !n_raw.is_empty()
            && n_raw.len() <= G3_MAX_ROWS
            && n_raw.len() == n_comp.len()
            && n_raw.iter().zip(n_comp).all(|(&r, &c)| {
                r >= 0 && c >= 0 && (c as u32) <= G3_MAX_COMP && (r + c) as u32 <= G3_MAX_KEYS
            })
    }

    /// One launch: `xq` / `xscale` (and `heads_out` when given) receive what the chain's
    /// `rope_inv_quant_q8` leaves in `heads_xq` / `heads_xscale` (/ `heads`). `zero_row`: >= 512
    /// f16 zeros (the row invalid keys read). `xq` / `xscale` / `heads_out` must NOT alias `q`: other
    /// work-groups still read q while one writes its epilogue (production's R2 `heads_xq` lives
    /// inside `q_normed`; the call site uses the R1 `g3_heads_*` views). Grid
    /// (n_head / 16, b) x 512. Requires n_head 64, head_dim 512, n_rot 64, b <= 16 and every row
    /// within [`Self::rows_ok`] (hard errors).
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Stream,
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
    ) -> eyre::Result<()> {
        self.launch_epi(G3Epilogue::Q8_0, stream, heads_out, xq, xscale, zero_row, sinks, q, raw_kv, comp, rows,
                        pos_per_b, n_head, head_dim, n_rot, params)
    }

    /// [`Self::launch`] with the epilogue `epi` (`attn_dec_g3` / `attn_dec_g3_o8`: the same block and
    /// grid; only the bytes written into `xq` / `xscale` differ).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_epi(
        &self,
        epi: G3Epilogue,
        stream: &Stream,
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
    ) -> eyre::Result<()> {
        if n_head != 64 || head_dim != 512 || n_rot != 64 {
            return Err(eyre!("attn_dec_g3: written for 64 x 512 heads, n_rot 64"));
        }
        let b = rows.n_raw.len();
        if b == 0 {
            return Ok(());
        }
        if rows.raw_off.len() != b || !Self::rows_ok(rows.n_raw, rows.n_comp) {
            return Err(eyre!(
                "attn_dec_g3: rows outside the kernel's limits (b <= {G3_MAX_ROWS}, n_comp <= {G3_MAX_COMP}, \
                 n_raw + n_comp <= {G3_MAX_KEYS}): n_raw {:?} n_comp {:?}",
                rows.n_raw, rows.n_comp
            ));
        }
        let n = b * 32768;
        if xq.len() < n || xscale.len() < n / 32 || q.len() < n || pos_per_b.len() < b
            || heads_out.as_ref().is_some_and(|h| h.len() < n)
        {
            return Err(eyre!("attn_dec_g3: buffers too small for b={b}"));
        }
        if zero_row.len() < head_dim as usize || (zero_row.raw() as usize) % 16 != 0 {
            return Err(eyre!("attn_dec_g3: zero_row must be >= {head_dim} 16-B aligned f16 zeros"));
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
                if sel.len() < b * sel_stride as usize
                    || rows.n_comp.iter().any(|&c| c as u32 > sel_stride)
                {
                    return Err(eyre!("attn_dec_g3: selection [{}] too small for b={b} x {sel_stride}", sel.len()));
                }
                (store.raw(), sel.raw(), sel_stride, base.map(|x| x.raw()).unwrap_or(null), 0u32)
            }
            G3Comp::Dense { store, base, batch_stride } => {
                (store.raw(), null, 0u32, base.map(|x| x.raw()).unwrap_or(null), batch_stride)
            }
        };
        let (theta_scale, mscale_eff, corr_low, corr_high) = crate::RopeTail::device_args(params, n_rot);
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let heads_ptr = heads_out.map(|h| h.raw()).unwrap_or(std::ptr::null_mut());
        let function = self.module.get_function(match epi {
            G3Epilogue::Q8_0 => "attn_dec_g3",
            G3Epilogue::Fp8 => "attn_dec_g3_o8",
        })?;
        if epi == G3Epilogue::Fp8 {
            crate::fp8_wo_a::count(crate::fp8_wo_a::Fp8Site::G3);
        }
        let cfg = LaunchConfig { grid: (n_head / 16, b as u32, 1), block: (512, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            heads_ptr, xq.raw(), xscale.raw(), sinks.raw(), q.raw(), raw_kv.raw(),
            comp_ptr, sel_ptr, sel_stride, base_ptr, batch_stride, pos_per_b.raw(),
            theta_scale, params.freq_scale, params.ext_factor, mscale_eff, corr_low, corr_high,
            kq_scale, n_head, zero_row.raw(), meta
        ])
    }
}
