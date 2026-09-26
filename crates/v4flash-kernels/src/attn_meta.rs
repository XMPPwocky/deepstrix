//! Attention per-row metadata upload as one kernel launch
//! (kernels/attn_meta.hip): up to 4 int32 arrays of <= `ATTN_META_MAX_ROWS`
//! entries written from the kernel ARGUMENTS, replacing up to five pageable
//! `hipMemcpyAsync` H2D copies per lane-layer (~4.1 us of device timeline each
//! at decode shapes, tests/bench_decode_latency_breakdown.rs). The device bytes
//! are exactly what the copies wrote.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const ATTN_META_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ATTN_META_GFX1201"));
const ATTN_META_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ATTN_META_GFX1151"));

/// Rows one launch can carry (the kernel's `AMF_MAX_ROWS`).
pub const ATTN_META_MAX_ROWS: usize = 16;
const N_DST: usize = 4;

#[repr(C)]
#[derive(Clone, Copy)]
struct AmfArgs {
    v: [[i32; ATTN_META_MAX_ROWS]; N_DST],
}

/// One destination: the first `vals.len()` 4-byte elements of `buf` get `vals`
/// (bit patterns; u32 buffers take the `as i32` of their values).
pub struct MetaDst<'a> {
    ptr: v4flash_hip::sys::hipDeviceptr_t,
    vals: &'a [i32],
}

impl<'a> MetaDst<'a> {
    pub fn new<T>(buf: &'a mut DeviceBuffer<T>, vals: &'a [i32]) -> eyre::Result<Self> {
        if std::mem::size_of::<T>() != 4 {
            return Err(eyre!("attn_meta: element size {} != 4", std::mem::size_of::<T>()));
        }
        if vals.len() > buf.len() {
            return Err(eyre!("attn_meta: {} values > buffer len {}", vals.len(), buf.len()));
        }
        Ok(Self { ptr: buf.raw(), vals })
    }
}

pub struct AttnMetaFill {
    module: Module,
}

impl AttnMetaFill {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ATTN_META_GFX1201
        } else if arch.starts_with("gfx1151") {
            ATTN_META_GFX1151
        } else {
            return Err(eyre!("unsupported arch for attn_meta: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// Write every destination's values with ONE launch on `stream` (stream
    /// order, like the async copies it replaces). All destinations must carry
    /// the same number of values, `1..=ATTN_META_MAX_ROWS`.
    pub fn launch(&self, stream: &Stream, dsts: [Option<MetaDst<'_>>; N_DST]) -> eyre::Result<()> {
        let mut args = AmfArgs { v: [[0; ATTN_META_MAX_ROWS]; N_DST] };
        let mut ptrs = [std::ptr::null_mut(); N_DST];
        let mut n: Option<usize> = None;
        for (k, d) in dsts.iter().enumerate() {
            if let Some(d) = d {
                let len = d.vals.len();
                if len == 0 || len > ATTN_META_MAX_ROWS || n.is_some_and(|m| m != len) {
                    return Err(eyre!("attn_meta: destination {k} has {len} values (need 1..={ATTN_META_MAX_ROWS}, all equal)"));
                }
                n = Some(len);
                args.v[k][..len].copy_from_slice(d.vals);
                ptrs[k] = d.ptr;
            }
        }
        let n = n.ok_or_else(|| eyre!("attn_meta: no destination"))? as u32;
        let function = self.module.get_function("attn_meta_fill")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [ptrs[0], ptrs[1], ptrs[2], ptrs[3], n, args])
    }
}
