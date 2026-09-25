//! Router readback pack (kernels/readback_pack.hip): the host-bound router
//! outputs gathered by one kernel into one host-mapped pinned buffer, on the
//! compute stream before the router-done event. The event wait then IS the
//! readback -- no copies and no second stream (tests/readback_pack.rs,
//! tests/bench_router_readback_ab.rs).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, PinnedBuffer, Stream};

const READBACK_PACK_GFX1201: &[u8] = include_bytes!(env!("KERNEL_READBACK_PACK_GFX1201"));
const READBACK_PACK_GFX1151: &[u8] = include_bytes!(env!("KERNEL_READBACK_PACK_GFX1151"));

/// Segments one launch can gather (the kernel's `s0..s9`).
pub const RB_PACK_MAX_SEG: usize = 10;

/// One source segment: the first `words` 32-bit words of a device buffer,
/// borrowed for `'a` so a segment cannot outlive its buffer.
#[derive(Clone, Copy, Debug)]
pub struct PackSeg<'a> {
    ptr: v4flash_hip::sys::hipDeviceptr_t,
    words: u32,
    _buf: std::marker::PhantomData<&'a ()>,
}

impl<'a> PackSeg<'a> {
    /// The first `n` elements of a buffer of 4-byte elements (i32 / f32 / u32).
    pub fn words<T>(buf: &'a DeviceBuffer<T>, n: usize) -> eyre::Result<Self> {
        if std::mem::size_of::<T>() != 4 {
            return Err(eyre!("PackSeg::words: element size {} != 4", std::mem::size_of::<T>()));
        }
        if n > buf.len() {
            return Err(eyre!("PackSeg::words: {n} > buffer len {}", buf.len()));
        }
        Ok(Self { ptr: buf.raw(), words: n as u32, _buf: std::marker::PhantomData })
    }

    /// The first `n` bytes of a byte buffer; `n` must be a multiple of 4.
    pub fn bytes(buf: &'a DeviceBuffer<u8>, n: usize) -> eyre::Result<Self> {
        if n % 4 != 0 || n > buf.len() {
            return Err(eyre!("PackSeg::bytes: {n} bytes (buffer {}), need a multiple of 4 within it", buf.len()));
        }
        Ok(Self { ptr: buf.raw(), words: (n / 4) as u32, _buf: std::marker::PhantomData })
    }

    pub fn words_len(&self) -> u32 {
        self.words
    }
}

/// The segments of one launch, in order, with where each lands.
#[derive(Default)]
pub struct PackPlan<'a> {
    segs: Vec<PackSeg<'a>>,
    words: u32,
}

impl<'a> PackPlan<'a> {
    /// Append `sg`; returns its `(word offset, words)` in the destination.
    pub fn push(&mut self, sg: PackSeg<'a>) -> (u32, u32) {
        let at = (self.words, sg.words);
        self.words += sg.words;
        self.segs.push(sg);
        at
    }

    pub fn segs(&self) -> &[PackSeg<'a>] {
        &self.segs
    }
}

pub struct ReadbackPack {
    module: Module,
}

impl ReadbackPack {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            READBACK_PACK_GFX1201
        } else if arch.starts_with("gfx1151") {
            READBACK_PACK_GFX1151
        } else {
            return Err(eyre!("unsupported arch for readback_pack: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// Gather `segs` back to back into `dst` words `[0, sum of words)`, in
    /// order. The words are host-visible once a later event on `stream`
    /// (recorded without `hipEventDisableSystemFence`) has completed.
    pub fn launch(&self, stream: &Stream, dst: &mut PinnedBuffer<u32>, segs: &[PackSeg<'_>]) -> eyre::Result<()> {
        if segs.len() > RB_PACK_MAX_SEG {
            return Err(eyre!("readback_pack: {} segments > {RB_PACK_MAX_SEG}", segs.len()));
        }
        let mut src = [std::ptr::null_mut(); RB_PACK_MAX_SEG];
        let mut end = [0u32; RB_PACK_MAX_SEG];
        let mut at = 0u64;
        for k in 0..RB_PACK_MAX_SEG {
            if let Some(s) = segs.get(k) {
                src[k] = s.ptr;
                at += s.words as u64;
            }
            end[k] = at as u32;
        }
        if at > dst.len() as u64 {
            return Err(eyre!("readback_pack: {at} words > staging {}", dst.len()));
        }
        if at == 0 {
            return Ok(());
        }
        let function = self.module.get_function("readback_pack")?;
        let cfg = LaunchConfig { grid: (at.div_ceil(256) as u32, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            dst.device_ptr(),
            src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9],
            end[0], end[1], end[2], end[3], end[4], end[5], end[6], end[7], end[8], end[9]
        ])
    }
}
